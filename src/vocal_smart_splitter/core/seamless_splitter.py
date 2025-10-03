#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/seamless_splitter.py
# AI-SUMMARY: 无缝分割主流程，调度分离、停顿检测与切点精炼并桥接轨道特征缓存。

import os
import numpy as np
import librosa
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import time
import tempfile
import soundfile as sf

from audio_cut.analysis import TrackFeatureCache, build_feature_cache
from audio_cut.utils.gpu_pipeline import PipelineConfig, PipelineContext, build_pipeline_context
from audio_cut.cutting.refine import CutContext, CutPoint, finalize_cut_points

from ..utils.config_manager import get_config
from ..utils.audio_processor import AudioProcessor
from .pure_vocal_pause_detector import PureVocalPauseDetector
from .quality_controller import QualityController
from .enhanced_vocal_separator import EnhancedVocalSeparator

logger = logging.getLogger(__name__)

class SeamlessSplitter:
    """
    [v2.3 统一指挥中心]
    无缝分割器 - 负责编排所有分割模式的唯一引擎。
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate)
        self.pure_vocal_detector = PureVocalPauseDetector(sample_rate)  # 用于v2.2模式
        self.quality_controller = QualityController(sample_rate)
        self.separator = EnhancedVocalSeparator(sample_rate)
        self._last_segment_classification_debug: List[Dict[str, Any]] = []
        self._last_guard_shift_stats: Dict[str, float] = self._blank_guard_stats()
        logger.info(f"无缝分割器统一指挥中心初始化完成 (SR: {self.sample_rate}) - 已加载纯人声检测器")

    def split_audio_seamlessly(self, input_path: str, output_dir: str, mode: str = 'v2.2_mdd') -> Dict:
        """执行无缝分割的主入口，支持纯人声分离与 v2.2 MDD 模式"""
        logger.info(f"开始无缝分割: {input_path} (模式: {mode})")

        try:
            if mode == 'vocal_separation':
                return self._process_vocal_separation_only(input_path, output_dir)
            if mode != 'v2.2_mdd':
                logger.warning(f"未知模式 {mode}，使用默认 v2.2 MDD 模式")
            return self._process_pure_vocal_split(input_path, output_dir, 'v2.2_mdd')

        except Exception as e:
            logger.error(f"无缝分割失败: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'input_file': input_path}
    def _load_and_resample_if_needed(self, input_path: str):
        """加载音频并根据需要重采样"""
        original_audio, sr = self.audio_processor.load_audio(input_path, normalize=False)
        if sr != self.sample_rate:
            logger.info(f"音频采样率 {sr}Hz 与目标 {self.sample_rate}Hz 不符，将进行重采样。")
            original_audio = librosa.resample(original_audio, orig_sr=sr, target_sr=self.sample_rate)
        return original_audio

    def _process_pure_vocal_split(self, input_path: str, output_dir: str, mode: str) -> Dict:
        """
        [v2.9 终极修正版] 处理v2.2 MDD模式的核心逻辑
        关键修复: 调用新的双音频输入接口，确保音乐分析在 original_audio 上，而VAD检测在 vocal_track 上。
        """
        logger.info(f"[{mode.upper()}] 执行纯人声分割流程...")
        overall_start_time = time.time()

        # 1. 加载音频
        original_audio = self._load_and_resample_if_needed(input_path)

        gpu_cfg = self._get_gpu_pipeline_config()
        gpu_context: Optional[PipelineContext] = None
        if gpu_cfg.enable:
            try:
                gpu_context = self._build_gpu_pipeline_context(original_audio, gpu_cfg)
            except Exception as exc:
                gpu_context = None
                logger.warning("[GPU Pipeline] 初始化失败，回退 CPU: %s", exc, exc_info=True)

        if gpu_context and gpu_context.enabled:
            logger.info(
                "[GPU Pipeline] enabled: device=%s, chunk_count=%d",
                gpu_context.device,
                len(gpu_context.plans),
            )
        else:
            logger.info("[GPU Pipeline] disabled; continue with CPU path.")

        # 2. 高质量人声分离
        logger.info(f"[{mode.upper()}-STEP1] 执行高质量人声分离...")
        separation_start = time.time()
        separation_result = self.separator.separate_for_detection(
            original_audio,
            gpu_context=gpu_context,
        )
        separation_time = time.time() - separation_start

        if separation_result.vocal_track is None:
            gpu_meta = dict(separation_result.gpu_meta or {})
            failure = {
                'success': False,
                'error': '��������ʧ��',
                'input_file': input_path,
            }
            failure.update(gpu_meta)
            return failure

        vocal_track = separation_result.vocal_track
        logger.info(f"[{mode.upper()}-STEP1] 人声分离完成 - 后端: {separation_result.backend_used}, 质量: {separation_result.separation_confidence:.3f}, 耗时: {separation_time:.1f}s")

        gpu_meta = dict(separation_result.gpu_meta or {})
        if gpu_meta.get('gpu_pipeline_used'):
            logger.info(
                "[GPU Pipeline] used device=%s chunks=%s",
                gpu_meta.get('gpu_pipeline_device'),
                gpu_meta.get('gpu_pipeline_chunks'),
            )
        elif gpu_meta.get('gpu_pipeline_enabled'):
            logger.info(
                "[GPU Pipeline] not used, reason=%s",
                gpu_meta.get('fallback_reason', gpu_meta.get('gpu_pipeline_failures', 'n/a')),
            )

        # 3. 关键修复：使用正确的PureVocalPauseDetector进行停顿检测
        logger.info(f"[{mode.upper()}-STEP2] 使用PureVocalPauseDetector在[纯人声轨道]上进行多维特征检测...")

        # 检查是否为v2.2 MDD模式
        enable_mdd = (mode == 'v2.2_mdd')
        feature_cache: Optional[TrackFeatureCache] = separation_result.feature_cache
        if feature_cache is None:
            try:
                feature_cache = build_feature_cache(original_audio, vocal_track, self.sample_rate)
                logger.debug(
                    "Feature cache built (fallback): frames=%d, global_mdd=%.4f",
                    feature_cache.frame_count(),
                    feature_cache.global_mdd,
                )
            except Exception as exc:
                logger.warning("Failed to build feature cache: %s", exc)
                feature_cache = None

        vocal_pauses = self.pure_vocal_detector.detect_pure_vocal_pauses(
            vocal_track,
            enable_mdd_enhancement=enable_mdd,
            original_audio=original_audio,
            feature_cache=feature_cache,
            vad_segments=separation_result.vad_segments,
        )

        if not vocal_pauses:
            has_vocal = self._estimate_vocal_presence(vocal_track)
            return self._create_single_segment_result(
                original_audio,
                input_path,
                output_dir,
                "δ�ڴ��������ҵ�ͣ��",
                is_vocal=has_vocal,
                gpu_meta=gpu_meta,
            )

        cut_points_samples: List[int] = []
        for pause in vocal_pauses:
            if hasattr(pause, 'cut_point'):
                cut_points_samples.append(int(pause.cut_point * self.sample_rate))
            else:
                center_time = (pause.start_time + pause.end_time) / 2
                cut_points_samples.append(int(center_time * self.sample_rate))

        logger.info(f"[{mode.upper()}-STEP3] 生成{len(cut_points_samples)}个候选分割点")
        # 将样本点转换为(time, score)形式，暂以score=confidence占位（若有）
        cut_candidates = []
        markers = getattr(separation_result, 'quality_metrics', {}) or {}
        marker_times = []
        try:
            marker_times = [float(t) for t in markers.get('vocal_presence_cut_points_sec', []) if t is not None]
        except Exception:
            marker_times = []
        protected_marker_samples = set()
        for p in vocal_pauses:
            t = float(getattr(p, 'cut_point', (p.start_time + p.end_time)/2))
            s = float(getattr(p, 'confidence', 1.0))
            # 过滤疑似间奏的超长静默
            dur = float(getattr(p, 'duration', (p.end_time - p.start_time)))
            interlude_min_s = get_config('pure_vocal_detection.pause_stats_adaptation.interlude_min_s', 4.0)
            if False and dur >= interlude_min_s:
                continue
            cut_candidates.append((t, s))


        # 纯音乐（无人声）长区间 → 加入边界（极简方案：只看无人声时长阈值）
        try:
            min_pure_music = float(get_config('quality_control.pure_music_min_duration', 0.0))
        except Exception:
            min_pure_music = 0.0
        if min_pure_music > 0.0:
            spans = self._find_no_vocal_runs(vocal_track, min_pure_music)
            if spans:
                logger.info(f"[{mode.upper()}-STEP3] 纯音乐无人声区间: {len(spans)} 段满足 >= {min_pure_music:.2f}s")
            else:
                logger.info(f"[{mode.upper()}-STEP3] 未发现满足 >= {min_pure_music:.2f}s 的无人声区间，自动跳过")
            for a, b in spans:
                cut_candidates.append((float(a), 1.0))
                cut_candidates.append((float(b), 1.0))
        else:
            logger.info(f"[{mode.upper()}-STEP3] 纯音乐无人声检测未启用 (quality_control.pure_music_min_duration<=0)")
        audio_duration = len(original_audio) / self.sample_rate
        for t in marker_times:
            if t <= 0.0 or t >= audio_duration:
                continue
            cut_candidates.append((t, 1.0))
            protected_marker_samples.add(int(round(t * self.sample_rate)))

        final_cut_points = self._finalize_and_filter_cuts_v2(
            cut_candidates,
            original_audio,
            pure_vocal_audio=vocal_track
        )

        if protected_marker_samples:
            total_samples = len(original_audio)
            augmented = set(int(point) for point in final_cut_points)
            for sample_idx in protected_marker_samples:
                sample_idx = int(min(max(sample_idx, 0), total_samples))
                if sample_idx not in (0, total_samples):
                    augmented.add(sample_idx)
            final_cut_points = sorted(augmented)

        instrumental_audio = separation_result.instrumental_track
        if isinstance(instrumental_audio, np.ndarray):
            if len(instrumental_audio) != len(vocal_track):
                min_len = min(len(instrumental_audio), len(vocal_track))
                if min_len > 0:
                    instrumental_audio = instrumental_audio[:min_len]
                    if len(vocal_track) > min_len:
                        instrumental_audio = np.pad(instrumental_audio, (0, len(vocal_track) - min_len))
                else:
                    instrumental_audio = None
        else:
            instrumental_audio = None
        segment_vocal_flags = self._classify_segments_vocal_presence(
            vocal_track,
            final_cut_points,
            marker_segments=markers.get('vocal_presence_segments'),
            pure_music_segments=markers.get('pure_music_segments'),
            instrumental_audio=instrumental_audio,
            original_audio=original_audio
        )
        classification_debug = list(getattr(self, '_last_segment_classification_debug', []))
        segments, segment_vocal_flags, merged_debug = self._split_at_sample_level(
            original_audio,
            final_cut_points,
            segment_flags=segment_vocal_flags,
            debug_entries=classification_debug,
        )
        segment_durations = [len(seg) / float(self.sample_rate) for seg in segments]
        if merged_debug is not None:
            classification_debug = merged_debug
            self._last_segment_classification_debug = merged_debug
        else:
            self._last_segment_classification_debug = classification_debug

        mix_segment_files = self._save_segments(segments, output_dir, segment_is_vocal=segment_vocal_flags)

        vocal_segments, _, _ = self._split_at_sample_level(vocal_track, final_cut_points)
        vocal_segment_files = self._save_segments(
            vocal_segments,
            output_dir,
            segment_is_vocal=segment_vocal_flags,
            subdir='segments_vocal',
            file_suffix='_vocal'
        )

        saved_files = mix_segment_files + vocal_segment_files

        # 5. 保存完整的分离文件
        input_name = Path(input_path).stem
        full_vocal_file = Path(output_dir) / f"{input_name}_{mode}_vocal_full.wav"
        sf.write(full_vocal_file, vocal_track, self.sample_rate, subtype='PCM_24')
        saved_files.append(str(full_vocal_file))

        if separation_result.instrumental_track is not None:
            full_instrumental_file = Path(output_dir) / f"{input_name}_{mode}_instrumental.wav"
            sf.write(full_instrumental_file, separation_result.instrumental_track, self.sample_rate, subtype='PCM_24')
            saved_files.append(str(full_instrumental_file))

        total_time = time.time() - overall_start_time

        result = {
            'success': True, 'method': f'pure_vocal_split_{mode}', 'num_segments': len(segments),
            'saved_files': saved_files, 'mix_segment_files': mix_segment_files, 'vocal_segment_files': vocal_segment_files,
            'backend_used': separation_result.backend_used,
            'separation_confidence': separation_result.separation_confidence, 'processing_time': total_time,
            'segment_durations': segment_durations,
            'segment_vocal_flags': segment_vocal_flags,
            'segment_labels': ['human' if flag else 'music' for flag in segment_vocal_flags],
            'segment_classification_debug': getattr(self, '_last_segment_classification_debug', []),
            'guard_shift_stats': self._get_guard_shift_stats(),
            'input_file': input_path, 'output_dir': output_dir
        }
        result.update(gpu_meta)
        return result

    def _get_gpu_pipeline_config(self) -> PipelineConfig:
        try:
            mapping = get_config('gpu_pipeline', {})
        except Exception:
            mapping = {}
        if not isinstance(mapping, dict):
            mapping = {}
        return PipelineConfig.from_mapping(mapping)

    def _build_gpu_pipeline_context(self, audio: np.ndarray, cfg: PipelineConfig) -> Optional[PipelineContext]:
        if not cfg.enable or self.sample_rate <= 0:
            return None
        duration_s = float(audio.shape[-1]) / float(self.sample_rate)
        ctx = build_pipeline_context(duration_s, cfg)
        if not ctx.enabled:
            return None
        return ctx

    def _process_vocal_separation_only(self, input_path: str, output_dir: str) -> Dict:
        """处理纯人声分离模式"""
        logger.info("[VOCAL_SEPARATION] 执行纯人声分离...")
        start_time = time.time()
        original_audio = self._load_and_resample_if_needed(input_path)
        separation_result = self.separator.separate_for_detection(original_audio)

        if separation_result.vocal_track is None:
            return {'success': False, 'error': '人声分离失败', 'input_file': input_path}

        input_name = Path(input_path).stem
        saved_files = []
        vocal_file = Path(output_dir) / f"{input_name}_vocal.wav"
        sf.write(vocal_file, separation_result.vocal_track, self.sample_rate, subtype='PCM_24')
        saved_files.append(str(vocal_file))

        if separation_result.instrumental_track is not None:
            instrumental_file = Path(output_dir) / f"{input_name}_instrumental.wav"
            sf.write(instrumental_file, separation_result.instrumental_track, self.sample_rate, subtype='PCM_24')
            saved_files.append(str(instrumental_file))

        processing_time = time.time() - start_time

        return {
            'success': True, 'method': 'vocal_separation_only', 'num_segments': 0, 'saved_files': saved_files,
            'backend_used': separation_result.backend_used, 'separation_confidence': separation_result.separation_confidence,
            'processing_time': processing_time, 'segment_durations': [],
            'guard_shift_stats': self._get_guard_shift_stats(), 'input_file': input_path, 'output_dir': output_dir
        } | dict(separation_result.gpu_meta or {})

    def _find_no_vocal_runs(self, vocal_audio: np.ndarray, min_duration: float):
        """
        基于人声轨的能量包络构造 voice_active 掩码，提取连续“无人声”区间（秒）。
        修复：改用鲁棒阈值（噪底与人声分布中值之间的中点）+ 轻度形态学，避免阈值过低导致全程“活跃”。
        """
        sr = self.sample_rate
        hop = max(1, int(0.01 * sr))  # 10ms
        # RMS 包络 → dB
        rms = librosa.feature.rms(y=vocal_audio, hop_length=hop)[0]
        db = 20.0 * np.log10(rms + 1e-12)
        # 鲁棒阈值：在噪声地板与人声上分位之间取中点
        try:
            noise_pct = float(get_config('quality_control.enforce_quiet_cut.floor_percentile', 10))
        except Exception:
            noise_pct = 10.0
        try:
            voice_pct = float(get_config('pure_vocal_detection.pause_stats_adaptation.voice_percentile_hint', 90))
        except Exception:
            voice_pct = 90.0
        noise_db = float(np.percentile(db, np.clip(noise_pct, 0, 50)))
        voice_db = float(np.percentile(db, np.clip(voice_pct, 50, 100)))
        delta_db = float(get_config('pure_vocal_detection.pause_stats_adaptation.delta_db', 3.0))
        thr_mid = 0.5 * (noise_db + voice_db)
        thr_db = max(noise_db + delta_db, thr_mid)
        active = db > thr_db
        # 轻度形态学：闭后开（毫秒 → 帧）
        close_ms = int(get_config('pure_vocal_detection.pause_stats_adaptation.morph_close_ms', 150))
        open_ms = int(get_config('pure_vocal_detection.pause_stats_adaptation.morph_open_ms', 50))
        frame_sec = hop / float(sr)
        close_k = max(1, int(close_ms / 1000.0 / frame_sec))
        open_k = max(1, int(open_ms / 1000.0 / frame_sec))
        def fill_false_runs(m: np.ndarray, max_len: int) -> np.ndarray:
            m = m.astype(bool).copy(); n = len(m); i = 0
            while i < n:
                if not m[i]:
                    j = i
                    while j < n and not m[j]:
                        j += 1
                    if (j - i) <= max_len:
                        m[i:j] = True
                    i = j
                else:
                    i += 1
            return m
        def remove_true_runs(m: np.ndarray, max_len: int) -> np.ndarray:
            m = m.astype(bool).copy(); n = len(m); i = 0
            while i < n:
                if m[i]:
                    j = i
                    while j < n and m[j]:
                        j += 1
                    if (j - i) <= max_len:
                        m[i:j] = False
                    i = j
                else:
                    i += 1
            return m
        active = fill_false_runs(active, close_k)
        active = remove_true_runs(active, open_k)
        inactive = ~active
        # 帧时间轴
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
        spans = []
        in_run = False
        start_t = 0.0
        for i, flag in enumerate(inactive):
            if flag and not in_run:
                in_run = True
                start_t = float(times[i])
            elif (not flag) and in_run:
                end_t = float(times[i])
                if end_t - start_t >= float(min_duration):
                    spans.append((start_t, end_t))
                in_run = False
        # 收尾
        if in_run:
            end_t = float(len(vocal_audio) / float(sr))
            if end_t - start_t >= float(min_duration):
                spans.append((start_t, end_t))
        try:
            n_inactive = int(np.sum(inactive.astype(int)))
            logger.info(f"[NoVocalRuns] thr_db={thr_db:.2f}, noise_db={noise_db:.2f}, voice_db={voice_db:.2f}, inactive_frames={n_inactive}/{len(inactive)}")
        except Exception:
            pass
        return spans

    def _finalize_and_filter_cuts_v2(
        self,
        cut_candidates,
        audio_for_split: np.ndarray,
        pure_vocal_audio: Optional[np.ndarray] = None
    ) -> List[int]:
        """Refined candidate filtering: leverage shared refiner for guards/NMS."""

        sr = self.sample_rate
        if sr <= 0 or audio_for_split.size == 0:
            self._last_guard_shift_stats = self._blank_guard_stats()
            return [0, len(audio_for_split)]

        points: List[CutPoint] = []
        if isinstance(cut_candidates, list) and cut_candidates:
            first = cut_candidates[0]
            if isinstance(first, tuple) and len(first) >= 2:
                for candidate in cut_candidates:
                    t, s = candidate[:2]
                    points.append(CutPoint(t=float(t), score=float(s)))
            elif isinstance(first, int):
                for sample_idx in cut_candidates:
                    points.append(CutPoint(t=float(sample_idx) / float(sr), score=1.0))
            else:
                for t in cut_candidates:
                    points.append(CutPoint(t=float(t), score=1.0))

        if not points:
            self._last_guard_shift_stats = self._blank_guard_stats()
            return [0, len(audio_for_split)]

        if os.environ.get('CUT_REFINER_LEGACY') == '1':
            legacy_samples = [int(round(p.t * self.sample_rate)) for p in points]
            logger.warning('CUT_REFINER_LEGACY=1，使用旧版切点精炼路径')
            return self._finalize_and_filter_cuts_legacy(legacy_samples, audio_for_split)

        min_gap_s = float(get_config('quality_control.min_split_gap', 1.0))
        try:
            max_keep = int(get_config('pure_vocal_detection.valley_scoring.max_kept_after_nms', 150))
        except Exception:
            max_keep = None

        guard_enabled = bool(get_config('quality_control.enforce_quiet_cut.enable', False))
        guard_db = float(get_config('quality_control.enforce_quiet_cut.guard_db', 2.5))
        search_right_ms = float(get_config('quality_control.enforce_quiet_cut.search_right_ms', 150))
        guard_win_ms = float(get_config('quality_control.enforce_quiet_cut.win_ms', 80))

        floor_db = -60.0
        if guard_enabled:
            try:
                floor_cfg = get_config('quality_control.enforce_quiet_cut.floor_percentile', 5)
                floor_pct = float(floor_cfg) / 100.0 if float(floor_cfg) > 1 else float(floor_cfg)
            except Exception:
                floor_pct = 0.05
            mono = audio_for_split if audio_for_split.ndim == 1 else np.mean(audio_for_split, axis=0)
            if mono.size > 0:
                hop_length = max(1, int(sr * 0.01))
                rms = librosa.feature.rms(y=mono, hop_length=hop_length)[0]
                rms_db = 20.0 * np.log10(rms + 1e-12)
                floor_db = float(np.percentile(rms_db, max(0.0, min(100.0, floor_pct * 100.0))))

        ctx = CutContext(sr=sr, mix_wave=audio_for_split, vocal_wave=pure_vocal_audio)
        use_vocal_guard = pure_vocal_audio is not None
        nms_topk_cfg = get_config('quality_control.nms_topk_per_10s', None)
        topk_per_10s = int(nms_topk_cfg) if nms_topk_cfg is not None else None
        nms_window_s = float(get_config('quality_control.nms_window_s', 10.0))

        result = finalize_cut_points(
            ctx,
            points,
            use_vocal_guard_first=use_vocal_guard,
            min_gap_s=min_gap_s,
            max_keep=max_keep,
            topk_per_10s=topk_per_10s,
            nms_window_s=nms_window_s,
            guard_db=guard_db,
            search_right_ms=search_right_ms,
            guard_win_ms=guard_win_ms,
            floor_db=floor_db,
            enable_mix_guard=guard_enabled,
            enable_vocal_guard=(guard_enabled and use_vocal_guard),
        )

        adjustments = result.adjustments
        if adjustments:
            total_shifts = [adj.final_shift_ms for adj in adjustments]
            vocal_shifts = [adj.guard_shift_ms for adj in adjustments]
            mix_shifts = [adj.final_shift_ms - adj.guard_shift_ms for adj in adjustments]

            def _avg_abs(values: List[float]) -> float:
                return float(sum(abs(v) for v in values) / len(values)) if values else 0.0

            def _avg_positive(values: List[float]) -> float:
                positives = [v for v in values if v > 0]
                return float(sum(positives) / len(positives)) if positives else 0.0

            self._last_guard_shift_stats = {
                'avg_shift_ms': _avg_abs(total_shifts),
                'max_shift_ms': float(max((abs(v) for v in total_shifts), default=0.0)),
                'avg_guard_only_shift_ms': _avg_positive(total_shifts),
                'avg_vocal_guard_shift_ms': _avg_positive(vocal_shifts),
                'avg_mix_guard_shift_ms': _avg_positive(mix_shifts),
                'count': len(adjustments),
            }
        else:
            self._last_guard_shift_stats = self._blank_guard_stats()

        boundaries = result.sample_boundaries or [0, len(audio_for_split)]
        return sorted(set(boundaries))
    def _finalize_and_filter_cuts_legacy(self, cut_points_samples: List[int], audio: np.ndarray) -> List[int]:
        """Legacy 切点精炼路径：保留旧守卫/NMS 逻辑，供 CUT_REFINER_LEGACY 回退使用。"""
        audio_duration_s = len(audio) / self.sample_rate
        cut_times = sorted(list(set([p / self.sample_rate for p in cut_points_samples])))
        validated_times = [t for t in cut_times if self.quality_controller.enforce_quiet_cut(audio, self.sample_rate, t) >= 0]
        # 修复：降低最小间隔限制，允许更密集的切割
        min_interval = get_config('quality_control.min_split_gap', 1.0)  # 从2.0降到1.0秒
        final_times = self.quality_controller.pure_filter_cut_points(
            validated_times, audio_duration_s, min_interval=min_interval
        )
        final_samples = [0] + [int(t * self.sample_rate) for t in final_times] + [len(audio)]
        return sorted(list(set(final_samples)))

    def _split_at_sample_level(
        self,
        audio: np.ndarray,
        final_cut_points: List[int],
        *,
        segment_flags: Optional[List[bool]] = None,
        debug_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[np.ndarray], Optional[List[bool]], Optional[List[Dict[str, Any]]]]:
        """按照样本级切点拆分音频，并可选合并标记与调试信息."""
        segments: List[np.ndarray] = []
        merged_flags: Optional[List[bool]] = [] if segment_flags is not None else None
        merged_debug: Optional[List[Dict[str, Any]]] = [] if debug_entries is not None else None

        min_keep_samples = max(1, int(0.01 * self.sample_rate))
        carry_audio: Optional[np.ndarray] = None
        carry_flag: Optional[bool] = None
        carry_debug: Optional[Dict[str, Any]] = None

        def _flag_at(index: int) -> bool:
            if segment_flags is not None and index < len(segment_flags):
                return bool(segment_flags[index])
            return True

        def _debug_at(index: int) -> Optional[Dict[str, Any]]:
            if debug_entries is not None and index < len(debug_entries):
                entry = dict(debug_entries[index])
                entry.setdefault('merged_from_segments', [index])
                entry.setdefault('decision_reason', entry.get('decision_reason', entry.get('reason')))
                return entry
            return None

        def _merge_debug(base: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if base is None:
                return extra
            if extra is None:
                return base
            merged = dict(base)
            merged['start_s'] = min(base.get('start_s', merged.get('start_s')), extra.get('start_s', merged.get('start_s')))
            merged['end_s'] = max(base.get('end_s', merged.get('end_s')), extra.get('end_s', merged.get('end_s')))
            merged_duration = float(base.get('duration_s', 0.0)) + float(extra.get('duration_s', 0.0))
            merged_seconds = float(base.get('vocal_activity_seconds', 0.0)) + float(extra.get('vocal_activity_seconds', 0.0))
            merged['duration_s'] = merged_duration
            merged['vocal_activity_seconds'] = merged_seconds
            merged['vocal_activity_ratio'] = merged_seconds / merged_duration if merged_duration > 0 else 0.0
            merged['decision'] = bool(base.get('decision')) or bool(extra.get('decision'))
            reason = extra.get('decision_reason') or base.get('decision_reason') or extra.get('reason') or base.get('reason')
            if reason:
                merged['reason'] = reason
                merged['decision_reason'] = reason
            merged_from = list(base.get('merged_from_segments', []))
            merged_from += list(extra.get('merged_from_segments', []))
            merged['merged_from_segments'] = sorted({seg for seg in merged_from if seg is not None})
            merged['activity_ratio_threshold'] = extra.get('activity_ratio_threshold', base.get('activity_ratio_threshold'))
            merged['activity_threshold_db'] = extra.get('activity_threshold_db', base.get('activity_threshold_db'))
            return merged

        for idx in range(len(final_cut_points) - 1):
            start = final_cut_points[idx]
            end = final_cut_points[idx + 1]
            chunk = audio[start:end]
            flag = _flag_at(idx)
            entry = _debug_at(idx)

            if carry_audio is not None:
                if chunk.size:
                    chunk = np.concatenate((carry_audio, chunk))
                else:
                    chunk = carry_audio
                flag = bool(carry_flag) or bool(flag)
                entry = _merge_debug(carry_debug, entry)
                carry_audio = None
                carry_flag = None
                carry_debug = None

            if end - start >= min_keep_samples and chunk.size:
                segments.append(chunk)
                if merged_flags is not None:
                    merged_flags.append(flag)
                if merged_debug is not None:
                    if entry is None:
                        entry = {
                            'start_s': start / self.sample_rate,
                            'end_s': end / self.sample_rate,
                            'duration_s': (end - start) / self.sample_rate,
                            'vocal_activity_ratio': 1.0 if flag else 0.0,
                            'vocal_activity_seconds': ((end - start) / self.sample_rate) if flag else 0.0,
                            'activity_ratio_threshold': None,
                            'activity_threshold_db': None,
                            'decision': flag,
                            'decision_reason': 'inferred_flag',
                            'reason': 'inferred_flag',
                            'merged_from_segments': [idx],
                        }
                    merged_debug.append(entry)
            else:
                if chunk.size:
                    carry_audio = chunk if carry_audio is None else np.concatenate((carry_audio, chunk))
                    carry_flag = flag if carry_flag is None else (bool(carry_flag) or bool(flag))
                    if merged_debug is not None:
                        carry_debug = _merge_debug(carry_debug, entry)
                else:
                    if merged_debug is not None:
                        carry_debug = _merge_debug(carry_debug, entry)

        if carry_audio is not None:
            if segments:
                segments[-1] = np.concatenate((segments[-1], carry_audio))
                if merged_flags is not None:
                    merged_flags[-1] = bool(merged_flags[-1]) or bool(carry_flag)
                if merged_debug is not None:
                    merged_debug[-1] = _merge_debug(merged_debug[-1], carry_debug)
            else:
                segments.append(carry_audio)
                if merged_flags is not None:
                    merged_flags.append(bool(carry_flag))
                if merged_debug is not None:
                    entry = carry_debug or {
                        'start_s': 0.0,
                        'end_s': len(carry_audio) / self.sample_rate,
                        'duration_s': len(carry_audio) / self.sample_rate,
                        'vocal_activity_ratio': 1.0 if carry_flag else 0.0,
                        'vocal_activity_seconds': (len(carry_audio) / self.sample_rate) if carry_flag else 0.0,
                        'activity_ratio_threshold': None,
                        'activity_threshold_db': None,
                        'decision': bool(carry_flag),
                        'decision_reason': 'merged_trailing_segment',
                        'reason': 'merged_trailing_segment',
                        'merged_from_segments': [],
                    }
                    merged_debug.append(entry)

        if merged_debug is not None:
            for new_idx, entry in enumerate(merged_debug):
                entry['index'] = new_idx

        if not isinstance(self, SeamlessSplitter):
            return segments
        return segments, merged_flags, merged_debug
    def _classify_segments_vocal_presence(
        self,
        vocal_audio: np.ndarray,
        cut_points: List[int],
        marker_segments: Optional[List[Dict]] = None,
        pure_music_segments: Optional[List[Dict]] = None,
        instrumental_audio: Optional[np.ndarray] = None,
        original_audio: Optional[np.ndarray] = None,
    ) -> List[bool]:
        """Classify segments via pure vocal occupancy for _human/_music suffixes."""
        num_segments = max(len(cut_points) - 1, 0)
        self._last_segment_classification_debug = []
        if num_segments == 0:
            return []

        sr = self.sample_rate
        if sr <= 0 or vocal_audio is None or getattr(vocal_audio, 'size', 0) == 0:
            flags = [True] * num_segments
            for idx in range(num_segments):
                self._last_segment_classification_debug.append({
                    'index': idx,
                    'reason': 'fallback_invalid_input',
                    'decision': True,
                })
            return flags

        try:
            legacy_ratio = get_config('quality_control.vocal_presence_ratio_threshold')
        except KeyError:
            legacy_ratio = None
        if legacy_ratio is not None:
            activity_ratio_threshold = float(legacy_ratio)
        else:
            activity_ratio_threshold = float(get_config('quality_control.segment_vocal_activity_ratio', 0.10))
        threshold_db = float(get_config('quality_control.segment_vocal_threshold_db', -50.0))
        hop = max(1, int(0.02 * sr))
        frame_length = max(hop * 2, int(0.05 * sr))
        min_samples = max(1, int(0.02 * sr))

        def _slice_audio(array: Optional[np.ndarray], start_idx: int, end_idx: int) -> Optional[np.ndarray]:
            if array is None or not isinstance(array, np.ndarray):
                return None
            n = len(array)
            if n == 0:
                return None
            start_i = max(0, min(start_idx, n))
            end_i = max(start_i, min(end_idx, n))
            if end_i <= start_i:
                return None
            return array[start_i:end_i]

        flags: List[bool] = []
        debug_entries: List[Dict[str, Any]] = []

        for i in range(num_segments):
            start_idx = max(0, min(int(cut_points[i]), len(vocal_audio)))
            end_idx = max(start_idx, min(int(cut_points[i + 1]), len(vocal_audio)))
            seg_start_s = start_idx / sr
            seg_end_s = end_idx / sr
            seg_duration = max(seg_end_s - seg_start_s, 1e-6)

            vocal_segment = _slice_audio(vocal_audio, start_idx, end_idx)

            vocal_activity_ratio = 0.0
            vocal_activity_seconds = 0.0
            segment_rms_db = None

            if vocal_segment is not None and len(vocal_segment) >= frame_length:
                try:
                    rms_frames = librosa.feature.rms(y=vocal_segment, frame_length=frame_length, hop_length=hop)[0]
                except Exception:
                    rms_scalar = float(np.sqrt(np.mean(np.square(vocal_segment)) + 1e-12))
                    rms_frames = np.full(max(1, len(vocal_segment) // hop + 1), rms_scalar, dtype=np.float32)
                rms_db_frames = 20.0 * np.log10(rms_frames + 1e-12)
                active_mask = rms_db_frames > threshold_db
                if active_mask.size > 0:
                    vocal_activity_ratio = float(np.mean(active_mask))
                    active_seconds = float(active_mask.sum()) * (hop / sr)
                    vocal_activity_seconds = float(min(seg_duration, active_seconds))
            elif vocal_segment is not None and len(vocal_segment) > 0:
                rms_scalar = float(np.sqrt(np.mean(np.square(vocal_segment)) + 1e-12))
                segment_rms_db = 20.0 * np.log10(rms_scalar)
                if segment_rms_db > threshold_db:
                    vocal_activity_ratio = 1.0
                    vocal_activity_seconds = seg_duration

            if vocal_segment is not None and segment_rms_db is None and len(vocal_segment) > 0:
                rms_scalar = float(np.sqrt(np.mean(np.square(vocal_segment)) + 1e-12))
                segment_rms_db = 20.0 * np.log10(rms_scalar)

            decision = vocal_activity_ratio >= activity_ratio_threshold
            decision_reason = 'vocal_activity_ratio_gte_threshold' if decision else 'vocal_activity_ratio_lt_threshold'

            # Energy ratio vote temporarily disabled; keep placeholders for future reinstatement.
            energy_ratio = None
            energy_vote = None
            energy_reason = None

            debug_entries.append({
                'index': i,
                'start_s': seg_start_s,
                'end_s': seg_end_s,
                'duration_s': seg_duration,
                'vocal_activity_ratio': vocal_activity_ratio,
                'vocal_activity_seconds': vocal_activity_seconds,
                'activity_ratio_threshold': activity_ratio_threshold,
                'activity_threshold_db': threshold_db,
                'marker_vote': None,
                'marker_reason': None,
                'vocal_overlap_s': None,
                'music_overlap_s': None,
                'vocal_ratio': None,
                'music_ratio': None,
                'energy_ratio': energy_ratio,
                'energy_vote': energy_vote,
                'presence_ratio': None,
                'presence_vote': None,
                'presence_baseline_db': None,
                'vocal_energy': None,
                'instrumental_energy': None,
                'total_energy': None,
                'rms_db': segment_rms_db,
                'rms_vote': None,
                'decision': decision,
                'decision_reason': decision_reason,
                'reason': decision_reason,
                'decision_threshold_db': threshold_db,
                'threshold_source': 'vocal_activity_ratio',
                'noise_floor_db': None,
                'energy_reason': energy_reason,
            })
            flags.append(bool(decision))

        self._last_segment_classification_debug = debug_entries
        return flags
    def _estimate_vocal_presence(self, vocal_audio: np.ndarray) -> bool:
        """估算给定人声轨整体是否包含人声"""
        if vocal_audio is None or getattr(vocal_audio, 'size', 0) == 0:
            return False
        flags = self._classify_segments_vocal_presence(vocal_audio, [0, len(vocal_audio)])
        return bool(flags[0]) if flags else False

    def _save_segments(self, segments: List[np.ndarray], output_dir: str, segment_is_vocal: Optional[List[bool]] = None, *, subdir: Optional[str] = None, file_suffix: str = '') -> List[str]:
        """保存分割后的片段，并根据人声识别结果命名"""
        base_dir = Path(output_dir)
        if subdir:
            base_dir = base_dir / subdir
            base_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        for i, segment_audio in enumerate(segments):
            is_vocal = True
            if segment_is_vocal is not None and i < len(segment_is_vocal):
                is_vocal = bool(segment_is_vocal[i])
            label = 'human' if is_vocal else 'music'
            output_path = base_dir / f"segment_{i+1:03d}_{label}{file_suffix}.wav"
            sf.write(output_path, segment_audio, self.sample_rate, subtype='PCM_24')
            saved_files.append(str(output_path))
        return saved_files

    @staticmethod
    def _blank_guard_stats() -> Dict[str, float]:
        return {
            'avg_shift_ms': 0.0,
            'max_shift_ms': 0.0,
            'avg_guard_only_shift_ms': 0.0,
            'avg_vocal_guard_shift_ms': 0.0,
            'avg_mix_guard_shift_ms': 0.0,
            'count': 0,
        }

    def _get_guard_shift_stats(self) -> Dict[str, float]:
        stats = getattr(self, '_last_guard_shift_stats', None)
        if not stats:
            return self._blank_guard_stats()
        return dict(stats)

    def _create_single_segment_result(self, audio: np.ndarray, input_path: str, output_dir: str, reason: str, is_vocal: bool = True, gpu_meta: Optional[Dict[str, Any]] = None) -> Dict:
        """当无法分割时，创建单个片段的结果"""
        logger.warning(f"{reason}，将输出为单个文件。")
        self._last_guard_shift_stats = self._blank_guard_stats()
        sample_count = audio.shape[-1] if hasattr(audio, 'shape') else len(audio)
        duration_s = float(sample_count) / float(self.sample_rate) if self.sample_rate > 0 else 0.0
        saved_files = self._save_segments([audio], output_dir, segment_is_vocal=[is_vocal])
        result = {
            'success': True, 'num_segments': 1, 'saved_files': saved_files,
            'segment_durations': [duration_s],
            'segment_vocal_flags': [is_vocal],
            'segment_labels': ['human' if is_vocal else 'music'],
            'segment_classification_debug': getattr(self, '_last_segment_classification_debug', []),
            'guard_shift_stats': self._get_guard_shift_stats(),
            'note': reason, 'input_file': input_path, 'output_dir': output_dir
        }
        if gpu_meta:
            result.update(gpu_meta)
        return result





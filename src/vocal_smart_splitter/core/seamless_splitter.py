#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/seamless_splitter.py

import os
import numpy as np
import librosa
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import time
import tempfile
import soundfile as sf

from ..utils.config_manager import get_config
from ..utils.audio_processor import AudioProcessor
from ..utils.signal_ops import rtrim_trailing_zeros
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

        # 2. 高质量人声分离
        logger.info(f"[{mode.upper()}-STEP1] 执行高质量人声分离...")
        separation_start = time.time()
        separation_result = self.separator.separate_for_detection(original_audio)
        separation_time = time.time() - separation_start

        if separation_result.vocal_track is None:
            return {'success': False, 'error': '人声分离失败', 'input_file': input_path}

        vocal_track = separation_result.vocal_track
        logger.info(f"[{mode.upper()}-STEP1] 人声分离完成 - 后端: {separation_result.backend_used}, 质量: {separation_result.separation_confidence:.3f}, 耗时: {separation_time:.1f}s")

        # 3. 关键修复：使用正确的PureVocalPauseDetector进行停顿检测
        logger.info(f"[{mode.upper()}-STEP2] 使用PureVocalPauseDetector在[纯人声轨道]上进行多维特征检测...")

        # 检查是否为v2.2 MDD模式
        enable_mdd = (mode == 'v2.2_mdd')
        vocal_pauses = self.pure_vocal_detector.detect_pure_vocal_pauses(
            vocal_track,
            enable_mdd_enhancement=enable_mdd,
            original_audio=original_audio  # 提供原始音频用于MDD分析
        )

        if not vocal_pauses:
            has_vocal = self._estimate_vocal_presence(vocal_track)
            return self._create_single_segment_result(original_audio, input_path, output_dir, "未在纯人声中找到停顿", is_vocal=has_vocal)

        # 4. 生成、过滤并分割 (在 vocal_track 上进行最终验证)
        # 转换PureVocalPause到样本点
        cut_points_samples = []
        for p in vocal_pauses:
            if hasattr(p, 'cut_point'):
                cut_points_samples.append(int(p.cut_point * self.sample_rate))
            else:
                # 兜底：使用停顿中心
                center_time = (p.start_time + p.end_time) / 2
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
        segments = self._split_at_sample_level(original_audio, final_cut_points)
        mix_segment_files = self._save_segments(segments, output_dir, segment_is_vocal=segment_vocal_flags)

        vocal_segments = self._split_at_sample_level(vocal_track, final_cut_points)
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

        return {
            'success': True, 'method': f'pure_vocal_split_{mode}', 'num_segments': len(segments),
            'saved_files': saved_files, 'mix_segment_files': mix_segment_files, 'vocal_segment_files': vocal_segment_files,
            'backend_used': separation_result.backend_used,
            'separation_confidence': separation_result.separation_confidence, 'processing_time': total_time,
            'segment_vocal_flags': segment_vocal_flags,
            'segment_labels': ['human' if flag else 'music' for flag in segment_vocal_flags],
            'segment_classification_debug': getattr(self, '_last_segment_classification_debug', []),
            'input_file': input_path, 'output_dir': output_dir
        }

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
            'processing_time': processing_time, 'input_file': input_path, 'output_dir': output_dir
        }

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

    def _finalize_and_filter_cuts_v2(self,
                                     cut_candidates,
                                     audio_for_split: np.ndarray,
                                     pure_vocal_audio: Optional[np.ndarray] = None) -> List[int]:
        """
        方案2：带评分候选的加权NMS + 守卫校正 + 长短段治理（直接在此函数内完成）。

        - 输入可为：
          1) [(time_sec, score), ...]
          2) 样本点列表（int）
          3) 时间列表（float，单位秒）
        """
        sr = self.sample_rate
        x_guard = pure_vocal_audio if pure_vocal_audio is not None else audio_for_split
        duration_s = len(audio_for_split) / sr

        # 解析候选为 time, score
        times: List[float] = []
        scores: List[float] = []
        if isinstance(cut_candidates, list) and cut_candidates:
            first = cut_candidates[0]
            if isinstance(first, tuple) and len(first) >= 2:
                for t, s in cut_candidates:
                    times.append(float(t))
                    scores.append(float(s))
            elif isinstance(first, int):
                times = [float(p) / sr for p in cut_candidates]
                scores = [1.0] * len(times)
            else:
                times = [float(p) for p in cut_candidates]
                scores = [1.0] * len(times)

        # 加权NMS（可控开关）
        use_weighted_nms = get_config('pure_vocal_detection.valley_scoring.use_weighted_nms', True)
        if times and use_weighted_nms:
            logger.info(f"[FinalizeV2] 加权NMS准备: candidates={len(times)}")
            pairs: List[Tuple[float, float]] = list(zip(times, scores))
            try:
                pairs.sort(key=lambda x: x[1], reverse=True)
            except Exception:
                pairs = [(t, 1.0) for t in times]
            min_gap = get_config('quality_control.min_split_gap', 1.0)
            kept: List[Tuple[float, float]] = []
            before_n = len(pairs)
            for t, s in pairs:
                if all(abs(t - kt) >= min_gap for kt, _ in kept):
                    kept.append((t, s))
            # 安全上限：防止极端情况下候选仍然过多导致后续计算量爆炸
            try:
                max_kept = int(get_config('pure_vocal_detection.valley_scoring.max_kept_after_nms', 150))
            except Exception:
                max_kept = 150
            if len(kept) > max_kept:
                kept = kept[:max_kept]
            after_n = len(kept)
            top_score = kept[0][1] if kept else (pairs[0][1] if pairs else None)
            try:
                if top_score is not None:
                    logger.info(f"[NMS] candidates={before_n} -> kept={after_n}, top_score={float(top_score):.3f}")
                else:
                    logger.info(f"[NMS] candidates={before_n} -> kept={after_n}")
            except Exception:
                pass
            times = sorted([t for t, _ in kept])
        elif times:
            # 基础去重
            times = sorted(set(times))

        if not times:
            return [0, len(audio_for_split)]

        # 守卫右推与零交叉吸附
        logger.info(f"[FinalizeV2] 守卫校正开始: {len(times)} 个候选")
        guard_primary = pure_vocal_audio if pure_vocal_audio is not None else audio_for_split
        mix_guard = audio_for_split
        quiet_cut_enabled = getattr(self.quality_controller, 'quiet_cut_enabled', True)

        def _fallback_local_min(source_audio: Optional[np.ndarray], t0: float) -> float:
            if source_audio is None:
                return t0
            search_ms = get_config('quality_control.enforce_quiet_cut.search_right_ms', 300)
            win = int(sr * (search_ms / 1000.0))
            c = int(t0 * sr)
            a = c
            b = min(len(source_audio), c + max(win, int(0.05 * sr)))
            if b <= a + 1:
                return t0
            seg = np.abs(source_audio[a:b])
            idx = int(np.argmin(seg))
            return (a + idx) / sr

        corrected: List[float] = []
        for idx_t, t in enumerate(times):
            t_adj = float(t)
            try:
                if guard_primary is not None:
                    candidate = self.quality_controller.safe_zero_crossing_align(guard_primary, sr, t_adj, window_ms=10)
                    if candidate is None or candidate < 0:
                        candidate = self.quality_controller.enforce_quiet_cut_fast(guard_primary, sr, t_adj)
                    if candidate is None or candidate < 0:
                        candidate = _fallback_local_min(guard_primary, t_adj)
                    if candidate is not None and candidate >= 0:
                        t_adj = float(candidate)
                if quiet_cut_enabled and mix_guard is not None:
                    mix_candidate = self.quality_controller.enforce_quiet_cut_fast(mix_guard, sr, t_adj)
                    if mix_candidate is None or mix_candidate < 0:
                        mix_candidate = _fallback_local_min(mix_guard, t_adj)
                    else:
                        zc_candidate = self.quality_controller.safe_zero_crossing_align(mix_guard, sr, mix_candidate, window_ms=10)
                        if zc_candidate is not None and zc_candidate >= 0:
                            mix_candidate = float(zc_candidate)
                    if mix_candidate is not None and mix_candidate >= 0:
                        t_adj = float(mix_candidate)
                t_adj = float(np.clip(t_adj, 0.0 + 1e-3, duration_s - 1e-3))
                corrected.append(t_adj)
                if idx_t % 5 == 0:
                    logger.info(f"[FinalizeV2] 守卫校正进度 {idx_t+1}/{len(times)}: {t:.3f}s -> {t_adj:.3f}s")
            except Exception:
                corrected.append(float(np.clip(t, 0.0 + 1e-3, duration_s - 1e-3)))

        # 最小间隔过滤（再次稳固）
        min_interval = get_config('quality_control.min_split_gap', 1.0)
        filtered_times = self.quality_controller.pure_filter_cut_points(
            corrected, duration_s, min_interval=min_interval
        )
        logger.info(f"[FinalizeV2] 最小间隔过滤: {len(corrected)} -> {len(filtered_times)}")

        # 构建边界
        boundaries: List[int] = [0] + [int(t * sr) for t in filtered_times] + [len(audio_for_split)]
        boundaries = sorted(set(boundaries))
        logger.info(f"[FinalizeV2] 初始边界数: {len(boundaries)}")
        seg_max = float(get_config('quality_control.segment_max_duration', 18.0))
        min_gap_samples = max(1, int(get_config('quality_control.min_split_gap', 1.0) * sr))

        def _find_energy_min_cut(start_idx: int, end_idx: int, source_audio: np.ndarray) -> Optional[int]:
            span = end_idx - start_idx
            if span <= min_gap_samples * 2:
                return None
            search_start = start_idx + span // 4
            search_end = start_idx + (3 * span) // 4
            if search_end - search_start <= min_gap_samples:
                return None
            sub = source_audio[search_start:search_end]
            if sub.size == 0:
                return None
            win = max(1, int(0.02 * sr))
            sq = np.square(sub)
            if sub.size >= win > 1:
                kernel = np.ones(win, dtype=sq.dtype) / float(win)
                energy = np.convolve(sq, kernel, mode='same')
            else:
                energy = sq
            min_offset = int(np.argmin(energy))
            return search_start + min_offset

        def _split_long_segments(bounds: List[int]) -> List[int]:
            if seg_max <= 0:
                return bounds
            updated = sorted(set(bounds))
            target_audio = x_guard if x_guard is not None else audio_for_split
            max_passes = 8
            passes = 0
            changed = True
            while changed and passes < max_passes:
                changed = False
                passes += 1
                new_points: List[int] = []
                for i in range(len(updated) - 1):
                    start_idx = updated[i]
                    end_idx = updated[i + 1]
                    if (end_idx - start_idx) / sr <= seg_max:
                        continue
                    cut_idx = _find_energy_min_cut(start_idx, end_idx, target_audio)
                    if cut_idx is None:
                        cut_idx = start_idx + (end_idx - start_idx) // 2
                    cut_idx = int(np.clip(cut_idx, start_idx + min_gap_samples, end_idx - min_gap_samples))
                    if cut_idx <= start_idx or cut_idx >= end_idx:
                        continue
                    if all(abs(cut_idx - existing) >= min_gap_samples for existing in updated):
                        new_points.append(cut_idx)
                if new_points:
                    updated.extend(new_points)
                    updated = sorted(set(updated))
                    changed = True
            if passes >= max_passes and changed:
                logger.warning(f'[FinalizeV2] long-segment splitting reached max passes {max_passes}; segments may still exceed {seg_max:.3f}s')
            return updated

        boundaries = _split_long_segments(boundaries)

        # VPP一次判定：仅做“合并短段”，不再做任何二次插点/强拆，避免破坏首次优选切点
        seg_min = float(get_config('quality_control.segment_min_duration', 4.0))
        changed_local = True
        while changed_local and len(boundaries) > 2:
            changed_local = False
            durs_local = [(boundaries[i+1]-boundaries[i]) / sr for i in range(len(boundaries)-1)]
            for i, d in enumerate(durs_local):
                if d < seg_min:
                    rm_idx = i+1 if i+1 < len(boundaries)-1 else i
                    logger.info(f"[FinalizeV2] 合并短段: 移除边界索引 {rm_idx}，片段时长 {d:.3f}s < {seg_min:.3f}s")
                    del boundaries[rm_idx]
                    changed_local = True
                    break
        logger.info(f"[FinalizeV2] 最终边界数: {len(boundaries)}")
        return boundaries

    def _finalize_and_filter_cuts(self, cut_points_samples: List[int], audio: np.ndarray) -> List[int]:
        """对切割点进行最终的排序、去重和安全校验"""
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

    def _split_at_sample_level(self, audio: np.ndarray, final_cut_points: List[int]) -> List[np.ndarray]:
        """执行样本级分割"""
        segments = []
        carry = None
        min_keep_samples = max(1, int(0.01 * self.sample_rate))
        for i in range(len(final_cut_points) - 1):
            start = final_cut_points[i]
            end = final_cut_points[i+1]
            chunk = audio[start:end]
            if carry is not None:
                if chunk.size:
                    chunk = np.concatenate((carry, chunk))
                else:
                    chunk = carry
                carry = None
            if end - start >= min_keep_samples:
                if chunk.size:
                    segments.append(chunk)
            else:
                if chunk.size:
                    carry = chunk if carry is None else np.concatenate((carry, chunk))
        if carry is not None:
            if segments:
                segments[-1] = np.concatenate((segments[-1], carry))
            else:
                segments.append(carry)
        return segments

    def _classify_segments_vocal_presence(
        self,
        vocal_audio: np.ndarray,
        cut_points: List[int],
        marker_segments: Optional[List[Dict]] = None,
        pure_music_segments: Optional[List[Dict]] = None,
        instrumental_audio: Optional[np.ndarray] = None,
        original_audio: Optional[np.ndarray] = None,
    ) -> List[bool]:
        """综合 MDX23 分离结果与能量占比判断片段是否包含人声"""
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

        overlap_vocal_ratio = float(get_config('quality_control.segment_vocal_overlap_ratio', 0.35))
        overlap_music_ratio = float(get_config('quality_control.segment_music_overlap_ratio', 0.55))
        energy_vocal_ratio = float(get_config('quality_control.segment_vocal_energy_ratio', 0.6))
        energy_music_ratio = float(get_config('quality_control.segment_music_energy_ratio', 0.4))
        presence_vocal_ratio = float(get_config('quality_control.segment_vocal_presence_ratio', 0.35))
        presence_music_ratio = float(get_config('quality_control.segment_music_presence_ratio', 0.15))
        noise_margin_db = float(get_config('quality_control.segment_noise_margin_db', 6.0))
        threshold_db = float(get_config('quality_control.segment_vocal_threshold_db', -50.0))
        min_samples = max(1, int(0.02 * sr))
        energy_floor = 1e-10

        def _normalize_segments(raw_segments, fallback_state=None):
            normalized = []
            if not raw_segments:
                return normalized
            for seg in raw_segments:
                try:
                    start = float(seg.get('start', 0.0))
                    end = float(seg.get('end', start))
                    is_vocal = bool(seg.get('is_vocal', fallback_state)) if fallback_state is not None else bool(seg.get('is_vocal', False))
                except Exception:
                    continue
                if end <= start:
                    continue
                normalized.append((start, end, is_vocal))
            return normalized

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

        marker_segments = _normalize_segments(marker_segments or [], fallback_state=True)
        music_segments = [(start, end) for start, end, is_vocal in marker_segments if not is_vocal]
        if pure_music_segments:
            for seg in pure_music_segments:
                try:
                    start = float(seg.get('start', 0.0))
                    end = float(seg.get('end', start))
                except Exception:
                    continue
                if end > start:
                    music_segments.append((start, end))
        vocal_segments = [(start, end) for start, end, is_vocal in marker_segments if is_vocal]

        def _total_overlap(seg_start: float, seg_end: float, segments: List[tuple]) -> float:
            total = 0.0
            for s_start, s_end in segments:
                overlap = min(seg_end, s_end) - max(seg_start, s_start)
                if overlap > 0:
                    total += overlap
            return total

        def _to_index(time_sec: float) -> int:
            return int(max(0, min(round(time_sec * sr), len(vocal_audio))))

        noise_floor_db = None
        if music_segments:
            noise_samples = []
            for start, end in music_segments:
                start_idx = _to_index(start)
                end_idx = _to_index(end)
                segment = _slice_audio(vocal_audio, start_idx, end_idx)
                if segment is None or len(segment) < min_samples:
                    continue
                noise_rms = float(np.sqrt(np.mean(np.square(segment)) + 1e-12))
                noise_samples.append(20.0 * np.log10(noise_rms))
            if noise_samples:
                noise_floor_db = float(np.median(noise_samples))

        flags: List[bool] = []
        debug_entries: List[Dict[str, Any]] = []

        hop = max(1, int(0.02 * sr))
        frame_length = max(hop * 2, int(0.05 * sr))

        for i in range(num_segments):
            start_idx = max(0, min(int(cut_points[i]), len(vocal_audio)))
            end_idx = max(start_idx, min(int(cut_points[i + 1]), len(vocal_audio)))
            seg_start_s = start_idx / sr
            seg_end_s = end_idx / sr
            seg_duration = max(seg_end_s - seg_start_s, 1e-6)

            vocal_segment = _slice_audio(vocal_audio, start_idx, end_idx)
            # 与导出一致：在判定前对人声音频片段做严格零尾部修剪（最多100ms），
            # 使判定素材与输出/segments_vocal 实际保存的内容等价，避免因保存端尾部处理而产生偏差。
            if vocal_segment is not None and len(vocal_segment) > 0:
                try:
                    vocal_segment = rtrim_trailing_zeros(
                        np.ascontiguousarray(vocal_segment),
                        floor=0.0,
                        max_strip_samples=int(0.1 * sr),
                    )
                except Exception:
                    pass
            instrumental_segment = _slice_audio(instrumental_audio, start_idx, end_idx)
            original_segment = _slice_audio(original_audio, start_idx, end_idx)

            vocal_energy = float(np.mean(np.square(vocal_segment)) + 1e-12) if vocal_segment is not None else 0.0
            inst_energy = float(np.mean(np.square(instrumental_segment)) + 1e-12) if instrumental_segment is not None else None
            orig_energy = float(np.mean(np.square(original_segment)) + 1e-12) if original_segment is not None else None
            if inst_energy is None and orig_energy is not None:
                inst_energy = max(orig_energy - vocal_energy, 0.0)

            total_energy = None
            if inst_energy is not None:
                total_energy = vocal_energy + inst_energy
            elif orig_energy is not None:
                total_energy = orig_energy

            marker_vote = None
            marker_reason = None
            vocal_overlap = music_overlap = None
            vocal_ratio = music_ratio = None
            if vocal_segments or music_segments:
                vocal_overlap = _total_overlap(seg_start_s, seg_end_s, vocal_segments)
                music_overlap = _total_overlap(seg_start_s, seg_end_s, music_segments)
                vocal_ratio = vocal_overlap / seg_duration if seg_duration > 0 else 0.0
                music_ratio = music_overlap / seg_duration if seg_duration > 0 else 0.0
                if vocal_ratio >= overlap_vocal_ratio:
                    marker_vote = True
                    marker_reason = 'marker_vocal_overlap'
                elif music_ratio >= overlap_music_ratio:
                    marker_vote = False
                    marker_reason = 'marker_music_overlap'

            energy_ratio = None
            energy_vote = None
            energy_reason = None
            if total_energy is not None and total_energy > energy_floor:
                energy_ratio = float(vocal_energy / (total_energy + 1e-12))
                if energy_ratio >= energy_vocal_ratio:
                    energy_vote = True
                    energy_reason = 'energy_vocal_ratio'
                elif energy_ratio <= energy_music_ratio:
                    energy_vote = False
                    energy_reason = 'energy_music_ratio'

            presence_ratio = None
            presence_vote = None
            presence_reason = None
            presence_baseline_db = None
            if vocal_segment is not None and len(vocal_segment) >= frame_length:
                try:
                    rms_frames = librosa.feature.rms(y=vocal_segment, frame_length=frame_length, hop_length=hop)[0]
                except Exception:
                    rms_frames = np.sqrt(np.mean(np.square(vocal_segment) + 1e-12)) * np.ones(max(1, len(vocal_segment) // hop + 1))
                rms_db_frames = 20.0 * np.log10(rms_frames + 1e-12)
                baseline_db = noise_floor_db
                residual_segment = None
                if original_segment is not None and vocal_segment is not None and len(original_segment) == len(vocal_segment):
                    residual_segment = original_segment - vocal_segment
                if baseline_db is None:
                    candidate_levels: List[float] = []
                    if instrumental_segment is not None and len(instrumental_segment) >= min_samples:
                        inst_rms = float(np.sqrt(np.mean(np.square(instrumental_segment)) + 1e-12))
                        candidate_levels.append(20.0 * np.log10(inst_rms))
                    if residual_segment is not None and len(residual_segment) >= min_samples:
                        res_rms = float(np.sqrt(np.mean(np.square(residual_segment)) + 1e-12))
                        candidate_levels.append(20.0 * np.log10(res_rms))
                    if candidate_levels:
                        baseline_db = float(np.median(candidate_levels))
                    else:
                        baseline_db = float(np.percentile(rms_db_frames, 25))
                presence_baseline_db = float(baseline_db) if baseline_db is not None else None
                presence_threshold_db = presence_baseline_db + noise_margin_db if presence_baseline_db is not None else noise_margin_db
                presence_ratio = float(np.mean(rms_db_frames > presence_threshold_db))
                if presence_ratio >= presence_vocal_ratio:
                    presence_vote = True
                    presence_reason = 'presence_vocal_ratio'
                elif presence_ratio <= presence_music_ratio:
                    presence_vote = False
                    presence_reason = 'presence_music_ratio'

            segment = vocal_segment
            rms_db = None
            rms_vote = None
            rms_reason = None
            decision_threshold = threshold_db
            threshold_source = 'static_threshold'
            if segment is None or len(segment) < min_samples:
                rms_vote = False
                rms_reason = 'segment_insufficient_samples'
            else:
                rms = float(np.sqrt(np.mean(np.square(segment)) + 1e-12))
                rms_db = 20.0 * np.log10(rms)
                if noise_floor_db is not None:
                    decision_threshold = noise_floor_db + noise_margin_db
                    threshold_source = 'noise_floor'
                rms_vote = rms_db > decision_threshold
                rms_reason = 'rms_vs_threshold'

            decision = None
            decision_reason = 'unset'

            if energy_vote is not None:
                decision = energy_vote
                decision_reason = energy_reason
            elif presence_vote is not None:
                decision = presence_vote
                decision_reason = presence_reason
            elif marker_vote is not None:
                decision = marker_vote
                decision_reason = marker_reason
            elif rms_vote is not None:
                decision = rms_vote
                decision_reason = rms_reason
            else:
                decision = False
                decision_reason = 'default_music'

            if decision is False and energy_vote is False:
                if marker_vote is True:
                    decision = True
                    decision_reason = 'marker_override_energy'
                elif presence_vote is True and (energy_ratio is None or energy_ratio >= presence_music_ratio):
                    decision = True
                    decision_reason = 'presence_override_energy'

            debug_entries.append({
                'index': i,
                'start_s': seg_start_s,
                'end_s': seg_end_s,
                'duration_s': seg_duration,
                'vocal_overlap_s': vocal_overlap,
                'music_overlap_s': music_overlap,
                'vocal_ratio': vocal_ratio,
                'music_ratio': music_ratio,
                'marker_vote': marker_vote,
                'energy_ratio': energy_ratio,
                'energy_vote': energy_vote,
                'presence_ratio': presence_ratio,
                'presence_vote': presence_vote,
                'presence_baseline_db': presence_baseline_db,
                'vocal_energy': vocal_energy,
                'instrumental_energy': inst_energy,
                'total_energy': total_energy,
                'rms_db': rms_db,
                'rms_vote': rms_vote,
                'decision': decision,
                'reason': decision_reason,
                'decision_threshold_db': decision_threshold,
                'threshold_source': threshold_source,
                'noise_floor_db': noise_floor_db,
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
        total_samples = 0
        for i, segment_audio in enumerate(segments):
            is_vocal = True
            if segment_is_vocal is not None and i < len(segment_is_vocal):
                is_vocal = bool(segment_is_vocal[i])
            label = 'human' if is_vocal else 'music'
            output_path = base_dir / f"segment_{i+1:03d}_{label}{file_suffix}.wav"

            # 防御式修复：去除片段尾部被错误补上的 0（最多 100ms）
            trimmed = rtrim_trailing_zeros(
                np.ascontiguousarray(segment_audio),
                floor=0.0,
                max_strip_samples=int(0.1 * self.sample_rate),
            )
            total_samples += len(trimmed)

            sf.write(output_path, trimmed, self.sample_rate, subtype='PCM_24')
            saved_files.append(str(output_path))
        try:
            logger.info(f"[Integrity] sum(segment_len)={total_samples} samples @{self.sample_rate}Hz")
        except Exception:
            pass
        return saved_files

    def _create_single_segment_result(self, audio: np.ndarray, input_path: str, output_dir: str, reason: str, is_vocal: bool = True) -> Dict:
        """当无法分割时，创建单个片段的结果"""
        logger.warning(f"{reason}，将输出为单个文件。")
        saved_files = self._save_segments([audio], output_dir, segment_is_vocal=[is_vocal])
        return {
            'success': True, 'num_segments': 1, 'saved_files': saved_files,
            'segment_vocal_flags': [is_vocal],
            'segment_labels': ['human' if is_vocal else 'music'],
            'segment_classification_debug': getattr(self, '_last_segment_classification_debug', []),
            'note': reason, 'input_file': input_path, 'output_dir': output_dir
        }

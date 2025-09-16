#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/seamless_splitter.py

import os
import numpy as np
import librosa
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time
import tempfile
import soundfile as sf

from ..utils.config_manager import get_config
from ..utils.audio_processor import AudioProcessor
from .vocal_pause_detector import VocalPauseDetectorV2
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
        self.pause_detector = VocalPauseDetectorV2(sample_rate)  # 用于smart_split模式
        self.pure_vocal_detector = PureVocalPauseDetector(sample_rate)  # 用于v2.1/v2.2模式
        self.quality_controller = QualityController(sample_rate)
        self.separator = EnhancedVocalSeparator(sample_rate)
        logger.info(f"无缝分割器统一指挥中心初始化完成 (SR: {self.sample_rate}) - 已加载双检测器")

    def split_audio_seamlessly(self, input_path: str, output_dir: str, mode: str = 'v2.2_mdd') -> Dict:
        """
        执行无缝分割的主入口。

        Args:
            input_path: 输入文件路径
            output_dir: 输出目录
            mode: 分割模式 ('v2.1', 'v2.2_mdd', 'smart_split', 'vocal_separation')

        Returns:
            分割结果信息
        """
        logger.info(f"开始无缝分割: {input_path} (模式: {mode})")

        try:
            if mode == 'vocal_separation':
                return self._process_vocal_separation_only(input_path, output_dir)
            elif mode in ['v2.1', 'v2.2_mdd']:
                return self._process_pure_vocal_split(input_path, output_dir, mode)
            elif mode == 'smart_split':
                return self._process_smart_split(input_path, output_dir)
            else:
                logger.warning(f"未知模式 {mode}，使用默认v2.2 MDD模式")
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
        [v2.9 终极修正版] 处理v2.1和v2.2 MDD模式的核心逻辑
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
            return self._create_single_segment_result(original_audio, input_path, output_dir, "未在纯人声中找到停顿")

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

        final_cut_points = self._finalize_and_filter_cuts_v2(
            cut_candidates,
            original_audio,
            pure_vocal_audio=vocal_track
        )

        # 使用最终确定的分割点来切割原始音频
        segments = self._split_at_sample_level(original_audio, final_cut_points)
        saved_files = self._save_segments(segments, output_dir)

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
            'saved_files': saved_files, 'backend_used': separation_result.backend_used,
            'separation_confidence': separation_result.separation_confidence, 'processing_time': total_time,
            'input_file': input_path, 'output_dir': output_dir
        }

    def _process_smart_split(self, input_path: str, output_dir: str) -> Dict:
        """处理传统智能分割模式"""
        logger.info("[SMART_SPLIT] 执行传统智能分割...")
        original_audio = self._load_and_resample_if_needed(input_path)
        vocal_pauses = self.pause_detector.detect_vocal_pauses(original_audio)
        if not vocal_pauses:
            return self._create_single_segment_result(original_audio, input_path, output_dir, "未找到符合条件的停顿")

        cut_points_samples = [int(p.cut_point * self.sample_rate) for p in vocal_pauses]
        final_cut_points = self._finalize_and_filter_cuts_v2(
            cut_points_samples,
            original_audio,
            pure_vocal_audio=None
        )
        segments = self._split_at_sample_level(original_audio, final_cut_points)
        saved_files = self._save_segments(segments, output_dir)

        return {'success': True, 'method': 'smart_split', 'num_segments': len(segments), 'saved_files': saved_files, 'input_file': input_path, 'output_dir': output_dir}

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
        def _fallback_local_min(t0: float) -> float:
            search_ms = get_config('quality_control.enforce_quiet_cut.search_right_ms', 300)
            win = int(sr * (search_ms / 1000.0))
            c = int(t0 * sr)
            a = c
            b = min(len(x_guard), c + max(win, int(0.05 * sr)))
            if b <= a + 1:
                return t0
            seg = np.abs(x_guard[a:b])
            idx = int(np.argmin(seg))
            return (a + idx) / sr

        corrected: List[float] = []
        for idx_t, t in enumerate(times):
            try:
                t_adj = self.quality_controller.safe_zero_crossing_align(x_guard, sr, t, window_ms=10)
                if t_adj is None or t_adj < 0:
                    t_adj = self.quality_controller.enforce_quiet_cut_fast(x_guard, sr, t)
                if t_adj is None or t_adj < 0:
                    t_adj = _fallback_local_min(t)
                t_adj = float(np.clip(t_adj, 0.0 + 1e-3, duration_s - 1e-3))
                corrected.append(t_adj)
                if idx_t % 5 == 0:
                    logger.info(f"[FinalizeV2] 守卫校正进度 {idx_t+1}/{len(times)}: {t:.3f}s -> {t_adj:.3f}s")
            except Exception:
                corrected.append(t)

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

        # 新治理策略：
        # 1) 仅使用首次高质量能量谷作为切点；
        # 2) 合并短段（< segment_min_duration）；
        # 3) 对超长段（> segment_max_duration）在该段内做“二次能量谷检测”，再合并一次短段；
        seg_min = float(get_config('quality_control.segment_min_duration', 4.0))
        seg_max = float(get_config('quality_control.segment_max_duration', 18.0))
        min_gap_s = float(get_config('quality_control.min_split_gap', 1.0))
        min_gap_samples = max(1, int(min_gap_s * sr))

        def _merge_short(bounds: List[int]) -> List[int]:
            changed_local = True
            while changed_local and len(bounds) > 2:
                changed_local = False
                durs_local = [(bounds[i+1]-bounds[i]) / sr for i in range(len(bounds)-1)]
                for i, d in enumerate(durs_local):
                    if d < seg_min:
                        rm_idx = i+1 if i+1 < len(bounds)-1 else i
                        logger.info(f"[FinalizeV2] 合并短段: 移除边界索引 {rm_idx}，片段时长 {d:.3f}s < {seg_min:.3f}s")
                        del bounds[rm_idx]
                        changed_local = True
                        break
            return bounds

        def _second_valley_for_long(bounds: List[int]) -> List[int]:
            new_bounds = list(bounds)
            for i in range(len(bounds)-1):
                start = bounds[i]
                end = bounds[i+1]
                span_s = (end - start) / sr
                if span_s > seg_max:
                    n_add = int(np.floor(span_s / seg_max))
                    if n_add <= 0:
                        continue
                    step = span_s / (n_add + 1)
                    for k in range(1, n_add+1):
                        t_candidate = (start / sr) + step * k
                        valley_idx = None
                        try:
                            local_rms_ms = int(get_config('vocal_pause_splitting.local_rms_window_ms', 25))
                            guard_ms = int(get_config('vocal_pause_splitting.lookahead_guard_ms', 120))
                            floor_pct = float(get_config('vocal_pause_splitting.silence_floor_percentile', 5))
                            valley_idx = self.pause_detector._select_valley_cut_point(
                                x_guard, start, end, sr, local_rms_ms, guard_ms, floor_pct
                            )
                        except Exception:
                            valley_idx = None
                        if valley_idx is None:
                            t_adj = _fallback_local_min(t_candidate)
                            valley_idx = int(round(t_adj * sr))
                        lo = start + min_gap_samples
                        hi = end - min_gap_samples
                        if hi <= lo:
                            logger.warning(f"[FinalizeV2] 长段过窄无法插点 [{start},{end}]，跳过")
                            continue
                        valley_idx = int(np.clip(valley_idx, lo, hi))
                        if any(abs(valley_idx - bb) < min_gap_samples for bb in (start, end)):
                            valley_idx = int(np.clip((start + end)//2, lo, hi))
                        new_bounds.append(valley_idx)
                        logger.info(f"[FinalizeV2] 二次检测切点: 区间[{i},{i+1}] 追加 valley@{valley_idx/sr:.3f}s (span={span_s:.3f}s)")
            # 去重与样本间隔清理
            new_bounds = sorted(set(new_bounds))
            cleaned = [new_bounds[0]]
            for b in new_bounds[1:]:
                if b - cleaned[-1] >= min_gap_samples:
                    cleaned.append(b)
                else:
                    logger.info(f"[FinalizeV2] 移除过近边界(二次): prev={cleaned[-1]} this={b}")
            if cleaned[-1] != new_bounds[-1]:
                if new_bounds[-1] - cleaned[-1] < min_gap_samples:
                    if len(cleaned) >= 2:
                        cleaned[-1] = new_bounds[-1]
                    else:
                        cleaned.append(new_bounds[-1])
                else:
                    cleaned.append(new_bounds[-1])
            return cleaned

        # 迭代版二次谷值：
        def _second_valley_iterative(bounds: List[int], max_passes: int = 3) -> List[int]:
            cur = list(bounds)
            # 记录第一轮的超长区间（样本坐标），后续仅对这些区间再做2轮
            def _get_long_spans(bds: List[int]) -> List[Tuple[int, int]]:
                out = []
                for ii in range(len(bds)-1):
                    st, ed = bds[ii], bds[ii+1]
                    if (ed - st) / sr > seg_max:
                        out.append((st, ed))
                return out

            initial_long_spans = _get_long_spans(cur)

            def _in_initial_long(st: int, ed: int) -> bool:
                for a, b in initial_long_spans:
                    if st >= a and ed <= b:
                        return True
                return False

            passes = 0
            while passes < max_passes:
                passes += 1
                added = False
                new_bounds = list(cur)
                for ii in range(len(cur)-1):
                    start, end = cur[ii], cur[ii+1]
                    span_s = (end - start) / sr
                    if span_s <= seg_max:
                        continue
                    # 仅对第一轮发现的超长区间在后续轮次继续处理
                    if passes > 1 and not _in_initial_long(start, end):
                        continue
                    # 使用 ceil 方案，严格收敛到 seg_max
                    n_add = int(np.ceil(span_s / seg_max)) - 1
                    if n_add <= 0:
                        continue
                    step = span_s / (n_add + 1)
                    for k in range(1, n_add + 1):
                        t_candidate = (start / sr) + step * k
                        valley_idx = None
                        try:
                            local_rms_ms = int(get_config('vocal_pause_splitting.local_rms_window_ms', 25))
                            guard_ms = int(get_config('vocal_pause_splitting.lookahead_guard_ms', 120))
                            floor_pct = float(get_config('vocal_pause_splitting.silence_floor_percentile', 5))
                            valley_idx = self.pause_detector._select_valley_cut_point(
                                x_guard, start, end, sr, local_rms_ms, guard_ms, floor_pct
                            )
                        except Exception:
                            valley_idx = None
                        if valley_idx is None:
                            t_adj = _fallback_local_min(t_candidate)
                            valley_idx = int(round(t_adj * sr))
                        # 间隔与范围保护
                        lo = start + min_gap_samples
                        hi = end - min_gap_samples
                        if hi <= lo:
                            logger.warning(f"[FinalizeV2] 长段过窄无法插点 [{start},{end}]，跳过")
                            continue
                        valley_idx = int(np.clip(valley_idx, lo, hi))
                        if any(abs(valley_idx - bb) < min_gap_samples for bb in (start, end)):
                            valley_idx = int(np.clip((start + end)//2, lo, hi))
                        new_bounds.append(valley_idx)
                        added = True
                        logger.info(f"[FinalizeV2] 二次检测切点(p#{passes}): 区间[{ii},{ii+1}] 追加 valley@{valley_idx/sr:.3f}s (span={span_s:.3f}s)")
                # 去重与间隔清理
                new_bounds = sorted(set(new_bounds))
                cleaned = [new_bounds[0]]
                for b in new_bounds[1:]:
                    if b - cleaned[-1] >= min_gap_samples:
                        cleaned.append(b)
                    else:
                        logger.info(f"[FinalizeV2] 移除过近边界(二次p#{passes}): prev={cleaned[-1]} this={b}")
                if cleaned[-1] != new_bounds[-1]:
                    if new_bounds[-1] - cleaned[-1] < min_gap_samples:
                        if len(cleaned) >= 2:
                            cleaned[-1] = new_bounds[-1]
                        else:
                            cleaned.append(new_bounds[-1])
                    else:
                        cleaned.append(new_bounds[-1])
                cur = cleaned
                # 检查是否已满足上限
                max_span = 0.0
                for ii in range(len(cur)-1):
                    span_s = (cur[ii+1] - cur[ii]) / sr
                    if span_s > max_span:
                        max_span = span_s
                if max_span <= seg_max:
                    break
                # 若本轮没有新增，提前退出
                if not added:
                    break
                # 仅对第一轮标记的超长片段继续后续轮次
                if passes == 1:
                    initial_long_spans = _get_long_spans(cur)
            return cur

        # 1) 合并短段（尊重首次能量谷切点，不新增切点）
        boundaries = _merge_short(boundaries)
        # 2) 迭代对超长段做局部二次谷值檢测（最多3轮，且仅对第一轮判定的超长区间再做后续2轮）
        boundaries = _second_valley_iterative(boundaries, max_passes=3)
        # 3) 再合并一次短段，确保满足 segment_min_duration
        boundaries = _merge_short(boundaries)

        logger.info(f"[FinalizeV2] 最终边界数: {len(boundaries)}")
        return boundaries

        # 长短段治理
        seg_min = get_config('quality_control.segment_min_duration', 4.0)
        seg_max = get_config('quality_control.segment_max_duration', 18.0)

        changed = True
        merge_pass = 0
        while changed and len(boundaries) > 2:
            changed = False
            merge_pass += 1
            durs = [(boundaries[i+1]-boundaries[i]) / sr for i in range(len(boundaries)-1)]
            for i, d in enumerate(durs):
                if d < seg_min and len(boundaries) > 2:
                    rm_idx = i+1 if i+1 < len(boundaries)-1 else i
                    del boundaries[rm_idx]
                    changed = True
                    logger.info(f"[FinalizeV2] 合并短段 pass#{merge_pass}: 移除边界索引 {rm_idx}，片段时长 {d:.3f}s")
                    break

        def _insert_split(boundaries: List[int], start_idx: int, count: int):
            start = boundaries[start_idx]
            end = boundaries[start_idx+1]
            span_s = (end - start) / sr
            step = span_s / (count + 1)
            min_gap_s = float(get_config('quality_control.min_split_gap', 1.0))
            min_gap_samples = max(1, int(min_gap_s * sr))
            for k in range(1, count+1):
                t_candidate = (start / sr) + step * k
                t_adj = self.quality_controller.enforce_quiet_cut_fast(x_guard, sr, t_candidate)
                if t_adj is None or t_adj < 0:
                    t_adj = _fallback_local_min(t_candidate)
                # 转为样本并与边界保持最小间隔，避免 0s 片段
                b_idx = int(round(t_adj * sr))
                lo = start + min_gap_samples
                hi = end - min_gap_samples
                if hi <= lo:
                    # 区间太窄，不插入，避免制造0s片段
                    logger.warning(f"[FinalizeV2] 区间[{start},{end}] 太短，跳过插入切点")
                    continue
                # 将切点夹紧到有效范围
                b_idx = int(np.clip(b_idx, lo, hi))
                # 去重：避免与现有边界过近
                if any(abs(b_idx - bb) < min_gap_samples for bb in boundaries[start_idx:start_idx+2]):
                    # 回退到中点
                    b_idx = int((start + end) // 2)
                    b_idx = int(np.clip(b_idx, lo, hi))
                boundaries.append(b_idx)

        changed = True
        insert_pass = 0
        max_insert_pass = 10
        while changed and insert_pass < max_insert_pass:
            changed = False
            insert_pass += 1
            boundaries.sort()
            i = 0
            while i < len(boundaries) - 1:
                span_s = (boundaries[i+1] - boundaries[i]) / sr
                if span_s > seg_max:
                    n_add = int(np.floor(span_s / seg_max))
                    _insert_split(boundaries, i, n_add)
                    changed = True
                    logger.info(f"[FinalizeV2] 强拆长段 pass#{insert_pass}: 在区间[{i},{i+1}] 追加 {n_add} 个切点 (span={span_s:.3f}s)")
                    break
                i += 1

        # 二次清理：去重与最小样本间隔，避免 0s 片段
        min_gap_s = float(get_config('quality_control.min_split_gap', 1.0))
        min_gap_samples = max(1, int(min_gap_s * sr))
        boundaries = sorted(set(boundaries))
        cleaned = [boundaries[0]]
        for b in boundaries[1:]:
            if b - cleaned[-1] >= min_gap_samples:
                cleaned.append(b)
            else:
                logger.info(f"[FinalizeV2] 移除过近边界: prev={cleaned[-1]} this={b} (<{min_gap_samples} samples)")
        # 确保保留结尾边界
        if cleaned[-1] != boundaries[-1]:
            if boundaries[-1] - cleaned[-1] < min_gap_samples:
                # 如果最后一段过短，保留结尾，移除上一个
                if len(cleaned) >= 2:
                    cleaned[-1] = boundaries[-1]
                else:
                    cleaned.append(boundaries[-1])
            else:
                cleaned.append(boundaries[-1])
        boundaries = cleaned
        # 最终一次短段合并：确保所有片段长度≥segment_min_duration
        changed = True
        seg_min = float(get_config('quality_control.segment_min_duration', 4.0))
        while changed and len(boundaries) > 2:
            changed = False
            durs = [(boundaries[i+1]-boundaries[i]) / sr for i in range(len(boundaries)-1)]
            for i, d in enumerate(durs):
                if d < seg_min and len(boundaries) > 2:
                    rm_idx = i+1 if i+1 < len(boundaries)-1 else i
                    logger.info(f"[FinalizeV2] 合并短段(最终): 移除边界索引 {rm_idx}，片段时长 {d:.3f}s < {seg_min:.3f}s")
                    del boundaries[rm_idx]
                    changed = True
                    break
        if insert_pass >= max_insert_pass:
            logger.warning(f"[FinalizeV2] 强拆循环达到上限 {max_insert_pass} 次，提前退出")
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
        # 避免 0s 片段：至少保持10ms样本长度
        min_keep_samples = max(1, int(0.01 * self.sample_rate))
        for i in range(len(final_cut_points) - 1):
            start = final_cut_points[i]
            end = final_cut_points[i+1]
            if end - start >= min_keep_samples:
                segments.append(audio[start:end])
            else:
                logger.info(f"[Split] 跳过过短片段 idx={i} len_samples={end-start}")
        return segments

    def _save_segments(self, segments: List[np.ndarray], output_dir: str) -> List[str]:
        """保存分割后的片段"""
        saved_files = []
        for i, segment_audio in enumerate(segments):
            output_path = Path(output_dir) / f"segment_{i+1:03d}.wav"
            sf.write(output_path, segment_audio, self.sample_rate, subtype='PCM_24')
            saved_files.append(str(output_path))
        return saved_files

    def _create_single_segment_result(self, audio: np.ndarray, input_path: str, output_dir: str, reason: str) -> Dict:
        """当无法分割时，创建单个片段的结果"""
        logger.warning(f"{reason}，将输出为单个文件。")
        saved_files = self._save_segments([audio], output_dir)
        return {
            'success': True, 'num_segments': 1, 'saved_files': saved_files,
            'note': reason, 'input_file': input_path, 'output_dir': output_dir
        }

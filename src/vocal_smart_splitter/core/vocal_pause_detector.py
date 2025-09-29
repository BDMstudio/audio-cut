#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/vocal_pause_detector.py

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ..utils.config_manager import get_config
from ..utils.adaptive_parameter_calculator import create_adaptive_calculator, AdaptiveParameters

logger = logging.getLogger(__name__)

try:
    from .adaptive_vad_enhancer import AdaptiveVADEnhancer, BPMFeatures
    ADAPTIVE_VAD_AVAILABLE = True
    logger.info("自适应VAD增强器可用")
except ImportError as e:
    logger.warning(f"自适应VAD增强器不可用: {e}")
    ADAPTIVE_VAD_AVAILABLE = False
    BPMFeatures = None

@dataclass
class VocalPause:
    start_time: float
    end_time: float
    duration: float
    position_type: str
    confidence: float
    cut_point: float

class VocalPauseDetectorV2:
    """[v2.9 终极修复版] 改进的人声停顿检测器 - 集成BPM自适应能力"""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.adaptive_calculator = create_adaptive_calculator()
        self.current_adaptive_params: Optional[AdaptiveParameters] = None
        self.head_offset = get_config('vocal_pause_splitting.head_offset', -0.5)
        self.tail_offset = get_config('vocal_pause_splitting.tail_offset', 0.5)
        
        self.enable_bpm_adaptation = get_config('vocal_pause_splitting.enable_bpm_adaptation', True) and ADAPTIVE_VAD_AVAILABLE
        if self.enable_bpm_adaptation:
            self.adaptive_enhancer = AdaptiveVADEnhancer(sample_rate)
            logger.info("BPM自适应增强器已启用")
        else:
            self.adaptive_enhancer = None
            logger.info("BPM自适应已禁用或不可用，将使用固定阈值模式")

        self._init_silero_vad()
        logger.info(f"VocalPauseDetectorV2 初始化完成 (SR: {sample_rate})")

    def _init_silero_vad(self):
        try:
            import torch
            torch.set_num_threads(1)
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
            )
            (self.get_speech_timestamps, _, _, _, _) = self.vad_utils
            self._vad_device = 'cuda' if torch.cuda.is_available() and get_config('advanced_vad.use_cuda', True) else 'cpu'
            self._silero_use_fp16 = bool(get_config('advanced_vad.silero_use_fp16', True)) and self._vad_device == 'cuda'
            target_device = torch.device(self._vad_device)
            self.vad_model = self.vad_model.to(target_device)
            if self._silero_use_fp16:
                self.vad_model = self.vad_model.half()
            else:
                self.vad_model = self.vad_model.float()
            logger.info("Silero VAD模型加载成功")
        except Exception as e:
            self.vad_model = None
            self._vad_device = 'cpu'
            self._silero_use_fp16 = False
            logger.error(f"Silero VAD初始化失败: {e}")

    def detect_vocal_pauses(self, detection_target_audio: np.ndarray, context_audio: Optional[np.ndarray] = None) -> List[VocalPause]:
        """
        主检测流程，同时使用背景音频和目标音频。
        
        Args:
            detection_target_audio: 用于精细检测的音频 (如: vocal_track)
            context_audio: 用于音乐背景分析的音频 (如: original_audio)
        """
        logger.info("开始BPM感知的人声停顿检测...")
        if self.vad_model is None:
            logger.error("Silero VAD模型未加载，无法继续")
            return []

        # 如果没有提供背景音频，则使用目标音频进行分析（兼容旧的smart_split模式）
        if context_audio is None:
            context_audio = detection_target_audio
            logger.info("未提供背景音频，将在目标音频上进行音乐分析。")

        bpm_features = None
        if self.enable_bpm_adaptation and self.adaptive_enhancer:
            logger.info("步骤 1/5: 在[背景音频]上执行BPM和编曲复杂度分析...")
            complexity_segments, bpm_features = self.adaptive_enhancer.analyze_arrangement_complexity(context_audio)
            if bpm_features:
                logger.info(f"🎵 音乐分析完成: {float(bpm_features.main_bpm):.1f} BPM ({bpm_features.bpm_category})")
                instrument_analyzer = getattr(self.adaptive_enhancer, 'instrument_analyzer', None)
                if instrument_analyzer:
                    instrument_complexity = instrument_analyzer.analyze_instrument_complexity(context_audio)
                    self.current_adaptive_params = self.adaptive_calculator.calculate_all_parameters(
                        float(bpm_features.main_bpm), float(instrument_complexity.get('overall_complexity', 0.5)), int(instrument_complexity.get('instrument_count', 3))
                    )
        
        logger.info("步骤 2/5: 在[目标音频]上使用自适应参数进行VAD语音检测...")
        speech_timestamps = self._detect_speech_timestamps(detection_target_audio)

        logger.info("步骤 3/5: 计算语音间的停顿区域...")
        pause_segments = self._calculate_pause_segments(speech_timestamps, len(detection_target_audio))

        logger.info("步骤 4/5: 使用动态阈值过滤有效停顿...")
        valid_pauses = self._filter_adaptive_pauses(pause_segments, bpm_features)
        
        logger.info("步骤 5/5: 分类停顿并计算最终切点...")
        vocal_pauses = self._classify_pause_positions(valid_pauses, len(detection_target_audio))
        vocal_pauses = self._calculate_cut_points(vocal_pauses, bpm_features=bpm_features, waveform=detection_target_audio)
        
        logger.info(f"检测完成，找到 {len(vocal_pauses)} 个有效人声停顿")
        return vocal_pauses

    # ... 省略 _init_silero_vad, _detect_speech_timestamps, _calculate_pause_segments ...
    # ... 它们的内容保持不变 ...

    def _detect_speech_timestamps(self, audio: np.ndarray) -> List[Dict]:
        """使用 Silero VAD 计算语音时间戳，并在 GPU 时优先尝试 FP16 推理。"""
        try:
            import torch
            import librosa
        except Exception as exc:
            logger.error(f"Silero VAD 依赖未就绪: {exc}", exc_info=True)
            return []

        if self.vad_model is None:
            logger.error("Silero VAD模型未初始化，无法检测语音片段")
            return []

        target_sr = 16000
        audio_16k = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr).astype(np.float32)

        device_name = getattr(self, '_vad_device', 'cpu')
        use_fp16 = bool(getattr(self, '_silero_use_fp16', False) and device_name == 'cuda')
        device = torch.device(device_name)

        tensor = torch.from_numpy(audio_16k).to(device)
        tensor_fp = tensor.half() if use_fp16 else tensor.float()

        self.vad_model = self.vad_model.to(device)
        self.vad_model = self.vad_model.half() if use_fp16 else self.vad_model.float()

        if self.current_adaptive_params:
            params = self.current_adaptive_params
            vad_params = {
                'threshold': params.vad_threshold,
                'min_speech_duration_ms': get_config('advanced_vad.silero_min_speech_ms', 250),
                'min_silence_duration_ms': int(params.min_pause_duration * 1000),
                'window_size_samples': get_config('advanced_vad.silero_window_size_samples', 512),
                'speech_pad_ms': int(params.speech_pad_ms),
            }
            logger.info(f"应用自适应 VAD 参数: {vad_params}")
        else:
            vad_params = {
                'threshold': get_config('advanced_vad.silero_prob_threshold_down', 0.35),
                'min_speech_duration_ms': get_config('advanced_vad.silero_min_speech_ms', 250),
                'min_silence_duration_ms': get_config('advanced_vad.silero_min_silence_ms', 700),
                'window_size_samples': get_config('advanced_vad.silero_window_size_samples', 512),
                'speech_pad_ms': get_config('advanced_vad.silero_speech_pad_ms', 150),
            }
            logger.info(f"应用静态 VAD 参数: {vad_params}")

        def _run_inference(input_tensor, model):
            with torch.inference_mode():
                return self.get_speech_timestamps(input_tensor, model, sampling_rate=target_sr, **vad_params)

        try:
            speech_timestamps_16k = _run_inference(tensor_fp, self.vad_model)
        except RuntimeError as err:
            if use_fp16:
                logger.warning(f"Silero FP16 推理失败，回退至 FP32: {err}")
                self._silero_use_fp16 = False
                tensor_fp32 = tensor.float()
                self.vad_model = self.vad_model.float()
                speech_timestamps_16k = _run_inference(tensor_fp32, self.vad_model)
            else:
                logger.error(f"Silero VAD推理失败: {err}", exc_info=True)
                return []

        scale_factor = self.sample_rate / target_sr
        for ts in speech_timestamps_16k:
            ts['start'] = int(ts['start'] * scale_factor)
            ts['end'] = int(ts['end'] * scale_factor)
        return speech_timestamps_16k

    def _calculate_pause_segments(self, speech_timestamps: List[Dict], audio_length: int) -> List[Dict]:
        pause_segments = []
        if not speech_timestamps:
            pause_segments.append({'start': 0, 'end': audio_length})
            return pause_segments
        if speech_timestamps[0]['start'] > 0:
            pause_segments.append({'start': 0, 'end': speech_timestamps[0]['start']})
        for i in range(len(speech_timestamps) - 1):
            if speech_timestamps[i+1]['start'] > speech_timestamps[i]['end']:
                pause_segments.append({'start': speech_timestamps[i]['end'], 'end': speech_timestamps[i+1]['start']})
        if speech_timestamps[-1]['end'] < audio_length:
            pause_segments.append({'start': speech_timestamps[-1]['end'], 'end': audio_length})
        return pause_segments

    # 关键的 _filter_adaptive_pauses 函数保持我们上一轮修复后的 v2.5 版本即可
    def _filter_adaptive_pauses(self, pause_segments: List[Dict], bpm_features: Optional[BPMFeatures]) -> List[Dict]:
        """
        [v2.5 终极修复版] 基于鲁棒统计学的智能裁决系统
        技术: 统一使用75分位数作为基础动态阈值，彻底解决前奏长静音对统计模型的污染问题。
        """
        if not self.enable_bpm_adaptation or not bpm_features or not self.current_adaptive_params:
            min_pause_duration = get_config('vocal_pause_splitting.min_pause_duration', 1.0)
            min_pause_samples = int(min_pause_duration * self.sample_rate)
            valid_pauses = [p for p in pause_segments if (p['end'] - p['start']) >= min_pause_samples]
            for p in valid_pauses:
                p['duration'] = (p['end'] - p['start']) / self.sample_rate
            logger.info(f"BPM自适应禁用，使用静态阈值 {min_pause_duration}s，过滤后剩 {len(valid_pauses)} 个停顿")
            return valid_pauses

        absolute_min_pause_s = get_config('vocal_pause_splitting.statistical_filter.absolute_min_pause', 0.3)
        min_pause_samples = int(absolute_min_pause_s * self.sample_rate)
        
        all_candidate_durations = []
        for p in pause_segments:
            duration = (p['end'] - p['start']) / self.sample_rate
            p['duration'] = duration
            if duration >= absolute_min_pause_s:
                all_candidate_durations.append(duration)

        if not all_candidate_durations:
            logger.warning(f"在应用最小初筛 ({absolute_min_pause_s}s) 后，没有找到任何候选停顿。")
            return []

        percentile_75 = np.percentile(all_candidate_durations, 75)
        median_pause = np.median(all_candidate_durations)
        logger.info(f"鲁棒停顿时长统计模型: 75分位={percentile_75:.3f}s, 中位数={median_pause:.3f}s")
        
        valid_pauses = []
        total_audio_length = pause_segments[-1]['end'] if pause_segments else 0
        segments_with_mdd = getattr(self.adaptive_enhancer, 'last_analyzed_segments', [])
        
        base_threshold_ratio = get_config('vocal_pause_splitting.statistical_filter.base_threshold_ratio', 0.7)
        chorus_multiplier = get_config('vocal_pause_splitting.statistical_filter.chorus_multiplier', 1.0)
        mdd_threshold_multiplier = get_config('musical_dynamic_density.threshold_multiplier', 0.2)
        
        base_dynamic_threshold = percentile_75 * base_threshold_ratio

        for pause in pause_segments:
            duration_s = pause['duration']
            if duration_s < absolute_min_pause_s:
                continue

            current_time = pause['start'] / self.sample_rate
            current_mdd = 0.5
            if segments_with_mdd:
                for seg in segments_with_mdd:
                    if seg.start_time <= current_time < seg.end_time:
                        current_mdd = seg.dynamic_density_score
                        break
            
            is_head = (pause.get('start', 0) == 0)
            is_tail = (pause.get('end', 0) >= total_audio_length * 0.95)
            
            dynamic_threshold = base_dynamic_threshold

            if self.current_adaptive_params.category in ['fast', 'very_fast']:
                if current_mdd > 0.7:
                    mdd_adjustment = 1.0 - (current_mdd * mdd_threshold_multiplier)
                    dynamic_threshold = max(median_pause, dynamic_threshold * mdd_adjustment)
            else:
                if current_mdd > 0.6:
                    mdd_adjustment = 1.0 + (current_mdd - 0.6) * mdd_threshold_multiplier
                    dynamic_threshold *= mdd_adjustment
            
            final_threshold = max(dynamic_threshold * chorus_multiplier, absolute_min_pause_s)

            if duration_s >= final_threshold or is_head or is_tail:
                valid_pauses.append(pause)
        
        logger.info(f"鲁棒统计裁决完成: {len(all_candidate_durations)}个候选 -> {len(valid_pauses)}个最终分割点")
        return valid_pauses

    def _classify_pause_positions(self, valid_pauses: List[Dict], audio_length: int) -> List[VocalPause]:
        vocal_pauses = []
        for pause in valid_pauses:
            start_time = pause['start'] / self.sample_rate
            end_time = pause['end'] / self.sample_rate
            duration = pause['duration']
            position_type = 'head' if pause['start'] == 0 else 'tail' if pause['end'] == audio_length else 'middle'
            min_pause_duration = self.current_adaptive_params.min_pause_duration if self.current_adaptive_params else 1.0
            confidence = min(1.0, duration / (min_pause_duration * 2))
            vocal_pauses.append(VocalPause(start_time, end_time, duration, position_type, confidence, 0.0))
        return vocal_pauses

    def _calculate_cut_points(self, vocal_pauses: List[VocalPause], bpm_features: Optional['BPMFeatures'] = None, waveform: Optional[np.ndarray] = None) -> List[VocalPause]:
        logger.info(f"计算 {len(vocal_pauses)} 个停顿的切割点...")
        for i, pause in enumerate(vocal_pauses):
            search_start, search_end = self._define_search_range(pause)
            valley_point_s = self._find_energy_valley(waveform, search_start, search_end)
            if valley_point_s is None:
                valley_point_s = (pause.start_time + pause.end_time) / 2
            final_cut_point_s = valley_point_s
            if bpm_features and self.current_adaptive_params:
                final_cut_point_s = self._smart_beat_align(waveform, valley_point_s, bpm_features, search_start, search_end)
            pause.cut_point = final_cut_point_s
        return vocal_pauses

    def _define_search_range(self, pause: VocalPause) -> Tuple[float, float]:
        search_start, search_end = pause.start_time, pause.end_time
        if pause.position_type == 'head':
            search_start = max(search_start, pause.end_time + self.head_offset - 0.5)
            search_end = min(search_end, pause.end_time + self.head_offset + 0.5)
        elif pause.position_type == 'tail':
            search_start = max(search_start, pause.start_time + self.tail_offset - 0.5)
            search_end = min(search_end, pause.start_time + self.tail_offset + 0.5)
        return (search_start, search_end) if search_end > search_start else (pause.start_time, pause.end_time)

    def _find_energy_valley(self, waveform: Optional[np.ndarray], start_s: float, end_s: float) -> Optional[float]:
        if waveform is None: return None
        local_rms_ms = get_config('vocal_pause_splitting.local_rms_window_ms', 25)
        guard_ms = get_config('vocal_pause_splitting.lookahead_guard_ms', 120)
        floor_pct = get_config('vocal_pause_splitting.silence_floor_percentile', 5)
        l_idx, r_idx = int(start_s * self.sample_rate), int(end_s * self.sample_rate)
        if r_idx > l_idx:
            valley_idx = self._select_valley_cut_point(waveform, l_idx, r_idx, self.sample_rate, local_rms_ms, guard_ms, floor_pct)
            return valley_idx / self.sample_rate if valley_idx is not None else None
        return None

    def _get_mdd_score_for_pause(self, pause: Dict) -> float:
        segments = getattr(self, 'adaptive_enhancer', None)
        last_segments = getattr(segments, 'last_analyzed_segments', None) if segments else None
        if not last_segments:
            return 0.0
        sr = self.sample_rate
        if 'start_time' in pause and 'end_time' in pause:
            start_s = float(pause.get('start_time', 0.0))
            end_s = float(pause.get('end_time', start_s))
        else:
            start_s = float(pause.get('start', 0.0)) / float(sr)
            end_s = float(pause.get('end', start_s)) / float(sr)
        if end_s <= start_s:
            return 0.0
        scores = []
        for seg in last_segments:
            seg_start = float(getattr(seg, 'start_time', 0.0))
            seg_end = float(getattr(seg, 'end_time', seg_start))
            overlap_start = max(start_s, seg_start)
            overlap_end = min(end_s, seg_end)
            if overlap_end > overlap_start:
                scores.append(float(getattr(seg, 'dynamic_density_score', 0.0)))
        return float(np.mean(scores)) if scores else 0.0

    def _smart_beat_align(self, waveform: np.ndarray, valley_point_s: float, bpm_features: 'BPMFeatures', search_start_s: float, search_end_s: float) -> float:
        beat_interval = 60.0 / float(bpm_features.main_bpm)
        nearest_beat_s = round(valley_point_s / beat_interval) * beat_interval
        if not (search_start_s <= nearest_beat_s <= search_end_s):
            return valley_point_s
        valley_idx, beat_idx = int(valley_point_s * self.sample_rate), int(nearest_beat_s * self.sample_rate)
        win_size = int(0.05 * self.sample_rate)
        valley_energy = np.mean(waveform[max(0, valley_idx - win_size//2) : valley_idx + win_size//2]**2)
        beat_energy = np.mean(waveform[max(0, beat_idx - win_size//2) : beat_idx + win_size//2]**2)
        if beat_energy <= valley_energy * 1.3:
            return nearest_beat_s
        return valley_point_s

    def _select_valley_cut_point(self, waveform: np.ndarray, left_idx: int, right_idx: int, sample_rate: int, local_rms_ms: int, guard_ms: int, floor_percentile: float) -> Optional[int]:
        left_idx, right_idx = max(0, int(left_idx)), min(len(waveform), int(right_idx))
        if right_idx - left_idx <= 8: return None
        win_samples = max(1, int(local_rms_ms / 1000.0 * sample_rate))
        guard_samples = max(1, int(guard_ms / 1000.0 * sample_rate)) if guard_ms > 0 else 0
        segment = waveform[left_idx:right_idx]
        rms = self._compute_rms_envelope(segment, win_samples)
        floor_val = np.percentile(np.abs(segment), floor_percentile)
        order = np.argsort(rms)
        margin_samples = max(1, int(0.02 * sample_rate))
        enable_valley_mode = bool(get_config('vocal_pause_splitting.enable_valley_mode', True))
        auto_fallback = bool(get_config('vocal_pause_splitting.auto_valley_fallback', True))
        enforce_margin = enable_valley_mode or auto_fallback
        min_margin = margin_samples if enforce_margin else 0
        for j in order:
            if min_margin <= j < (len(rms) - min_margin):
                if guard_samples == 0 or not enforce_margin or self._future_silence_guard(rms, j, guard_samples, floor_val):
                    return left_idx + j
        for j in order:
            if min_margin <= j < (len(rms) - min_margin):
                return left_idx + j
        if order.size == 0:
            return None
        fallback = left_idx + order[0]
        enforce_margin = bool(get_config('vocal_pause_splitting.enable_valley_mode', True))
        if enforce_margin:
            if (fallback - left_idx) < margin_samples:
                fallback = left_idx + margin_samples
            if (right_idx - fallback) < margin_samples:
                fallback = right_idx - margin_samples
        return max(left_idx, min(fallback, right_idx))



    def _compute_rms_envelope(self, waveform: np.ndarray, win_samples: int) -> np.ndarray:
        if win_samples <= 1: return np.abs(waveform).astype(np.float32)
        kernel = np.ones(int(win_samples), dtype=np.float32) / float(max(1, int(win_samples)))
        return np.sqrt(np.convolve((waveform.astype(np.float32) ** 2), kernel, mode='same'))

    def _future_silence_guard(self, rms: np.ndarray, start_idx: int, guard_samples: int, floor_val: float, allowance: float = 1.2, ratio: float = 0.7) -> bool:
        end_idx = min(start_idx + guard_samples, len(rms))
        if end_idx <= start_idx: return False
        window = rms[start_idx:end_idx]
        return (np.sum(window <= (floor_val * allowance)) / float(window.size)) >= ratio if window.size > 0 else False



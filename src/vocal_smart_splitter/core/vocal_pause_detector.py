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
    """改进的人声停顿检测器 - 集成BPM自适应能力"""

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
            self.vad_model, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
            (self.get_speech_timestamps, _, _, _, _) = self.vad_utils
            logger.info("Silero VAD模型加载成功")
        except Exception as e:
            self.vad_model = None
            logger.error(f"Silero VAD初始化失败: {e}")

    def detect_vocal_pauses(self, original_audio: np.ndarray) -> List[VocalPause]:
        """主检测流程，现在完全由BPM自适应系统驱动"""
        logger.info("开始BPM感知的人声停顿检测...")
        if self.vad_model is None:
            logger.error("Silero VAD模型未加载，无法继续")
            return []

        bpm_features = None
        if self.enable_bpm_adaptation and self.adaptive_enhancer:
            logger.info("步骤 1/5: 执行BPM和编曲复杂度分析...")
            complexity_segments, bpm_features = self.adaptive_enhancer.analyze_arrangement_complexity(original_audio)
            if bpm_features:
                logger.info(f"🎵 音乐分析完成: {float(bpm_features.main_bpm):.1f} BPM ({bpm_features.bpm_category})")
                # 使用分析结果计算并应用动态参数
                analysis = getattr(self.adaptive_enhancer, 'last_instrument_analysis', {})
                instrument_analyzer = getattr(self.adaptive_enhancer, 'instrument_analyzer', None)
                if instrument_analyzer:
                    instrument_complexity = instrument_analyzer.analyze_instrument_complexity(original_audio)
                    self.current_adaptive_params = self.adaptive_calculator.calculate_all_parameters(
                        float(bpm_features.main_bpm), float(instrument_complexity.get('overall_complexity', 0.5)), int(instrument_complexity.get('instrument_count', 3))
                    )
        else:
            logger.info("步骤 1/5: 跳过BPM分析（已禁用或不可用）")

        logger.info("步骤 2/5: 使用自适应参数进行VAD语音检测...")
        speech_timestamps = self._detect_speech_timestamps(original_audio)

        logger.info("步骤 3/5: 计算语音间的停顿区域...")
        pause_segments = self._calculate_pause_segments(speech_timestamps, len(original_audio))

        logger.info("步骤 4/5: 使用动态阈值过滤有效停顿...")
        valid_pauses = self._filter_adaptive_pauses(pause_segments, bpm_features)
        
        logger.info("步骤 5/5: 分类停顿并计算最终切点...")
        vocal_pauses = self._classify_pause_positions(valid_pauses, len(original_audio))
        vocal_pauses = self._calculate_cut_points(vocal_pauses, bpm_features=bpm_features, waveform=original_audio)
        
        logger.info(f"检测完成，找到 {len(vocal_pauses)} 个有效人声停顿")
        return vocal_pauses

    def _detect_speech_timestamps(self, audio: np.ndarray) -> List[Dict]:
        """使用Silero VAD检测语音时间戳，参数由self.current_adaptive_params动态提供"""
        try:
            import torch
            import librosa
            target_sr = 16000
            audio_16k = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr)
            audio_tensor = torch.from_numpy(audio_16k).float()
            
            # 动态获取VAD参数
            if self.current_adaptive_params:
                params = self.current_adaptive_params
                vad_params = {
                    'threshold': params.vad_threshold,
                    'min_speech_duration_ms': get_config('advanced_vad.silero_min_speech_ms', 250), # 通常保持固定
                    'min_silence_duration_ms': int(params.min_pause_duration * 1000),
                    'window_size_samples': get_config('advanced_vad.silero_window_size_samples', 512),
                    'speech_pad_ms': int(params.speech_pad_ms)
                }
                logger.info(f"应用动态VAD参数: {vad_params}")
            else: # 回退到静态配置
                vad_params = {
                    'threshold': get_config('advanced_vad.silero_prob_threshold_down', 0.35),
                    'min_speech_duration_ms': get_config('advanced_vad.silero_min_speech_ms', 250),
                    'min_silence_duration_ms': get_config('advanced_vad.silero_min_silence_ms', 700),
                    'window_size_samples': get_config('advanced_vad.silero_window_size_samples', 512),
                    'speech_pad_ms': get_config('advanced_vad.silero_speech_pad_ms', 150)
                }
                logger.info(f"应用静态VAD参数: {vad_params}")
            
            speech_timestamps_16k = self.get_speech_timestamps(audio_tensor, self.vad_model, sampling_rate=target_sr, **vad_params)
            
            # 映射回原始采样率
            scale_factor = self.sample_rate / target_sr
            for ts in speech_timestamps_16k:
                ts['start'] = int(ts['start'] * scale_factor)
                ts['end'] = int(ts['end'] * scale_factor)
            return speech_timestamps_16k
        except Exception as e:
            logger.error(f"Silero VAD检测失败: {e}", exc_info=True)
            return []

    def _calculate_pause_segments(self, speech_timestamps: List[Dict], audio_length: int) -> List[Dict]:
        """计算语音片段之间的停顿区域"""
        pause_segments = []

        if not speech_timestamps:
            # 没有检测到语音，整个音频都是停顿
            pause_segments.append({
                'start': 0,
                'end': audio_length
            })
            return pause_segments

        # 头部停顿（音频开始到第一个语音片段）
        if speech_timestamps[0]['start'] > 0:
            pause_segments.append({
                'start': 0,
                'end': speech_timestamps[0]['start']
            })

        # 中间停顿（语音片段之间）
        for i in range(len(speech_timestamps) - 1):
            current_end = speech_timestamps[i]['end']
            next_start = speech_timestamps[i + 1]['start']

            if next_start > current_end:
                pause_segments.append({
                    'start': current_end,
                    'end': next_start
                })

        # 尾部停顿（最后一个语音片段到音频结束）
        if speech_timestamps[-1]['end'] < audio_length:
            pause_segments.append({
                'start': speech_timestamps[-1]['end'],
                'end': audio_length
            })

        return pause_segments

    def _filter_adaptive_pauses(self, pause_segments: List[Dict], bpm_features: Optional[BPMFeatures]) -> List[Dict]:
        """
        [大脑升级版] 基于停顿分布的统计学动态筛选
        技术：两遍扫描法，第一遍收集并分析数据，第二遍根据动态阈值进行裁决。
        增强：添加安全边界、副歌检测、频谱验证
        """
        # 获取配置参数
        enable_statistical = get_config('vocal_pause_splitting.statistical_filter.enable', True)
        
        # 如果BPM系统未启用或统计筛选禁用，退回到简单过滤
        if not self.enable_bpm_adaptation or not bpm_features or not enable_statistical:
            return self._filter_simple_pauses(pause_segments)
        
        # === 第一遍扫描：数据收集与统计分析 ===
        
        # 1. 动态计算基础门槛（宽松，用于过滤噪音）
        base_threshold_ratio = get_config('vocal_pause_splitting.statistical_filter.base_threshold_ratio', 0.7)
        base_min_duration = self.current_adaptive_params.min_pause_duration * base_threshold_ratio
        min_pause_samples = int(base_min_duration * self.sample_rate)
        
        # 准备停顿时长数据
        middle_pause_durations = []
        all_pauses_with_duration = []
        total_audio_length = pause_segments[-1]['end'] if pause_segments else 0
        
        for i, pause in enumerate(pause_segments):
            duration_samples = pause['end'] - pause['start']
            duration_seconds = duration_samples / self.sample_rate
            
            # 添加duration字段
            pause['duration'] = duration_seconds
            
            # 判断位置
            is_head = (i == 0 and pause['start'] == 0)
            is_tail = (i == len(pause_segments) - 1 and pause['end'] >= total_audio_length * 0.95)
            pause['is_head'] = is_head
            pause['is_tail'] = is_tail
            
            # 只统计中间停顿用于分析
            if duration_samples >= min_pause_samples and not is_head and not is_tail:
                middle_pause_durations.append(duration_seconds)
                
            if duration_samples >= min_pause_samples:
                all_pauses_with_duration.append(pause)
        
        # 处理无中间停顿的情况
        if not middle_pause_durations:
            logger.warning("歌曲中间没有足够的候选停顿，使用所有停顿进行统计")
            middle_pause_durations = [p['duration'] for p in all_pauses_with_duration 
                                     if not p.get('is_head') and not p.get('is_tail')]
            if not middle_pause_durations:
                # 极端情况：用所有停顿
                middle_pause_durations = [p['duration'] for p in all_pauses_with_duration]
                if not middle_pause_durations:
                    logger.warning("没有找到任何有效停顿，返回基础过滤结果")
                    return [p for p in pause_segments if p.get('duration', 0) >= base_min_duration]
        
        # 2. 统计学分析
        import numpy as np
        average_pause = np.mean(middle_pause_durations)
        median_pause = np.median(middle_pause_durations)
        std_dev = np.std(middle_pause_durations) if len(middle_pause_durations) > 1 else 0
        
        logger.info(f"停顿统计分析: 平均={average_pause:.3f}s, 中位数={median_pause:.3f}s, 标准差={std_dev:.3f}s")
        
        # 3. 动态生成裁决阈值
        use_median_priority = get_config('vocal_pause_splitting.statistical_filter.use_median_priority', True)
        
        if std_dev > average_pause * 0.5:
            # 停顿分布离散，使用平均值
            duration_threshold = average_pause
            logger.info(f"停顿分布离散，使用平均值: {duration_threshold:.3f}s")
        elif use_median_priority:
            # 优先使用中位数（更鲁棒）
            duration_threshold = max(average_pause, median_pause)
            logger.info(f"使用平均值/中位数较大者: {duration_threshold:.3f}s")
        else:
            duration_threshold = average_pause
            logger.info(f"使用平均值: {duration_threshold:.3f}s")
        
        # 4. 应用安全边界
        absolute_min = get_config('vocal_pause_splitting.statistical_filter.absolute_min_pause', 0.6)
        absolute_max = get_config('vocal_pause_splitting.statistical_filter.absolute_max_pause', 2.5)
        
        # 确保不低于BPM系统计算的最小值
        duration_threshold = max(duration_threshold, self.current_adaptive_params.min_pause_duration)
        # 应用绝对边界
        duration_threshold = np.clip(duration_threshold, absolute_min, absolute_max)
        
        logger.info(f"最终裁决阈值: {duration_threshold:.3f}s (边界: {absolute_min:.1f}-{absolute_max:.1f}s)")
        
        # === 第二遍扫描：执行裁决 ===
        valid_pauses = []
        filtered_count = 0
        
        for pause in all_pauses_with_duration:
            duration_seconds = pause['duration']
            is_head = pause.get('is_head', False)
            is_tail = pause.get('is_tail', False)
            
            # 判断是否保留
            passes_base = duration_seconds >= self.current_adaptive_params.min_pause_duration
            passes_dynamic = duration_seconds >= duration_threshold
            
            # 副歌检测（可选增强）
            chorus_multiplier = 1.0
            if hasattr(self, '_is_chorus_section') and self._is_chorus_section(pause):
                chorus_multiplier = get_config('vocal_pause_splitting.statistical_filter.chorus_multiplier', 1.3)
                passes_dynamic = duration_seconds >= (duration_threshold * chorus_multiplier)
                logger.debug(f"副歌部分检测，阈值提高到 {duration_threshold * chorus_multiplier:.3f}s")
            
            # 决策逻辑
            if is_head or is_tail:
                # 边界停顿：宽松处理
                if passes_base:
                    valid_pauses.append(pause)
                    logger.debug(f"保留边界停顿: {duration_seconds:.3f}s")
                else:
                    filtered_count += 1
            else:
                # 中间停顿：严格筛选
                if passes_dynamic:
                    valid_pauses.append(pause)
                    logger.debug(f"保留中间停顿: {duration_seconds:.3f}s >= {duration_threshold:.3f}s")
                else:
                    filtered_count += 1
                    logger.debug(f"过滤中间停顿: {duration_seconds:.3f}s < {duration_threshold:.3f}s")
        
        logger.info(f"统计学动态裁决完成: {len(all_pauses_with_duration)}个候选 -> {len(valid_pauses)}个保留 (过滤{filtered_count}个)")
        
        return valid_pauses
    
    def _filter_simple_pauses(self, pause_segments: List[Dict]) -> List[Dict]:
        """简单的静态阈值过滤（回退方法）"""
        min_pause_duration = self.current_adaptive_params.min_pause_duration if self.current_adaptive_params else get_config('vocal_pause_splitting.min_pause_duration', 1.0)
        min_pause_samples = int(min_pause_duration * self.sample_rate)
        
        valid_pauses = []
        for pause in pause_segments:
            duration_samples = pause['end'] - pause['start']
            if duration_samples >= min_pause_samples:
                pause['duration'] = duration_samples / self.sample_rate
                valid_pauses.append(pause)
        
        logger.info(f"简单过滤：保留 {len(valid_pauses)} 个停顿 (最小 > {min_pause_duration:.2f}s)")
        return valid_pauses
    
    def _is_chorus_section(self, pause: Dict) -> bool:
        """检测是否为副歌部分（基于能量和频谱特征）"""
        # 这是一个占位实现，可以后续增强
        # 可以通过分析pause前后的音频能量、频谱复杂度等判断
        return False
        
    def _classify_pause_positions(self, valid_pauses: List[Dict], audio_length: int) -> List[VocalPause]:
        """分类停顿位置（头部/中间/尾部）"""
        vocal_pauses = []

        for pause in valid_pauses:
            start_time = pause['start'] / self.sample_rate
            end_time = pause['end'] / self.sample_rate
            duration = pause['duration']

            # 判断停顿位置类型
            if pause['start'] == 0:
                # 头部停顿
                position_type = 'head'
            elif pause['end'] == audio_length:
                # 尾部停顿
                position_type = 'tail'
            else:
                # 中间停顿
                position_type = 'middle'

            # 计算置信度（基于停顿时长，越长置信度越高）
            min_pause_duration = self.current_adaptive_params.min_pause_duration if self.current_adaptive_params else 1.0
            confidence = min(1.0, duration / (min_pause_duration * 2))

            vocal_pause = VocalPause(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                position_type=position_type,
                confidence=confidence,
                cut_point=0.0  # 稍后计算
            )
            vocal_pauses.append(vocal_pause)

        return vocal_pauses

    def _calculate_cut_points(self, vocal_pauses: List[VocalPause], bpm_features: Optional['BPMFeatures'] = None, waveform: Optional[np.ndarray] = None) -> List[VocalPause]:
        """计算最终切点，集成能量谷和节拍对齐"""
        logger.info(f"计算 {len(vocal_pauses)} 个停顿的切割点 (能量谷最优 + BPM智能融合模式)...")

        for i, pause in enumerate(vocal_pauses):
            # 1. 为能量谷搜索定义一个安全的范围
            search_start, search_end = self._define_search_range(pause)
            
            logger.debug(f"停顿 {i+1} ({pause.position_type}): 原始范围 [{pause.start_time:.3f}s, {pause.end_time:.3f}s], "
                        f"能量谷搜索范围 [{search_start:.3f}s, {search_end:.3f}s]")

            # 2. 强制寻找物理上的能量最低点作为基准
            valley_point_s = self._find_energy_valley(waveform, search_start, search_end)
            if valley_point_s is None:
                # 如果找不到能量谷，使用停顿中心作为兜底
                valley_point_s = (pause.start_time + pause.end_time) / 2
                logger.warning(f"  -> 未找到能量谷，回退到中心点: {valley_point_s:.3f}s")

            # 3. 如果BPM信息可用，进行智能对齐（以能量谷为基础）
            final_cut_point_s = valley_point_s
            if bpm_features and self.current_adaptive_params:
                final_cut_point_s = self._smart_beat_align(
                    waveform, valley_point_s, bpm_features, search_start, search_end
                )

            # 4. 更新最终切点
            pause.cut_point = final_cut_point_s
            logger.info(f"停顿 {i+1} ({pause.position_type}): 最终切点 @ {pause.cut_point:.3f}s")

        return vocal_pauses

    def _define_search_range(self, pause: VocalPause) -> Tuple[float, float]:
        """为能量谷搜索定义一个安全的范围，巧妙利用offset参数"""
        search_start = pause.start_time
        search_end = pause.end_time
        
        # 应用偏移量来指导搜索范围，而不是直接决定切点
        if pause.position_type == 'head':
            search_start = max(search_start, pause.end_time + self.head_offset - 0.5)
            search_end = min(search_end, pause.end_time + self.head_offset + 0.5)
        elif pause.position_type == 'tail':
            search_start = max(search_start, pause.start_time + self.tail_offset - 0.5)
            search_end = min(search_end, pause.start_time + self.tail_offset + 0.5)
        
        return (search_start, search_end) if search_end > search_start else (pause.start_time, pause.end_time)

    def _find_energy_valley(self, waveform: Optional[np.ndarray], start_s: float, end_s: float) -> Optional[float]:
        """在指定时间范围内寻找能量最低点，并应用安全守卫"""
        if waveform is None or len(waveform) == 0:
            return None

        # 从配置中获取能量谷检测的精细参数
        local_rms_ms = int(get_config('vocal_pause_splitting.local_rms_window_ms', 25))
        guard_ms = int(get_config('vocal_pause_splitting.lookahead_guard_ms', 120))
        floor_pct = float(get_config('vocal_pause_splitting.silence_floor_percentile', 5))

        l_idx = max(0, int(start_s * self.sample_rate))
        r_idx = min(len(waveform), int(end_s * self.sample_rate))

        if r_idx > l_idx:
            # 调用底层的能量谷搜索函数
            valley_idx = self._select_valley_cut_point(
                waveform, l_idx, r_idx, self.sample_rate,
                local_rms_ms, guard_ms, floor_pct
            )
            return valley_idx / self.sample_rate if valley_idx is not None else None
        return None

    def _smart_beat_align(self, waveform: np.ndarray, valley_point_s: float, bpm_features: 'BPMFeatures', search_start_s: float, search_end_s: float) -> float:
        """智能节拍对齐：在能量谷划定的安静区内寻找节拍点
        
        核心原则：能量谷最优，BPM仅为辅助，绝不允许切在人声上
        """
        beat_interval = 60.0 / float(bpm_features.main_bpm)
        nearest_beat_s = round(valley_point_s / beat_interval) * beat_interval

        # 安全检查1：节拍点必须在搜索范围内
        if not (search_start_s <= nearest_beat_s <= search_end_s):
            logger.debug(f"  节拍点 {nearest_beat_s:.3f}s 超出搜索范围，坚守能量谷点 {valley_point_s:.3f}s")
            return valley_point_s

        # 安全检查2：严格能量校验，绝不允许切在人声上
        valley_idx = int(valley_point_s * self.sample_rate)
        beat_idx = int(nearest_beat_s * self.sample_rate)
        
        win_size = int(0.05 * self.sample_rate) # 50ms能量比较窗口
        
        valley_energy = np.mean(waveform[max(0, valley_idx - win_size//2) : valley_idx + win_size//2]**2)
        beat_energy = np.mean(waveform[max(0, beat_idx - win_size//2) : beat_idx + win_size//2]**2)

        # 关键修复：严格的能量容忍度，优先物理静音
        energy_tolerance_ratio = 1.3  # 降低容忍度，更严格

        if beat_energy <= valley_energy * energy_tolerance_ratio:
            logger.debug(f"  智能对齐：节拍点 {nearest_beat_s:.3f}s 能量验证通过 (Beat={beat_energy:.2e} ≤ Valley*{energy_tolerance_ratio}={valley_energy*energy_tolerance_ratio:.2e})")
            return nearest_beat_s
        else:
            logger.debug(f"  智能对齐拒绝：节拍点能量过高 (Beat={beat_energy:.2e} > Valley*{energy_tolerance_ratio}={valley_energy*energy_tolerance_ratio:.2e})，坚守能量谷")
            return valley_point_s

    def _select_valley_cut_point(self, waveform: np.ndarray, left_idx: int, right_idx: int,
                                 sample_rate: int, local_rms_ms: int, guard_ms: int, floor_percentile: float) -> Optional[int]:
        """在[left_idx, right_idx]内选择RMS谷值切点，带未来静默守卫；若失败返回None。"""
        left_idx = max(0, int(left_idx)); right_idx = min(len(waveform), int(right_idx))
        if right_idx - left_idx <= 8:
            return None
        
        win_samples = max(1, int(local_rms_ms / 1000.0 * sample_rate))
        guard_samples = max(1, int(guard_ms / 1000.0 * sample_rate)) if guard_ms and guard_ms > 0 else 0
        segment = waveform[left_idx:right_idx]
        rms = self._compute_rms_envelope(segment, win_samples)
        floor_val = np.percentile(np.abs(segment), floor_percentile)
        
        # 候选按RMS升序尝试，优先更"静"的点
        order = np.argsort(rms)
        max_try = min(200, len(order))
        # 边界保护：至少距左右边界各20ms，避免贴边切
        margin_samples = max(1, int(0.02 * sample_rate))
        
        for k in range(max_try):
            j = int(order[k])
            if j < margin_samples or j > (len(rms) - margin_samples):
                continue
                
            if guard_samples > 0:
                if self._future_silence_guard(rms, j, guard_samples, floor_val):
                    return left_idx + j
            else:
                return left_idx + j
                
        # 兜底：选择满足边界保护的RMS最小点
        if order.size > 0:
            for jj in order:
                j = int(jj)
                if j >= margin_samples and j <= (len(rms) - margin_samples):
                    return left_idx + j
            j = int(order[0])
            j = max(margin_samples, min(len(rms) - margin_samples, j))
            return left_idx + j
        return None

    def _compute_rms_envelope(self, waveform: np.ndarray, win_samples: int) -> np.ndarray:
        """计算简易滑动RMS包络（same对齐）。"""
        if win_samples <= 1:
            return np.abs(waveform).astype(np.float32)
        kernel = np.ones(int(win_samples), dtype=np.float32) / float(max(1, int(win_samples)))
        return np.sqrt(np.convolve((waveform.astype(np.float32) ** 2), kernel, mode='same'))

    def _future_silence_guard(self, rms: np.ndarray, start_idx: int, guard_samples: int, floor_val: float,
                               allowance: float = 1.2, ratio: float = 0.7) -> bool:
        """未来静默守卫：在 [start_idx, start_idx+guard] 内多数样本低于 floor×allowance。"""
        end_idx = min(start_idx + max(0, int(guard_samples)), len(rms))
        if end_idx <= start_idx:
            return False
        window = rms[start_idx:end_idx]
        if window.size == 0:
            return False
        below = np.sum(window <= (floor_val * allowance))
        return (below / float(window.size)) >= ratio
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
        [v2.3 最终版] 双模式智能裁决系统
        技术: 引入"极度宽松"的初筛，并根据歌曲类型（快/慢）和动态（主/副歌）选择不同的统计策略。
        """
        if not self.enable_bpm_adaptation or not bpm_features or not self.current_adaptive_params:
            # 回退到最简单的静态过滤
            min_pause_duration = get_config('vocal_pause_splitting.min_pause_duration', 1.0)
            min_pause_samples = int(min_pause_duration * self.sample_rate)
            valid_pauses = [p for p in pause_segments if (p['end'] - p['start']) >= min_pause_samples]
            for p in valid_pauses:
                p['duration'] = (p['end'] - p['start']) / self.sample_rate
            logger.info(f"BPM自适应禁用，使用静态阈值 {min_pause_duration}s，过滤后剩 {len(valid_pauses)} 个停顿")
            return valid_pauses

        # === 步骤 1: 极度宽松的初筛，收集所有潜在的"微停顿" ===
        # 核心修复：使用一个非常小且固定的值（如0.3s），而不是动态计算的值，来确保快歌的短气口能进入候选池。
        ABSOLUTE_MIN_PAUSE_S = 0.3
        min_pause_samples = int(ABSOLUTE_MIN_PAUSE_S * self.sample_rate)

        all_candidate_durations = []
        # 在对 all_candidate_durations 进行操作之前，需要先对 pause_segments 里的 duration 进行计算
        for p in pause_segments:
            p['duration'] = (p['end'] - p['start']) / self.sample_rate

        for pause in pause_segments:
            if pause['duration'] >= ABSOLUTE_MIN_PAUSE_S:
                all_candidate_durations.append(pause['duration'])

        if not all_candidate_durations:
            logger.warning("在应用极度宽松的初筛后，仍然没有找到任何候选停顿。歌曲可能过于连续。")
            return []

        # === 步骤 2: 统计学建模，理解这首歌的"停顿语言" ===
        average_pause = np.mean(all_candidate_durations)
        median_pause = np.median(all_candidate_durations)
        std_dev = np.std(all_candidate_durations)
        
        # 使用百分位数为快歌寻找"异常长"的停顿，这通常是真正的分割点
        # 对于快歌，75%的停顿可能都是0.4s的呼吸，而第90%的那个0.8s的停顿才是我们要找的
        percentile_75 = np.percentile(all_candidate_durations, 75)
        percentile_90 = np.percentile(all_candidate_durations, 90)

        logger.info(f"停顿时长统计模型: 平均值={average_pause:.3f}s, 中位数={median_pause:.3f}s, 75分位={percentile_75:.3f}s, 90分位={percentile_90:.3f}s")
        logger.info(f"[DEBUG] 初筛候选数量: {len(all_candidate_durations)}, BPM类别: {getattr(self.current_adaptive_params, 'category', 'unknown')}")

        # === 步骤 3: "双模式"智能裁决 ===
        valid_pauses = []
        total_audio_length = pause_segments[-1]['end'] if pause_segments else 0

        # 获取MDD分析结果，这需要 adaptive_enhancer 在上游被调用并存储结果
        # 我们假设 self.adaptive_enhancer.last_analyzed_segments 存在
        segments_with_mdd = getattr(self.adaptive_enhancer, 'last_analyzed_segments', [])

        for pause in pause_segments:
            duration_s = pause['duration']
            if duration_s < ABSOLUTE_MIN_PAUSE_S:
                continue

            # 确定当前停顿所处的音乐环境 (MDD)
            current_time = (pause['start'] / self.sample_rate)
            current_mdd = 0.5 # 默认中等密度
            if segments_with_mdd:
                for seg in segments_with_mdd:
                    if seg.start_time <= current_time < seg.end_time:
                        current_mdd = seg.dynamic_density_score
                        break
            
            # 决策逻辑
            is_head = (pause.get('start', 0) == 0)
            is_tail = (pause.get('end', 0) >= total_audio_length * 0.95)
            
            # 模式一：快歌裁决 (BPM > 120) - 寻找统计上的"异常长停顿"
            if self.current_adaptive_params.category in ['fast', 'very_fast']:
                # 核心修复：对于快歌，我们的标准是"比大部分呼吸都长"
                # 我们使用75分位数作为基础阈值，因为它能代表这首歌里"比较长"的停顿是多长
                dynamic_threshold = percentile_75 
                mode_type = "快歌模式"
                # 对于非常激烈的副歌部分（高MDD），我们甚至可能需要放宽到中位数，只求有得切
                if current_mdd > 0.7:
                    dynamic_threshold = median_pause
                    mode_type = "快歌+高密度模式"
                
            # 模式二：慢歌/中速歌裁决 (BPM <= 120) - 寻找"足够长且结构合理"的停顿
            else:
                # 对于慢歌，我们使用更严格的标准，要求停顿必须显著长于平均呼吸
                dynamic_threshold = max(average_pause, median_pause)
                mode_type = "慢歌模式"
                # 在激烈的副歌部分（高MDD），我们提高标准，避免乱切
                if current_mdd > 0.6:
                    old_threshold = dynamic_threshold
                    dynamic_threshold *= (1 + (current_mdd - 0.6) * 0.5) # MDD越高，阈值越高
                    mode_type = "慢歌+高密度模式"
                    logger.debug(f"[DEBUG] 慢歌MDD调整: {old_threshold:.3f}s -> {dynamic_threshold:.3f}s (MDD={current_mdd:.2f})")

            # 最终裁决
            final_threshold = max(dynamic_threshold, ABSOLUTE_MIN_PAUSE_S) # 保证不低于绝对下限
            
            # 详细决策日志
            if duration_s >= final_threshold or is_head or is_tail:
                logger.debug(f"[KEEP] @{current_time:.2f}s: {duration_s:.3f}s >= {final_threshold:.3f}s ({mode_type}, MDD={current_mdd:.2f})")
            else:
                logger.debug(f"[FILTER] @{current_time:.2f}s: {duration_s:.3f}s < {final_threshold:.3f}s ({mode_type}, MDD={current_mdd:.2f})")

            if duration_s >= final_threshold or is_head or is_tail:
                valid_pauses.append(pause)

        logger.info(f"双模式智能裁决完成: {len(pause_segments)}个候选 -> {len(valid_pauses)}个最终分割点")
        return valid_pauses
    
    def _get_mdd_score_for_pause(self, pause: Dict) -> float:
        """为停顿点获取MDD（音乐动态密度）评分
        
        Args:
            pause: 停顿信息字典
            
        Returns:
            MDD评分 (0-1, 越高表示音乐越激烈，越不应该切割)
        """
        if not self.adaptive_enhancer or not hasattr(self.adaptive_enhancer, 'last_analyzed_segments'):
            return 0.5  # 默认中等密度
        
        # 计算停顿的中心时间点
        pause_center_time = ((pause['start'] + pause['end']) / 2.0) / self.sample_rate
        
        # 在分析的片段中找到对应的MDD评分
        for segment in self.adaptive_enhancer.last_analyzed_segments:
            if segment.start_time <= pause_center_time < segment.end_time:
                return segment.dynamic_density_score
        
        # 如果没找到对应片段，使用相邻片段的平均值
        segments = self.adaptive_enhancer.last_analyzed_segments
        if not segments:
            return 0.5
        
        # 找到最近的片段
        closest_segment = min(segments, key=lambda s: min(
            abs(s.start_time - pause_center_time),
            abs(s.end_time - pause_center_time)
        ))
        
        return closest_segment.dynamic_density_score
    
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
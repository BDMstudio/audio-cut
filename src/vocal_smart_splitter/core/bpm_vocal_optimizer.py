#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/bpm_vocal_optimizer.py
# AI-SUMMARY: BPM人声优化器 - 结合音乐节拍理论优化停顿检测，实现节拍对齐和音乐风格适配

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import librosa

from ..utils.config_manager import get_config
from ..utils.adaptive_parameter_calculator import AdaptiveParameters

logger = logging.getLogger(__name__)

@dataclass
class BPMInfo:
    """BPM信息结构"""
    bpm: float                      # 每分钟节拍数
    beat_interval: float            # 节拍间隔(秒)
    beat_positions: np.ndarray      # 节拍位置时间戳
    downbeat_positions: np.ndarray  # 强拍位置
    category: str                   # 音乐风格类别
    time_signature: Tuple[int, int]  # 拍号
    phrase_length: int              # 乐句长度(拍数)

@dataclass
class OptimizedPause:
    """优化后的停顿"""
    start_time: float
    end_time: float
    duration: float
    original_start: float  # 原始开始时间
    original_end: float    # 原始结束时间
    cut_point: float       # 优化的切割点
    confidence: float
    alignment_score: float  # 节拍对齐分数
    optimization_reason: str  # 优化原因

class BPMVocalOptimizer:
    """BPM驱动的人声停顿优化器
    
    核心功能：
    1. 节拍对齐 - 将切点调整到音乐节拍边界
    2. 风格适配 - 根据音乐风格调整停顿策略
    3. 时长优化 - 确保片段符合目标时长范围
    4. 乐句感知 - 保持音乐乐句的完整性
    """
    
    def __init__(self, sample_rate: int = 44100):
        """初始化BPM优化器
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        
        # 从配置加载参数
        self.enable_beat_alignment = get_config('bpm_vocal_optimizer.enable_beat_alignment', True)
        self.beat_snap_tolerance = get_config('bpm_vocal_optimizer.beat_snap_tolerance', 0.1)  # 100ms
        self.enable_style_adaptation = get_config('bpm_vocal_optimizer.enable_style_adaptation', True)
        self.target_segment_range = get_config('bpm_vocal_optimizer.target_segment_range', [8, 15])
        self.min_segment_duration = get_config('bpm_vocal_optimizer.min_segment_duration', 5.0)
        self.max_segment_duration = get_config('bpm_vocal_optimizer.max_segment_duration', 20.0)
        
        # 音乐风格参数
        self.style_params = {
            'slow': {
                'breath_filter_threshold': 0.2,  # 保留更多换气
                'phrase_priority': 0.8,          # 优先保持乐句完整
                'beat_alignment_strength': 0.6   # 节拍对齐中等
            },
            'medium': {
                'breath_filter_threshold': 0.3,
                'phrase_priority': 0.7,
                'beat_alignment_strength': 0.7
            },
            'fast': {
                'breath_filter_threshold': 0.4,  # 过滤更多换气
                'phrase_priority': 0.6,
                'beat_alignment_strength': 0.8   # 强节拍对齐
            },
            'very_fast': {
                'breath_filter_threshold': 0.5,
                'phrase_priority': 0.5,
                'beat_alignment_strength': 0.9
            }
        }
        
        logger.info(f"BPM人声优化器初始化完成 (节拍对齐: {self.enable_beat_alignment})")
    
    def optimize_with_bpm(self, pauses: List, audio: np.ndarray,
                         bpm_params: Optional[AdaptiveParameters] = None) -> List[OptimizedPause]:
        """基于BPM优化停顿
        
        Args:
            pauses: 原始停顿列表
            audio: 音频信号
            bpm_params: BPM自适应参数
            
        Returns:
            优化后的停顿列表
        """
        if not pauses:
            return []
        
        logger.info(f"开始BPM优化: {len(pauses)}个停顿")
        
        # 1. 提取BPM信息
        bpm_info = self._extract_bpm_info(audio, bpm_params)
        
        # 2. 节拍对齐
        if self.enable_beat_alignment:
            aligned_pauses = self._align_to_beats(pauses, bpm_info)
        else:
            aligned_pauses = self._convert_to_optimized(pauses)
        
        # 3. 音乐风格适配
        if self.enable_style_adaptation:
            style_adjusted = self._adjust_for_style(aligned_pauses, bpm_info)
        else:
            style_adjusted = aligned_pauses
        
        # 4. 时长优化
        duration_optimized = self._optimize_durations(style_adjusted, bpm_info)
        
        # 5. 乐句边界优化
        phrase_optimized = self._optimize_phrase_boundaries(duration_optimized, bpm_info)
        
        logger.info(f"BPM优化完成: {len(phrase_optimized)}个优化停顿")
        return phrase_optimized
    
    def _extract_bpm_info(self, audio: np.ndarray, 
                         bpm_params: Optional[AdaptiveParameters]) -> BPMInfo:
        """提取BPM信息
        
        Args:
            audio: 音频信号
            bpm_params: 预设的BPM参数
            
        Returns:
            BPM信息
        """
        if bpm_params and hasattr(bpm_params, 'bpm'):
            # 使用预设的BPM
            bpm = float(bpm_params.bpm)
            category = bpm_params.category
            logger.debug(f"使用预设BPM: {bpm:.1f} ({category})")
        else:
            # 自动检测BPM
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            bpm = float(tempo)
            
            # 分类
            if bpm < 70:
                category = 'slow'
            elif bpm < 100:
                category = 'medium'
            elif bpm < 140:
                category = 'fast'
            else:
                category = 'very_fast'
            
            logger.debug(f"自动检测BPM: {bpm:.1f} ({category})")
        
        # 计算节拍间隔
        beat_interval = 60.0 / bpm
        
        # 检测节拍位置
        _, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate, 
                                                 trim=False)
        beat_positions = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
        
        # 估计强拍(假设4/4拍，每4拍一个强拍)
        downbeat_positions = beat_positions[::4] if len(beat_positions) > 4 else beat_positions
        
        # 估计乐句长度(通常8或16拍)
        phrase_length = 8 if category in ['fast', 'very_fast'] else 16
        
        return BPMInfo(
            bpm=bpm,
            beat_interval=beat_interval,
            beat_positions=beat_positions,
            downbeat_positions=downbeat_positions,
            category=category,
            time_signature=(4, 4),  # 假设4/4拍
            phrase_length=phrase_length
        )
    
    def _convert_to_optimized(self, pauses: List) -> List[OptimizedPause]:
        """将原始停顿转换为优化格式
        
        Args:
            pauses: 原始停顿列表
            
        Returns:
            OptimizedPause列表
        """
        optimized = []
        for pause in pauses:
            # 适配不同的输入格式
            if hasattr(pause, 'start_time'):
                start = pause.start_time
                end = pause.end_time
                duration = pause.duration
                confidence = getattr(pause, 'confidence', 0.5)
            else:
                # 字典格式
                start = pause.get('start_time', 0)
                end = pause.get('end_time', 0)
                duration = pause.get('duration', end - start)
                confidence = pause.get('confidence', 0.5)
            
            optimized.append(OptimizedPause(
                start_time=start,
                end_time=end,
                duration=duration,
                original_start=start,
                original_end=end,
                cut_point=(start + end) / 2,  # 默认中点
                confidence=confidence,
                alignment_score=0.0,
                optimization_reason="converted"
            ))
        
        return optimized
    
    def _align_to_beats(self, pauses: List, bpm_info: BPMInfo) -> List[OptimizedPause]:
        """将停顿对齐到节拍
        
        Args:
            pauses: 停顿列表
            bpm_info: BPM信息
            
        Returns:
            对齐后的停顿列表
        """
        aligned = []
        
        for pause in pauses:
            # 转换格式
            if not isinstance(pause, OptimizedPause):
                opt_pause = self._convert_to_optimized([pause])[0]
            else:
                opt_pause = pause
            
            # 找最近的节拍
            cut_time = opt_pause.cut_point
            
            # 计算到各节拍的距离
            distances = np.abs(bpm_info.beat_positions - cut_time)
            nearest_idx = np.argmin(distances)
            nearest_beat = bpm_info.beat_positions[nearest_idx]
            distance = distances[nearest_idx]
            
            # 如果距离在容差范围内，对齐到节拍
            if distance <= self.beat_snap_tolerance:
                # 调整切点
                adjustment = nearest_beat - cut_time
                opt_pause.cut_point = nearest_beat
                
                # 调整开始和结束时间
                opt_pause.start_time += adjustment
                opt_pause.end_time += adjustment
                
                # 计算对齐分数
                opt_pause.alignment_score = 1.0 - (distance / self.beat_snap_tolerance)
                opt_pause.optimization_reason = f"aligned to beat (moved {adjustment:.3f}s)"
                
                logger.debug(f"对齐停顿 {opt_pause.original_start:.2f}s -> "
                           f"节拍 {nearest_beat:.2f}s (移动 {adjustment:.3f}s)")
            else:
                # 检查是否接近强拍
                downbeat_distances = np.abs(bpm_info.downbeat_positions - cut_time)
                if len(downbeat_distances) > 0:
                    nearest_downbeat_idx = np.argmin(downbeat_distances)
                    nearest_downbeat = bpm_info.downbeat_positions[nearest_downbeat_idx]
                    downbeat_distance = downbeat_distances[nearest_downbeat_idx]
                    
                    if downbeat_distance <= self.beat_snap_tolerance * 2:
                        # 对齐到强拍(容差更大)
                        adjustment = nearest_downbeat - cut_time
                        opt_pause.cut_point = nearest_downbeat
                        opt_pause.start_time += adjustment
                        opt_pause.end_time += adjustment
                        opt_pause.alignment_score = 0.8 * (1.0 - downbeat_distance / (self.beat_snap_tolerance * 2))
                        opt_pause.optimization_reason = f"aligned to downbeat (moved {adjustment:.3f}s)"
                    else:
                        opt_pause.alignment_score = 0.0
                        opt_pause.optimization_reason = "no alignment needed"
            
            aligned.append(opt_pause)
        
        return aligned
    
    def _adjust_for_style(self, pauses: List[OptimizedPause], 
                         bpm_info: BPMInfo) -> List[OptimizedPause]:
        """根据音乐风格调整停顿
        
        Args:
            pauses: 停顿列表
            bpm_info: BPM信息
            
        Returns:
            调整后的停顿列表
        """
        style_params = self.style_params.get(bpm_info.category, self.style_params['medium'])
        adjusted = []
        
        for pause in pauses:
            # 根据时长和置信度决定是否保留
            if pause.duration < 0.3:  # 短换气
                # 根据风格决定是否过滤
                if pause.confidence < style_params['breath_filter_threshold']:
                    logger.debug(f"过滤短换气: {pause.start_time:.2f}s (置信度 {pause.confidence:.2f})")
                    continue
            
            # 加强节拍对齐
            if pause.alignment_score > 0:
                pause.confidence *= (1.0 + style_params['beat_alignment_strength'] * pause.alignment_score)
                pause.confidence = min(1.0, pause.confidence)
            
            adjusted.append(pause)
        
        logger.debug(f"风格调整: {len(pauses)} -> {len(adjusted)} 个停顿")
        return adjusted
    
    def _optimize_durations(self, pauses: List[OptimizedPause], 
                          bpm_info: BPMInfo) -> List[OptimizedPause]:
        """优化片段时长
        
        Args:
            pauses: 停顿列表
            bpm_info: BPM信息
            
        Returns:
            时长优化后的停顿列表
        """
        if not pauses:
            return pauses
        
        optimized = []
        
        # 按时间排序
        pauses = sorted(pauses, key=lambda p: p.start_time)
        
        # 计算片段时长
        for i in range(len(pauses)):
            current = pauses[i]
            
            # 计算到下一个停顿的时长
            if i < len(pauses) - 1:
                segment_duration = pauses[i + 1].start_time - current.end_time
            else:
                # 最后一个片段，假设还有10秒音频
                segment_duration = 10.0
            
            # 检查是否在目标范围内
            if self.target_segment_range[0] <= segment_duration <= self.target_segment_range[1]:
                # 理想时长
                current.optimization_reason += f" | ideal duration ({segment_duration:.1f}s)"
                optimized.append(current)
            elif segment_duration < self.min_segment_duration:
                # 太短，考虑合并
                if i < len(pauses) - 1:
                    # 跳过当前停顿，与下一段合并
                    logger.debug(f"跳过停顿以合并片段: {current.start_time:.2f}s")
                    continue
                else:
                    optimized.append(current)
            elif segment_duration > self.max_segment_duration:
                # 太长，考虑分割(这里只标记，实际分割需要额外检测)
                current.optimization_reason += f" | long segment ({segment_duration:.1f}s)"
                optimized.append(current)
            else:
                optimized.append(current)
        
        return optimized
    
    def _optimize_phrase_boundaries(self, pauses: List[OptimizedPause], 
                                   bpm_info: BPMInfo) -> List[OptimizedPause]:
        """优化乐句边界
        
        Args:
            pauses: 停顿列表
            bpm_info: BPM信息
            
        Returns:
            乐句优化后的停顿列表
        """
        if not pauses or not self.enable_beat_alignment:
            return pauses
        
        optimized = []
        phrase_duration = bpm_info.phrase_length * bpm_info.beat_interval
        
        for pause in pauses:
            # 检查是否接近乐句边界
            phrase_position = pause.cut_point % phrase_duration
            distance_to_boundary = min(phrase_position, phrase_duration - phrase_position)
            
            if distance_to_boundary < bpm_info.beat_interval:
                # 接近乐句边界，提高优先级
                pause.confidence *= 1.2
                pause.confidence = min(1.0, pause.confidence)
                pause.optimization_reason += " | phrase boundary"
                logger.debug(f"乐句边界停顿: {pause.cut_point:.2f}s")
            
            optimized.append(pause)
        
        return optimized
    
    def calculate_segment_quality(self, pauses: List[OptimizedPause], 
                                 total_duration: float) -> Dict:
        """计算分割质量指标
        
        Args:
            pauses: 优化后的停顿列表
            total_duration: 音频总时长
            
        Returns:
            质量指标字典
        """
        if not pauses:
            return {
                'num_segments': 0,
                'avg_segment_duration': 0,
                'duration_variance': 0,
                'beat_alignment_rate': 0,
                'quality_score': 0
            }
        
        # 计算片段时长
        segment_durations = []
        pauses = sorted(pauses, key=lambda p: p.start_time)
        
        # 第一个片段
        if pauses[0].start_time > 0:
            segment_durations.append(pauses[0].start_time)
        
        # 中间片段
        for i in range(len(pauses) - 1):
            duration = pauses[i + 1].start_time - pauses[i].end_time
            segment_durations.append(duration)
        
        # 最后一个片段
        if pauses[-1].end_time < total_duration:
            segment_durations.append(total_duration - pauses[-1].end_time)
        
        # 计算指标
        avg_duration = np.mean(segment_durations) if segment_durations else 0
        duration_variance = np.var(segment_durations) if segment_durations else 0
        
        # 节拍对齐率
        aligned_count = sum(1 for p in pauses if p.alignment_score > 0.5)
        alignment_rate = aligned_count / len(pauses) if pauses else 0
        
        # 目标范围内的片段比例
        in_target_range = sum(1 for d in segment_durations 
                            if self.target_segment_range[0] <= d <= self.target_segment_range[1])
        target_rate = in_target_range / len(segment_durations) if segment_durations else 0
        
        # 综合质量分数
        quality_score = (
            0.3 * target_rate +           # 时长合适性
            0.3 * alignment_rate +         # 节拍对齐
            0.2 * (1.0 - min(1.0, duration_variance / 25)) +  # 时长一致性
            0.2 * min(1.0, len(pauses) / 10)  # 片段数量合理性
        )
        
        return {
            'num_segments': len(segment_durations),
            'avg_segment_duration': avg_duration,
            'duration_variance': duration_variance,
            'beat_alignment_rate': alignment_rate,
            'target_range_rate': target_rate,
            'quality_score': quality_score,
            'segment_durations': segment_durations
        }
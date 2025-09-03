#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/pause_priority_splitter.py
# AI-SUMMARY: 基于停顿优先级的精准分割器，优先在最长停顿处分割

"""
停顿优先级分割器

基于用户建议的新算法：
1. 全局扫描所有停顿点
2. 按停顿质量排序（停顿时长为主要因素）
3. 贪心选择最佳停顿点进行分割
4. 先保证精准度，再考虑时长约束
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..utils.config_manager import get_config


@dataclass
class PausePoint:
    """停顿点数据结构"""
    time: float           # 停顿时间点（秒）
    duration: float       # 停顿持续时间（秒）
    vocal_intensity: float # 停顿处人声强度（越低越好）
    energy_drop: float    # 能量下降程度
    spectral_stability: float # 频谱稳定性
    priority_score: float # 综合优先级评分
    selected: bool = False # 是否被选中作为分割点


class PausePrioritySplitter:
    """停顿优先级分割器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("停顿优先级分割器初始化完成")
    
    def create_split_plan(self, vocal_audio: np.ndarray, 
                         sample_rate: int,
                         pause_points: List[Dict],
                         content_segments: List[Dict]) -> Dict:
        """
        创建基于停顿优先级的分割方案
        
        Args:
            vocal_audio: 人声音频数据
            sample_rate: 采样率
            pause_points: 检测到的停顿点
            content_segments: 内容片段信息
            
        Returns:
            分割方案字典
        """
        self.logger.info("开始创建停顿优先级分割方案...")
        
        # 第一阶段：停顿点质量评估和排序
        evaluated_pauses = self._evaluate_pause_quality(
            vocal_audio, sample_rate, pause_points
        )
        
        # 第二阶段：基于优先级选择分割点
        selected_splits = self._select_split_points(evaluated_pauses)
        
        # 第三阶段：生成最终分割方案
        split_plan = self._generate_split_plan(
            selected_splits, len(vocal_audio), sample_rate
        )
        
        self.logger.info(f"停顿优先级分割方案创建完成，共 {len(selected_splits)} 个分割点")
        
        return split_plan
    
    def _evaluate_pause_quality(self, vocal_audio: np.ndarray, 
                               sample_rate: int,
                               pause_points: List[Dict]) -> List[PausePoint]:
        """评估停顿点质量并排序"""
        self.logger.info(f"开始评估 {len(pause_points)} 个停顿点的质量...")
        
        evaluated_pauses = []
        
        for pause in pause_points:
            # 处理不同的停顿点数据格式
            if isinstance(pause, dict):
                start_time = pause['start']
                end_time = pause['end']
            elif isinstance(pause, (tuple, list)) and len(pause) >= 2:
                start_time = pause[0]
                end_time = pause[1]
            else:
                self.logger.warning(f"无法解析停顿点格式: {pause}")
                continue

            duration = end_time - start_time
            
            # 转换为样本索引
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            center_idx = (start_idx + end_idx) // 2
            
            # 计算各项质量指标
            vocal_intensity = self._calculate_vocal_intensity(
                vocal_audio, center_idx, sample_rate
            )
            
            energy_drop = self._calculate_energy_drop(
                vocal_audio, center_idx, sample_rate
            )
            
            spectral_stability = self._calculate_spectral_stability(
                vocal_audio, center_idx, sample_rate
            )
            
            # 计算综合优先级评分
            priority_score = self._calculate_priority_score(
                duration, vocal_intensity, energy_drop, spectral_stability
            )
            
            pause_point = PausePoint(
                time=start_time + duration / 2,  # 停顿中心点
                duration=duration,
                vocal_intensity=vocal_intensity,
                energy_drop=energy_drop,
                spectral_stability=spectral_stability,
                priority_score=priority_score
            )
            
            evaluated_pauses.append(pause_point)
        
        # 按优先级评分排序（从高到低）
        evaluated_pauses.sort(key=lambda x: x.priority_score, reverse=True)
        
        self.logger.info("停顿点质量评估完成，按优先级排序")
        self._log_top_pauses(evaluated_pauses[:10])  # 记录前10个最佳停顿点
        
        return evaluated_pauses
    
    def _calculate_vocal_intensity(self, vocal_audio: np.ndarray, 
                                  center_idx: int, sample_rate: int) -> float:
        """计算停顿点处的人声强度（越低越好）"""
        window_size = int(0.1 * sample_rate)  # 100ms窗口
        start_idx = max(0, center_idx - window_size // 2)
        end_idx = min(len(vocal_audio), center_idx + window_size // 2)
        
        if start_idx >= end_idx:
            return 1.0
        
        window_audio = vocal_audio[start_idx:end_idx]
        rms = np.sqrt(np.mean(window_audio ** 2))
        
        return float(rms)
    
    def _calculate_energy_drop(self, vocal_audio: np.ndarray, 
                              center_idx: int, sample_rate: int) -> float:
        """计算能量下降程度（相对于前后区域）"""
        window_size = int(0.1 * sample_rate)  # 100ms窗口
        context_size = int(0.5 * sample_rate)  # 500ms上下文
        
        # 停顿点能量
        pause_start = max(0, center_idx - window_size // 2)
        pause_end = min(len(vocal_audio), center_idx + window_size // 2)
        pause_energy = np.mean(vocal_audio[pause_start:pause_end] ** 2)
        
        # 前后上下文能量
        before_start = max(0, center_idx - context_size)
        before_end = max(0, center_idx - window_size // 2)
        after_start = min(len(vocal_audio), center_idx + window_size // 2)
        after_end = min(len(vocal_audio), center_idx + context_size)
        
        before_energy = 0
        after_energy = 0
        
        if before_end > before_start:
            before_energy = np.mean(vocal_audio[before_start:before_end] ** 2)
        
        if after_end > after_start:
            after_energy = np.mean(vocal_audio[after_start:after_end] ** 2)
        
        context_energy = (before_energy + after_energy) / 2
        
        if context_energy > 0:
            energy_drop = (context_energy - pause_energy) / context_energy
            return max(0, float(energy_drop))
        
        return 0.0
    
    def _calculate_spectral_stability(self, vocal_audio: np.ndarray, 
                                     center_idx: int, sample_rate: int) -> float:
        """计算频谱稳定性（停顿点前后的频谱相似性）"""
        window_size = int(0.2 * sample_rate)  # 200ms窗口
        
        before_start = max(0, center_idx - window_size)
        before_end = center_idx
        after_start = center_idx
        after_end = min(len(vocal_audio), center_idx + window_size)
        
        if before_end <= before_start or after_end <= after_start:
            return 0.0
        
        # 简化的频谱相似性计算
        before_segment = vocal_audio[before_start:before_end]
        after_segment = vocal_audio[after_start:after_end]
        
        # 使用能量分布作为简化的频谱特征
        before_energy = np.mean(before_segment ** 2)
        after_energy = np.mean(after_segment ** 2)
        
        if before_energy + after_energy > 0:
            similarity = 1 - abs(before_energy - after_energy) / (before_energy + after_energy)
            return float(similarity)
        
        return 1.0
    
    def _calculate_priority_score(self, duration: float, vocal_intensity: float,
                                 energy_drop: float, spectral_stability: float) -> float:
        """计算综合优先级评分"""
        # 权重配置
        duration_weight = get_config('pause_priority.duration_weight', 0.5)
        intensity_weight = get_config('pause_priority.intensity_weight', 0.2)
        energy_weight = get_config('pause_priority.energy_weight', 0.2)
        stability_weight = get_config('pause_priority.stability_weight', 0.1)
        
        # 归一化各项指标
        duration_score = min(1.0, duration / 2.0)  # 2秒为满分
        intensity_score = max(0, 1.0 - vocal_intensity * 10)  # 人声强度越低越好
        energy_score = min(1.0, energy_drop)
        stability_score = spectral_stability
        
        # 计算加权总分
        total_score = (
            duration_score * duration_weight +
            intensity_score * intensity_weight +
            energy_score * energy_weight +
            stability_score * stability_weight
        )
        
        return float(total_score)
    
    def _select_split_points(self, evaluated_pauses: List[PausePoint]) -> List[PausePoint]:
        """基于优先级选择分割点（贪心算法）"""
        self.logger.info("开始选择最佳分割点...")
        
        selected_splits = []
        min_interval = get_config('pause_priority.min_split_interval', 3.0)  # 最小分割间隔
        
        for pause in evaluated_pauses:
            # 检查是否与已选择的分割点冲突
            conflict = False
            for selected in selected_splits:
                if abs(pause.time - selected.time) < min_interval:
                    conflict = True
                    break
            
            if not conflict:
                pause.selected = True
                selected_splits.append(pause)
                self.logger.debug(f"选择分割点: {pause.time:.2f}s, 评分: {pause.priority_score:.3f}")
        
        self.logger.info(f"选择了 {len(selected_splits)} 个分割点")
        return selected_splits
    
    def _generate_split_plan(self, selected_splits: List[PausePoint], 
                           audio_length: int, sample_rate: int) -> Dict:
        """生成最终的分割方案"""
        split_times = [pause.time for pause in selected_splits]
        split_times.sort()
        
        # 添加开始和结束时间
        total_duration = audio_length / sample_rate
        if not split_times or split_times[0] > 1.0:
            split_times.insert(0, 0.0)
        if not split_times or split_times[-1] < total_duration - 1.0:
            split_times.append(total_duration)
        
        # 生成片段信息
        segments = []
        for i in range(len(split_times) - 1):
            start_time = split_times[i]
            end_time = split_times[i + 1]
            
            segments.append({
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'start_sample': int(start_time * sample_rate),
                'end_sample': int(end_time * sample_rate)
            })
        
        # 生成质量控制器期望的分割点格式
        formatted_split_points = []
        for split_time in split_times[1:-1]:  # 排除开始和结束点
            formatted_split_points.append({
                'split_time': split_time,
                'quality_score': 0.8,  # 默认质量评分
                'pause_duration': 0.1,  # 默认停顿时长
                'confidence': 0.8  # 默认置信度
            })

        return {
            'segments': segments,
            'split_points': formatted_split_points,
            'selected_pauses': selected_splits,
            'total_segments': len(segments)
        }
    
    def _log_top_pauses(self, top_pauses: List[PausePoint]):
        """记录最佳停顿点信息"""
        self.logger.info("前10个最佳停顿点:")
        for i, pause in enumerate(top_pauses, 1):
            self.logger.info(
                f"  {i}. 时间: {pause.time:.2f}s, "
                f"时长: {pause.duration:.3f}s, "
                f"评分: {pause.priority_score:.3f}"
            )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/content_analyzer.py
# AI-SUMMARY: 内容分析核心模块，分析人声内容的完整性和连续性

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

from vocal_smart_splitter.utils.config_manager import get_config
from vocal_smart_splitter.utils.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """人声内容分析器，专门分析人声内容的完整性和连续性"""
    
    def __init__(self, sample_rate: int = 22050):
        """初始化内容分析器
        
        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate)
        
        # 从配置加载参数
        self.min_vocal_segment = get_config('content_analysis.min_vocal_segment', 1.0)
        self.max_vocal_segment = get_config('content_analysis.max_vocal_segment', 8.0)
        self.continuity_threshold = get_config('content_analysis.continuity_threshold', 0.5)
        self.semantic_gap_max = get_config('content_analysis.semantic_gap_max', 1.0)
        self.vocal_clarity_min = get_config('content_analysis.vocal_clarity_min', 0.6)
        self.content_completeness = get_config('content_analysis.content_completeness', 0.8)
        
        logger.info("内容分析器初始化完成")
    
    def analyze_vocal_content(self, vocal_track: np.ndarray, 
                             pauses: List[Tuple[float, float, float]]) -> Dict:
        """分析人声内容
        
        Args:
            vocal_track: 人声轨道
            pauses: 检测到的停顿点
            
        Returns:
            内容分析结果
        """
        logger.info("开始分析人声内容...")
        
        try:
            # 1. 识别人声片段
            vocal_segments = self._identify_vocal_segments(vocal_track, pauses)
            
            # 2. 分析内容连续性
            continuity_analysis = self._analyze_continuity(vocal_segments, vocal_track)
            
            # 3. 评估内容完整性
            completeness_analysis = self._analyze_completeness(vocal_segments, vocal_track)
            
            # 4. 分析语义边界
            semantic_boundaries = self._analyze_semantic_boundaries(vocal_segments, vocal_track)
            
            # 5. 生成内容分组建议
            content_groups = self._suggest_content_groups(
                vocal_segments, pauses, continuity_analysis, semantic_boundaries
            )
            
            logger.info(f"内容分析完成，识别到 {len(vocal_segments)} 个人声片段")
            
            return {
                'vocal_segments': vocal_segments,
                'continuity_analysis': continuity_analysis,
                'completeness_analysis': completeness_analysis,
                'semantic_boundaries': semantic_boundaries,
                'content_groups': content_groups,
                'quality_score': self._calculate_content_quality(vocal_segments, continuity_analysis)
            }
            
        except Exception as e:
            logger.error(f"内容分析失败: {e}")
            raise
    
    def _identify_vocal_segments(self, vocal_track: np.ndarray, 
                               pauses: List[Tuple[float, float, float]]) -> List[Dict]:
        """识别人声片段
        
        Args:
            vocal_track: 人声轨道
            pauses: 停顿点
            
        Returns:
            人声片段列表
        """
        total_duration = len(vocal_track) / self.sample_rate
        
        # 创建时间轴上的停顿掩码
        pause_mask = np.zeros(int(total_duration * 100))  # 10ms精度
        
        for start, end, _ in pauses:
            start_idx = int(start * 100)
            end_idx = int(end * 100)
            if end_idx < len(pause_mask):
                pause_mask[start_idx:end_idx] = 1
        
        # 识别人声活动区域
        vocal_activity = self.feature_extractor.detect_voice_activity(vocal_track)
        
        # 将人声活动映射到时间轴
        if len(vocal_activity) != len(pause_mask):
            vocal_activity_resampled = np.interp(
                np.linspace(0, 1, len(pause_mask)),
                np.linspace(0, 1, len(vocal_activity)),
                vocal_activity.astype(float)
            ) > 0.5
        else:
            vocal_activity_resampled = vocal_activity
        
        # 找到人声片段（非停顿且有人声活动的区域）
        vocal_regions = vocal_activity_resampled & (pause_mask == 0)
        
        # 提取连续的人声片段
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, is_vocal in enumerate(vocal_regions):
            time_pos = i / 100.0  # 转换为秒
            
            if is_vocal and not in_segment:
                segment_start = time_pos
                in_segment = True
            elif not is_vocal and in_segment:
                segment_end = time_pos
                duration = segment_end - segment_start
                
                if duration >= self.min_vocal_segment:
                    segment_info = self._analyze_segment(
                        vocal_track, segment_start, segment_end
                    )
                    segments.append(segment_info)
                
                in_segment = False
        
        # 处理最后一个片段
        if in_segment:
            segment_end = total_duration
            duration = segment_end - segment_start
            if duration >= self.min_vocal_segment:
                segment_info = self._analyze_segment(
                    vocal_track, segment_start, segment_end
                )
                segments.append(segment_info)
        
        logger.debug(f"识别到 {len(segments)} 个人声片段")
        return segments
    
    def _analyze_segment(self, vocal_track: np.ndarray, 
                        start_time: float, end_time: float) -> Dict:
        """分析单个人声片段
        
        Args:
            vocal_track: 人声轨道
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            片段分析结果
        """
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        if end_sample > len(vocal_track):
            end_sample = len(vocal_track)
        
        segment_audio = vocal_track[start_sample:end_sample]
        
        # 提取片段特征
        vocal_features = self.feature_extractor.extract_vocal_features(segment_audio)
        energy_features = self.feature_extractor.extract_energy_features(segment_audio)
        
        # 计算片段质量指标
        avg_clarity = np.mean(vocal_features['vocal_clarity'])
        avg_energy = np.mean(energy_features['energy'])
        energy_stability = 1.0 - (np.std(energy_features['energy']) / (np.mean(energy_features['energy']) + 1e-8))
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'clarity': avg_clarity,
            'energy': avg_energy,
            'stability': energy_stability,
            'audio_segment': segment_audio,
            'quality_score': (avg_clarity + energy_stability) / 2
        }
    
    def _analyze_continuity(self, vocal_segments: List[Dict], 
                          vocal_track: np.ndarray) -> Dict:
        """分析内容连续性
        
        Args:
            vocal_segments: 人声片段
            vocal_track: 人声轨道
            
        Returns:
            连续性分析结果
        """
        if len(vocal_segments) < 2:
            return {
                'continuity_score': 1.0,
                'gaps': [],
                'transitions': []
            }
        
        gaps = []
        transitions = []
        
        for i in range(len(vocal_segments) - 1):
            current_segment = vocal_segments[i]
            next_segment = vocal_segments[i + 1]
            
            gap_start = current_segment['end_time']
            gap_end = next_segment['start_time']
            gap_duration = gap_end - gap_start
            
            # 分析间隔
            gap_info = {
                'start_time': gap_start,
                'end_time': gap_end,
                'duration': gap_duration,
                'is_natural': gap_duration <= self.semantic_gap_max,
                'severity': 'low' if gap_duration <= 0.5 else 'medium' if gap_duration <= 1.0 else 'high'
            }
            gaps.append(gap_info)
            
            # 分析过渡
            transition_quality = self._analyze_transition(
                current_segment, next_segment, vocal_track
            )
            transitions.append(transition_quality)
        
        # 计算整体连续性分数
        natural_gaps = sum(1 for gap in gaps if gap['is_natural'])
        continuity_score = natural_gaps / len(gaps) if gaps else 1.0
        
        return {
            'continuity_score': continuity_score,
            'gaps': gaps,
            'transitions': transitions
        }
    
    def _analyze_transition(self, segment1: Dict, segment2: Dict, 
                          vocal_track: np.ndarray) -> Dict:
        """分析两个片段之间的过渡
        
        Args:
            segment1: 第一个片段
            segment2: 第二个片段
            vocal_track: 人声轨道
            
        Returns:
            过渡分析结果
        """
        # 分析能量过渡
        energy_diff = abs(segment2['energy'] - segment1['energy'])
        energy_transition = 'smooth' if energy_diff < 0.1 else 'abrupt'
        
        # 分析清晰度过渡
        clarity_diff = abs(segment2['clarity'] - segment1['clarity'])
        clarity_transition = 'smooth' if clarity_diff < 0.2 else 'abrupt'
        
        # 综合评分
        transition_score = 1.0
        if energy_transition == 'abrupt':
            transition_score -= 0.3
        if clarity_transition == 'abrupt':
            transition_score -= 0.3
        
        return {
            'energy_transition': energy_transition,
            'clarity_transition': clarity_transition,
            'transition_score': max(transition_score, 0.0)
        }
    
    def _analyze_completeness(self, vocal_segments: List[Dict], 
                            vocal_track: np.ndarray) -> Dict:
        """分析内容完整性
        
        Args:
            vocal_segments: 人声片段
            vocal_track: 人声轨道
            
        Returns:
            完整性分析结果
        """
        total_duration = len(vocal_track) / self.sample_rate
        vocal_duration = sum(segment['duration'] for segment in vocal_segments)
        vocal_coverage = vocal_duration / total_duration
        
        # 分析片段质量分布
        quality_scores = [segment['quality_score'] for segment in vocal_segments]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        quality_consistency = 1.0 - (np.std(quality_scores) / (np.mean(quality_scores) + 1e-8)) if quality_scores else 0.0
        
        # 分析片段长度分布
        durations = [segment['duration'] for segment in vocal_segments]
        avg_duration = np.mean(durations) if durations else 0.0
        
        # 完整性评分
        completeness_score = (vocal_coverage + avg_quality + quality_consistency) / 3
        
        return {
            'vocal_coverage': vocal_coverage,
            'avg_quality': avg_quality,
            'quality_consistency': quality_consistency,
            'avg_segment_duration': avg_duration,
            'completeness_score': completeness_score,
            'num_segments': len(vocal_segments)
        }
    
    def _analyze_semantic_boundaries(self, vocal_segments: List[Dict], 
                                   vocal_track: np.ndarray) -> List[Dict]:
        """分析语义边界
        
        Args:
            vocal_segments: 人声片段
            vocal_track: 人声轨道
            
        Returns:
            语义边界列表
        """
        boundaries = []
        
        for i, segment in enumerate(vocal_segments):
            # 分析片段的语义特征
            semantic_strength = self._calculate_semantic_strength(segment)
            
            boundary_info = {
                'segment_index': i,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'semantic_strength': semantic_strength,
                'is_boundary': semantic_strength > self.continuity_threshold,
                'boundary_type': self._classify_boundary_type(segment, semantic_strength)
            }
            
            boundaries.append(boundary_info)
        
        return boundaries
    
    def _calculate_semantic_strength(self, segment: Dict) -> float:
        """计算语义强度
        
        Args:
            segment: 人声片段
            
        Returns:
            语义强度分数
        """
        # 基于多个因素计算语义强度
        clarity_factor = segment['clarity']
        stability_factor = segment['stability']
        duration_factor = min(segment['duration'] / self.max_vocal_segment, 1.0)
        
        # 综合计算
        semantic_strength = (clarity_factor * 0.4 + 
                           stability_factor * 0.3 + 
                           duration_factor * 0.3)
        
        return semantic_strength
    
    def _classify_boundary_type(self, segment: Dict, semantic_strength: float) -> str:
        """分类边界类型
        
        Args:
            segment: 人声片段
            semantic_strength: 语义强度
            
        Returns:
            边界类型
        """
        if semantic_strength > 0.8:
            return 'strong'
        elif semantic_strength > 0.6:
            return 'medium'
        elif semantic_strength > 0.4:
            return 'weak'
        else:
            return 'none'
    
    def _suggest_content_groups(self, vocal_segments: List[Dict],
                              pauses: List[Tuple[float, float, float]],
                              continuity_analysis: Dict,
                              semantic_boundaries: List[Dict]) -> List[Dict]:
        """建议内容分组
        
        Args:
            vocal_segments: 人声片段
            pauses: 停顿点
            continuity_analysis: 连续性分析
            semantic_boundaries: 语义边界
            
        Returns:
            内容分组建议
        """
        groups = []
        current_group = []
        current_group_start = 0
        
        for i, segment in enumerate(vocal_segments):
            current_group.append(segment)
            
            # 检查是否应该结束当前分组
            should_end_group = False
            
            # 基于语义边界
            if i < len(semantic_boundaries):
                boundary = semantic_boundaries[i]
                if boundary['is_boundary'] and boundary['boundary_type'] in ['strong', 'medium']:
                    should_end_group = True
            
            # 基于分组长度
            if current_group:
                group_duration = segment['end_time'] - current_group[0]['start_time']
                if group_duration >= self.max_vocal_segment:
                    should_end_group = True
            
            # 基于连续性间断
            if i < len(continuity_analysis['gaps']):
                gap = continuity_analysis['gaps'][i]
                if not gap['is_natural']:
                    should_end_group = True
            
            # 结束当前分组
            if should_end_group or i == len(vocal_segments) - 1:
                if current_group:
                    group_info = self._create_group_info(current_group, current_group_start)
                    groups.append(group_info)
                    
                    current_group = []
                    current_group_start = i + 1
        
        return groups
    
    def _create_group_info(self, segments: List[Dict], group_index: int) -> Dict:
        """创建分组信息
        
        Args:
            segments: 分组中的片段
            group_index: 分组索引
            
        Returns:
            分组信息
        """
        if not segments:
            return {}
        
        start_time = segments[0]['start_time']
        end_time = segments[-1]['end_time']
        duration = end_time - start_time
        
        # 计算分组质量
        avg_clarity = np.mean([seg['clarity'] for seg in segments])
        avg_stability = np.mean([seg['stability'] for seg in segments])
        group_quality = (avg_clarity + avg_stability) / 2
        
        return {
            'group_index': group_index,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'num_segments': len(segments),
            'segments': segments,
            'avg_clarity': avg_clarity,
            'avg_stability': avg_stability,
            'group_quality': group_quality,
            'is_complete': duration >= self.min_vocal_segment
        }
    
    def _calculate_content_quality(self, vocal_segments: List[Dict], 
                                 continuity_analysis: Dict) -> float:
        """计算内容质量分数
        
        Args:
            vocal_segments: 人声片段
            continuity_analysis: 连续性分析
            
        Returns:
            内容质量分数
        """
        if not vocal_segments:
            return 0.0
        
        # 片段质量分数
        segment_scores = [seg['quality_score'] for seg in vocal_segments]
        avg_segment_quality = np.mean(segment_scores)
        
        # 连续性分数
        continuity_score = continuity_analysis['continuity_score']
        
        # 片段数量合理性
        num_segments = len(vocal_segments)
        if 3 <= num_segments <= 15:
            quantity_score = 1.0
        elif num_segments < 3:
            quantity_score = num_segments / 3.0
        else:
            quantity_score = max(0.5, 15.0 / num_segments)
        
        # 综合评分
        overall_quality = (avg_segment_quality * 0.5 + 
                          continuity_score * 0.3 + 
                          quantity_score * 0.2)
        
        return min(overall_quality, 1.0)

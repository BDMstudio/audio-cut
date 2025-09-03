#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/smart_splitter.py
# AI-SUMMARY: 智能分割核心模块，基于人声内容和停顿进行智能分割决策

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

from vocal_smart_splitter.utils.config_manager import get_config
from .pause_priority_splitter import PausePrioritySplitter
from .precise_voice_splitter import PreciseVoiceSplitter

logger = logging.getLogger(__name__)

class SmartSplitter:
    """智能分割器，基于人声内容进行智能分割决策"""
    
    def __init__(self, sample_rate: int = 22050):
        """初始化智能分割器

        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate

        # 初始化停顿优先级分割器
        self.pause_priority_splitter = PausePrioritySplitter()

        # 初始化精确人声分割器
        self.precise_voice_splitter = PreciseVoiceSplitter(sample_rate)

        # 从配置加载参数
        self.min_segment_length = get_config('smart_splitting.min_segment_length', 5)
        self.max_segment_length = get_config('smart_splitting.max_segment_length', 15)
        self.target_segment_length = get_config('smart_splitting.target_segment_length', 10)
        
        self.prefer_natural_breaks = get_config('smart_splitting.prefer_natural_breaks', True)
        self.allow_content_extension = get_config('smart_splitting.allow_content_extension', True)
        self.strict_time_limit = get_config('smart_splitting.strict_time_limit', False)
        
        self.split_point_buffer = get_config('smart_splitting.split_point_buffer', 0.1)
        self.split_quality_threshold = get_config('smart_splitting.split_quality_threshold', 0.8)
        
        # 优先级权重
        self.complete_phrase_priority = get_config('smart_splitting.complete_phrase_priority', 0.9)
        self.natural_pause_priority = get_config('smart_splitting.natural_pause_priority', 0.8)
        self.time_constraint_priority = get_config('smart_splitting.time_constraint_priority', 0.6)
        
        logger.info("智能分割器初始化完成")
    
    def create_smart_splits(self, audio_duration: float,
                           breath_results: Dict,
                           content_results: Dict,
                           vocal_audio: Optional[np.ndarray] = None,
                           sample_rate: Optional[int] = None) -> List[Dict]:
        """创建智能分割方案
        
        Args:
            audio_duration: 音频总时长
            breath_results: 换气检测结果
            content_results: 内容分析结果
            
        Returns:
            分割方案列表
        """
        logger.info("开始创建智能分割方案...")

        # 检查使用哪种分割算法
        use_pause_priority = get_config('smart_splitting.use_pause_priority_algorithm', False)
        use_precise_voice = get_config('smart_splitting.use_precise_voice_algorithm', True)

        if use_precise_voice and vocal_audio is not None and sample_rate is not None:
            logger.info("使用精确人声分割算法...")
            return self._create_precise_voice_splits(
                vocal_audio, sample_rate, breath_results, content_results
            )
        elif use_pause_priority and vocal_audio is not None and sample_rate is not None:
            logger.info("使用停顿优先级算法...")
            return self._create_pause_priority_splits(
                vocal_audio, sample_rate, breath_results, content_results
            )
        else:
            logger.info("使用传统智能分割算法...")
            return self._create_traditional_splits(audio_duration, breath_results, content_results)

    def _create_precise_voice_splits(self, vocal_audio: np.ndarray,
                                   sample_rate: int,
                                   breath_results: Dict,
                                   content_results: Dict) -> List[Dict]:
        """使用精确人声分割算法创建分割方案 - 改进版本：使用真实人声停顿"""
        try:
            # 提取breath_detector检测到的真实人声停顿点
            breath_pauses = breath_results.get('pauses', [])
            
            # 使用精确人声分割器，并传入真实停顿点
            split_plan = self.precise_voice_splitter.split_by_voice_activity(
                vocal_audio, sample_rate, breath_pauses=breath_pauses
            )

            logger.info(f"精确人声分割: {self.precise_voice_splitter.get_split_summary(split_plan)}")

            return split_plan['split_points']

        except Exception as e:
            logger.error(f"精确人声分割失败: {e}")
            # 回退到停顿优先级算法
            return self._create_pause_priority_splits(
                vocal_audio, sample_rate, breath_results, content_results
            )

    def _create_pause_priority_splits(self, vocal_audio: np.ndarray,
                                     sample_rate: int,
                                     breath_results: Dict,
                                     content_results: Dict) -> List[Dict]:
        """使用停顿优先级算法创建分割方案"""
        try:
            # 使用新的停顿优先级分割器
            split_plan = self.pause_priority_splitter.create_split_plan(
                vocal_audio, sample_rate, breath_results['pauses'], content_results['vocal_segments']
            )

            return split_plan['split_points']

        except Exception as e:
            logger.error(f"停顿优先级分割失败: {e}")
            # 回退到传统算法
            audio_duration = len(vocal_audio) / sample_rate
            return self._create_traditional_splits(audio_duration, breath_results, content_results)

    def _create_traditional_splits(self, audio_duration: float, breath_results: Dict, content_results: Dict) -> List[Dict]:
        """传统的智能分割算法（保持向后兼容）"""
        try:
            # 1. 提取关键信息
            pauses = breath_results['pauses']
            content_groups = content_results['content_groups']
            vocal_segments = content_results['vocal_segments']
            
            # 2. 生成候选分割点
            candidate_splits = self._generate_candidate_splits(
                pauses, content_groups, vocal_segments, audio_duration
            )
            
            # 3. 评估分割点质量
            evaluated_splits = self._evaluate_split_candidates(
                candidate_splits, content_results, breath_results
            )
            
            # 4. 选择最优分割方案
            optimal_splits = self._select_optimal_splits(
                evaluated_splits, audio_duration
            )
            
            # 5. 验证和调整分割方案
            final_splits = self._validate_and_adjust_splits(
                optimal_splits, vocal_segments, pauses
            )
            
            logger.info(f"智能分割方案创建完成，共 {len(final_splits)} 个分割点")
            
            return final_splits
            
        except Exception as e:
            logger.error(f"智能分割方案创建失败: {e}")
            raise
    
    def _generate_candidate_splits(self, pauses: List[Tuple[float, float, float]],
                                 content_groups: List[Dict],
                                 vocal_segments: List[Dict],
                                 audio_duration: float) -> List[Dict]:
        """生成候选分割点
        
        Args:
            pauses: 停顿点
            content_groups: 内容分组
            vocal_segments: 人声片段
            audio_duration: 音频时长
            
        Returns:
            候选分割点列表
        """
        candidates = []
        
        # 1. 基于停顿的候选点
        for start, end, confidence in pauses:
            split_time = (start + end) / 2  # 停顿中心点
            
            candidate = {
                'split_time': split_time,
                'type': 'pause_based',
                'confidence': confidence,
                'pause_info': (start, end, confidence),
                'quality_factors': {
                    'natural_break': True,
                    'pause_confidence': confidence,
                    'content_boundary': False
                }
            }
            candidates.append(candidate)
        
        # 2. 基于内容分组的候选点
        for i, group in enumerate(content_groups):
            if i > 0:  # 不在开头分割
                split_time = group['start_time']
                
                candidate = {
                    'split_time': split_time,
                    'type': 'content_based',
                    'confidence': group['group_quality'],
                    'group_info': group,
                    'quality_factors': {
                        'natural_break': False,
                        'pause_confidence': 0.0,
                        'content_boundary': True,
                        'content_quality': group['group_quality']
                    }
                }
                candidates.append(candidate)
        
        # 3. 基于时间约束的候选点
        if self.strict_time_limit:
            time_based_candidates = self._generate_time_based_candidates(
                vocal_segments, pauses, audio_duration
            )
            candidates.extend(time_based_candidates)
        
        # 按时间排序
        candidates.sort(key=lambda x: x['split_time'])
        
        logger.debug(f"生成了 {len(candidates)} 个候选分割点")
        return candidates
    
    def _generate_time_based_candidates(self, vocal_segments: List[Dict],
                                      pauses: List[Tuple[float, float, float]],
                                      audio_duration: float) -> List[Dict]:
        """生成基于时间约束的候选点
        
        Args:
            vocal_segments: 人声片段
            pauses: 停顿点
            audio_duration: 音频时长
            
        Returns:
            时间约束候选点列表
        """
        candidates = []
        current_time = 0.0
        
        while current_time < audio_duration:
            target_time = current_time + self.target_segment_length
            max_time = current_time + self.max_segment_length
            
            if target_time >= audio_duration:
                break
            
            # 在目标时间附近寻找最佳分割点
            best_split = self._find_best_split_near_time(
                target_time, max_time, pauses, vocal_segments
            )
            
            if best_split:
                candidates.append(best_split)
                current_time = best_split['split_time']
            else:
                current_time = max_time
        
        return candidates
    
    def _find_best_split_near_time(self, target_time: float, max_time: float,
                                 pauses: List[Tuple[float, float, float]],
                                 vocal_segments: List[Dict]) -> Optional[Dict]:
        """在指定时间附近寻找最佳分割点
        
        Args:
            target_time: 目标时间
            max_time: 最大时间
            pauses: 停顿点
            vocal_segments: 人声片段
            
        Returns:
            最佳分割点或None
        """
        best_candidate = None
        best_score = 0.0
        
        # 在停顿中寻找
        for start, end, confidence in pauses:
            split_time = (start + end) / 2
            
            if target_time - 2.0 <= split_time <= max_time:
                # 计算分数：越接近目标时间越好
                time_distance = abs(split_time - target_time)
                time_score = max(0, 1.0 - time_distance / 2.0)
                
                total_score = time_score * 0.6 + confidence * 0.4
                
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = {
                        'split_time': split_time,
                        'type': 'time_constrained',
                        'confidence': total_score,
                        'pause_info': (start, end, confidence),
                        'quality_factors': {
                            'natural_break': True,
                            'pause_confidence': confidence,
                            'time_score': time_score
                        }
                    }
        
        # 如果没有找到合适的停顿，在人声片段边界寻找
        if not best_candidate:
            for segment in vocal_segments:
                for boundary_time in [segment['start_time'], segment['end_time']]:
                    if target_time - 1.0 <= boundary_time <= max_time:
                        time_distance = abs(boundary_time - target_time)
                        time_score = max(0, 1.0 - time_distance / 1.0)
                        
                        if time_score > best_score:
                            best_score = time_score
                            best_candidate = {
                                'split_time': boundary_time,
                                'type': 'segment_boundary',
                                'confidence': time_score * 0.7,
                                'quality_factors': {
                                    'natural_break': False,
                                    'segment_boundary': True,
                                    'time_score': time_score
                                }
                            }
        
        return best_candidate
    
    def _evaluate_split_candidates(self, candidates: List[Dict],
                                 content_results: Dict,
                                 breath_results: Dict) -> List[Dict]:
        """评估分割点候选的质量
        
        Args:
            candidates: 候选分割点
            content_results: 内容分析结果
            breath_results: 换气检测结果
            
        Returns:
            评估后的候选点列表
        """
        evaluated_candidates = []
        
        for candidate in candidates:
            # 计算综合质量分数
            quality_score = self._calculate_split_quality(
                candidate, content_results, breath_results
            )
            
            candidate['quality_score'] = quality_score
            candidate['is_viable'] = quality_score >= self.split_quality_threshold
            
            evaluated_candidates.append(candidate)
        
        # 按质量分数排序
        evaluated_candidates.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return evaluated_candidates
    
    def _calculate_split_quality(self, candidate: Dict,
                               content_results: Dict,
                               breath_results: Dict) -> float:
        """计算分割点质量分数
        
        Args:
            candidate: 候选分割点
            content_results: 内容分析结果
            breath_results: 换气检测结果
            
        Returns:
            质量分数
        """
        split_time = candidate['split_time']
        quality_factors = candidate['quality_factors']
        
        score = 0.0
        
        # 1. 自然停顿优先级
        if quality_factors.get('natural_break', False):
            pause_confidence = quality_factors.get('pause_confidence', 0.0)
            score += self.natural_pause_priority * pause_confidence
        
        # 2. 内容边界优先级
        if quality_factors.get('content_boundary', False):
            content_quality = quality_factors.get('content_quality', 0.0)
            score += self.complete_phrase_priority * content_quality
        
        # 3. 时间约束优先级
        time_score = quality_factors.get('time_score', 0.0)
        score += self.time_constraint_priority * time_score
        
        # 4. 避免在人声活动高峰期分割
        vocal_activity_penalty = self._calculate_vocal_activity_penalty(
            split_time, content_results['vocal_segments']
        )
        score -= vocal_activity_penalty
        
        # 5. 分割点周围的静音质量
        silence_quality = self._calculate_silence_quality(
            split_time, breath_results['pauses']
        )
        score += silence_quality * 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _calculate_vocal_activity_penalty(self, split_time: float,
                                        vocal_segments: List[Dict]) -> float:
        """计算人声活动惩罚分数
        
        Args:
            split_time: 分割时间
            vocal_segments: 人声片段
            
        Returns:
            惩罚分数
        """
        penalty = 0.0
        
        for segment in vocal_segments:
            # 如果分割点在人声片段中间，给予惩罚
            if segment['start_time'] < split_time < segment['end_time']:
                # 计算分割点在片段中的位置
                segment_progress = (split_time - segment['start_time']) / segment['duration']
                
                # 在片段中间分割惩罚最大
                if 0.2 <= segment_progress <= 0.8:
                    penalty += 0.5 * segment['clarity']
                else:
                    penalty += 0.2 * segment['clarity']
                
                break
        
        return penalty
    
    def _calculate_silence_quality(self, split_time: float,
                                 pauses: List[Tuple[float, float, float]]) -> float:
        """计算分割点周围的静音质量
        
        Args:
            split_time: 分割时间
            pauses: 停顿点
            
        Returns:
            静音质量分数
        """
        # 寻找最近的停顿
        min_distance = float('inf')
        closest_pause_confidence = 0.0
        
        for start, end, confidence in pauses:
            # 计算分割点到停顿的距离
            if start <= split_time <= end:
                # 分割点在停顿内
                return confidence
            else:
                # 分割点在停顿外
                distance = min(abs(split_time - start), abs(split_time - end))
                if distance < min_distance:
                    min_distance = distance
                    closest_pause_confidence = confidence
        
        # 基于距离衰减质量分数
        if min_distance < self.split_point_buffer:
            return closest_pause_confidence * (1.0 - min_distance / self.split_point_buffer)
        else:
            return 0.0
    
    def _select_optimal_splits(self, evaluated_candidates: List[Dict],
                             audio_duration: float) -> List[Dict]:
        """选择最优分割方案
        
        Args:
            evaluated_candidates: 评估后的候选点
            audio_duration: 音频时长
            
        Returns:
            最优分割点列表
        """
        selected_splits = []
        current_time = 0.0
        
        while current_time < audio_duration:
            # 寻找下一个最佳分割点
            next_split = self._find_next_optimal_split(
                current_time, evaluated_candidates, audio_duration
            )
            
            if next_split:
                selected_splits.append(next_split)
                current_time = next_split['split_time']
            else:
                break
        
        return selected_splits
    
    def _find_next_optimal_split(self, current_time: float,
                               candidates: List[Dict],
                               audio_duration: float) -> Optional[Dict]:
        """寻找下一个最优分割点
        
        Args:
            current_time: 当前时间
            candidates: 候选点列表
            audio_duration: 音频时长
            
        Returns:
            下一个最优分割点或None
        """
        min_next_time = current_time + self.min_segment_length
        max_next_time = current_time + self.max_segment_length
        target_next_time = current_time + self.target_segment_length
        
        # 如果剩余时间不足，不再分割
        if min_next_time >= audio_duration:
            return None
        
        # 在合适的时间范围内寻找最佳候选点
        viable_candidates = []
        
        for candidate in candidates:
            split_time = candidate['split_time']
            
            # 必须在合理的时间范围内
            if min_next_time <= split_time <= min(max_next_time, audio_duration):
                # 计算时间偏好分数
                time_preference = 1.0 - abs(split_time - target_next_time) / self.target_segment_length
                
                # 综合分数
                combined_score = candidate['quality_score'] * 0.7 + time_preference * 0.3
                
                viable_candidates.append({
                    **candidate,
                    'combined_score': combined_score
                })
        
        if not viable_candidates:
            # 回退策略：
            # 1) 尝试在当前窗口[min_next_time, max_next_time]内选择最近停顿中心
            fallback = None
            best_dist = float('inf')
            for cand in candidates:
                st = cand['split_time']
                if min_next_time <= st <= min(max_next_time, audio_duration):
                    d = abs(st - target_next_time)
                    if d < best_dist:
                        best_dist = d
                        fallback = cand
            if fallback is not None:
                return {**fallback, 'fallback': True}
            return None
        
        # 选择综合分数最高的候选点
        best_candidate = max(viable_candidates, key=lambda x: x['combined_score'])
        
        return best_candidate
    
    def _validate_and_adjust_splits(self, splits: List[Dict],
                                  vocal_segments: List[Dict],
                                  pauses: List[Tuple[float, float, float]]) -> List[Dict]:
        """验证和调整分割方案
        
        Args:
            splits: 分割点列表
            vocal_segments: 人声片段
            pauses: 停顿点
            
        Returns:
            调整后的分割点列表
        """
        adjusted_splits = []
        
        for split in splits:
            # 验证分割点质量
            if self._validate_split_point(split, vocal_segments, pauses):
                # 微调分割点位置
                adjusted_split = self._fine_tune_split_point(split, pauses)
                adjusted_splits.append(adjusted_split)
            else:
                logger.debug(f"分割点 {split['split_time']:.2f}s 验证失败，已移除")
        
        return adjusted_splits
    
    def _validate_split_point(self, split: Dict,
                            vocal_segments: List[Dict],
                            pauses: List[Tuple[float, float, float]]) -> bool:
        """验证分割点的有效性
        
        Args:
            split: 分割点
            vocal_segments: 人声片段
            pauses: 停顿点
            
        Returns:
            是否有效
        """
        split_time = split['split_time']
        
        # 检查是否在人声片段的中心位置（应该避免）
        for segment in vocal_segments:
            if segment['start_time'] < split_time < segment['end_time']:
                segment_progress = (split_time - segment['start_time']) / segment['duration']
                if 0.3 <= segment_progress <= 0.7:  # 在片段中心
                    return False
        
        # 检查质量分数
        if split['quality_score'] < self.split_quality_threshold:
            return False
        
        return True
    
    def _fine_tune_split_point(self, split: Dict,
                             pauses: List[Tuple[float, float, float]]) -> Dict:
        """微调分割点位置
        
        Args:
            split: 分割点
            pauses: 停顿点
            
        Returns:
            调整后的分割点
        """
        split_time = split['split_time']
        
        # 如果分割点接近停顿，调整到停顿中心
        for start, end, confidence in pauses:
            if abs(split_time - (start + end) / 2) < self.split_point_buffer:
                split['split_time'] = (start + end) / 2
                split['fine_tuned'] = True
                break
        
        return split

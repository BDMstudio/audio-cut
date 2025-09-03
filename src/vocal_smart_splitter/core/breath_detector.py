#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/breath_detector.py
# AI-SUMMARY: 换气和停顿检测核心模块，专门检测人声中的自然停顿点

import numpy as np
import librosa
import logging
from typing import List, Tuple, Dict, Optional
from scipy import signal
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter1d

from vocal_smart_splitter.utils.config_manager import get_config
from vocal_smart_splitter.utils.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class BreathDetector:
    """换气和停顿检测器，专门针对人声分析优化"""
    
    def __init__(self, sample_rate: int = 22050):
        """初始化换气检测器
        
        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate)
        
        # 从配置加载参数
        self.energy_window = get_config('breath_detection.energy_window', 0.05)
        self.energy_hop = get_config('breath_detection.energy_hop', 0.01)
        self.energy_threshold = get_config('breath_detection.energy_threshold', 0.02)
        
        self.spectral_window = get_config('breath_detection.spectral_window', 0.1)
        self.spectral_threshold = get_config('breath_detection.spectral_threshold', 0.3)
        
        self.min_pause_duration = get_config('breath_detection.min_pause_duration', 0.15)
        self.max_pause_duration = get_config('breath_detection.max_pause_duration', 2.0)
        self.pause_merge_threshold = get_config('breath_detection.pause_merge_threshold', 0.1)
        
        self.breath_energy_drop = get_config('breath_detection.breath_energy_drop', 0.7)
        self.breath_duration_min = get_config('breath_detection.breath_duration_min', 0.1)
        self.breath_duration_max = get_config('breath_detection.breath_duration_max', 0.8)
        
        logger.info("换气检测器初始化完成")
    
    def detect_breath_points(self, vocal_track: np.ndarray, 
                           original_audio: Optional[np.ndarray] = None) -> Dict:
        """检测换气和停顿点
        
        Args:
            vocal_track: 分离的人声轨道
            original_audio: 原始音频（用于验证）
            
        Returns:
            包含检测结果的字典
        """
        logger.info("开始检测换气和停顿点...")
        
        try:
            # 1. 提取多种特征
            features = self._extract_breath_features(vocal_track)
            
            # 2. 检测能量下降点
            energy_drops = self._detect_energy_drops(features)
            
            # 3. 检测低能量区域
            low_energy_regions = self._detect_low_energy_regions(features)
            
            # 4. 检测频谱变化点
            spectral_changes = self._detect_spectral_changes(vocal_track)
            
            # 5. 综合分析得到候选停顿点
            candidate_pauses = self._combine_detections(
                energy_drops, low_energy_regions, spectral_changes, features
            )
            
            # 6. 验证和过滤停顿点
            validated_pauses = self._validate_pauses(
                candidate_pauses, vocal_track, original_audio
            )
            
            # 7. 分类停顿类型
            classified_pauses = self._classify_pauses(validated_pauses, features)
            
            logger.info(f"检测完成，发现 {len(validated_pauses)} 个有效停顿点")
            
            return {
                'pauses': validated_pauses,
                'classified_pauses': classified_pauses,
                'features': features,
                'quality_score': self._calculate_detection_quality(validated_pauses, features)
            }
            
        except Exception as e:
            logger.error(f"换气检测失败: {e}")
            raise
    
    def _extract_breath_features(self, vocal_track: np.ndarray) -> Dict:
        """提取换气相关特征
        
        Args:
            vocal_track: 人声轨道
            
        Returns:
            特征字典
        """
        # 基础能量特征
        energy_features = self.feature_extractor.extract_energy_features(
            vocal_track, self.energy_window, self.energy_hop
        )
        
        # 人声特征
        vocal_features = self.feature_extractor.extract_vocal_features(vocal_track)
        
        # 换气特征
        breath_features = self.feature_extractor.extract_breath_features(vocal_track)
        
        # 合并所有特征
        combined_features = {
            **energy_features,
            **vocal_features,
            **breath_features
        }
        
        # 确保时间轴一致
        time_axis = energy_features['time_axis']
        combined_features['time_axis'] = time_axis
        
        return combined_features
    
    def _detect_energy_drops(self, features: Dict) -> List[Tuple[float, float]]:
        """检测能量急剧下降的点
        
        Args:
            features: 特征字典
            
        Returns:
            能量下降区域列表 [(开始时间, 结束时间), ...]
        """
        energy = features['energy_smooth']
        energy_derivative = features['energy_derivative']
        time_axis = features['time_axis']
        
        # 检测急剧下降
        drop_threshold = -np.std(energy_derivative) * 2
        drop_points = energy_derivative < drop_threshold
        
        # 形态学处理
        drop_points = binary_opening(drop_points, structure=np.ones(3))
        
        # 找到连续的下降区域
        drop_regions = []
        in_drop = False
        drop_start = 0
        
        for i, is_drop in enumerate(drop_points):
            if is_drop and not in_drop:
                drop_start = time_axis[i]
                in_drop = True
            elif not is_drop and in_drop:
                drop_end = time_axis[i]
                if drop_end - drop_start >= self.breath_duration_min:
                    drop_regions.append((drop_start, drop_end))
                in_drop = False
        
        # 处理最后一个区域
        if in_drop:
            drop_end = time_axis[-1]
            if drop_end - drop_start >= self.breath_duration_min:
                drop_regions.append((drop_start, drop_end))
        
        logger.debug(f"检测到 {len(drop_regions)} 个能量下降区域")
        return drop_regions
    
    def _detect_low_energy_regions(self, features: Dict) -> List[Tuple[float, float]]:
        """检测低能量区域
        
        Args:
            features: 特征字典
            
        Returns:
            低能量区域列表
        """
        energy = features['energy_smooth']
        time_axis = features['time_axis']
        
        # 动态阈值
        energy_nonzero = energy[energy > 0]
        if len(energy_nonzero) > 0:
            dynamic_threshold = np.percentile(energy_nonzero, 15) * self.energy_threshold
        else:
            dynamic_threshold = self.energy_threshold
        
        # 检测低能量点
        low_energy_mask = energy < dynamic_threshold
        
        # 形态学处理
        low_energy_mask = binary_closing(low_energy_mask, structure=np.ones(5))
        low_energy_mask = binary_opening(low_energy_mask, structure=np.ones(3))
        
        # 找到连续的低能量区域
        low_energy_regions = []
        in_low_energy = False
        region_start = 0
        
        for i, is_low in enumerate(low_energy_mask):
            if is_low and not in_low_energy:
                region_start = time_axis[i]
                in_low_energy = True
            elif not is_low and in_low_energy:
                region_end = time_axis[i]
                duration = region_end - region_start
                if self.min_pause_duration <= duration <= self.max_pause_duration:
                    low_energy_regions.append((region_start, region_end))
                in_low_energy = False
        
        # 处理最后一个区域
        if in_low_energy:
            region_end = time_axis[-1]
            duration = region_end - region_start
            if self.min_pause_duration <= duration <= self.max_pause_duration:
                low_energy_regions.append((region_start, region_end))
        
        logger.debug(f"检测到 {len(low_energy_regions)} 个低能量区域")
        return low_energy_regions
    
    def _detect_spectral_changes(self, vocal_track: np.ndarray) -> List[Tuple[float, float]]:
        """检测频谱变化点
        
        Args:
            vocal_track: 人声轨道
            
        Returns:
            频谱变化区域列表
        """
        try:
            # 计算短时傅里叶变换
            hop_length = int(self.spectral_window * self.sample_rate)
            stft = np.abs(librosa.stft(vocal_track, hop_length=hop_length))
            
            # 计算频谱变化
            spectral_diff = np.diff(stft, axis=1)
            spectral_change = np.sum(np.abs(spectral_diff), axis=0)
            
            # 平滑变化曲线
            spectral_change = gaussian_filter1d(spectral_change, sigma=2)
            
            # 检测变化点
            change_threshold = np.mean(spectral_change) + np.std(spectral_change) * self.spectral_threshold
            change_points = spectral_change > change_threshold
            
            # 转换为时间
            time_axis = librosa.frames_to_time(
                np.arange(len(change_points)), sr=self.sample_rate, hop_length=hop_length
            )
            
            # 找到变化区域
            change_regions = []
            in_change = False
            change_start = 0
            
            for i, is_change in enumerate(change_points):
                if is_change and not in_change:
                    change_start = time_axis[i]
                    in_change = True
                elif not is_change and in_change:
                    change_end = time_axis[i]
                    if change_end - change_start >= self.breath_duration_min:
                        change_regions.append((change_start, change_end))
                    in_change = False
            
            logger.debug(f"检测到 {len(change_regions)} 个频谱变化区域")
            return change_regions
            
        except Exception as e:
            logger.warning(f"频谱变化检测失败: {e}")
            return []
    
    def _combine_detections(self, energy_drops: List[Tuple[float, float]],
                           low_energy_regions: List[Tuple[float, float]],
                           spectral_changes: List[Tuple[float, float]],
                           features: Dict) -> List[Tuple[float, float, float]]:
        """综合多种检测结果
        
        Args:
            energy_drops: 能量下降区域
            low_energy_regions: 低能量区域
            spectral_changes: 频谱变化区域
            features: 特征字典
            
        Returns:
            候选停顿点列表 [(开始时间, 结束时间, 置信度), ...]
        """
        candidate_pauses = []
        
        # 收集所有候选区域
        all_regions = []
        
        # 添加能量下降区域（高权重）
        for start, end in energy_drops:
            all_regions.append((start, end, 0.8, 'energy_drop'))
        
        # 添加低能量区域（中等权重）
        for start, end in low_energy_regions:
            all_regions.append((start, end, 0.6, 'low_energy'))
        
        # 添加频谱变化区域（低权重）
        for start, end in spectral_changes:
            all_regions.append((start, end, 0.4, 'spectral_change'))
        
        # 按时间排序
        all_regions.sort(key=lambda x: x[0])
        
        # 合并重叠区域
        merged_regions = self._merge_overlapping_regions(all_regions)
        
        # 计算综合置信度
        for start, end, base_confidence, region_type in merged_regions:
            # 基于区域特征调整置信度
            confidence = self._calculate_pause_confidence(start, end, features, base_confidence)
            
            if confidence > 0.3:  # 最小置信度阈值
                candidate_pauses.append((start, end, confidence))
        
        # 按置信度排序
        candidate_pauses.sort(key=lambda x: x[2], reverse=True)
        
        logger.debug(f"生成 {len(candidate_pauses)} 个候选停顿点")
        return candidate_pauses
    
    def _merge_overlapping_regions(self, regions: List[Tuple[float, float, float, str]]) -> List[Tuple[float, float, float, str]]:
        """合并重叠的区域
        
        Args:
            regions: 区域列表
            
        Returns:
            合并后的区域列表
        """
        if not regions:
            return []
        
        merged = []
        current_start, current_end, current_conf, current_type = regions[0]
        
        for start, end, conf, region_type in regions[1:]:
            if start <= current_end + self.pause_merge_threshold:
                # 合并区域
                current_end = max(current_end, end)
                current_conf = max(current_conf, conf)  # 取最高置信度
            else:
                # 添加当前区域，开始新区域
                merged.append((current_start, current_end, current_conf, current_type))
                current_start, current_end, current_conf, current_type = start, end, conf, region_type
        
        # 添加最后一个区域
        merged.append((current_start, current_end, current_conf, current_type))
        
        return merged
    
    def _calculate_pause_confidence(self, start: float, end: float, 
                                  features: Dict, base_confidence: float) -> float:
        """计算停顿点的置信度
        
        Args:
            start: 开始时间
            end: 结束时间
            features: 特征字典
            base_confidence: 基础置信度
            
        Returns:
            调整后的置信度
        """
        try:
            time_axis = features['time_axis']
            energy = features['energy_smooth']
            vocal_clarity = features['vocal_clarity']
            
            # 找到对应的时间索引
            start_idx = np.argmin(np.abs(time_axis - start))
            end_idx = np.argmin(np.abs(time_axis - end))
            
            if start_idx >= end_idx:
                return base_confidence * 0.5
            
            # 区域内的特征
            region_energy = energy[start_idx:end_idx]
            region_clarity = vocal_clarity[start_idx:end_idx] if len(vocal_clarity) > end_idx else vocal_clarity
            
            # 调整因子
            confidence = base_confidence
            
            # 基于能量的调整
            if len(region_energy) > 0:
                avg_energy = np.mean(region_energy)
                total_avg_energy = np.mean(energy)
                energy_ratio = avg_energy / (total_avg_energy + 1e-8)
                
                if energy_ratio < 0.1:  # 非常低的能量
                    confidence *= 1.2
                elif energy_ratio < 0.3:  # 较低的能量
                    confidence *= 1.1
            
            # 基于人声清晰度的调整
            if len(region_clarity) > 0:
                avg_clarity = np.mean(region_clarity)
                if avg_clarity < 0.2:  # 人声不清晰
                    confidence *= 1.1
            
            # 基于持续时间的调整
            duration = end - start
            if self.min_pause_duration <= duration <= self.breath_duration_max:
                confidence *= 1.0
            elif duration > self.breath_duration_max:
                confidence *= 0.8  # 太长的停顿降低置信度
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"置信度计算失败: {e}")
            return base_confidence * 0.5
    
    def _validate_pauses(self, candidate_pauses: List[Tuple[float, float, float]],
                        vocal_track: np.ndarray,
                        original_audio: Optional[np.ndarray] = None) -> List[Tuple[float, float, float]]:
        """验证停顿点的有效性
        
        Args:
            candidate_pauses: 候选停顿点
            vocal_track: 人声轨道
            original_audio: 原始音频
            
        Returns:
            验证通过的停顿点
        """
        validated_pauses = []
        
        for start, end, confidence in candidate_pauses:
            # 基本验证
            if end - start < self.min_pause_duration:
                continue
            
            if end - start > self.max_pause_duration:
                continue
            
            # 在人声轨道中验证
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            
            if end_sample > len(vocal_track):
                end_sample = len(vocal_track)
            
            if start_sample >= end_sample:
                continue
            
            pause_region = vocal_track[start_sample:end_sample]
            
            # 验证这个区域确实是低能量的
            region_energy = np.sum(pause_region ** 2) / len(pause_region)
            total_energy = np.sum(vocal_track ** 2) / len(vocal_track)
            energy_ratio = region_energy / (total_energy + 1e-8)
            
            if energy_ratio > 0.2:  # 能量太高，不是停顿
                continue
            
            # 如果有原始音频，进一步验证
            if original_audio is not None:
                if end_sample > len(original_audio):
                    end_sample = len(original_audio)
                
                original_region = original_audio[start_sample:end_sample]
                original_energy = np.sum(original_region ** 2) / len(original_region)
                original_total_energy = np.sum(original_audio ** 2) / len(original_audio)
                original_energy_ratio = original_energy / (original_total_energy + 1e-8)
                
                # 在原始音频中也应该相对安静
                if original_energy_ratio > 0.3:
                    confidence *= 0.7  # 降低置信度但不完全排除
            
            validated_pauses.append((start, end, confidence))
        
        logger.debug(f"验证通过 {len(validated_pauses)} 个停顿点")
        return validated_pauses
    
    def _classify_pauses(self, pauses: List[Tuple[float, float, float]], 
                        features: Dict) -> Dict:
        """分类停顿类型
        
        Args:
            pauses: 停顿点列表
            features: 特征字典
            
        Returns:
            分类结果字典
        """
        classified = {
            'breath_pauses': [],  # 换气停顿
            'phrase_pauses': [],  # 短语停顿
            'long_pauses': []     # 长停顿
        }
        
        for start, end, confidence in pauses:
            duration = end - start
            
            if duration <= self.breath_duration_max:
                if confidence > 0.7:
                    classified['breath_pauses'].append((start, end, confidence))
                else:
                    classified['phrase_pauses'].append((start, end, confidence))
            else:
                classified['long_pauses'].append((start, end, confidence))
        
        return classified
    
    def _calculate_detection_quality(self, pauses: List[Tuple[float, float, float]], 
                                   features: Dict) -> float:
        """计算检测质量分数
        
        Args:
            pauses: 检测到的停顿点
            features: 特征字典
            
        Returns:
            质量分数 (0-1)
        """
        if not pauses:
            return 0.0
        
        # 基于停顿数量的评分
        num_pauses = len(pauses)
        if 3 <= num_pauses <= 20:  # 合理的停顿数量
            quantity_score = 1.0
        elif num_pauses < 3:
            quantity_score = num_pauses / 3.0
        else:
            quantity_score = max(0.5, 20.0 / num_pauses)
        
        # 基于平均置信度的评分
        avg_confidence = np.mean([conf for _, _, conf in pauses])
        confidence_score = avg_confidence
        
        # 基于停顿分布的评分
        pause_intervals = []
        for i in range(len(pauses) - 1):
            interval = pauses[i+1][0] - pauses[i][1]
            pause_intervals.append(interval)
        
        if pause_intervals:
            avg_interval = np.mean(pause_intervals)
            if 3 <= avg_interval <= 15:  # 合理的间隔
                distribution_score = 1.0
            else:
                distribution_score = 0.7
        else:
            distribution_score = 0.8
        
        # 综合评分
        overall_score = (quantity_score * 0.4 + confidence_score * 0.4 + distribution_score * 0.2)
        
        return min(overall_score, 1.0)

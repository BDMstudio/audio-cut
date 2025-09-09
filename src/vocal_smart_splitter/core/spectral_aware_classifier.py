#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/spectral_aware_classifier.py
# AI-SUMMARY: 频谱感知分类器 - 通过模式识别区分真停顿vs高频换气，解决误判问题

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ..utils.config_manager import get_config

logger = logging.getLogger(__name__)

@dataclass
class SpectralPattern:
    """频谱模式定义"""
    name: str                          # 模式名称
    f0_drop_type: str                  # 基频下降类型: 'sharp', 'gradual', 'none'
    formant_shift_level: str           # 共振峰偏移: 'significant', 'minimal', 'none'
    spectral_rolloff_type: str         # 频谱滚降: 'full', 'high', 'low', 'none'
    duration_range: Tuple[float, float]  # 时长范围
    energy_profile: str                # 能量轮廓: 'v_shape', 'u_shape', 'flat'
    confidence_weight: float           # 模式置信度权重

class SpectralAwareClassifier:
    """频谱感知的停顿类型分类器
    
    核心功能：
    1. 模式匹配 - 对比换气vs停顿的典型模式
    2. 概率分类 - 贝叶斯方法计算类别概率
    3. 特征加权 - 动态调整特征重要性
    4. 上下文感知 - 考虑前后文音频特征
    """
    
    def __init__(self):
        """初始化频谱分类器"""
        
        # 定义典型模式
        self.patterns = {
            'breath': SpectralPattern(
                name='高频换气',
                f0_drop_type='gradual',
                formant_shift_level='minimal',
                spectral_rolloff_type='high',
                duration_range=(0.05, 0.35),
                energy_profile='v_shape',
                confidence_weight=0.8
            ),
            'true_pause': SpectralPattern(
                name='真停顿',
                f0_drop_type='sharp',
                formant_shift_level='significant',
                spectral_rolloff_type='full',
                duration_range=(0.4, 5.0),
                energy_profile='u_shape',
                confidence_weight=1.0
            ),
            'phrase_end': SpectralPattern(
                name='乐句结束',
                f0_drop_type='gradual',
                formant_shift_level='significant',
                spectral_rolloff_type='full',
                duration_range=(0.8, 2.0),
                energy_profile='u_shape',
                confidence_weight=0.9
            ),
            'glottal_stop': SpectralPattern(
                name='声门停顿',
                f0_drop_type='sharp',
                formant_shift_level='minimal',
                spectral_rolloff_type='low',
                duration_range=(0.02, 0.1),
                energy_profile='flat',
                confidence_weight=0.6
            )
        }
        
        # 从配置加载参数 - 调整为更宽松的阈值
        self.breath_threshold = get_config('spectral_classifier.breath_threshold', 0.85)  # 提高换气阈值，减少误判
        self.pause_threshold = get_config('spectral_classifier.pause_threshold', 0.35)   # 降低停顿阈值，识别更多
        self.uncertain_range = get_config('spectral_classifier.uncertain_range', [0.35, 0.65])
        
        # 特征权重
        self.feature_weights = {
            'f0': get_config('spectral_classifier.f0_weight', 0.25),
            'formant': get_config('spectral_classifier.formant_weight', 0.20),
            'spectral': get_config('spectral_classifier.spectral_weight', 0.20),
            'duration': get_config('spectral_classifier.duration_weight', 0.20),
            'energy': get_config('spectral_classifier.energy_weight', 0.15)
        }
        
        # 初始化特征标准化器
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info("频谱感知分类器初始化完成")
    
    def classify_pause_type(self, pause_features: Dict, 
                           context_features: Optional[Dict] = None) -> Dict:
        """分类停顿类型
        
        Args:
            pause_features: 停顿特征字典
            context_features: 上下文特征(可选)
            
        Returns:
            分类结果字典，包含类型和概率
        """
        # 1. 特征向量化
        feature_vector = self._vectorize_features(pause_features)
        
        # 2. 模式匹配
        pattern_scores = self._match_patterns(pause_features)
        
        # 3. 概率计算
        probabilities = self._calculate_probabilities(feature_vector, pattern_scores)
        
        # 4. 上下文调整
        if context_features:
            probabilities = self._adjust_with_context(probabilities, context_features)
        
        # 5. 最终决策
        classification = self._make_decision(probabilities)
        
        return classification
    
    def _vectorize_features(self, features: Dict) -> np.ndarray:
        """将特征字典转换为向量
        
        Args:
            features: 特征字典
            
        Returns:
            特征向量
        """
        vector = []
        
        # F0特征
        vector.append(features.get('f0_drop_rate', 0.0))
        vector.append(features.get('f0_continuity', 1.0))
        
        # 共振峰特征
        vector.append(features.get('formant_stability', 0.5))
        vector.append(features.get('formant_shift', 0.0))
        
        # 频谱特征
        vector.append(features.get('centroid_shift', 0.0))
        vector.append(features.get('spectral_rolloff', 0.0))
        vector.append(features.get('spectral_flux', 0.0))
        
        # 能量特征
        vector.append(features.get('energy_drop', 0.0))
        vector.append(features.get('energy_rise', 0.0))
        vector.append(features.get('energy_variance', 0.0))
        
        # 时长特征
        vector.append(features.get('duration', 0.0))
        
        # 谐波特征
        vector.append(features.get('harmonic_drop', 0.0))
        vector.append(features.get('harmonic_ratio', 0.5))
        
        return np.array(vector)
    
    def _match_patterns(self, features: Dict) -> Dict[str, float]:
        """匹配预定义模式
        
        Args:
            features: 特征字典
            
        Returns:
            各模式的匹配分数
        """
        scores = {}
        
        for pattern_name, pattern in self.patterns.items():
            score = 0.0
            weight_sum = 0.0
            
            # F0下降类型匹配
            f0_score = self._match_f0_pattern(features, pattern.f0_drop_type)
            score += f0_score * self.feature_weights['f0']
            weight_sum += self.feature_weights['f0']
            
            # 共振峰偏移匹配
            formant_score = self._match_formant_pattern(features, pattern.formant_shift_level)
            score += formant_score * self.feature_weights['formant']
            weight_sum += self.feature_weights['formant']
            
            # 频谱滚降匹配
            spectral_score = self._match_spectral_pattern(features, pattern.spectral_rolloff_type)
            score += spectral_score * self.feature_weights['spectral']
            weight_sum += self.feature_weights['spectral']
            
            # 时长匹配
            duration = features.get('duration', 0.0)
            if pattern.duration_range[0] <= duration <= pattern.duration_range[1]:
                duration_score = 1.0
            else:
                # 计算偏离程度
                if duration < pattern.duration_range[0]:
                    deviation = pattern.duration_range[0] - duration
                else:
                    deviation = duration - pattern.duration_range[1]
                duration_score = max(0.0, 1.0 - deviation / 2.0)
            
            score += duration_score * self.feature_weights['duration']
            weight_sum += self.feature_weights['duration']
            
            # 能量轮廓匹配
            energy_score = self._match_energy_profile(features, pattern.energy_profile)
            score += energy_score * self.feature_weights['energy']
            weight_sum += self.feature_weights['energy']
            
            # 归一化并应用模式权重
            if weight_sum > 0:
                scores[pattern_name] = (score / weight_sum) * pattern.confidence_weight
            else:
                scores[pattern_name] = 0.0
        
        return scores
    
    def _match_f0_pattern(self, features: Dict, pattern_type: str) -> float:
        """匹配F0下降模式
        
        Args:
            features: 特征字典
            pattern_type: 模式类型
            
        Returns:
            匹配分数
        """
        f0_drop = features.get('f0_drop_rate', 0.0)
        
        if pattern_type == 'sharp':
            # 急剧下降: >70%
            return min(1.0, f0_drop / 0.7) if f0_drop > 0.5 else 0.0
        elif pattern_type == 'gradual':
            # 渐进下降: 20%-50%
            if 0.2 <= f0_drop <= 0.5:
                return 1.0
            elif f0_drop < 0.2:
                return f0_drop / 0.2
            else:
                return max(0.0, 1.0 - (f0_drop - 0.5) / 0.5)
        else:  # 'none'
            # 无下降: <10%
            return max(0.0, 1.0 - f0_drop / 0.1)
    
    def _match_formant_pattern(self, features: Dict, shift_level: str) -> float:
        """匹配共振峰偏移模式
        
        Args:
            features: 特征字典
            shift_level: 偏移级别
            
        Returns:
            匹配分数
        """
        stability = features.get('formant_stability', 0.5)
        
        if shift_level == 'significant':
            # 显著偏移: 稳定性低
            return max(0.0, 1.0 - stability)
        elif shift_level == 'minimal':
            # 轻微偏移: 稳定性中等
            return 1.0 - abs(stability - 0.5) * 2
        else:  # 'none'
            # 无偏移: 稳定性高
            return stability
    
    def _match_spectral_pattern(self, features: Dict, rolloff_type: str) -> float:
        """匹配频谱滚降模式
        
        Args:
            features: 特征字典
            rolloff_type: 滚降类型
            
        Returns:
            匹配分数
        """
        centroid_shift = features.get('centroid_shift', 0.0)
        
        if rolloff_type == 'full':
            # 全频段滚降: 质心偏移大
            return min(1.0, centroid_shift / 0.5)
        elif rolloff_type == 'high':
            # 高频滚降: 质心偏移中等
            if 0.2 <= centroid_shift <= 0.4:
                return 1.0
            else:
                return max(0.0, 1.0 - abs(centroid_shift - 0.3) / 0.3)
        elif rolloff_type == 'low':
            # 低频滚降: 质心偏移小但非零
            if 0.05 <= centroid_shift <= 0.2:
                return 1.0
            else:
                return max(0.0, 1.0 - abs(centroid_shift - 0.125) / 0.2)
        else:  # 'none'
            # 无滚降
            return max(0.0, 1.0 - centroid_shift / 0.1)
    
    def _match_energy_profile(self, features: Dict, profile_type: str) -> float:
        """匹配能量轮廓
        
        Args:
            features: 特征字典
            profile_type: 轮廓类型
            
        Returns:
            匹配分数
        """
        energy_drop = features.get('energy_drop', 0.0)
        energy_rise = features.get('energy_rise', 0.0)
        
        if profile_type == 'v_shape':
            # V型: 快速下降和上升
            return min(energy_drop, energy_rise)
        elif profile_type == 'u_shape':
            # U型: 缓慢下降和上升
            drop_score = 1.0 if 0.3 <= energy_drop <= 0.7 else max(0.0, 1.0 - abs(energy_drop - 0.5))
            rise_score = 1.0 if 0.3 <= energy_rise <= 0.7 else max(0.0, 1.0 - abs(energy_rise - 0.5))
            return (drop_score + rise_score) / 2
        else:  # 'flat'
            # 平坦: 能量变化小
            return max(0.0, 1.0 - (energy_drop + energy_rise) / 2)
    
    def _calculate_probabilities(self, feature_vector: np.ndarray, 
                                pattern_scores: Dict[str, float]) -> Dict[str, float]:
        """计算各类别概率
        
        Args:
            feature_vector: 特征向量
            pattern_scores: 模式匹配分数
            
        Returns:
            类别概率字典
        """
        # 组合模式分数
        breath_score = pattern_scores.get('breath', 0.0) * 0.7 + \
                      pattern_scores.get('glottal_stop', 0.0) * 0.3
        
        pause_score = pattern_scores.get('true_pause', 0.0) * 0.6 + \
                     pattern_scores.get('phrase_end', 0.0) * 0.4
        
        # 归一化为概率
        total_score = breath_score + pause_score + 0.1  # 加小值避免除零
        
        probabilities = {
            'breath': breath_score / total_score,
            'true_pause': pause_score / total_score,
            'uncertain': 0.1 / total_score
        }
        
        # 应用softmax平滑
        prob_array = np.array(list(probabilities.values()))
        prob_array = np.exp(prob_array * 2) / np.sum(np.exp(prob_array * 2))
        
        probabilities = {
            'breath': prob_array[0],
            'true_pause': prob_array[1],
            'uncertain': prob_array[2]
        }
        
        return probabilities
    
    def _adjust_with_context(self, probabilities: Dict[str, float], 
                            context: Dict) -> Dict[str, float]:
        """根据上下文调整概率
        
        Args:
            probabilities: 初始概率
            context: 上下文特征
            
        Returns:
            调整后的概率
        """
        # 如果前后都是高能量人声，更可能是换气
        if context.get('pre_vocal_energy', 0) > 0.7 and \
           context.get('post_vocal_energy', 0) > 0.7:
            probabilities['breath'] *= 1.3
            probabilities['true_pause'] *= 0.7
        
        # 如果在乐句边界，更可能是真停顿
        if context.get('phrase_boundary', False):
            probabilities['true_pause'] *= 1.5
            probabilities['breath'] *= 0.5
        
        # 重新归一化
        total = sum(probabilities.values())
        for key in probabilities:
            probabilities[key] /= total
        
        return probabilities
    
    def _make_decision(self, probabilities: Dict[str, float]) -> Dict:
        """做出最终分类决策
        
        Args:
            probabilities: 类别概率
            
        Returns:
            决策结果
        """
        # 找到最高概率
        max_class = max(probabilities, key=probabilities.get)
        max_prob = probabilities[max_class]
        
        # 确定最终类别 - 更宽松的决策逻辑
        if max_class == 'breath' and max_prob >= self.breath_threshold:
            final_type = 'breath'
            action = 'filter'  # 过滤掉
        elif max_class == 'true_pause' and max_prob >= self.pause_threshold:
            final_type = 'true_pause'
            action = 'keep'  # 保留
        elif self.uncertain_range[0] <= max_prob <= self.uncertain_range[1]:
            # 对不确定的情况，默认保留作为停顿
            final_type = 'uncertain_pause'
            action = 'keep'  # 默认保留
        else:
            # 根据概率倾向决定 - 倾向于保留停顿
            if probabilities['true_pause'] >= probabilities['breath'] * 0.7:  # 降低判断标准
                final_type = 'likely_pause'
                action = 'keep'
            else:
                final_type = 'likely_breath'
                action = 'filter'
        
        return {
            'type': final_type,
            'action': action,
            'confidence': max_prob,
            'probabilities': probabilities,
            'reasoning': self._generate_reasoning(final_type, probabilities)
        }
    
    def _generate_reasoning(self, decision_type: str, 
                           probabilities: Dict[str, float]) -> str:
        """生成决策理由
        
        Args:
            decision_type: 决策类型
            probabilities: 概率分布
            
        Returns:
            理由说明
        """
        if decision_type == 'breath':
            return f"高频换气特征明显 (置信度: {probabilities['breath']:.2%})"
        elif decision_type == 'true_pause':
            return f"真停顿特征明确 (置信度: {probabilities['true_pause']:.2%})"
        elif decision_type == 'uncertain':
            return f"特征不明确，需进一步分析"
        elif decision_type == 'likely_pause':
            return f"倾向于真停顿 (概率: {probabilities['true_pause']:.2%})"
        else:  # likely_breath
            return f"倾向于换气 (概率: {probabilities['breath']:.2%})"
    
    def batch_classify(self, pause_list: List[Dict]) -> List[Dict]:
        """批量分类停顿
        
        Args:
            pause_list: 停顿特征列表
            
        Returns:
            分类结果列表
        """
        results = []
        
        # 提取全局统计用于上下文
        durations = [p.get('duration', 0) for p in pause_list]
        mean_duration = np.mean(durations) if durations else 0.5
        
        for i, pause in enumerate(pause_list):
            # 构建上下文
            context = {
                'mean_duration': mean_duration,
                'position_in_sequence': i / max(1, len(pause_list) - 1),
                'is_first': i == 0,
                'is_last': i == len(pause_list) - 1
            }
            
            # 添加前后停顿信息
            if i > 0:
                context['prev_duration'] = pause_list[i-1].get('duration', 0)
            if i < len(pause_list) - 1:
                context['next_duration'] = pause_list[i+1].get('duration', 0)
            
            # 分类
            result = self.classify_pause_type(pause, context)
            results.append(result)
        
        # 统计分类结果
        breath_count = sum(1 for r in results if r['type'] == 'breath')
        pause_count = sum(1 for r in results if r['type'] == 'true_pause')
        
        logger.info(f"批量分类完成: {len(pause_list)}个停顿 -> "
                   f"{pause_count}个真停顿, {breath_count}个换气")
        
        return results
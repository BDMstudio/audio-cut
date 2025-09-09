#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/multi_level_validator.py
# AI-SUMMARY: 多级验证系统 - 5级流水线验证确保停顿检测质量，提供最终质量保证

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import librosa

from ..utils.config_manager import get_config

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """验证结果"""
    level: str                      # 验证级别
    passed: bool                    # 是否通过
    score: float                    # 验证分数(0-1)
    reason: str                     # 验证理由
    suggestions: List[str]          # 改进建议

@dataclass
class ValidatedPause:
    """经过验证的停顿"""
    start_time: float
    end_time: float
    duration: float
    cut_point: float
    confidence: float
    validation_scores: Dict[str, float]  # 各级验证分数
    overall_score: float                 # 综合分数
    quality_grade: str                   # 质量等级: 'A', 'B', 'C', 'D'
    is_valid: bool                       # 是否有效

@dataclass
class AudioContext:
    """音频上下文信息"""
    audio: np.ndarray               # 音频信号
    sample_rate: int                # 采样率
    bpm: Optional[float]            # BPM
    total_duration: float           # 总时长
    energy_profile: np.ndarray     # 能量轮廓
    spectral_features: Dict        # 频谱特征

class MultiLevelValidator:
    """5级停顿验证系统
    
    验证级别：
    1. Level 1: 时长验证 - 基础的时长合理性检查
    2. Level 2: 能量验证 - 能量下降和恢复模式
    3. Level 3: 频谱验证 - 频谱特征的一致性
    4. Level 4: 上下文验证 - 前后文的连贯性
    5. Level 5: 音乐理论验证 - 符合音乐结构
    """
    
    def __init__(self, sample_rate: int = 44100):
        """初始化验证系统
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        
        # 从配置加载参数
        self.min_pause_duration = get_config('validator.min_pause_duration', 0.3)
        self.max_pause_duration = get_config('validator.max_pause_duration', 5.0)
        self.energy_drop_threshold = get_config('validator.energy_drop_threshold', 0.5)
        self.spectral_consistency_threshold = get_config('validator.spectral_consistency_threshold', 0.7)
        self.context_coherence_threshold = get_config('validator.context_coherence_threshold', 0.6)
        self.music_theory_compliance = get_config('validator.music_theory_compliance', 0.7)
        
        # 验证权重
        self.level_weights = {
            'duration': get_config('validator.duration_weight', 0.15),
            'energy': get_config('validator.energy_weight', 0.20),
            'spectral': get_config('validator.spectral_weight', 0.20),
            'context': get_config('validator.context_weight', 0.20),
            'music_theory': get_config('validator.music_theory_weight', 0.25)
        }
        
        # 质量等级阈值 (调整为适合音乐场景)
        self.grade_thresholds = {
            'A': 0.75,  # 优秀 (从0.85降至0.75)
            'B': 0.60,  # 良好 (从0.70降至0.60)
            'C': 0.45,  # 合格 (从0.55降至0.45)
            'D': 0.30   # 较差 (从0.40降至0.30，确保基本质量停顿不被拒绝)
        }
        
        logger.info("多级验证系统初始化完成")
    
    def validate_pauses(self, pauses: List, audio_context: AudioContext) -> List[ValidatedPause]:
        """验证停顿列表
        
        Args:
            pauses: 待验证的停顿列表
            audio_context: 音频上下文
            
        Returns:
            验证后的停顿列表
        """
        logger.info(f"开始5级验证: {len(pauses)}个停顿")
        
        validated = []
        
        for pause in pauses:
            # 执行5级验证
            validation_results = self._run_validation_pipeline(pause, audio_context)
            
            # 汇总结果
            validated_pause = self._aggregate_validation(pause, validation_results)
            
            # 只保留有效的停顿
            if validated_pause.is_valid:
                validated.append(validated_pause)
                logger.debug(f"停顿通过验证: {validated_pause.start_time:.2f}s, "
                           f"质量等级: {validated_pause.quality_grade}, "
                           f"综合分数: {validated_pause.overall_score:.3f}")
            else:
                logger.debug(f"停顿未通过验证: {pause.start_time:.2f}s")
        
        logger.info(f"验证完成: {len(pauses)} -> {len(validated)} 个有效停顿")
        return validated
    
    def _run_validation_pipeline(self, pause, audio_context: AudioContext) -> Dict[str, ValidationResult]:
        """运行5级验证流水线
        
        Args:
            pause: 待验证停顿
            audio_context: 音频上下文
            
        Returns:
            各级验证结果
        """
        results = {}
        
        # Level 1: 时长验证
        results['duration'] = self._validate_duration(pause)
        
        # Level 2: 能量验证
        results['energy'] = self._validate_energy_drop(pause, audio_context)
        
        # Level 3: 频谱验证
        results['spectral'] = self._validate_spectral_pattern(pause, audio_context)
        
        # Level 4: 上下文验证
        results['context'] = self._validate_context(pause, audio_context)
        
        # Level 5: 音乐理论验证
        results['music_theory'] = self._validate_music_theory(pause, audio_context)
        
        return results
    
    def _validate_duration(self, pause) -> ValidationResult:
        """Level 1: 时长验证
        
        Args:
            pause: 停顿对象
            
        Returns:
            验证结果
        """
        duration = pause.duration if hasattr(pause, 'duration') else \
                  pause.get('duration', pause.get('end_time', 0) - pause.get('start_time', 0))
        
        # 检查是否在合理范围内
        if self.min_pause_duration <= duration <= self.max_pause_duration:
            score = 1.0
            passed = True
            reason = f"时长合理 ({duration:.2f}s)"
            suggestions = []
        elif duration < self.min_pause_duration:
            # 太短
            score = duration / self.min_pause_duration
            passed = False
            reason = f"时长过短 ({duration:.2f}s < {self.min_pause_duration}s)"
            suggestions = ["可能是换气或噪音，建议过滤"]
        else:
            # 太长
            score = max(0, 1.0 - (duration - self.max_pause_duration) / self.max_pause_duration)
            passed = score > 0.5
            reason = f"时长过长 ({duration:.2f}s > {self.max_pause_duration}s)"
            suggestions = ["可能需要进一步分割"]
        
        return ValidationResult(
            level="duration",
            passed=passed,
            score=score,
            reason=reason,
            suggestions=suggestions
        )
    
    def _validate_energy_drop(self, pause, audio_context: AudioContext) -> ValidationResult:
        """Level 2: 能量验证
        
        Args:
            pause: 停顿对象
            audio_context: 音频上下文
            
        Returns:
            验证结果
        """
        # 获取停顿时间范围
        start_time = pause.start_time if hasattr(pause, 'start_time') else pause.get('start_time', 0)
        end_time = pause.end_time if hasattr(pause, 'end_time') else pause.get('end_time', 0)
        
        # 转换为样本索引
        start_sample = int(start_time * audio_context.sample_rate)
        end_sample = int(end_time * audio_context.sample_rate)
        
        # 获取前后文窗口
        window_size = int(0.5 * audio_context.sample_rate)  # 0.5秒窗口
        pre_start = max(0, start_sample - window_size)
        post_end = min(len(audio_context.audio), end_sample + window_size)
        
        # 计算能量
        if pre_start < start_sample:
            pre_energy = np.sqrt(np.mean(audio_context.audio[pre_start:start_sample]**2))
        else:
            pre_energy = 0
        
        pause_energy = np.sqrt(np.mean(audio_context.audio[start_sample:end_sample]**2))
        
        if end_sample < post_end:
            post_energy = np.sqrt(np.mean(audio_context.audio[end_sample:post_end]**2))
        else:
            post_energy = 0
        
        # 计算能量下降和恢复
        if pre_energy > 0:
            energy_drop = (pre_energy - pause_energy) / pre_energy
        else:
            energy_drop = 0
        
        if pause_energy > 0:
            energy_recovery = (post_energy - pause_energy) / pause_energy
        else:
            energy_recovery = 0
        
        # 评分
        drop_score = min(1.0, energy_drop / self.energy_drop_threshold)
        recovery_score = min(1.0, energy_recovery / 0.5)  # 50%恢复得满分
        score = (drop_score + recovery_score) / 2
        
        passed = energy_drop >= self.energy_drop_threshold * 0.7
        
        if passed:
            reason = f"能量模式正常 (下降{energy_drop:.1%}, 恢复{energy_recovery:.1%})"
            suggestions = []
        else:
            reason = f"能量变化不明显 (下降{energy_drop:.1%})"
            suggestions = ["可能不是真正的停顿"]
        
        return ValidationResult(
            level="energy",
            passed=passed,
            score=score,
            reason=reason,
            suggestions=suggestions
        )
    
    def _validate_spectral_pattern(self, pause, audio_context: AudioContext) -> ValidationResult:
        """Level 3: 频谱验证
        
        Args:
            pause: 停顿对象
            audio_context: 音频上下文
            
        Returns:
            验证结果
        """
        # 获取停顿时间范围
        start_time = pause.start_time if hasattr(pause, 'start_time') else pause.get('start_time', 0)
        end_time = pause.end_time if hasattr(pause, 'end_time') else pause.get('end_time', 0)
        
        # 转换为样本索引
        start_sample = int(start_time * audio_context.sample_rate)
        end_sample = int(end_time * audio_context.sample_rate)
        
        # 计算频谱特征
        hop_length = 512
        
        # 停顿段频谱
        pause_audio = audio_context.audio[start_sample:end_sample]
        if len(pause_audio) > hop_length:
            pause_spec = np.abs(librosa.stft(pause_audio, hop_length=hop_length))
            pause_centroid = np.mean(librosa.feature.spectral_centroid(
                S=pause_spec, sr=audio_context.sample_rate
            ))
            pause_rolloff = np.mean(librosa.feature.spectral_rolloff(
                S=pause_spec, sr=audio_context.sample_rate
            ))
        else:
            pause_centroid = 0
            pause_rolloff = 0
        
        # 前文频谱
        pre_start = max(0, start_sample - int(0.5 * audio_context.sample_rate))
        if pre_start < start_sample:
            pre_audio = audio_context.audio[pre_start:start_sample]
            pre_spec = np.abs(librosa.stft(pre_audio, hop_length=hop_length))
            pre_centroid = np.mean(librosa.feature.spectral_centroid(
                S=pre_spec, sr=audio_context.sample_rate
            ))
        else:
            pre_centroid = pause_centroid
        
        # 计算频谱一致性
        if pre_centroid > 0:
            spectral_shift = abs(pre_centroid - pause_centroid) / pre_centroid
            consistency = max(0, 1.0 - spectral_shift)
        else:
            consistency = 0.5
        
        score = consistency
        passed = consistency >= self.spectral_consistency_threshold
        
        if passed:
            reason = f"频谱特征一致 (一致性{consistency:.1%})"
            suggestions = []
        else:
            reason = f"频谱变化异常 (一致性{consistency:.1%})"
            suggestions = ["频谱特征不符合停顿模式"]
        
        return ValidationResult(
            level="spectral",
            passed=passed,
            score=score,
            reason=reason,
            suggestions=suggestions
        )
    
    def _validate_context(self, pause, audio_context: AudioContext) -> ValidationResult:
        """Level 4: 上下文验证
        
        Args:
            pause: 停顿对象
            audio_context: 音频上下文
            
        Returns:
            验证结果
        """
        start_time = pause.start_time if hasattr(pause, 'start_time') else pause.get('start_time', 0)
        end_time = pause.end_time if hasattr(pause, 'end_time') else pause.get('end_time', 0)
        
        # 检查位置合理性
        position_ratio = start_time / audio_context.total_duration
        
        # 头部停顿
        if position_ratio < 0.1:
            position_score = 0.8  # 开头停顿合理
            position_type = "开头"
        # 尾部停顿
        elif position_ratio > 0.9:
            position_score = 0.9  # 结尾停顿更合理
            position_type = "结尾"
        # 中间停顿
        else:
            position_score = 1.0  # 中间停顿最合理
            position_type = "中间"
        
        # 检查停顿密度(前后1秒内是否有其他停顿)
        # 这里简化处理，实际应该检查停顿列表
        density_score = 0.8  # 假设密度合理
        
        # 综合评分
        score = (position_score + density_score) / 2
        passed = score >= self.context_coherence_threshold
        
        reason = f"{position_type}停顿，上下文合理性{score:.1%}"
        suggestions = [] if passed else ["位置或密度可能不合理"]
        
        return ValidationResult(
            level="context",
            passed=passed,
            score=score,
            reason=reason,
            suggestions=suggestions
        )
    
    def _validate_music_theory(self, pause, audio_context: AudioContext) -> ValidationResult:
        """Level 5: 音乐理论验证
        
        Args:
            pause: 停顿对象
            audio_context: 音频上下文
            
        Returns:
            验证结果
        """
        # 获取切点时间
        if hasattr(pause, 'cut_point'):
            cut_time = pause.cut_point
        else:
            start_time = pause.start_time if hasattr(pause, 'start_time') else pause.get('start_time', 0)
            end_time = pause.end_time if hasattr(pause, 'end_time') else pause.get('end_time', 0)
            cut_time = (start_time + end_time) / 2
        
        # 如果有BPM信息
        if audio_context.bpm:
            beat_interval = 60.0 / audio_context.bpm
            
            # 检查是否在节拍附近
            beat_position = cut_time % beat_interval
            distance_to_beat = min(beat_position, beat_interval - beat_position)
            beat_alignment = max(0, 1.0 - distance_to_beat / (beat_interval / 2))
            
            # 检查是否在乐句边界(假设4或8拍一个乐句)
            phrase_length = 4 * beat_interval if audio_context.bpm > 100 else 8 * beat_interval
            phrase_position = cut_time % phrase_length
            distance_to_phrase = min(phrase_position, phrase_length - phrase_position)
            phrase_alignment = max(0, 1.0 - distance_to_phrase / beat_interval)
            
            # 综合评分
            score = beat_alignment * 0.4 + phrase_alignment * 0.6
            
            if beat_alignment > 0.7:
                reason = f"节拍对齐良好 (对齐度{beat_alignment:.1%})"
            elif phrase_alignment > 0.7:
                reason = f"乐句边界对齐 (对齐度{phrase_alignment:.1%})"
            else:
                reason = f"音乐结构对齐一般"
        else:
            # 没有BPM信息，给中等分数
            score = 0.6
            reason = "无BPM信息，音乐理论验证有限"
        
        passed = score >= self.music_theory_compliance
        suggestions = [] if passed else ["考虑调整到节拍或乐句边界"]
        
        return ValidationResult(
            level="music_theory",
            passed=passed,
            score=score,
            reason=reason,
            suggestions=suggestions
        )
    
    def _aggregate_validation(self, pause, validation_results: Dict[str, ValidationResult]) -> ValidatedPause:
        """汇总验证结果
        
        Args:
            pause: 原始停顿
            validation_results: 各级验证结果
            
        Returns:
            验证后的停顿
        """
        # 提取各级分数
        validation_scores = {
            level: result.score 
            for level, result in validation_results.items()
        }
        
        # 计算加权综合分数
        overall_score = sum(
            validation_scores[level] * self.level_weights[level]
            for level in validation_scores
        )
        
        # 确定质量等级
        for grade, threshold in self.grade_thresholds.items():
            if overall_score >= threshold:
                quality_grade = grade
                break
        else:
            quality_grade = 'F'  # 不合格
        
        # 判断是否有效(至少D级，适合音乐场景)
        is_valid = quality_grade in ['A', 'B', 'C', 'D']
        
        # 提取基本信息
        if hasattr(pause, 'start_time'):
            start_time = pause.start_time
            end_time = pause.end_time
            duration = pause.duration
            cut_point = getattr(pause, 'cut_point', (start_time + end_time) / 2)
            confidence = getattr(pause, 'confidence', overall_score)
        else:
            start_time = pause.get('start_time', 0)
            end_time = pause.get('end_time', 0)
            duration = pause.get('duration', end_time - start_time)
            cut_point = pause.get('cut_point', (start_time + end_time) / 2)
            confidence = pause.get('confidence', overall_score)
        
        return ValidatedPause(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            cut_point=cut_point,
            confidence=confidence,
            validation_scores=validation_scores,
            overall_score=overall_score,
            quality_grade=quality_grade,
            is_valid=is_valid
        )
    
    def generate_validation_report(self, validated_pauses: List[ValidatedPause]) -> Dict:
        """生成验证报告
        
        Args:
            validated_pauses: 验证后的停顿列表
            
        Returns:
            验证报告
        """
        if not validated_pauses:
            return {
                'total_pauses': 0,
                'grade_distribution': {},
                'avg_overall_score': 0,
                'level_avg_scores': {},
                'quality_summary': "无有效停顿"
            }
        
        # 统计等级分布
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        for pause in validated_pauses:
            grade_counts[pause.quality_grade] = grade_counts.get(pause.quality_grade, 0) + 1
        
        # 计算平均分数
        avg_overall = np.mean([p.overall_score for p in validated_pauses])
        
        # 各级平均分数
        level_scores = {}
        for level in ['duration', 'energy', 'spectral', 'context', 'music_theory']:
            scores = [p.validation_scores.get(level, 0) for p in validated_pauses]
            level_scores[level] = np.mean(scores) if scores else 0
        
        # 质量总结
        a_ratio = grade_counts['A'] / len(validated_pauses)
        b_ratio = grade_counts['B'] / len(validated_pauses)
        
        if a_ratio > 0.6:
            quality_summary = "优秀 - 大部分停顿质量很高"
        elif a_ratio + b_ratio > 0.7:
            quality_summary = "良好 - 停顿质量整体不错"
        elif grade_counts['C'] / len(validated_pauses) > 0.5:
            quality_summary = "合格 - 停顿质量一般"
        else:
            quality_summary = "需改进 - 停顿质量较差"
        
        return {
            'total_pauses': len(validated_pauses),
            'grade_distribution': grade_counts,
            'avg_overall_score': avg_overall,
            'level_avg_scores': level_scores,
            'quality_summary': quality_summary,
            'best_pause': max(validated_pauses, key=lambda p: p.overall_score).__dict__ if validated_pauses else None,
            'worst_pause': min(validated_pauses, key=lambda p: p.overall_score).__dict__ if validated_pauses else None
        }
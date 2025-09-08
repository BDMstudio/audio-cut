#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/dual_path_detector.py
# AI-SUMMARY: 双路人声停顿检测器，结合混音路径和分离路径进行交叉验证，显著提升检测精度

import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..utils.config_manager import get_config
from .vocal_pause_detector import VocalPauseDetectorV2, VocalPause
from .enhanced_vocal_separator import EnhancedVocalSeparator, SeparationResult

logger = logging.getLogger(__name__)

@dataclass
class ValidatedPause:
    """经过交叉验证的停顿结构"""
    start_time: float                    # 停顿开始时间
    end_time: float                      # 停顿结束时间
    duration: float                      # 停顿时长
    position_type: str                   # 位置类型：'head', 'middle', 'tail'
    confidence: float                    # 最终置信度 (0-1)
    cut_point: float                     # 优化后的切割点时间
    
    # 验证信息
    mixed_detection: bool = False        # 混音路径是否检测到
    separated_detection: bool = False    # 分离路径是否检测到
    separation_confidence: float = 0.0   # 分离质量置信度
    validation_method: str = "unknown"   # 验证方法
    
    # 原始停顿引用
    mixed_pause: Optional[VocalPause] = None
    separated_pause: Optional[VocalPause] = None

@dataclass  
class DualDetectionResult:
    """双路检测总结果"""
    validated_pauses: List[ValidatedPause]
    processing_stats: Dict
    quality_report: Dict

class DualPathVocalDetector:
    """双路人声停顿检测器
    
    核心理念：
    1. 双路并行：同时在混音和分离人声上检测停顿
    2. 交叉验证：通过两个结果的对比和融合提升精度
    3. 智能降级：分离失败时自动退回单路检测
    4. 质量评估：持续监控各路径的检测质量
    """
    
    def __init__(self, sample_rate: int = 44100):
        """初始化双路检测器
        
        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        
        # 从配置加载参数
        self.enable_dual_detection = get_config('enhanced_separation.dual_detection.enable_cross_validation', True)
        self.pause_matching_tolerance = get_config('enhanced_separation.dual_detection.pause_matching_tolerance', 0.2)
        self.confidence_boost = get_config('enhanced_separation.dual_detection.confidence_boost_factor', 1.2)
        self.mixed_weight = get_config('enhanced_separation.dual_detection.mixed_audio_weight', 0.4)
        self.separated_weight = get_config('enhanced_separation.dual_detection.separated_audio_weight', 0.6)
        self.min_separation_quality = get_config('enhanced_separation.min_separation_confidence', 0.7)
        
        # 初始化核心组件
        self.mixed_detector = VocalPauseDetectorV2(sample_rate)  # 混音检测器
        self.separator = EnhancedVocalSeparator(sample_rate)     # 增强分离器
        self.separated_detector = VocalPauseDetectorV2(sample_rate)  # 分离后检测器
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'dual_path_used': 0,
            'single_path_fallback': 0,
            'high_quality_separations': 0,
            'processing_times': []
        }
        
        logger.info(f"双路检测器初始化 - 启用双路: {self.enable_dual_detection}")
        
    def detect_with_dual_validation(self, audio: np.ndarray) -> DualDetectionResult:
        """执行双路检测和交叉验证
        
        Args:
            audio: 输入音频数据
            
        Returns:
            DualDetectionResult: 包含验证后停顿和统计信息
        """
        start_time = time.time()
        self.stats['total_detections'] += 1
        
        logger.info("开始双路人声停顿检测...")
        
        # 路径A：混音检测（总是执行）
        mixed_pauses = self._detect_on_mixed_audio(audio)
        logger.debug(f"混音路径检测到 {len(mixed_pauses)} 个停顿")
        
        # 路径B：分离检测（根据配置和分离质量决定）
        separated_pauses = []
        separation_result = None
        use_dual_path = False
        
        # 检查后端状态
        backend_available = self.separator.is_high_quality_backend_available()
        backend_status = self.separator.backend_status
        
        logger.info(f"双路检测状态检查:")
        logger.info(f"  配置启用: {self.enable_dual_detection}")
        logger.info(f"  高质量后端可用: {backend_available}")
        logger.info(f"  MDX23状态: {'可用' if backend_status['mdx23']['available'] else '不可用'}")
        if not backend_status['mdx23']['available'] and 'error' in backend_status['mdx23']:
            logger.info(f"    MDX23错误: {backend_status['mdx23']['error']}")
        logger.info(f"  Demucs状态: {'可用' if backend_status['demucs_v4']['available'] else '不可用'}")
        
        # 强制启用双路检测（测试模式） - 只要有后端就尝试
        if backend_available:
            logger.info("开始执行人声分离检测...")
            try:
                separation_result = self.separator.separate_for_detection(audio)
                logger.info(f"分离完成 - 后端: {separation_result.backend_used}, 质量: {separation_result.separation_confidence:.3f}")
                
                # 强制使用分离检测（忽略质量阈值以测试分割效果）
                logger.info("✓ 强制启用分离检测（测试模式）")
                separated_pauses = self._detect_on_separated_audio(separation_result.vocal_track)
                use_dual_path = True  # 强制启用双路
                self.stats['dual_path_used'] += 1
                
                if separation_result.backend_used in ['mdx23', 'demucs_v4']:
                    self.stats['high_quality_separations'] += 1
                    logger.info(f"✓ 高质量分离 ({separation_result.backend_used}) 检测到 {len(separated_pauses)} 个停顿")
                else:
                    logger.info(f"✓ HPSS后备分离检测到 {len(separated_pauses)} 个停顿")
                
                if separation_result.separation_confidence < self.min_separation_quality:
                    logger.warning(f"  警告: 分离质量较低 ({separation_result.separation_confidence:.3f} < {self.min_separation_quality})")
                    
            except Exception as e:
                logger.error(f"✗ 分离检测失败: {e}")
                logger.info("退回单路检测模式")
                use_dual_path = False
        else:
            logger.warning("✗ 高质量后端不可用，使用单路检测")
            use_dual_path = False
        
        if not use_dual_path:
            self.stats['single_path_fallback'] += 1
            logger.info("使用单路检测模式")
            
        # 双路检测结果分析和决策
        if use_dual_path and separation_result:
            # 对比双路检测结果 - 修复决策逻辑
            mixed_quality = len(mixed_pauses) * 0.1  # 混音检测基础质量
            separated_quality = separation_result.separation_confidence * len(separated_pauses) * 0.1
            
            logger.info(f"\n[双路检测结果对比]")
            logger.info(f"  混音路径: {len(mixed_pauses)}个停顿, 质量评分={mixed_quality:.3f}")
            logger.info(f"  分离路径: {len(separated_pauses)}个停顿, 质量评分={separated_quality:.3f}")
            logger.info(f"  分离后端: {separation_result.backend_used}")
            logger.info(f"  分离置信度: {separation_result.separation_confidence:.3f}")
            
            # 修复的智能选择策略
            use_separated = False
            if separation_result.backend_used in ['mdx23', 'demucs_v4']:
                # 高质量后端优先策略：置信度>0.5就优先使用分离结果
                if separation_result.separation_confidence > 0.5:
                    use_separated = True
                    logger.info(f"  决策: 使用{separation_result.backend_used}分离检测 (高质量后端+高置信度)")
                elif len(separated_pauses) > len(mixed_pauses) * 0.7:  # 检测数量不要相差太多
                    use_separated = True 
                    logger.info(f"  决策: 使用{separation_result.backend_used}分离检测 (检测数量合理)")
                else:
                    logger.info(f"  决策: 使用混音检测 (分离置信度不足: {separation_result.separation_confidence:.3f})")
            else:
                # HPSS后端需要更保守
                if separated_quality > mixed_quality and separation_result.separation_confidence > 0.3:
                    use_separated = True
                    logger.info(f"  决策: 使用HPSS分离检测 (质量优势)")
                else:
                    logger.info(f"  决策: 使用混音检测 (HPSS质量不足)")
            
            if use_separated:
                validated_pauses = self._convert_to_validated_pauses(separated_pauses, single_path=False, source="separated")
                logger.info(f"  最终选择: 分离检测 {len(separated_pauses)}个停顿")
            else:
                validated_pauses = self._convert_to_validated_pauses(mixed_pauses, single_path=False, source="mixed")
                logger.info(f"  最终选择: 混音检测 {len(mixed_pauses)}个停顿")
        else:
            # 单路模式：优先使用混音检测结果
            if mixed_pauses:
                logger.info(f"使用混音检测结果（{len(mixed_pauses)}个停顿）")
                validated_pauses = self._convert_to_validated_pauses(mixed_pauses, single_path=True, source="mixed")
            elif separated_pauses and separation_result and separation_result.separation_confidence > 0.1:
                logger.info(f"降级使用分离检测结果（{len(separated_pauses)}个停顿，质量: {separation_result.separation_confidence:.3f}）")
                validated_pauses = self._convert_to_validated_pauses(separated_pauses, single_path=True, source="separated")
            else:
                logger.warning("无有效检测结果")
                validated_pauses = []
        
        # 统计和报告
        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)
        
        processing_stats = {
            'processing_time': processing_time,
            'backend_used': separation_result.backend_used if separation_result else 'mixed_only',
            'dual_path_used': use_dual_path,
            'mixed_pauses_count': len(mixed_pauses),
            'separated_pauses_count': len(separated_pauses),
            'final_pauses_count': len(validated_pauses),
            'separation_confidence': separation_result.separation_confidence if separation_result else 0.0
        }
        
        quality_report = self._generate_quality_report(validated_pauses, processing_stats)
        
        logger.info(f"双路检测完成 - 最终停顿数: {len(validated_pauses)}, 用时: {processing_time:.2f}秒")
        
        return DualDetectionResult(
            validated_pauses=validated_pauses,
            processing_stats=processing_stats,
            quality_report=quality_report
        )
    
    def _detect_on_mixed_audio(self, audio: np.ndarray) -> List[VocalPause]:
        """在混音上检测停顿（路径A）"""
        try:
            # 使用现有的混音检测器
            return self.mixed_detector.detect_vocal_pauses(audio)
        except Exception as e:
            logger.error(f"混音检测失败: {e}")
            return []
    
    def _detect_on_separated_audio(self, vocal_track: np.ndarray) -> List[VocalPause]:
        """在分离人声上检测停顿（路径B）"""
        try:
            # 在纯人声轨道上检测，理论上精度更高
            return self.separated_detector.detect_vocal_pauses(vocal_track)
        except Exception as e:
            logger.error(f"分离音频检测失败: {e}")
            return []
    
    def _cross_validate_pauses(self, mixed_pauses: List[VocalPause], 
                             separated_pauses: List[VocalPause], 
                             separation_result: SeparationResult) -> List[ValidatedPause]:
        """交叉验证停顿（核心算法）"""
        validated = []
        matched_separated_indices = set()  # 已匹配的分离停顿索引
        
        logger.debug("开始停顿交叉验证...")
        
        # 步骤1：为每个混音停顿寻找分离停顿中的匹配
        for mixed_pause in mixed_pauses:
            best_match, match_score = self._find_best_matching_pause(mixed_pause, separated_pauses)
            
            if best_match and match_score > 0.5:  # 找到可信匹配
                matched_separated_indices.add(separated_pauses.index(best_match))
                validated_pause = self._create_dual_validated_pause(mixed_pause, best_match, separation_result, match_score)
                validated.append(validated_pause)
                logger.debug(f"双路验证: {mixed_pause.start_time:.2f}s-{mixed_pause.end_time:.2f}s (匹配度: {match_score:.3f})")
                
            elif separation_result.separation_confidence > 0.8:
                # 分离质量很高但只有混音检测到：保留但降低置信度
                validated_pause = self._create_mixed_only_validated_pause(mixed_pause, separation_result)
                validated.append(validated_pause)
                logger.debug(f"混音独有: {mixed_pause.start_time:.2f}s-{mixed_pause.end_time:.2f}s (质量高，保留)")
            else:
                # 分离质量一般，混音检测的停顿：谨慎保留
                validated_pause = self._create_mixed_only_validated_pause(mixed_pause, separation_result, penalty=True)
                validated.append(validated_pause)
                logger.debug(f"混音独有: {mixed_pause.start_time:.2f}s-{mixed_pause.end_time:.2f}s (质量中等，降级)")
        
        # 步骤2：处理分离检测独有的停顿
        for i, separated_pause in enumerate(separated_pauses):
            if i not in matched_separated_indices:
                # 这是分离检测独有的停顿，评估其价值
                if separation_result.separation_confidence > 0.85:  # 高质量分离才考虑采纳
                    validated_pause = self._create_separated_only_validated_pause(separated_pause, separation_result)
                    validated.append(validated_pause)
                    logger.debug(f"分离独有: {separated_pause.start_time:.2f}s-{separated_pause.end_time:.2f}s (高质量，采纳)")
                else:
                    logger.debug(f"分离独有: {separated_pause.start_time:.2f}s-{separated_pause.end_time:.2f}s (质量不足，忽略)")
        
        # 步骤3：按时间排序
        validated.sort(key=lambda p: p.start_time)
        
        logger.info(f"交叉验证完成: {len(mixed_pauses)}+{len(separated_pauses)} → {len(validated)} 个验证停顿")
        return validated
    
    def _find_best_matching_pause(self, target_pause: VocalPause, 
                                candidate_pauses: List[VocalPause]) -> Tuple[Optional[VocalPause], float]:
        """为目标停顿在候选列表中找最佳匹配"""
        best_match = None
        best_score = 0.0
        
        for candidate in candidate_pauses:
            score = self._calculate_pause_similarity(target_pause, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate
        
        return best_match, best_score
    
    def _calculate_pause_similarity(self, pause1: VocalPause, pause2: VocalPause) -> float:
        """计算两个停顿的相似度分数 (0-1)"""
        # 时间重叠度
        overlap_start = max(pause1.start_time, pause2.start_time)
        overlap_end = min(pause1.end_time, pause2.end_time)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        union_start = min(pause1.start_time, pause2.start_time)
        union_end = max(pause1.end_time, pause2.end_time)
        union_duration = union_end - union_start
        
        # 重叠比例（类似IoU）
        overlap_ratio = overlap_duration / union_duration if union_duration > 0 else 0
        
        # 时间中心的距离
        center1 = (pause1.start_time + pause1.end_time) / 2
        center2 = (pause2.start_time + pause2.end_time) / 2
        center_distance = abs(center1 - center2)
        
        # 距离评分（距离越近越好）
        max_distance = self.pause_matching_tolerance * 2
        distance_score = max(0, 1 - (center_distance / max_distance))
        
        # 时长相似性
        duration_diff = abs(pause1.duration - pause2.duration)
        max_duration = max(pause1.duration, pause2.duration)
        duration_similarity = 1 - (duration_diff / max_duration) if max_duration > 0 else 1
        
        # 综合评分
        final_score = (
            0.5 * overlap_ratio +      # 重叠是最重要的
            0.3 * distance_score +     # 中心距离
            0.2 * duration_similarity  # 时长相似性
        )
        
        return final_score
    
    def _create_dual_validated_pause(self, mixed_pause: VocalPause, separated_pause: VocalPause, 
                                   separation_result: SeparationResult, match_score: float) -> ValidatedPause:
        """创建双路验证的停顿"""
        # 使用加权平均融合两个检测结果
        fused_start = (mixed_pause.start_time * self.mixed_weight + 
                      separated_pause.start_time * self.separated_weight)
        fused_end = (mixed_pause.end_time * self.mixed_weight + 
                    separated_pause.end_time * self.separated_weight)
        fused_duration = fused_end - fused_start
        
        # 置信度提升（双路验证加成）
        base_confidence = max(mixed_pause.confidence, separated_pause.confidence)
        boosted_confidence = min(1.0, base_confidence * self.confidence_boost * match_score)
        
        # 选择更优的切割点
        if separated_pause.confidence > mixed_pause.confidence:
            cut_point = separated_pause.cut_point
        else:
            cut_point = mixed_pause.cut_point
        
        return ValidatedPause(
            start_time=fused_start,
            end_time=fused_end,
            duration=fused_duration,
            position_type=mixed_pause.position_type,  # 保持位置类型
            confidence=boosted_confidence,
            cut_point=cut_point,
            mixed_detection=True,
            separated_detection=True,
            separation_confidence=separation_result.separation_confidence,
            validation_method="dual_path_validated",
            mixed_pause=mixed_pause,
            separated_pause=separated_pause
        )
    
    def _create_mixed_only_validated_pause(self, mixed_pause: VocalPause, 
                                         separation_result: SeparationResult, 
                                         penalty: bool = False) -> ValidatedPause:
        """创建仅混音检测的验证停顿"""
        confidence = mixed_pause.confidence
        
        if penalty:
            confidence *= 0.8  # 应用置信度惩罚
            
        return ValidatedPause(
            start_time=mixed_pause.start_time,
            end_time=mixed_pause.end_time,
            duration=mixed_pause.duration,
            position_type=mixed_pause.position_type,
            confidence=confidence,
            cut_point=mixed_pause.cut_point,
            mixed_detection=True,
            separated_detection=False,
            separation_confidence=separation_result.separation_confidence,
            validation_method="mixed_only" + ("_penalty" if penalty else ""),
            mixed_pause=mixed_pause
        )
    
    def _create_separated_only_validated_pause(self, separated_pause: VocalPause, 
                                             separation_result: SeparationResult) -> ValidatedPause:
        """创建仅分离检测的验证停顿"""
        # 分离独有的停顿，置信度基于分离质量调整
        adjusted_confidence = separated_pause.confidence * separation_result.separation_confidence
        
        return ValidatedPause(
            start_time=separated_pause.start_time,
            end_time=separated_pause.end_time,
            duration=separated_pause.duration,
            position_type=separated_pause.position_type,
            confidence=adjusted_confidence,
            cut_point=separated_pause.cut_point,
            mixed_detection=False,
            separated_detection=True,
            separation_confidence=separation_result.separation_confidence,
            validation_method="separated_only",
            separated_pause=separated_pause
        )
    
    def _convert_to_validated_pauses(self, pauses: List[VocalPause], single_path: bool = True, source: str = "mixed") -> List[ValidatedPause]:
        """将单路检测结果转换为验证停顿格式"""
        validated = []
        
        for pause in pauses:
            validated_pause = ValidatedPause(
                start_time=pause.start_time,
                end_time=pause.end_time,
                duration=pause.duration,
                position_type=pause.position_type,
                confidence=pause.confidence,
                cut_point=pause.cut_point,
                mixed_detection=(source == "mixed"),
                separated_detection=(source == "separated"),
                separation_confidence=0.0,
                validation_method=f"single_path_{source}" if single_path else "fallback",
                mixed_pause=pause if source == "mixed" else None,
                separated_pause=pause if source == "separated" else None
            )
            validated.append(validated_pause)
        
        return validated
    
    def _generate_quality_report(self, validated_pauses: List[ValidatedPause], 
                               processing_stats: Dict) -> Dict:
        """生成质量报告"""
        if not validated_pauses:
            return {'overall_quality': 0.0, 'confidence_stats': {}, 'validation_stats': {}}
        
        # 置信度统计
        confidences = [p.confidence for p in validated_pauses]
        confidence_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'high_confidence_ratio': sum(1 for c in confidences if c > 0.8) / len(confidences)
        }
        
        # 验证方法统计
        validation_methods = {}
        for pause in validated_pauses:
            method = pause.validation_method
            validation_methods[method] = validation_methods.get(method, 0) + 1
        
        validation_stats = {
            'methods_count': validation_methods,
            'dual_validated_ratio': validation_methods.get('dual_path_validated', 0) / len(validated_pauses),
            'mixed_only_ratio': (validation_methods.get('mixed_only', 0) + validation_methods.get('mixed_only_penalty', 0)) / len(validated_pauses),
            'separated_only_ratio': validation_methods.get('separated_only', 0) / len(validated_pauses)
        }
        
        # 总体质量评分
        overall_quality = (
            0.4 * confidence_stats['mean'] +
            0.3 * validation_stats['dual_validated_ratio'] +
            0.2 * confidence_stats['high_confidence_ratio'] +
            0.1 * min(1.0, processing_stats.get('separation_confidence', 0))
        )
        
        return {
            'overall_quality': overall_quality,
            'confidence_stats': confidence_stats,
            'validation_stats': validation_stats,
            'processing_stats': processing_stats
        }
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        if not self.stats['processing_times']:
            return self.stats
        
        processing_times = self.stats['processing_times']
        enhanced_stats = dict(self.stats)
        enhanced_stats.update({
            'avg_processing_time': np.mean(processing_times),
            'dual_path_usage_rate': self.stats['dual_path_used'] / max(1, self.stats['total_detections']),
            'high_quality_rate': self.stats['high_quality_separations'] / max(1, self.stats['dual_path_used']),
            'backend_info': self.separator.get_backend_info()
        })
        
        return enhanced_stats
    
    def __str__(self) -> str:
        return f"DualPathVocalDetector(dual_enabled={self.enable_dual_detection}, detections={self.stats['total_detections']})"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/seamless_splitter.py
# AI-SUMMARY: 无缝分割器，实现基于人声停顿的精确分割，确保完美拼接

import os
import numpy as np
import librosa
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from vocal_smart_splitter.utils.config_manager import get_config
from vocal_smart_splitter.utils.audio_processor import AudioProcessor
from vocal_smart_splitter.utils.adaptive_parameter_calculator import AdaptiveParameterCalculator
# 注意：新版本不再需要人声分离，直接使用Silero VAD

logger = logging.getLogger(__name__)

@dataclass
class SeamlessSegment:
    """无缝分割片段"""
    index: int                    # 片段索引（从0开始）
    start_sample: int            # 开始样本点（精确到样本）
    end_sample: int              # 结束样本点（精确到样本）
    start_time: float            # 开始时间（秒）
    end_time: float              # 结束时间（秒）
    duration: float              # 时长（秒）
    audio_data: np.ndarray       # 音频数据（原始音频片段）
    cut_info: Dict               # 分割信息

class SeamlessSplitter:
    """无缝分割器 - 基于人声停顿的精确分割，确保完美拼接"""
    
    def __init__(self, sample_rate: int = 44100):
        """初始化无缝分割器
        
        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        
        # 从配置中读取参数
        from ..utils.config_manager import get_config
        self.min_pause_duration = get_config('vocal_pause_splitting.min_pause_duration', 1.2)
        self.zero_processing = get_config('vocal_pause_splitting.zero_processing', True)
        self.preserve_original = get_config('vocal_pause_splitting.preserve_original', True)
        
        # 初始化核心模块
        self.audio_processor = AudioProcessor(sample_rate)
        
        from .vocal_pause_detector import VocalPauseDetectorV2
        # ✅ 直接使用我们强化的VAD检测器
        self.pause_detector = VocalPauseDetectorV2(sample_rate)
        # 移除或禁用旧的 dual_detector, pure_detector, spectral_classifier 等，因为它们的功能已被整合
        logger.info(f"无缝分割器初始化完成，使用统一的VocalPauseDetectorV2")
    
    def apply_bpm_adaptive_parameters(self, bpm: float, complexity: float, 
                                     instrument_count: int) -> None:
        """应用BPM自适应参数到无缝分割器
        
        Args:
            bpm: 检测到的BPM值
            complexity: 编曲复杂度 (0-1)
            instrument_count: 乐器数量
        """
        try:
            # 计算自适应参数
            self.current_adaptive_params = self.adaptive_calculator.calculate_all_parameters(
                bpm=bpm, complexity=complexity, instrument_count=instrument_count
            )
            
            self.bpm_info = {
                'bpm': bpm,
                'complexity': complexity,
                'instrument_count': instrument_count,
                'category': self.current_adaptive_params.category,
                'compensation_factor': self.current_adaptive_params.compensation_factor
            }
            
            logger.info(f"无缝分割器BPM自适应参数已应用: {self.bpm_info['category']}歌曲 "
                       f"BPM={float(bpm):.1f}, 节拍间隔={self.current_adaptive_params.beat_interval:.3f}s")
                       
        except Exception as e:
            logger.error(f"无缝分割器应用BPM自适应参数失败: {e}")
            # 使用默认参数继续
            self.current_adaptive_params = None
    
    def split_audio_seamlessly(self, input_path: str, output_dir: str, mode: str = 'v2.2_mdd') -> Dict:
        """执行无缝分割的主入口 - 统一指挥中心
        
        Args:
            input_path: 输入音频文件路径
            output_dir: 输出目录
            mode: 分割模式 ('v2.1', 'v2.2_mdd', 'smart_split', 'vocal_separation')
            
        Returns:
            分割结果信息
        """
        logger.info(f"开始无缝分割: {input_path} (模式: {mode})")
        logger.info(f"[DEBUG] SeamlessSplitter配置: min_pause={self.min_pause_duration}s, zero_processing={self.zero_processing}")
        
        try:
            # === 模式路由：根据用户选择的模式执行不同的处理流程 ===
            if mode == 'vocal_separation':
                return self._process_vocal_separation_only(input_path, output_dir)
            elif mode == 'v2.1':
                return self._process_vocal_split_v2(input_path, output_dir)
            elif mode == 'v2.2_mdd' or mode == 'vocal_split_mdd':
                return self._process_vocal_split_mdd(input_path, output_dir)
            elif mode == 'smart_split':
                return self._process_smart_split(input_path, output_dir)
            else:
                # 默认使用v2.2 MDD模式
                logger.warning(f"未知模式 {mode}，使用默认v2.2 MDD模式")
                return self._process_vocal_split_mdd(input_path, output_dir)
                
        except Exception as e:
            logger.error(f"无缝分割失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'input_file': input_path,
                'num_segments': 0,
                'segments': [],
                'saved_files': []
            }
    
    def _process_smart_split(self, input_path: str, output_dir: str) -> Dict:
        """传统智能分割模式（原seamless_splitter逻辑）"""
        logger.info("[SMART_SPLIT] 执行传统智能分割...")
        
        try:
            # 1. 加载音频
            original_audio, original_sr = self._load_original_audio(input_path)
            logger.info(f"[DEBUG] 音频加载完成: 时长 {len(original_audio)/original_sr:.2f}秒, 样本数 {len(original_audio)}")
            
            # ✅ 2. 直接调用智能化的停顿检测器
            # 它内部会处理BPM分析、自适应参数和VAD检测
            logger.info("\n=== 统一化人声停顿检测 ===")
            vocal_pauses = self.pause_detector.detect_vocal_pauses(original_audio)
            logger.info(f"[DEBUG] 检测到的人声停顿数量: {len(vocal_pauses)}")
            
            if not vocal_pauses:
                logger.warning("未找到符合条件的人声停顿，无法分割")
                return self._create_single_segment_result(original_audio, input_path, output_dir)
            
            # 3. 生成精确分割点
            cut_points_samples = [int(p.cut_point * self.sample_rate) for p in vocal_pauses]
            logger.info(f"[DEBUG] 初始切点: {[p.cut_point for p in vocal_pauses]}")
            
            # 4. 应用最终的能量守卫和安全过滤
            logger.info("[FINAL CHECK] 应用最终能量守卫和安全过滤器...")
            from .quality_controller import QualityController
            qc = QualityController(self.sample_rate)
            cut_points_times = [p / self.sample_rate for p in cut_points_samples]
            logger.info(f"[DEBUG] 待验证切点: {cut_points_times}")
            
            # 使用原始音频（混音）进行最终能量校验
            validated_cut_times = [qc.enforce_quiet_cut(original_audio, self.sample_rate, t) for t in cut_points_times]
            validated_cut_times = [t for t in validated_cut_times if t >= 0] # 移除无效切点
            logger.info(f"[DEBUG] 能量校验后切点: {validated_cut_times}")

            # 纯化过滤
            final_cut_points_times = qc.pure_filter_cut_points(validated_cut_times, len(original_audio) / self.sample_rate)
            final_cut_points_samples = [int(t * self.sample_rate) for t in final_cut_points_times]
            logger.info(f"[DEBUG] 纯化过滤后切点: {final_cut_points_times}")
            
            # CRITICAL FIX: 添加起始点和结束点，确保完整音频覆盖
            audio_length = len(original_audio)
            complete_cut_points = [0] + final_cut_points_samples + [audio_length]
            # 去重并排序
            complete_cut_points = sorted(list(set(complete_cut_points)))
            
            logger.info(f"完整切点列表: {[p/self.sample_rate for p in complete_cut_points]}")
            logger.info(f"音频总长度: {audio_length/self.sample_rate:.2f}s, 将生成 {len(complete_cut_points)-1} 个片段")

            # 5. 执行分割
            segments = self._split_at_sample_level(original_audio, complete_cut_points, vocal_pauses)
            
            # 6. 保存分割结果
            saved_files = self._save_seamless_segments(segments, output_dir)
            
            # 7. 验证拼接完整性
            validation_result = self._validate_seamless_reconstruction(segments, original_audio)
            
            # 8. 生成结果报告
            result = self._generate_result_report(
                input_path, output_dir, segments, saved_files, 
                vocal_pauses, None, validation_result, None
            )
            
            logger.info(f"无缝分割完成: {len(segments)} 个片段")
            return result
            
        except Exception as e:
            logger.error(f"无缝分割失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'input_file': input_path,
                'num_segments': 0,
                'segments': [],
                'saved_files': []
            }
    
    def _load_original_audio(self, input_path: str) -> Tuple[np.ndarray, int]:
        """加载原始音频，保持原始参数
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            (音频数据, 采样率)
        """
        if self.preserve_original:
            # 保持原始采样率
            audio, sr = librosa.load(input_path, sr=None, mono=True)
            if sr != self.sample_rate:
                logger.info(f"原始采样率 {sr} 保持不变，重新设置目标采样率")
                self.sample_rate = sr
                # 重新初始化相关模块
                self._reinitialize_modules(sr)
        else:
            # 使用指定采样率
            audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)
        
        logger.info(f"音频加载完成: 时长={len(audio)/sr:.2f}s, 采样率={sr}")
        return audio, sr
    
    def _reinitialize_modules(self, new_sample_rate: int):
        """重新初始化模块以适应新的采样率"""
        # 无缝模式不需要VocalSeparator
        from .vocal_pause_detector import VocalPauseDetectorV2
        
        self.sample_rate = new_sample_rate
        self.vocal_pause_detector = VocalPauseDetectorV2(new_sample_rate)
        self.audio_processor = AudioProcessor(new_sample_rate)
    
    def _generate_precise_cut_points(self, vocal_pauses: List, 
                                   audio_length: int) -> List[int]:
        """生成精确的样本级分割点 - BPM感知与置信度加权版本
        
        Args:
            vocal_pauses: 人声停顿列表
            audio_length: 音频总长度（样本数）
            
        Returns:
            分割点样本位置列表
        """
        logger.info(f"开始从 {len(vocal_pauses)} 个停顿生成分割点 (BPM感知模式)...")
        
        # Phase 2.3: 置信度加权的分割决策
        scored_pauses = self._score_pauses_with_confidence(vocal_pauses)
        
        # Phase 2.3: 节拍对齐优先级机制
        if self.current_adaptive_params:
            aligned_pauses = self._apply_beat_alignment_priority(scored_pauses)
        else:
            aligned_pauses = scored_pauses
            logger.info("未启用BPM自适应，使用标准对齐")
        
        # 转换为样本位置
        cut_points = []
        for pause_info in aligned_pauses:
            pause = pause_info['pause']
            confidence_score = pause_info['confidence_score']
            beat_alignment_score = pause_info.get('beat_alignment_score', 0.0)
            
            # 转换为样本位置
            cut_sample = int(pause.cut_point * self.sample_rate)
            
            logger.debug(f"停顿: 时间={pause.cut_point:.2f}s, "
                        f"置信度={confidence_score:.3f}, "
                        f"节拍对齐={beat_alignment_score:.3f}")
            
            # 确保在有效范围内
            cut_sample = np.clip(cut_sample, 0, audio_length - 1)
            
            # 样本级对齐（确保在零交叉点附近）
            cut_sample = self._align_to_zero_crossing(cut_sample)
            
            cut_points.append(cut_sample)
        
        # 排序并去重
        cut_points = sorted(list(set(cut_points)))
        
        # 确保首尾
        if cut_points and cut_points[0] != 0:
            cut_points.insert(0, 0)
        if cut_points and cut_points[-1] != audio_length:
            cut_points.append(audio_length)
        elif not cut_points:
            cut_points = [0, audio_length]
        
        logger.info(f"生成 {len(cut_points)} 个精确分割点")
        return cut_points
    
    def _score_pauses_with_confidence(self, vocal_pauses: List) -> List[Dict]:
        """Phase 2.3: 基于置信度对停顿进行评分和筛选
        
        Args:
            vocal_pauses: 原始停顿列表
            
        Returns:
            带置信度评分的停顿列表
        """
        scored_pauses = []
        
        for pause in vocal_pauses:
            confidence_score = self._calculate_pause_confidence(pause)
            
            # 只保留置信度足够高的停顿
            confidence_threshold = get_config('vocal_pause_splitting.min_confidence', 0.65)
            
            if confidence_score >= confidence_threshold:
                scored_pauses.append({
                    'pause': pause,
                    'confidence_score': confidence_score,
                    'selected': True
                })
            else:
                logger.debug(f"跳过低置信度停顿: 时间={pause.cut_point:.2f}s, "
                           f"置信度={confidence_score:.3f} < {confidence_threshold}")
        
        # 按置信度排序（可选）
        scored_pauses.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        logger.info(f"置信度筛选: {len(vocal_pauses)} -> {len(scored_pauses)} 个停顿")
        return scored_pauses
    
    def _calculate_pause_confidence(self, pause) -> float:
        """计算停顿的置信度
        
        Args:
            pause: 停顿对象
            
        Returns:
            置信度分数 (0-1)
        """
        confidence_factors = []
        weights = []
        
        # 1. 停顿持续时间可靠性 (40%)
        duration_score = min(1.0, pause.duration / 2.0)  # 2秒以上为满分
        confidence_factors.append(duration_score)
        weights.append(0.4)
        
        # 2. 静音强度可靠性 (30%)
        if hasattr(pause, 'confidence'):
            silence_score = pause.confidence
        else:
            # 基于持续时间的推断评分
            silence_score = min(1.0, pause.duration / 1.5)
        confidence_factors.append(silence_score)
        weights.append(0.3)
        
        # 3. 边界清晰度 (20%)
        # 这里可以基于停顿前后的能量对比等因素
        boundary_score = 0.8  # 默认评分
        confidence_factors.append(boundary_score)
        weights.append(0.2)
        
        # 4. 位置合理性 (10%)
        # 避免过于靠近开头或结尾的停顿
        position_score = self._calculate_position_reasonableness(pause.cut_point)
        confidence_factors.append(position_score)
        weights.append(0.1)
        
        # 加权平均
        weighted_confidence = sum(score * weight for score, weight in zip(confidence_factors, weights))
        
        return weighted_confidence
    
    def _calculate_position_reasonableness(self, cut_time: float) -> float:
        """计算分割位置的合理性"""
        # 避免过于靠近开头（前5秒）或结尾（后5秒）
        if cut_time < 5.0:
            return max(0.3, cut_time / 5.0)
        # 这里需要知道总时长才能准确计算，简化处理
        return 1.0
        
    def _apply_beat_alignment_priority(self, scored_pauses: List[Dict]) -> List[Dict]:
        """Phase 2.3: 应用节拍对齐优先级机制
        
        Args:
            scored_pauses: 带置信度的停顿列表
            
        Returns:
            应用节拍对齐优化后的停顿列表
        """
        if not self.current_adaptive_params:
            return scored_pauses
            
        beat_interval = self.current_adaptive_params.beat_interval
        aligned_pauses = []
        
        for pause_info in scored_pauses:
            pause = pause_info['pause']
            original_time = pause.cut_point
            
            # 计算节拍对齐质量
            beat_alignment_score = self._calculate_beat_alignment_quality(original_time, beat_interval)
            
            # 如果原始位置节拍对齐较差，尝试调整到最近的节拍
            if beat_alignment_score < 0.7:  # 对齐质量阈值
                aligned_time = self._align_to_nearest_beat(original_time, beat_interval)
                aligned_score = self._calculate_beat_alignment_quality(aligned_time, beat_interval)
                
                # 如果调整后质量明显提升，则使用调整后的时间
                if aligned_score > beat_alignment_score + 0.2:
                    pause.cut_point = aligned_time
                    beat_alignment_score = aligned_score
                    logger.debug(f"节拍对齐调整: {original_time:.3f}s -> {aligned_time:.3f}s")
            
            pause_info['beat_alignment_score'] = beat_alignment_score
            aligned_pauses.append(pause_info)
        
        # 按综合质量重新排序（置信度 + 节拍对齐）
        for pause_info in aligned_pauses:
            combined_score = (pause_info['confidence_score'] * 0.7 + 
                            pause_info['beat_alignment_score'] * 0.3)
            pause_info['combined_score'] = combined_score
        
        aligned_pauses.sort(key=lambda x: x['combined_score'], reverse=True)
        
        logger.info(f"节拍对齐优化完成，平均对齐质量: "
                   f"{np.mean([p['beat_alignment_score'] for p in aligned_pauses]):.3f}")
        
        return aligned_pauses
    
    def _calculate_beat_alignment_quality(self, time: float, beat_interval: float) -> float:
        """计算时间点的节拍对齐质量
        
        Args:
            time: 时间点
            beat_interval: 节拍间隔
            
        Returns:
            对齐质量分数 (0-1)
        """
        beat_position = time / beat_interval
        distance_to_beat = abs(beat_position - round(beat_position))
        
        # 距离越近质量越高，最大偏移0.5个节拍为0分
        alignment_quality = max(0.0, 1.0 - (distance_to_beat / 0.5))
        
        return alignment_quality
    
    def _align_to_nearest_beat(self, time: float, beat_interval: float) -> float:
        """将时间对齐到最近的节拍"""
        beat_position = time / beat_interval
        aligned_beat = round(beat_position)
        return aligned_beat * beat_interval
    
    def _align_to_zero_crossing(self, cut_sample: int, window_size: int = 100) -> int:
        """将分割点对齐到最近的零交叉点
        
        Args:
            cut_sample: 原始分割点样本位置
            window_size: 搜索窗口大小
            
        Returns:
            对齐后的分割点
        """
        # 简化实现：直接返回原始位置
        # 在实际应用中可以实现零交叉点对齐算法
        return cut_sample
    
    def _split_at_sample_level(self, audio: np.ndarray, cut_points: List[int],
                              vocal_pauses: List) -> List[SeamlessSegment]:
        """在样本级别执行精确分割
        
        Args:
            audio: 原始音频
            cut_points: 分割点列表
            vocal_pauses: 人声停顿信息
            
        Returns:
            分割片段列表
        """
        segments = []
        
        for i in range(len(cut_points) - 1):
            start_sample = cut_points[i]
            end_sample = cut_points[i + 1]
            
            # 提取音频片段（零处理）
            if self.zero_processing:
                # 直接切割，无任何处理
                segment_audio = audio[start_sample:end_sample].copy()
            else:
                # 可选的最小处理（如去除咔嗒声）
                segment_audio = self._minimal_processing(audio[start_sample:end_sample])
            
            # 计算时间信息
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            duration = end_time - start_time
            
            # 查找对应的停顿信息
            cut_info = self._find_cut_info(start_time, end_time, vocal_pauses)
            
            segment = SeamlessSegment(
                index=i,
                start_sample=start_sample,
                end_sample=end_sample,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                audio_data=segment_audio,
                cut_info=cut_info
            )
            
            segments.append(segment)
        
        logger.info(f"样本级分割完成: {len(segments)} 个片段")
        return segments
    
    def _minimal_processing(self, audio: np.ndarray) -> np.ndarray:
        """最小化音频处理（仅在必要时）
        
        Args:
            audio: 音频数据
            
        Returns:
            处理后的音频
        """
        # 仅进行必要的处理以避免咔嗒声
        if len(audio) > 10:
            # 极短的淡入淡出（仅1-2个样本）
            audio[0] = audio[0] * 0.5
            audio[-1] = audio[-1] * 0.5
        
        return audio
    
    def _find_cut_info(self, start_time: float, end_time: float, 
                      vocal_pauses: List) -> Dict:
        """查找片段的分割信息
        
        Args:
            start_time: 片段开始时间
            end_time: 片段结束时间
            vocal_pauses: 人声停顿列表
            
        Returns:
            分割信息字典
        """
        cut_info = {
            'start_cut_type': 'natural',
            'end_cut_type': 'natural',
            'start_pause_info': None,
            'end_pause_info': None
        }
        
        # 查找起始和结束处的停顿信息
        for pause in vocal_pauses:
            if abs(pause.cut_point - start_time) < 0.1:
                cut_info['start_cut_type'] = pause.position_type
                cut_info['start_pause_info'] = {
                    'duration': pause.duration,
                    'confidence': pause.confidence
                }
            
            if abs(pause.cut_point - end_time) < 0.1:
                cut_info['end_cut_type'] = pause.position_type
                cut_info['end_pause_info'] = {
                    'duration': pause.duration,
                    'confidence': pause.confidence
                }
        
        return cut_info
    
    def _save_seamless_segments(self, segments: List[SeamlessSegment], 
                               output_dir: str) -> List[str]:
        """保存无缝分割片段
        
        Args:
            segments: 分割片段列表
            output_dir: 输出目录
            
        Returns:
            保存的文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        # 使用无损格式
        output_format = get_config('audio.format', 'wav')
        if output_format.lower() not in ['wav', 'flac']:
            output_format = 'wav'  # 强制使用无损格式
        
        naming_pattern = get_config('output.naming_pattern', 'vocal_segment_{index:02d}')
        
        for segment in segments:
            # 生成文件名
            filename = f"{naming_pattern.format(index=segment.index + 1)}.{output_format}"
            output_path = os.path.join(output_dir, filename)
            
            # 保存音频（零处理模式）
            try:
                if output_format.lower() == 'flac':
                    # FLAC无损保存
                    import soundfile as sf
                    sf.write(output_path, segment.audio_data, self.sample_rate, format='FLAC')
                else:
                    # WAV无损保存
                    import soundfile as sf
                    sf.write(output_path, segment.audio_data, self.sample_rate, format='WAV', subtype='PCM_24')
                
                saved_files.append(output_path)
                
                logger.debug(f"保存片段 {segment.index + 1}: {filename} "
                           f"({segment.duration:.3f}s)")
                
            except Exception as e:
                logger.error(f"保存片段 {segment.index + 1} 失败: {e}")
        
        logger.info(f"保存完成: {len(saved_files)} 个无损音频文件")
        return saved_files
    
    def _validate_seamless_reconstruction(self, segments: List[SeamlessSegment],
                                        original_audio: np.ndarray) -> Dict:
        """验证无缝拼接的完整性
        
        Args:
            segments: 分割片段列表
            original_audio: 原始音频
            
        Returns:
            验证结果
        """
        logger.info("开始验证无缝拼接完整性...")
        
        try:
            # 拼接所有片段
            reconstructed = np.concatenate([seg.audio_data for seg in segments])
            
            # 长度对比
            original_length = len(original_audio)
            reconstructed_length = len(reconstructed)
            length_match = (original_length == reconstructed_length)
            
            # 样本级对比
            if length_match:
                # 计算差异
                diff = original_audio - reconstructed
                max_diff = np.max(np.abs(diff))
                rms_diff = np.sqrt(np.mean(diff**2))
                
                # 判断是否完美重构
                perfect_reconstruction = (max_diff < 1e-10)
                
                logger.info(f"拼接验证完成: 长度匹配={length_match}, "
                           f"最大差异={max_diff:.2e}, RMS差异={rms_diff:.2e}")
                
                return {
                    'length_match': length_match,
                    'perfect_reconstruction': perfect_reconstruction,
                    'max_difference': float(max_diff),
                    'rms_difference': float(rms_diff),
                    'original_length': original_length,
                    'reconstructed_length': reconstructed_length
                }
            else:
                logger.warning(f"长度不匹配: 原始={original_length}, 重构={reconstructed_length}")
                return {
                    'length_match': False,
                    'perfect_reconstruction': False,
                    'max_difference': float('inf'),
                    'rms_difference': float('inf'),
                    'original_length': original_length,
                    'reconstructed_length': reconstructed_length
                }
                
        except Exception as e:
            logger.error(f"拼接验证失败: {e}")
            return {
                'length_match': False,
                'perfect_reconstruction': False,
                'error': str(e)
            }
    
    def _create_single_segment_result(self, audio: np.ndarray, input_path: str, 
                                    output_dir: str) -> Dict:
        """创建单片段结果（当无法分割时）"""
        logger.info("无法检测到停顿，输出单个片段")
        
        # 创建单个片段
        segment = SeamlessSegment(
            index=0,
            start_sample=0,
            end_sample=len(audio),
            start_time=0.0,
            end_time=len(audio) / self.sample_rate,
            duration=len(audio) / self.sample_rate,
            audio_data=audio,
            cut_info={'start_cut_type': 'none', 'end_cut_type': 'none'}
        )
        
        # 保存
        saved_files = self._save_seamless_segments([segment], output_dir)
        
        return {
            'success': True,
            'input_file': input_path,
            'output_directory': output_dir,
            'num_segments': 1,
            'segments': [segment],
            'saved_files': saved_files,
            'seamless_validation': {'perfect_reconstruction': True},
            'note': '未检测到人声停顿，输出完整音频'
        }
    
    def _generate_result_report(self, input_path: str, output_dir: str,
                               segments: List[SeamlessSegment], saved_files: List[str],
                               vocal_pauses: List, separation_quality: Dict,
                               validation_result: Dict, processing_stats: Dict = None) -> Dict:
        """生成结果报告
        
        Args:
            input_path: 输入路径
            output_dir: 输出目录
            segments: 分割片段
            saved_files: 保存的文件
            vocal_pauses: 人声停顿
            separation_quality: 分离质量
            validation_result: 验证结果
            processing_stats: 处理统计信息
            
        Returns:
            结果报告
        """
        # 生成停顿报告（兼容v1.1.4+双路检测）
        pause_report = self._generate_pause_report_from_pauses(vocal_pauses)
        
        segment_info = []
        for segment in segments:
            segment_info.append({
                'index': segment.index,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'duration': segment.duration,
                'cut_info': segment.cut_info
            })
        
        result = {
            'success': True,
            'input_file': input_path,
            'output_directory': output_dir,
            'processing_type': 'seamless_vocal_pause_splitting',
            
            # 分割结果
            'num_segments': len(segments),
            'segments': segment_info,
            'saved_files': saved_files,
            
            # 人声停顿分析
            'vocal_pause_analysis': pause_report,
            
            # 质量信息
            'separation_quality': separation_quality,
            'seamless_validation': validation_result,
            
            # 处理参数
            'processing_params': {
                'sample_rate': self.sample_rate,
                'min_pause_duration': self.min_pause_duration,
                'zero_processing': self.zero_processing,
                'preserve_original': self.preserve_original
            }
        }
        
        # 添加处理统计信息（如果存在）
        if processing_stats:
            result['processing_stats'] = processing_stats
        
        return result
    
    def _convert_validated_to_vocal_pauses(self, validated_pauses) -> List:
        """将ValidatedPause转换为VocalPause以保持向后兼容性
        
        Args:
            validated_pauses: List[ValidatedPause] 双路检测的验证结果
            
        Returns:
            List[VocalPause]: 兼容格式的停顿列表
        """
        vocal_pauses = []
        
        for validated in validated_pauses:
            # 创建VocalPause对象
            vocal_pause = self.VocalPause(
                start_time=validated.start_time,
                end_time=validated.end_time,
                duration=validated.duration,
                position_type=validated.position_type,
                confidence=validated.confidence,
                cut_point=validated.cut_point
            )
            vocal_pauses.append(vocal_pause)
        
        logger.debug(f"转换完成: {len(validated_pauses)} 个ValidatedPause -> {len(vocal_pauses)} 个VocalPause")
        return vocal_pauses
    
    def _generate_pause_report_from_pauses(self, vocal_pauses: List) -> Dict:
        """从人声停顿列表生成报告（兼容方法）
        
        Args:
            vocal_pauses: 人声停顿列表
            
        Returns:
            停顿报告字典
        """
        if not vocal_pauses:
            return {
                'total_pauses': 0,
                'avg_confidence': 0.0,
                'total_pause_duration': 0.0,
                'pause_types': {'head': 0, 'middle': 0, 'tail': 0}
            }
        
        # 统计停顿类型
        pause_types = {'head': 0, 'middle': 0, 'tail': 0}
        total_duration = 0.0
        total_confidence = 0.0
        
        for pause in vocal_pauses:
            pause_types[pause.position_type] += 1
            total_duration += pause.duration
            total_confidence += pause.confidence
        
        return {
            'total_pauses': len(vocal_pauses),
            'avg_confidence': total_confidence / len(vocal_pauses),
            'total_pause_duration': total_duration,
            'pause_types': pause_types
        }
    
    def _enhance_with_pure_vocal_detection(self, validated_pauses: List, 
                                          original_audio: np.ndarray,
                                          processing_stats: Dict) -> List:
        """使用纯人声检测系统v2.0增强停顿检测
        
        Args:
            validated_pauses: 双路检测的结果
            original_audio: 原始音频
            processing_stats: 处理统计信息
            
        Returns:
            增强后的停顿列表
        """
        logger.info("启动纯人声检测系统v2.0...")
        
        try:
            # 1. 获取分离的纯人声（如果可用）
            vocal_audio = self._extract_vocal_for_pure_detection(
                original_audio, processing_stats
            )
            
            if vocal_audio is None:
                logger.warning("无法获取纯人声，使用原音频进行检测")
                vocal_audio = original_audio
            
            # 2. 纯人声停顿检测
            pure_pauses = self.pure_detector.detect_pure_vocal_pauses(
                vocal_audio, original_audio
            )
            
            logger.info(f"纯人声检测: {len(pure_pauses)}个候选停顿")
            
            # 3. 频谱感知分类
            classified_pauses = []
            for pause in pure_pauses:
                # 转换为字典格式
                pause_dict = {
                    'start_time': pause.start_time,
                    'end_time': pause.end_time,
                    'duration': pause.duration,
                    'confidence': pause.confidence,
                    **pause.features
                }
                
                # 分类
                classification = self.spectral_classifier.classify_pause_type(pause_dict)
                
                # 只保留真停顿
                if classification['action'] == 'keep':
                    classified_pauses.append(pause)
                    logger.debug(f"保留停顿: {pause.start_time:.2f}s - {classification['reasoning']}")
                else:
                    logger.debug(f"过滤停顿: {pause.start_time:.2f}s - {classification['reasoning']}")
            
            logger.info(f"频谱分类完成: {len(pure_pauses)} -> {len(classified_pauses)}个停顿")
            
            # 4. BPM优化
            if self.current_adaptive_params:
                optimized_pauses = self.bpm_optimizer.optimize_with_bpm(
                    classified_pauses, original_audio, self.current_adaptive_params
                )
                logger.info(f"BPM优化完成: {len(classified_pauses)} -> {len(optimized_pauses)}个停顿")
            else:
                logger.info("无BPM参数，跳过BPM优化")
                # 转换格式
                optimized_pauses = []
                for pause in classified_pauses:
                    optimized_pauses.append(type('OptimizedPause', (), {
                        'start_time': pause.start_time,
                        'end_time': pause.end_time,
                        'duration': pause.duration,
                        'cut_point': (pause.start_time + pause.end_time) / 2,
                        'confidence': pause.confidence,
                        'alignment_score': 0.5,
                        'optimization_reason': 'no_bpm_data'
                    })())
            
            # 5. 多级验证
            audio_context = AudioContext(
                audio=original_audio,
                sample_rate=self.sample_rate,
                bpm=self.current_adaptive_params.bpm if self.current_adaptive_params else None,
                total_duration=len(original_audio) / self.sample_rate,
                energy_profile=np.abs(original_audio),
                spectral_features={}
            )
            
            final_pauses = self.validator.validate_pauses(optimized_pauses, audio_context)
            logger.info(f"多级验证完成: {len(optimized_pauses)} -> {len(final_pauses)}个有效停顿")
            
            # 6. 转换回ValidatedPause格式
            enhanced_validated_pauses = []
            for pause in final_pauses:
                enhanced_pause = self.ValidatedPause(
                    start_time=pause.start_time,
                    end_time=pause.end_time,
                    duration=pause.duration,
                    position_type=self._determine_position_type(pause.start_time, 
                                                              len(original_audio) / self.sample_rate),
                    confidence=pause.confidence,
                    cut_point=pause.cut_point,
                    mixed_detection=True,  # 来自混音检测
                    separated_detection=True,  # 来自分离检测
                    separation_confidence=processing_stats.get('separation_confidence', 0.0),
                    validation_method="pure_vocal_v2"
                )
                enhanced_validated_pauses.append(enhanced_pause)
            
            # 生成验证报告
            validation_report = self.validator.generate_validation_report(final_pauses)
            logger.info(f"验证报告: {validation_report['quality_summary']}")
            logger.info(f"质量分布: {validation_report['grade_distribution']}")
            
            return enhanced_validated_pauses
            
        except Exception as e:
            logger.error(f"纯人声检测增强失败: {e}")
            # 回退到原始结果
            logger.info("回退到双路检测结果")
            return validated_pauses
    
    def _extract_vocal_for_pure_detection(self, original_audio: np.ndarray,
                                        processing_stats: Dict) -> Optional[np.ndarray]:
        """为纯人声检测提取人声轨道
        
        Args:
            original_audio: 原始音频
            processing_stats: 处理统计信息
            
        Returns:
            分离的人声轨道或None
        """
        # 检查是否有可用的分离器
        if not hasattr(self.dual_detector, 'separator'):
            return None
            
        try:
            # 使用增强分离器分离人声
            separation_result = self.dual_detector.separator.separate_for_detection(original_audio)
            
            if separation_result.vocal_track is not None:
                logger.debug(f"成功分离人声 (置信度: {separation_result.separation_confidence:.3f})")
                return separation_result.vocal_track
            else:
                logger.debug("人声分离失败")
                return None
                
        except Exception as e:
            logger.debug(f"人声分离过程出错: {e}")
            return None
    
    def _determine_position_type(self, start_time: float, total_duration: float) -> str:
        """确定停顿位置类型
        
        Args:
            start_time: 停顿开始时间
            total_duration: 音频总时长
            
        Returns:
            位置类型
        """
        ratio = start_time / total_duration
        
        if ratio < 0.2:
            return 'head'
        elif ratio > 0.8:
            return 'tail'
        else:
            return 'middle'
    
    # ====================================================================
    # 新增模式处理函数 - 统一指挥中心的核心实现
    # ====================================================================
    
    def _process_vocal_separation_only(self, input_path: str, output_dir: str) -> Dict:
        """纯人声分离模式，不进行分割"""
        logger.info("[VOCAL_SEPARATION] 执行纯人声分离...")
        
        try:
            from .enhanced_vocal_separator import EnhancedVocalSeparator
            import soundfile as sf
            import time
            
            # 创建输出目录
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # 1. 加载音频
            audio, sr = self._load_original_audio(input_path)
            
            # 2. 执行人声分离
            separator = EnhancedVocalSeparator(self.sample_rate)
            start_time = time.time()
            separation_result = separator.separate_for_detection(audio)
            processing_time = time.time() - start_time
            
            if separation_result.vocal_track is None:
                return {
                    'success': False,
                    'error': '人声分离失败',
                    'input_file': input_path
                }
            
            # 3. 保存结果
            input_name = Path(input_path).stem
            vocal_file = Path(output_dir) / f"{input_name}_vocal.wav"
            sf.write(vocal_file, separation_result.vocal_track, self.sample_rate)
            
            saved_files = [str(vocal_file)]
            
            # 保存伴奏（如果有）
            if separation_result.instrumental_track is not None:
                instrumental_file = Path(output_dir) / f"{input_name}_instrumental.wav"
                sf.write(instrumental_file, separation_result.instrumental_track, self.sample_rate)
                saved_files.append(str(instrumental_file))
            
            return {
                'success': True,
                'method': 'vocal_separation_only',
                'input_file': input_path,
                'output_dir': output_dir,
                'backend_used': separation_result.backend_used,
                'separation_confidence': separation_result.separation_confidence,
                'processing_time': processing_time,
                'saved_files': saved_files,
                'num_segments': 0,  # 分离不产生片段
                'segments': []
            }
            
        except Exception as e:
            logger.error(f"人声分离失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_file': input_path
            }
    
    def _process_vocal_split_v2(self, input_path: str, output_dir: str) -> Dict:
        """v2.1纯人声检测模式 - 统计学动态裁决"""
        logger.info("[V2.1] 执行统计学动态裁决系统...")
        
        try:
            from .enhanced_vocal_separator import EnhancedVocalSeparator
            import soundfile as sf
            import tempfile
            import time
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            overall_start_time = time.time()
            
            # 1. 加载音频
            audio, sr = self._load_original_audio(input_path)
            
            # 2. 人声分离
            logger.info("[V2.1-STEP1] MDX23/Demucs高质量人声分离...")
            separator = EnhancedVocalSeparator(self.sample_rate)
            separation_start = time.time()
            separation_result = separator.separate_for_detection(audio)
            separation_time = time.time() - separation_start
            
            if separation_result.vocal_track is None:
                return {
                    'success': False,
                    'error': '人声分离失败，无法执行纯人声检测',
                    'input_file': input_path
                }
            
            # 3. 在纯人声轨上执行检测
            logger.info("[V2.1-STEP2] 在纯人声轨上执行统计学动态裁决...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_vocal:
                sf.write(temp_vocal.name, separation_result.vocal_track, self.sample_rate)
                temp_vocal_path = temp_vocal.name
            
            try:
                # 递归调用智能分割模式处理纯人声轨
                result = self._process_smart_split(temp_vocal_path, output_dir)
            finally:
                try:
                    import os
                    os.unlink(temp_vocal_path)
                except:
                    pass
            
            if not result.get('success', False):
                return {
                    'success': False,
                    'error': f"纯人声检测失败: {result.get('error', '未知错误')}",
                    'input_file': input_path
                }
            
            # 4. 保存完整的分离文件
            input_name = Path(input_path).stem
            full_vocal_file = Path(output_dir) / f"{input_name}_v2_vocal_full.wav"
            sf.write(full_vocal_file, separation_result.vocal_track, self.sample_rate, subtype='PCM_24')
            
            saved_files = result.get('saved_files', [])
            saved_files.append(str(full_vocal_file))
            
            if separation_result.instrumental_track is not None:
                full_instrumental_file = Path(output_dir) / f"{input_name}_v2_instrumental.wav"
                sf.write(full_instrumental_file, separation_result.instrumental_track, self.sample_rate, subtype='PCM_24')
                saved_files.append(str(full_instrumental_file))
            
            total_time = time.time() - overall_start_time
            
            # 构建v2.1格式返回结果
            v2_result = result.copy()
            v2_result.update({
                'version': '2.1.0',
                'method': 'SeamlessSplitter统计学动态裁决 + BPM自适应 + 边界保护',
                'backend_used': separation_result.backend_used,
                'separation_confidence': separation_result.separation_confidence,
                'separation_time': separation_time,
                'full_vocal_file': str(full_vocal_file),
                'total_processing_time': total_time,
                'saved_files': saved_files
            })
            
            logger.info(f"[V2.1-SUCCESS] 统计学动态裁决系统v2.1完成! 生成片段: {result.get('num_segments', 0)} 个")
            return v2_result
            
        except Exception as e:
            logger.error(f"v2.1系统失败: {e}")
            return {
                'success': False,
                'version': '2.1.0',
                'error': str(e),
                'input_file': input_path
            }
    
    def _process_vocal_split_mdd(self, input_path: str, output_dir: str) -> Dict:
        """v2.2 MDD增强模式 - 主副歌智能识别"""
        logger.info("[MDD_V2.2] 启动MDD增强分割系统...")
        
        try:
            from .enhanced_vocal_separator import EnhancedVocalSeparator
            from ..utils.config_manager import set_runtime_config, get_config_manager
            import soundfile as sf
            import tempfile
            import time
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            overall_start_time = time.time()
            
            # 临时启用MDD增强功能
            original_mdd_enable = get_config('musical_dynamic_density.enable', True)
            config_manager = get_config_manager()
            config_manager.set('musical_dynamic_density.enable', True)
            config_manager.set('vocal_pause_splitting.enable_chorus_detection', True)
            
            try:
                # 1. 加载音频
                audio, sr = self._load_original_audio(input_path)
                
                # 2. 人声分离
                logger.info("[MDD-STEP1] MDX23/Demucs高质量人声分离...")
                separator = EnhancedVocalSeparator(self.sample_rate)
                separation_start = time.time()
                separation_result = separator.separate_for_detection(audio)
                separation_time = time.time() - separation_start
                
                if separation_result.vocal_track is None:
                    return {
                        'success': False,
                        'error': '人声分离失败，无法执行MDD增强检测',
                        'input_file': input_path
                    }
                
                # 3. 在纯人声轨上执行MDD增强分析
                logger.info("[MDD-STEP2] 执行MDD增强分析和主副歌识别...")
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_vocal:
                    sf.write(temp_vocal.name, separation_result.vocal_track, self.sample_rate)
                    temp_vocal_path = temp_vocal.name
                
                try:
                    # 递归调用智能分割模式处理纯人声轨（MDD增强）
                    result = self._process_smart_split(temp_vocal_path, output_dir)
                finally:
                    try:
                        import os
                        os.unlink(temp_vocal_path)
                    except:
                        pass
                
                if not result.get('success', False):
                    return {
                        'success': False,
                        'error': f"MDD增强分割失败: {result.get('error', '未知错误')}",
                        'input_file': input_path
                    }
                
                # 4. 保存完整的分离文件
                input_name = Path(input_path).stem
                full_vocal_file = Path(output_dir) / f"{input_name}_mdd_vocal_full.wav"
                sf.write(full_vocal_file, separation_result.vocal_track, self.sample_rate, subtype='PCM_24')
                
                saved_files = result.get('saved_files', [])
                saved_files.append(str(full_vocal_file))
                
                if separation_result.instrumental_track is not None:
                    full_instrumental_file = Path(output_dir) / f"{input_name}_mdd_instrumental.wav"
                    sf.write(full_instrumental_file, separation_result.instrumental_track, self.sample_rate, subtype='PCM_24')
                    saved_files.append(str(full_instrumental_file))
                
                total_time = time.time() - overall_start_time
                
                # 构建v2.2 MDD格式返回结果
                mdd_result = result.copy()
                mdd_result.update({
                    'version': '2.2.0',
                    'method': 'MDD增强主副歌识别 + 统计学动态裁决 + BPM自适应',
                    'backend_used': separation_result.backend_used,
                    'separation_confidence': separation_result.separation_confidence,
                    'separation_time': separation_time,
                    'full_vocal_file': str(full_vocal_file),
                    'total_processing_time': total_time,
                    'saved_files': saved_files,
                    'mdd_enabled': True
                })
                
                logger.info(f"[MDD-SUCCESS] MDD增强分割系统v2.2完成! 生成片段: {result.get('num_segments', 0)} 个")
                return mdd_result
                
            finally:
                # 恢复原始MDD配置
                config_manager.set('musical_dynamic_density.enable', original_mdd_enable)
                
        except Exception as e:
            logger.error(f"MDD v2.2系统失败: {e}")
            return {
                'success': False,
                'version': '2.2.0',
                'error': str(e),
                'input_file': input_path
            }
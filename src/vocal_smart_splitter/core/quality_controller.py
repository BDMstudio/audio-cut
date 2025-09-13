#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/quality_controller.py
# AI-SUMMARY: 质量控制核心模块，确保分割结果的质量和完整性

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

from vocal_smart_splitter.utils.config_manager import get_config
from vocal_smart_splitter.utils.audio_processor import AudioProcessor
from vocal_smart_splitter.utils.adaptive_parameter_calculator import AdaptiveParameterCalculator

logger = logging.getLogger(__name__)

class QualityController:
    """BPM感知的质量控制器，确保分割结果的质量"""
    
    def __init__(self, sample_rate: int = 22050):
        """初始化质量控制器
        
        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate)
        self.adaptive_calculator = AdaptiveParameterCalculator()
        
        # 从配置加载参数
        self.validate_split_points = get_config('quality_control.validate_split_points', True)
        
        # 动态参数（将被BPM自适应系统覆盖）
        self.current_adaptive_params = None
        self.bpm_info = None
        
        # 🔄 以下参数将被BPM自适应系统动态覆盖
        self.min_pause_at_split = get_config('quality_control.min_pause_at_split', 1.0)
        self.max_vocal_at_split = get_config('quality_control.max_vocal_at_split', 0.10)
        self.min_split_gap = get_config('quality_control.min_split_gap', 2.5)
        
        self.min_vocal_content_ratio = get_config('quality_control.min_vocal_content_ratio', 0.4)
        self.max_silence_ratio = get_config('quality_control.max_silence_ratio', 0.3)
        
        self.fade_in_duration = get_config('quality_control.fade_in_duration', 0.02)
        self.fade_out_duration = get_config('quality_control.fade_out_duration', 0.02)
        self.normalize_audio = get_config('quality_control.normalize_audio', True)
        
        self.remove_click_noise = get_config('quality_control.remove_click_noise', True)
        self.smooth_transitions = get_config('quality_control.smooth_transitions', True)
        
        logger.info("BPM感知质量控制器初始化完成")
    
    def apply_bpm_adaptive_parameters(self, bpm: float, complexity: float, 
                                     instrument_count: int) -> None:
        """应用BPM自适应参数
        
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
            
            # 动态覆盖质量控制参数
            self.min_pause_at_split = self.current_adaptive_params.min_pause_duration
            self.min_split_gap = self.current_adaptive_params.min_split_gap
            self.max_vocal_at_split = min(0.15, 0.10 + complexity * 0.05)  # 基于复杂度调整
            
            logger.info(f"BPM自适应参数已应用: {self.bpm_info['category']}歌曲 "
                       f"BPM={float(bpm):.1f}, 停顿要求={self.min_pause_at_split:.3f}s, "
                       f"分割间隙={self.min_split_gap:.3f}s")
                       
        except Exception as e:
            logger.error(f"应用BPM自适应参数失败: {e}")
            # 使用默认参数继续
            self.current_adaptive_params = None
    
    def get_current_quality_parameters(self) -> Dict:
        """获取当前质量控制参数信息"""
        if self.current_adaptive_params:
            return {
                'min_pause_at_split': self.min_pause_at_split,
                'min_split_gap': self.min_split_gap,
                'max_vocal_at_split': self.max_vocal_at_split,
                'adaptive_mode': True,
                'bpm_info': self.bpm_info
            }
        else:
            return {
                'min_pause_at_split': self.min_pause_at_split,
                'min_split_gap': self.min_split_gap,
                'max_vocal_at_split': self.max_vocal_at_split,
                'adaptive_mode': False,
                'bpm_info': None
            }
    
    def validate_and_process_segments(self, audio: np.ndarray,
                                    vocal_track: np.ndarray,
                                    split_points: List[Dict]) -> List[Dict]:
        """验证和处理音频片段
        
        Args:
            audio: 原始音频
            vocal_track: 人声轨道
            split_points: 分割点列表
            
        Returns:
            处理后的片段信息列表
        """
        logger.info("开始验证和处理音频片段...")
        
        try:
            # 1. 创建音频片段
            raw_segments = self._create_audio_segments(audio, split_points)
            
            # 2. 验证片段质量
            validated_segments = self._validate_segments(raw_segments, vocal_track)
            
            # 3. 处理音频质量
            processed_segments = self._process_audio_quality(validated_segments)
            
            # 4. 生成质量报告
            quality_report = self._generate_quality_report(processed_segments, vocal_track)
            
            logger.info(f"片段处理完成，共 {len(processed_segments)} 个有效片段")
            
            return {
                'segments': processed_segments,
                'quality_report': quality_report
            }
            
        except Exception as e:
            logger.error(f"片段验证和处理失败: {e}")
            raise
    
    def _create_audio_segments(self, audio: np.ndarray, 
                             split_points: List[Dict]) -> List[Dict]:
        """创建音频片段
        
        Args:
            audio: 原始音频
            split_points: 分割点列表
            
        Returns:
            音频片段列表
        """
        segments = []
        audio_duration = len(audio) / self.sample_rate
        
        # 添加开始和结束时间点
        time_points = [0.0] + [point['split_time'] for point in split_points] + [audio_duration]
        
        for i in range(len(time_points) - 1):
            start_time = time_points[i]
            end_time = time_points[i + 1]
            
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # 确保不超出音频范围
            end_sample = min(end_sample, len(audio))
            
            if end_sample > start_sample:
                segment_audio = audio[start_sample:end_sample]
                
                segment_info = {
                    'index': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'audio_data': segment_audio,
                    'original_length': len(segment_audio)
                }
                
                segments.append(segment_info)
        
        logger.debug(f"创建了 {len(segments)} 个原始片段")
        return segments
    
    def _validate_segments(self, segments: List[Dict], 
                         vocal_track: np.ndarray) -> List[Dict]:
        """验证片段质量
        
        Args:
            segments: 原始片段列表
            vocal_track: 人声轨道
            
        Returns:
            验证通过的片段列表
        """
        validated_segments = []
        
        for segment in segments:
            validation_result = self._validate_single_segment(segment, vocal_track)
            
            if validation_result['is_valid']:
                segment.update(validation_result)
                validated_segments.append(segment)
            else:
                logger.debug(f"片段 {segment['index']} 验证失败: {validation_result['failure_reason']}")
        
        return validated_segments
    
    def _validate_single_segment(self, segment: Dict, 
                               vocal_track: np.ndarray) -> Dict:
        """验证单个片段 - BPM感知版本：基于音乐理论的质量评估
        
        Args:
            segment: 片段信息
            vocal_track: 人声轨道
            
        Returns:
            验证结果
        """
        audio_data = segment['audio_data']
        start_sample = segment['start_sample']
        end_sample = segment['end_sample']
        
        # 提取对应的人声片段
        vocal_segment = vocal_track[start_sample:min(end_sample, len(vocal_track))]
        
        validation_result = {
            'is_valid': True,
            'failure_reason': None,
            'quality_metrics': {},
            'musical_assessment': {}
        }
        
        # BPM感知验证策略：基于音乐理论的质量评估
        
        # 1. 检查片段长度 - BPM自适应长度判断
        duration = segment['duration']
        
        # 使用BPM自适应长度标准（如果可用）
        if self.current_adaptive_params:
            # 基于音乐节拍的合理长度范围
            beat_interval = self.current_adaptive_params.beat_interval
            min_beats = 8  # 至少2个乐句（每乐句4拍）
            max_beats = 32 # 最多8个乐句
            
            adaptive_min_len = min_beats * beat_interval
            adaptive_max_len = max_beats * beat_interval
            
            # 音乐理论验证
            validation_result['musical_assessment']['expected_duration_range'] = (adaptive_min_len, adaptive_max_len)
            validation_result['musical_assessment']['beat_count'] = duration / beat_interval
            
            # 只在极端情况下拒绝
            if duration < adaptive_min_len * 0.5:  # 少于1个乐句
                validation_result['is_valid'] = False
                validation_result['failure_reason'] = f"音乐长度过短: {duration:.2f}s (少于{adaptive_min_len*0.5:.1f}s最小音乐单位)"
                return validation_result
            if duration > adaptive_max_len * 2:  # 超过16个乐句
                validation_result['is_valid'] = False
                validation_result['failure_reason'] = f"音乐长度过长: {duration:.2f}s (超过{adaptive_max_len*2:.1f}s最大音乐单位)"
                return validation_result
        else:
            # 回退到配置文件设置
            min_len = get_config('quality_control.min_segment_duration', None)
            max_len = get_config('quality_control.max_segment_duration', None)
            if min_len is not None and duration < float(min_len):
                validation_result['is_valid'] = False
                validation_result['failure_reason'] = f"片段过短: {duration:.2f}s (<{float(min_len):.2f}s)"
                return validation_result
            if max_len is not None and duration > float(max_len):
                validation_result['is_valid'] = False
                validation_result['failure_reason'] = f"片段过长: {duration:.2f}s (>{float(max_len):.2f}s)"
                return validation_result
        
        # 2. 检查音频数据完整性
        if len(audio_data) == 0:
            validation_result['is_valid'] = False
            validation_result['failure_reason'] = "音频数据为空"
            return validation_result
        
        # 3. 计算质量指标但不用于过滤 - 只用于信息记录
        vocal_content_ratio = self._calculate_vocal_content_ratio(vocal_segment)
        validation_result['quality_metrics']['vocal_content_ratio'] = vocal_content_ratio
        
        # 4. 计算音频质量指标
        audio_quality_dict = self._calculate_audio_quality(audio_data)
        validation_result['quality_metrics'].update(audio_quality_dict)
        
        # 计算综合音频质量分数
        audio_quality_score = (audio_quality_dict['rms_energy'] * 0.4 + 
                              min(audio_quality_dict['dynamic_range'] / 20.0, 1.0) * 0.3 +
                              audio_quality_dict['peak_level'] * 0.3)
        validation_result['quality_metrics']['audio_quality'] = audio_quality_score
        
        # 5. 计算静音比例
        silence_ratio = self._calculate_silence_ratio(audio_data)
        validation_result['quality_metrics']['silence_ratio'] = silence_ratio
        
        # 6. 设置分割质量（简化处理）
        validation_result['quality_metrics']['split_quality'] = 0.8  # 假设分割质量良好
        
        # 7. 计算综合质量分数 - 必需的，但不用于过滤
        validation_result['quality_metrics']['overall_quality'] = self._calculate_overall_quality(
            validation_result['quality_metrics']
        )
        
        # 🆕 基于音乐理论的额外质量评估
        if self.current_adaptive_params:
            musical_quality = self._assess_musical_quality(segment, validation_result['musical_assessment'])
            validation_result['quality_metrics']['musical_quality'] = musical_quality
        
        # 记录但不过滤 - 让所有在自然停顿处的分割都保留
        logger.debug(f"片段 {segment['index']}: {duration:.2f}s, 人声比例: {vocal_content_ratio:.2f}")
        
        return validation_result
    
    def _assess_musical_quality(self, segment: Dict, musical_assessment: Dict) -> float:
        """基于音乐理论评估片段质量
        
        Args:
            segment: 片段信息
            musical_assessment: 音乐评估数据
            
        Returns:
            音乐质量分数 (0-1)
        """
        if not self.current_adaptive_params:
            return 0.5  # 无BPM信息时的默认分数
            
        quality_score = 0.0
        total_weight = 0.0
        
        # 1. 节拍对齐质量 (30%)
        if 'beat_count' in musical_assessment:
            beat_count = musical_assessment['beat_count']
            # 更接近整数拍数的片段质量更高
            beat_alignment_quality = 1.0 - (abs(beat_count - round(beat_count)) / 0.5)
            beat_alignment_quality = max(0.0, min(1.0, beat_alignment_quality))
            quality_score += beat_alignment_quality * 0.3
            total_weight += 0.3
        
        # 2. 音乐长度合理性 (25%)
        if 'expected_duration_range' in musical_assessment:
            min_expected, max_expected = musical_assessment['expected_duration_range']
            duration = segment['duration']
            
            # 在期望范围内的质量最高
            if min_expected <= duration <= max_expected:
                duration_quality = 1.0
            else:
                # 超出范围的质量递减
                if duration < min_expected:
                    duration_quality = duration / min_expected
                else:
                    duration_quality = max_expected / duration
                duration_quality = max(0.2, min(1.0, duration_quality))
            
            quality_score += duration_quality * 0.25
            total_weight += 0.25
        
        # 3. BPM类别适应性 (20%)
        category_quality = self._assess_category_adaptation(segment)
        quality_score += category_quality * 0.20
        total_weight += 0.20
        
        # 4. 复杂度补偿效果 (15%)
        complexity_quality = self._assess_complexity_adaptation(segment)
        quality_score += complexity_quality * 0.15
        total_weight += 0.15
        
        # 5. 基础音频质量 (10%)
        if 'audio_quality' in segment.get('quality_metrics', {}):
            audio_quality = segment['quality_metrics']['audio_quality']
            quality_score += audio_quality * 0.10
            total_weight += 0.10
        
        return quality_score / total_weight if total_weight > 0 else 0.5
    
    def _assess_category_adaptation(self, segment: Dict) -> float:
        """评估BPM类别适应质量"""
        if not self.bpm_info:
            return 0.5
            
        duration = segment['duration']
        category = self.bpm_info['category']
        bpm = self.bpm_info['bpm']
        
        # 根据不同类别的期望特征评估
        if category == 'slow':
            # 慢歌：期望较长的片段，允许自然呼吸
            ideal_range = (8.0, 20.0)
        elif category == 'medium':
            # 中速：标准流行歌曲长度
            ideal_range = (6.0, 15.0)
        elif category == 'fast':
            # 快歌：较短的片段，紧凑节奏
            ideal_range = (4.0, 12.0)
        else:  # very_fast
            # 极快：很短的片段
            ideal_range = (3.0, 8.0)
        
        if ideal_range[0] <= duration <= ideal_range[1]:
            return 1.0
        elif duration < ideal_range[0]:
            return max(0.3, duration / ideal_range[0])
        else:
            return max(0.3, ideal_range[1] / duration)
    
    def _assess_complexity_adaptation(self, segment: Dict) -> float:
        """评估复杂度适应质量"""
        if not self.bpm_info:
            return 0.5
            
        complexity = self.bpm_info['complexity']
        compensation_factor = self.bpm_info['compensation_factor']
        
        # 复杂度越高，补偿因子应该越大
        expected_compensation = 1.0 + complexity * 0.5
        compensation_accuracy = 1.0 - abs(compensation_factor - expected_compensation) / expected_compensation
        
        return max(0.2, compensation_accuracy)
    
    def validate_split_gaps(self, split_points: List[Dict]) -> List[Dict]:
        """验证和调整分割间隙 - 节拍感知版本
        
        Args:
            split_points: 分割点列表
            
        Returns:
            调整后的分割点列表
        """
        if not split_points or not self.current_adaptive_params:
            return split_points
            
        validated_points = []
        beat_interval = self.current_adaptive_params.beat_interval
        min_gap = self.min_split_gap
        
        logger.info(f"开始节拍感知分割间隙验证，最小间隙: {min_gap:.3f}s")
        
        for i, point in enumerate(split_points):
            if i == 0:
                validated_points.append(point)
                continue
                
            prev_point = validated_points[-1]
            current_gap = point['split_time'] - prev_point['split_time']
            
            # 检查是否满足最小间隙要求
            if current_gap < min_gap:
                # 尝试节拍对齐调整
                adjusted_time = self._align_to_beat(
                    prev_point['split_time'] + min_gap, beat_interval
                )
                
                # 如果调整后的时间合理，则使用调整后的时间
                if adjusted_time < point['split_time'] + beat_interval:
                    point_copy = point.copy()
                    point_copy['split_time'] = adjusted_time
                    point_copy['adjustment_reason'] = f"节拍对齐间隙调整: {current_gap:.3f}s -> {min_gap:.3f}s"
                    validated_points.append(point_copy)
                    logger.debug(f"调整分割点 {i}: {point['split_time']:.3f}s -> {adjusted_time:.3f}s")
                else:
                    # 跳过此分割点
                    logger.debug(f"跳过过近的分割点 {i}: 间隙 {current_gap:.3f}s < {min_gap:.3f}s")
                    continue
            else:
                validated_points.append(point)
        
        logger.info(f"分割间隙验证完成: {len(split_points)} -> {len(validated_points)} 个分割点")
        return validated_points
    
    def _align_to_beat(self, time: float, beat_interval: float) -> float:
        """将时间对齐到最近的节拍"""
        beat_position = time / beat_interval
        aligned_beat = round(beat_position)
        return aligned_beat * beat_interval
    
    def _calculate_audio_quality(self, audio_data: np.ndarray) -> Dict:
        """计算音频质量指标 - 仅用于记录，不用于过滤
        
        Args:
            audio_data: 音频数据
            
        Returns:
            音频质量指标字典
        """
        if len(audio_data) == 0:
            return {'rms_energy': 0.0, 'dynamic_range': 0.0, 'peak_level': 0.0}
        
        # 计算基本音频质量指标
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        peak_level = np.max(np.abs(audio_data))
        
        # 计算动态范围
        if peak_level > 0:
            dynamic_range = 20 * np.log10(peak_level / (rms_energy + 1e-10))
        else:
            dynamic_range = 0.0
        
        return {
            'rms_energy': float(rms_energy),
            'dynamic_range': float(dynamic_range),
            'peak_level': float(peak_level)
        }
    
    def _calculate_vocal_content_ratio(self, vocal_segment: np.ndarray) -> float:
        """计算人声内容比例
        
        Args:
            vocal_segment: 人声片段
            
        Returns:
            人声内容比例
        """
        if len(vocal_segment) == 0:
            return 0.0
        
        # 计算RMS能量
        rms_energy = np.sqrt(np.mean(vocal_segment ** 2))
        
        # 计算有效人声的比例
        window_size = int(0.1 * self.sample_rate)  # 100ms窗口
        hop_size = window_size // 2
        
        vocal_frames = 0
        total_frames = 0
        
        for i in range(0, len(vocal_segment) - window_size, hop_size):
            window = vocal_segment[i:i + window_size]
            window_rms = np.sqrt(np.mean(window ** 2))
            
            # 如果窗口能量超过阈值，认为有人声
            if window_rms > rms_energy * 0.1:
                vocal_frames += 1
            
            total_frames += 1
        
        return vocal_frames / total_frames if total_frames > 0 else 0.0
    
    def _calculate_silence_ratio(self, audio_data: np.ndarray) -> float:
        """计算静音比例
        
        Args:
            audio_data: 音频数据
            
        Returns:
            静音比例
        """
        if len(audio_data) == 0:
            return 1.0
        
        # 计算整体RMS
        overall_rms = np.sqrt(np.mean(audio_data ** 2))
        silence_threshold = overall_rms * 0.05  # 5%的整体RMS作为静音阈值
        
        # 计算静音样本数
        silence_samples = np.sum(np.abs(audio_data) < silence_threshold)
        
        return silence_samples / len(audio_data)
    
    def _assess_audio_quality(self, audio_data: np.ndarray) -> float:
        """评估音频质量
        
        Args:
            audio_data: 音频数据
            
        Returns:
            音频质量分数
        """
        if len(audio_data) == 0:
            return 0.0
        
        quality_score = 0.0
        
        # 1. 动态范围
        dynamic_range = np.max(audio_data) - np.min(audio_data)
        if dynamic_range > 0.1:
            quality_score += 0.3
        
        # 2. 信噪比估计
        rms = np.sqrt(np.mean(audio_data ** 2))
        noise_floor = np.percentile(np.abs(audio_data), 10)
        snr_estimate = rms / (noise_floor + 1e-8)
        
        if snr_estimate > 10:
            quality_score += 0.4
        elif snr_estimate > 5:
            quality_score += 0.2
        
        # 3. 频谱完整性（简单检查）
        if len(audio_data) > 1024:
            fft = np.fft.fft(audio_data[:1024])
            spectrum = np.abs(fft)
            
            # 检查是否有足够的频谱内容
            if np.sum(spectrum) > 0:
                spectral_centroid = np.sum(spectrum * np.arange(len(spectrum))) / np.sum(spectrum)
                if 50 < spectral_centroid < 400:  # 合理的频谱质心
                    quality_score += 0.3
        
        return min(quality_score, 1.0)
    
    def _validate_split_boundaries(self, segment: Dict, 
                                 vocal_track: np.ndarray) -> float:
        """验证分割边界质量
        
        Args:
            segment: 片段信息
            vocal_track: 人声轨道
            
        Returns:
            边界质量分数
        """
        start_sample = segment['start_sample']
        end_sample = segment['end_sample']
        
        boundary_quality = 0.0
        
        # 检查开始边界
        start_quality = self._check_boundary_quality(
            vocal_track, start_sample, 'start'
        )
        boundary_quality += start_quality * 0.5
        
        # 检查结束边界
        end_quality = self._check_boundary_quality(
            vocal_track, end_sample, 'end'
        )
        boundary_quality += end_quality * 0.5
        
        return boundary_quality
    
    def _check_boundary_quality(self, vocal_track: np.ndarray, 
                              boundary_sample: int, 
                              boundary_type: str) -> float:
        """检查单个边界的质量
        
        Args:
            vocal_track: 人声轨道
            boundary_sample: 边界样本位置
            boundary_type: 边界类型 ('start' 或 'end')
            
        Returns:
            边界质量分数
        """
        check_window = int(self.min_pause_at_split * self.sample_rate)
        
        if boundary_type == 'start':
            # 检查开始前的静音
            start_idx = max(0, boundary_sample - check_window)
            end_idx = boundary_sample
        else:
            # 检查结束后的静音
            start_idx = boundary_sample
            end_idx = min(len(vocal_track), boundary_sample + check_window)
        
        if start_idx >= end_idx:
            return 0.5  # 边界情况
        
        boundary_region = vocal_track[start_idx:end_idx]
        
        # 计算边界区域的能量
        region_energy = np.sum(boundary_region ** 2) / len(boundary_region)
        total_energy = np.sum(vocal_track ** 2) / len(vocal_track)
        
        energy_ratio = region_energy / (total_energy + 1e-8)
        
        # 边界区域应该相对安静
        if energy_ratio < self.max_vocal_at_split:
            return 1.0
        elif energy_ratio < self.max_vocal_at_split * 2:
            return 0.7
        else:
            return 0.3
    
    def _calculate_overall_quality(self, quality_metrics: Dict) -> float:
        """计算综合质量分数
        
        Args:
            quality_metrics: 质量指标字典
            
        Returns:
            综合质量分数
        """
        weights = {
            'vocal_content_ratio': 0.3,
            'audio_quality': 0.3,
            'split_quality': 0.2,
            'silence_ratio': 0.2  # 静音比例越低越好
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_metrics:
                value = quality_metrics[metric]
                
                # 静音比例需要反转（越低越好）
                if metric == 'silence_ratio':
                    value = 1.0 - value
                
                weighted_score += value * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _process_audio_quality(self, segments: List[Dict]) -> List[Dict]:
        """处理音频质量
        
        Args:
            segments: 验证通过的片段列表
            
        Returns:
            处理后的片段列表
        """
        processed_segments = []
        
        for segment in segments:
            audio_data = segment['audio_data'].copy()
            
            # 1. 去除咔嗒声
            if self.remove_click_noise:
                audio_data = self._remove_click_noise(audio_data)
            
            # 2. 应用渐入渐出
            audio_data = self.audio_processor._apply_fades(
                audio_data, self.sample_rate, 
                self.fade_in_duration, self.fade_out_duration
            )
            
            # 3. 标准化音频
            if self.normalize_audio:
                audio_data = self.audio_processor._normalize_audio(audio_data)
            
            # 4. 平滑过渡
            if self.smooth_transitions:
                audio_data = self._smooth_transitions(audio_data)
            
            # 更新片段信息
            segment['processed_audio'] = audio_data
            segment['processing_applied'] = {
                'click_removal': self.remove_click_noise,
                'fade_applied': True,
                'normalized': self.normalize_audio,
                'smoothed': self.smooth_transitions
            }
            
            processed_segments.append(segment)
        
        return processed_segments
    
    def _remove_click_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """去除咔嗒声
        
        Args:
            audio_data: 音频数据
            
        Returns:
            处理后的音频
        """
        try:
            # 简单的咔嗒声检测和移除
            # 检测突然的幅度变化
            diff = np.diff(audio_data)
            threshold = np.std(diff) * 3
            
            click_indices = np.where(np.abs(diff) > threshold)[0]
            
            # 对检测到的咔嗒声进行平滑处理
            processed_audio = audio_data.copy()
            
            for idx in click_indices:
                if 0 < idx < len(processed_audio) - 1:
                    # 用相邻样本的平均值替换
                    processed_audio[idx] = (processed_audio[idx-1] + processed_audio[idx+1]) / 2
            
            return processed_audio
            
        except Exception as e:
            logger.warning(f"咔嗒声去除失败: {e}")
            return audio_data
    
    def _smooth_transitions(self, audio_data: np.ndarray) -> np.ndarray:
        """平滑过渡
        
        Args:
            audio_data: 音频数据
            
        Returns:
            平滑后的音频
        """
        try:
            # 对音频开头和结尾进行额外的平滑处理
            smooth_samples = int(0.01 * self.sample_rate)  # 10ms
            
            if len(audio_data) > smooth_samples * 2:
                # 开头平滑
                for i in range(smooth_samples):
                    weight = i / smooth_samples
                    audio_data[i] *= weight
                
                # 结尾平滑
                for i in range(smooth_samples):
                    weight = (smooth_samples - i) / smooth_samples
                    audio_data[-(i+1)] *= weight
            
            return audio_data
            
        except Exception as e:
            logger.warning(f"过渡平滑失败: {e}")
            return audio_data
    
    def _generate_quality_report(self, segments: List[Dict], 
                               vocal_track: np.ndarray) -> Dict:
        """生成质量报告
        
        Args:
            segments: 处理后的片段列表
            vocal_track: 人声轨道
            
        Returns:
            质量报告
        """
        if not segments:
            return {
                'overall_quality': 0.0,
                'segment_count': 0,
                'avg_duration': 0.0,
                'quality_distribution': {},
                'issues': ['没有有效片段']
            }
        
        # 统计信息
        durations = [seg['duration'] for seg in segments]
        quality_scores = [seg['quality_metrics']['overall_quality'] for seg in segments]
        
        # 质量分布
        quality_distribution = {
            'excellent': sum(1 for q in quality_scores if q >= 0.8),
            'good': sum(1 for q in quality_scores if 0.6 <= q < 0.8),
            'fair': sum(1 for q in quality_scores if 0.4 <= q < 0.6),
            'poor': sum(1 for q in quality_scores if q < 0.4)
        }
        
        # 问题检测
        issues = []
        if np.mean(durations) < 7:
            issues.append("平均片段长度偏短")
        if np.mean(durations) > 13:
            issues.append("平均片段长度偏长")
        if np.mean(quality_scores) < 0.6:
            issues.append("整体质量偏低")
        if len(segments) < 3:
            issues.append("片段数量过少")
        if len(segments) > 20:
            issues.append("片段数量过多")
        
        return {
            'overall_quality': np.mean(quality_scores),
            'segment_count': len(segments),
            'avg_duration': np.mean(durations),
            'duration_std': np.std(durations),
            'quality_distribution': quality_distribution,
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'issues': issues if issues else ['无明显问题']
        }

    def enforce_quiet_cut(self, x_mono, sr, t_sec,
                          win_ms=80, guard_db=3.0, floor_pct=0.05,
                          search_right_ms=400):
        """
        技术：局部RMS + 动态噪声地板。若 t 附近不够安静，只向右搜索第一个"够安静"的谷底。
        - win_ms: 评估窗口
        - guard_db: 相对地板的余量（地板+3dB 以内才算安静）
        - search_right_ms: 最多向右搜的距离（保证不提前）
        """
        # 允许从配置覆盖守卫参数
        try:
            win_ms = get_config('quality_control.enforce_quiet_cut.win_ms', win_ms)
            guard_db = get_config('quality_control.enforce_quiet_cut.guard_db', guard_db)
            search_right_ms = get_config('quality_control.enforce_quiet_cut.search_right_ms', search_right_ms)
            floor_cfg = get_config('quality_control.enforce_quiet_cut.floor_percentile', int(floor_pct*100))
            # 支持传入百分数(0-100)或比例(0-1)
            if isinstance(floor_cfg, (int, float)):
                floor_pct = float(floor_cfg) / 100.0 if floor_cfg > 1 else float(floor_cfg)
        except Exception:
            pass

        hop_ms = 10
        rms_db, t_axis = self._moving_rms_db(x_mono, sr, frame_ms=win_ms, hop_ms=hop_ms)
        rms_db = self._ema_smooth(rms_db, sr, hop_ms=hop_ms, smooth_ms=120)
        
        # 🔴 关键修复：使用全局最小值而非滚动分位数作为噪声地板
        # 滚动分位数在人声密集区域会产生过高的地板，导致无法正确识别高能量区
        global_floor = np.percentile(rms_db, floor_pct * 100)  # 全局5%分位数
        floor_db = np.full_like(rms_db, global_floor)

        def ok(idx):
            # 主要判断：相对于噪声地板的能量
            # 动态阈值：基于地板自适应
            # 如果地板很低（<-40dB），使用较严格的guard
            # 如果地板较高（>-30dB），使用较宽松的guard
            adaptive_guard = guard_db
            if global_floor < -40:
                adaptive_guard = guard_db  # 保持原值
            elif global_floor > -30:
                adaptive_guard = guard_db * 1.5  # 放宽1.5倍
            
            return rms_db[idx] <= floor_db[idx] + adaptive_guard

        # 找到 t 对应的帧
        idx = int(t_sec / (hop_ms/1000.0))
        if idx < 0 or idx >= len(rms_db):
            return t_sec

        if ok(idx):
            return t_sec  # 已够安静

        # 只向右找"安静谷底"
        max_step = int(search_right_ms / hop_ms)
        best = None
        for k in range(1, max_step+1):
            j = idx + k
            if j >= len(rms_db): break
            if ok(j):
                best = j
                break
        
        # 🔴 关键修复：如果在搜索范围内找不到安静点，扩大搜索到整个剩余音频
        if best is None:
            # 继续向右搜索整个剩余音频
            for j in range(idx + max_step + 1, len(rms_db)):
                if ok(j):
                    best = j
                    break
        
        if best is None:
            # 还是找不到，返回特殊值表示该切点无效（应被过滤）
            return -1.0  # 负值表示无效切点
        return best * (hop_ms/1000.0)

    def _moving_rms_db(self, x: np.ndarray, sr: int, frame_ms: int = 30, hop_ms: int = 10):
        """RMS 能量包络计算"""
        frame = int(sr * frame_ms / 1000)
        hop = int(sr * hop_ms / 1000)
        if frame <= 2: frame = 3
        if hop < 1: hop = 1
        n = (len(x) - frame) // hop + 1
        rms = np.zeros(n, dtype=np.float32)
        for i in range(n):
            seg = x[i*hop : i*hop + frame]
            if len(seg) == 0: break
            rms[i] = np.sqrt(np.mean(seg**2) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)
        t_axis = np.arange(n) * (hop / sr)
        return db, t_axis

    def _ema_smooth(self, x: np.ndarray, sr: int, hop_ms: int = 10, smooth_ms: int = 120):
        """指数滑动平均，技术: 概率/能量平滑以抑制抖动"""
        alpha = np.exp(- (hop_ms / smooth_ms))
        y = np.zeros_like(x, dtype=np.float32)
        acc = 0.0
        for i, v in enumerate(x):
            acc = alpha * acc + (1 - alpha) * v
            y[i] = acc
        return y

    def _rolling_percentile_db(self, x: np.ndarray, sr: int, hop_ms: int = 10, win_s: float = 30.0, p: float = 0.05):
        """动态噪声地板：滚动分位（默认 5%）"""
        win = int(win_s * 1000 / hop_ms)
        if win < 5: win = 5
        out = np.zeros_like(x, dtype=np.float32)
        half = win // 2
        for i in range(len(x)):
            a = max(0, i - half)
            b = min(len(x), i + half + 1)
            seg = np.sort(x[a:b])
            k = int(len(seg) * p)
            k = np.clip(k, 0, len(seg)-1)
            out[i] = seg[k]
        return out

    def safe_zero_crossing_align(self, x_mono, sr, t_sec, window_ms=10):
        """
        安全的零交叉对齐：先对齐到零交叉，再验证能量
        如果零交叉对齐后能量过高，则回退到原始切点
        """
        # 1. 找到最近的零交叉点
        w = int(sr * window_ms / 1000)
        c = int(t_sec * sr)
        a = max(1, c - w)
        b = min(len(x_mono)-1, c + w)
        
        if b <= a:
            return t_sec
        
        seg = x_mono[a:b]
        
        # 寻找符号变化的位置（零交叉）
        zero_crossings = []
        for i in range(len(seg)-1):
            if seg[i] * seg[i+1] <= 0:  # 符号变化或有一个为0
                zero_crossings.append(a + i)
        
        if not zero_crossings:
            return t_sec  # 没有零交叉，保持原切点
        
        # 找最接近中心的零交叉
        best_zc = min(zero_crossings, key=lambda zc: abs(zc - c))
        t_zc = best_zc / sr
        
        # 2. 验证零交叉对齐后的切点是否安静
        t_validated = self.enforce_quiet_cut(x_mono, sr, t_zc)
        
        # 3. 如果验证后偏离太多，说明零交叉对齐把我们拉到了高能量区，回退到原始切点
        if abs(t_validated - t_zc) > 0.050:  # 50ms tolerance
            return t_sec  # 回退到原始切点
        
        return t_validated

    def pure_filter_cut_points(self, cut_points, audio_duration, min_interval=0.5, min_segment_duration=0.5):
        """
        纯粹的切点过滤器：只过滤，不重定位（vocal_prime.md 核心要求）
        
        Args:
            cut_points: 切点列表（秒）
            audio_duration: 音频总时长
            min_interval: 最小间隔
            min_segment_duration: 最小片段时长
            
        Returns:
            过滤后的切点列表（只移除无效点，不做任何重定位）
        """
        if not cut_points:
            return []
        
        # 排序并去重
        sorted_points = sorted(set(cut_points))
        filtered_points = []
        
        # 过滤边界点
        for point in sorted_points:
            if point <= min_segment_duration or point >= (audio_duration - min_segment_duration):
                continue  # 过滤掉过于接近边界的点
            filtered_points.append(point)
        
        if len(filtered_points) <= 1:
            return filtered_points
        
        # 过滤间隔过小的点（保留第一个，丢弃后续的）
        final_points = [filtered_points[0]]
        
        for point in filtered_points[1:]:
            if point - final_points[-1] >= min_interval:
                final_points.append(point)
            # 否则直接丢弃，不做任何重定位
        
        return final_points

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/quality_controller.py
# AI-SUMMARY: 质量控制核心模块，确保分割结果的质量和完整性

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

from vocal_smart_splitter.utils.config_manager import get_config
from vocal_smart_splitter.utils.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class QualityController:
    """质量控制器，确保分割结果的质量"""
    
    def __init__(self, sample_rate: int = 22050):
        """初始化质量控制器
        
        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate)
        
        # 从配置加载参数
        self.validate_split_points = get_config('quality_control.validate_split_points', True)
        self.min_pause_at_split = get_config('quality_control.min_pause_at_split', 0.1)
        self.max_vocal_at_split = get_config('quality_control.max_vocal_at_split', 0.05)
        
        self.min_vocal_content_ratio = get_config('quality_control.min_vocal_content_ratio', 0.4)
        self.max_silence_ratio = get_config('quality_control.max_silence_ratio', 0.3)
        
        self.fade_in_duration = get_config('quality_control.fade_in_duration', 0.02)
        self.fade_out_duration = get_config('quality_control.fade_out_duration', 0.02)
        self.normalize_audio = get_config('quality_control.normalize_audio', True)
        
        self.remove_click_noise = get_config('quality_control.remove_click_noise', True)
        self.smooth_transitions = get_config('quality_control.smooth_transitions', True)
        
        logger.info("质量控制器初始化完成")
    
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
        """验证单个片段 - 修复版本：专注分割准确性而非片段质量
        
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
            'quality_metrics': {}
        }
        
        # 新的验证策略：只检查基本的技术问题，不强加人为的质量标准
        
        # 1. 检查片段长度 - 大幅放宽限制，允许自然停顿产生的短片段
        duration = segment['duration']
        
        # 按配置判断长度边界；如果未配置，则不过滤长度
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
        
        # 记录但不过滤 - 让所有在自然停顿处的分割都保留
        logger.debug(f"片段 {segment['index']}: {duration:.2f}s, 人声比例: {vocal_content_ratio:.2f}")
        
        return validation_result
    
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/vocal_separator.py
# AI-SUMMARY: 人声分离核心模块，专门针对歌曲进行高质量人声分离

import numpy as np
import librosa
import logging
from typing import Tuple, Dict, Optional, List
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy import signal

from vocal_smart_splitter.utils.config_manager import get_config
from vocal_smart_splitter.utils.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class VocalSeparator:
    """专业人声分离器，专门针对歌曲优化"""
    
    def __init__(self, sample_rate: int = 22050):
        """初始化人声分离器
        
        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate)
        
        # 从配置加载参数
        self.hpss_margin = get_config('vocal_separation.hpss_margin', 3.0)
        self.hpss_power = get_config('vocal_separation.hpss_power', 2.0)
        self.mask_threshold = get_config('vocal_separation.mask_threshold', 0.15)
        self.mask_smoothing = get_config('vocal_separation.mask_smoothing', 5)
        
        # 人声频率范围
        self.vocal_freq_min = get_config('vocal_separation.vocal_freq_min', 80)
        self.vocal_freq_max = get_config('vocal_separation.vocal_freq_max', 4000)
        self.vocal_core_min = get_config('vocal_separation.vocal_core_min', 200)
        self.vocal_core_max = get_config('vocal_separation.vocal_core_max', 1000)
        
        # 质量控制参数
        self.min_vocal_ratio = get_config('vocal_separation.min_vocal_ratio', 0.15)
        self.max_noise_ratio = get_config('vocal_separation.max_noise_ratio', 0.3)
        
        logger.info("人声分离器初始化完成")
    
    def separate_vocals(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """分离人声和伴奏
        
        Args:
            audio: 输入音频
            
        Returns:
            (人声轨道, 伴奏轨道, 分离质量信息)
        """
        logger.info("开始人声分离...")
        
        try:
            # 1. 谐波-冲击分离
            harmonic, percussive = self._harmonic_percussive_separation(audio)
            
            # 2. 频谱掩码分离
            vocal_track, accompaniment_track = self._spectral_mask_separation(audio, harmonic)
            
            # 3. 后处理优化
            vocal_track = self._post_process_vocal(vocal_track, audio)
            accompaniment_track = self._post_process_accompaniment(accompaniment_track, audio)
            
            # 4. 质量评估
            quality_info = self._evaluate_separation_quality(vocal_track, accompaniment_track, audio)
            
            markers = self._compute_vocal_presence_markers(vocal_track)
            quality_info.update(markers)

            logger.info(f"人声分离完成，质量评分: {quality_info['overall_score']:.3f}")

            
            return vocal_track, accompaniment_track, quality_info
            
        except Exception as e:
            logger.error(f"人声分离失败: {e}")
            raise
    
    def _harmonic_percussive_separation(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """谐波-冲击分离
        
        Args:
            audio: 输入音频
            
        Returns:
            (谐波成分, 冲击成分)
        """
        try:
            harmonic, percussive = librosa.effects.hpss(
                audio,
                margin=self.hpss_margin,
                power=self.hpss_power
            )
            
            logger.debug("谐波-冲击分离完成")
            return harmonic, percussive
            
        except Exception as e:
            logger.error(f"谐波-冲击分离失败: {e}")
            return audio, np.zeros_like(audio)
    
    def _spectral_mask_separation(self, audio: np.ndarray, 
                                 harmonic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """基于频谱掩码的人声分离
        
        Args:
            audio: 原始音频
            harmonic: 谐波成分
            
        Returns:
            (人声轨道, 伴奏轨道)
        """
        try:
            # 计算STFT
            hop_length = 512
            stft = librosa.stft(audio, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 计算谐波成分的STFT
            harmonic_stft = librosa.stft(harmonic, hop_length=hop_length)
            harmonic_magnitude = np.abs(harmonic_stft)
            
            # 提取音频特征
            spectral_features = self.feature_extractor.extract_spectral_features(audio)
            vocal_features = self.feature_extractor.extract_vocal_features(audio)
            
            # 创建人声掩码
            vocal_mask = self._create_vocal_mask(
                magnitude, harmonic_magnitude, spectral_features, vocal_features
            )
            
            # 应用掩码
            vocal_stft = stft * vocal_mask
            accompaniment_stft = stft * (1 - vocal_mask)
            
            # 重构音频
            vocal_track = librosa.istft(vocal_stft, hop_length=hop_length)
            accompaniment_track = librosa.istft(accompaniment_stft, hop_length=hop_length)

            # 确保长度一致
            min_length = min(len(vocal_track), len(accompaniment_track), len(audio))
            vocal_track = vocal_track[:min_length]
            accompaniment_track = accompaniment_track[:min_length]

            # 检查并修复无穷大或NaN值
            vocal_track = np.nan_to_num(vocal_track, nan=0.0, posinf=0.0, neginf=0.0)
            accompaniment_track = np.nan_to_num(accompaniment_track, nan=0.0, posinf=0.0, neginf=0.0)

            # 限制动态范围
            vocal_track = np.clip(vocal_track, -1.0, 1.0)
            accompaniment_track = np.clip(accompaniment_track, -1.0, 1.0)
            
            logger.debug("频谱掩码分离完成")
            return vocal_track, accompaniment_track
            
        except Exception as e:
            logger.error(f"频谱掩码分离失败: {e}")
            return audio * 0.5, audio * 0.5
    
    def _create_vocal_mask(self, magnitude: np.ndarray, 
                          harmonic_magnitude: np.ndarray,
                          spectral_features: Dict,
                          vocal_features: Dict) -> np.ndarray:
        """创建人声掩码
        
        Args:
            magnitude: 原始幅度谱
            harmonic_magnitude: 谐波幅度谱
            spectral_features: 频谱特征
            vocal_features: 人声特征
            
        Returns:
            人声掩码
        """
        vocal_mask = np.zeros_like(magnitude)
        
        # 频率轴
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
        
        # 获取特征
        spectral_centroids = spectral_features['spectral_centroids']
        mfcc = spectral_features['mfcc']
        vocal_clarity = vocal_features['vocal_clarity']
        
        # 确保时间维度一致
        n_frames = magnitude.shape[1]
        if len(spectral_centroids) != n_frames:
            spectral_centroids = np.interp(
                np.linspace(0, 1, n_frames),
                np.linspace(0, 1, len(spectral_centroids)),
                spectral_centroids
            )
        
        if len(vocal_clarity) != n_frames:
            vocal_clarity = np.interp(
                np.linspace(0, 1, n_frames),
                np.linspace(0, 1, len(vocal_clarity)),
                vocal_clarity
            )
        
        # 为每个时间帧创建掩码
        for t in range(n_frames):
            # 人声判断条件
            centroid = spectral_centroids[t]
            clarity = vocal_clarity[t]
            
            # 基于频率的人声判断
            is_vocal_freq = self.vocal_freq_min < centroid < self.vocal_freq_max
            
            # 基于清晰度的人声判断
            is_clear_vocal = clarity > 0.3
            
            # 基于MFCC的人声判断
            mfcc_vocal = mfcc[1, t] > -25 if t < mfcc.shape[1] else False
            
            # 基于谐波强度的判断
            harmonic_ratio = np.sum(harmonic_magnitude[:, t]) / (np.sum(magnitude[:, t]) + 1e-8)
            is_harmonic = harmonic_ratio > 0.2
            
            if is_vocal_freq and is_clear_vocal and mfcc_vocal and is_harmonic:
                # 创建频率选择性掩码
                freq_mask = self._create_frequency_mask(freqs, centroid)
                
                # 基于能量的掩码
                energy_mask = magnitude[:, t] > (self.mask_threshold * np.max(magnitude[:, t]))
                
                # 组合掩码
                combined_mask = freq_mask * energy_mask * clarity
                vocal_mask[:, t] = combined_mask
        
        # 平滑掩码
        vocal_mask = median_filter(vocal_mask, size=(1, self.mask_smoothing))
        
        # 确保掩码在0-1范围内
        vocal_mask = np.clip(vocal_mask, 0, 1)
        
        return vocal_mask
    
    def _create_frequency_mask(self, freqs: np.ndarray, centroid: float) -> np.ndarray:
        """创建基于频率的掩码
        
        Args:
            freqs: 频率数组
            centroid: 频谱质心
            
        Returns:
            频率掩码
        """
        freq_mask = np.zeros_like(freqs)
        
        for i, freq in enumerate(freqs):
            if self.vocal_freq_min <= freq <= self.vocal_freq_max:
                if self.vocal_core_min <= freq <= self.vocal_core_max:
                    # 核心人声频率范围
                    freq_mask[i] = 1.0
                elif freq < self.vocal_core_min:
                    # 低频衰减
                    freq_mask[i] = 0.6
                else:
                    # 高频衰减
                    freq_mask[i] = 0.7
        
        return freq_mask
    
    def _post_process_vocal(self, vocal_track: np.ndarray, 
                           original_audio: np.ndarray) -> np.ndarray:
        """人声轨道后处理
        
        Args:
            vocal_track: 分离的人声轨道
            original_audio: 原始音频
            
        Returns:
            处理后的人声轨道
        """
        # 去除噪声
        vocal_track = self._denoise_vocal(vocal_track)
        
        # 增强人声清晰度
        vocal_track = self._enhance_vocal_clarity(vocal_track)
        
        # 限制动态范围
        vocal_track = np.clip(vocal_track, -1.0, 1.0)
        
        return vocal_track
    
    def _post_process_accompaniment(self, accompaniment_track: np.ndarray,
                                  original_audio: np.ndarray) -> np.ndarray:
        """伴奏轨道后处理
        
        Args:
            accompaniment_track: 分离的伴奏轨道
            original_audio: 原始音频
            
        Returns:
            处理后的伴奏轨道
        """
        # 限制动态范围
        accompaniment_track = np.clip(accompaniment_track, -1.0, 1.0)
        
        return accompaniment_track
    
    def _denoise_vocal(self, vocal_track: np.ndarray) -> np.ndarray:
        """人声去噪
        
        Args:
            vocal_track: 人声轨道
            
        Returns:
            去噪后的人声轨道
        """
        try:
            # 使用Wiener滤波进行去噪
            # 这里使用简单的频域滤波
            stft = librosa.stft(vocal_track)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 噪声估计（使用低能量区域）
            noise_threshold = max(np.percentile(magnitude, 10), 1e-8)

            # Wiener滤波
            denominator = magnitude ** 2 + noise_threshold ** 2
            denominator = np.maximum(denominator, 1e-10)  # 避免除零
            wiener_filter = magnitude ** 2 / denominator
            wiener_filter = np.clip(wiener_filter, 0.0, 1.0)  # 限制滤波器范围
            
            # 应用滤波器
            filtered_magnitude = magnitude * wiener_filter
            filtered_stft = filtered_magnitude * np.exp(1j * phase)
            
            # 重构音频
            denoised_vocal = librosa.istft(filtered_stft)

            # 修复数值问题
            denoised_vocal = np.nan_to_num(denoised_vocal, nan=0.0, posinf=0.0, neginf=0.0)
            denoised_vocal = np.clip(denoised_vocal, -1.0, 1.0)

            return denoised_vocal
            
        except Exception as e:
            logger.warning(f"人声去噪失败: {e}")
            return vocal_track
    
    def _enhance_vocal_clarity(self, vocal_track: np.ndarray) -> np.ndarray:
        """增强人声清晰度
        
        Args:
            vocal_track: 人声轨道
            
        Returns:
            增强后的人声轨道
        """
        try:
            # 使用EQ增强人声频率范围
            # 这里使用简单的频域增强
            stft = librosa.stft(vocal_track)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
            
            # 创建EQ曲线
            eq_curve = np.ones_like(freqs)
            for i, freq in enumerate(freqs):
                if 200 <= freq <= 1000:  # 基频范围增强
                    eq_curve[i] = 1.2
                elif 1000 <= freq <= 3000:  # 共振峰范围增强
                    eq_curve[i] = 1.1
            
            # 应用EQ
            enhanced_magnitude = magnitude * eq_curve[:, np.newaxis]
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            
            # 重构音频
            enhanced_vocal = librosa.istft(enhanced_stft)

            # 修复数值问题
            enhanced_vocal = np.nan_to_num(enhanced_vocal, nan=0.0, posinf=0.0, neginf=0.0)
            enhanced_vocal = np.clip(enhanced_vocal, -1.0, 1.0)

            return enhanced_vocal
            
        except Exception as e:
            logger.warning(f"人声清晰度增强失败: {e}")
            return vocal_track
    
    def _evaluate_separation_quality(self, vocal_track: np.ndarray,
                                   accompaniment_track: np.ndarray,
                                   original_audio: np.ndarray) -> Dict:
        """评估分离质量
        
        Args:
            vocal_track: 人声轨道
            accompaniment_track: 伴奏轨道
            original_audio: 原始音频
            
        Returns:
            质量评估信息
        """
        try:
            # 能量比例
            vocal_energy = np.sum(vocal_track ** 2)
            accompaniment_energy = np.sum(accompaniment_track ** 2)
            total_energy = np.sum(original_audio ** 2)
            
            vocal_ratio = vocal_energy / (total_energy + 1e-8)
            accompaniment_ratio = accompaniment_energy / (total_energy + 1e-8)
            
            # 重构误差
            reconstructed = vocal_track + accompaniment_track
            min_length = min(len(reconstructed), len(original_audio))
            reconstruction_error = np.mean((reconstructed[:min_length] - original_audio[:min_length]) ** 2)
            
            # 人声清晰度
            vocal_features = self.feature_extractor.extract_vocal_features(vocal_track)
            avg_vocal_clarity = np.mean(vocal_features['vocal_clarity'])
            
            # 综合评分
            quality_score = 0.0
            
            # 人声比例合理性 (15%-60%)
            if 0.15 <= vocal_ratio <= 0.6:
                quality_score += 0.3
            
            # 重构误差小
            if reconstruction_error < 0.01:
                quality_score += 0.3
            
            # 人声清晰度高
            if avg_vocal_clarity > 0.4:
                quality_score += 0.4
            
            quality_info = {
                'vocal_ratio': vocal_ratio,
                'accompaniment_ratio': accompaniment_ratio,
                'reconstruction_error': reconstruction_error,
                'vocal_clarity': avg_vocal_clarity,
                'overall_score': quality_score
            }
            
            return quality_info
            
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            return {
                'vocal_ratio': 0.0,
                'accompaniment_ratio': 0.0,
                'reconstruction_error': 1.0,
                'vocal_clarity': 0.0,
                'overall_score': 0.0
            }
    def _compute_vocal_presence_markers(self, vocal_audio: np.ndarray) -> Dict:
        """计算人声出现/消失的关键切点信息"""
        sr = self.sample_rate
        if sr <= 0 or vocal_audio is None:
            return {
                'vocal_presence_cut_points_sec': [],
                'vocal_presence_cut_points_samples': [],
                'vocal_presence_segments': [],
                'pure_music_segments': []
            }
        duration = float(len(vocal_audio)) / sr if len(vocal_audio) > 0 else 0.0
        threshold_db = float(get_config('quality_control.segment_vocal_threshold_db', -50.0))
        pure_music_min = float(get_config('quality_control.pure_music_min_duration', 0.0))
        hop = max(1, int(0.02 * sr))
        frame_length = max(hop * 2, int(0.05 * sr))
        if len(vocal_audio) == 0:
            return {
                'vocal_presence_cut_points_sec': [],
                'vocal_presence_cut_points_samples': [],
                'vocal_presence_segments': [],
                'pure_music_segments': []
            }
        try:
            rms = librosa.feature.rms(y=vocal_audio, frame_length=frame_length, hop_length=hop)[0]
        except Exception:
            rms = np.sqrt(np.mean(np.square(vocal_audio) + 1e-12)) * np.ones(max(1, len(vocal_audio) // hop + 1))
        rms_db = 20.0 * np.log10(rms + 1e-12)
        vocal_mask = rms_db > threshold_db
        if vocal_mask.size == 0:
            return {
                'vocal_presence_cut_points_sec': [],
                'vocal_presence_cut_points_samples': [],
                'vocal_presence_segments': [],
                'pure_music_segments': []
            }
        times = librosa.frames_to_time(np.arange(len(vocal_mask)), sr=sr, hop_length=hop)
        segments: List[Dict] = []
        current_state = bool(vocal_mask[0])
        current_start = 0.0
        for idx in range(1, len(vocal_mask)):
            state = bool(vocal_mask[idx])
            if state != current_state:
                end_time = float(times[idx])
                segments.append({'start': current_start, 'end': end_time, 'is_vocal': current_state})
                current_start = float(times[idx])
                current_state = state
        segments.append({'start': current_start, 'end': duration, 'is_vocal': current_state})
        def clamp_time(value: float) -> float:
            return float(min(max(value, 0.0), duration))
        cut_points = set()
        first_vocal = next((seg for seg in segments if seg['is_vocal'] and seg['end'] > seg['start']), None)
        if first_vocal is not None:
            cut_points.add(clamp_time(first_vocal['start'] - 1.0))
        for prev, nxt in zip(segments, segments[1:]):
            if not prev['is_vocal'] and nxt['is_vocal']:
                if (prev['end'] - prev['start']) >= pure_music_min:
                    candidate = clamp_time(nxt['start'] - 1.0)
                    if candidate >= prev['start']:
                        cut_points.add(candidate)
        last_vocal = next((seg for seg in reversed(segments) if seg['is_vocal'] and seg['end'] > seg['start']), None)
        if last_vocal is not None:
            cut_points.add(clamp_time(last_vocal['end'] + 1.0))
        cut_points_sec = sorted({cp for cp in cut_points if 0.0 <= cp <= duration})
        cut_points_samples = [int(round(cp * sr)) for cp in cut_points_sec]
        return {
            'vocal_presence_cut_points_sec': cut_points_sec,
            'vocal_presence_cut_points_samples': cut_points_samples,
            'vocal_presence_segments': segments,
            'pure_music_segments': [seg for seg in segments if not seg['is_vocal'] and seg['end'] > seg['start']]
        }


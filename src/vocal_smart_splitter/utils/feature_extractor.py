#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/utils/feature_extractor.py
# AI-SUMMARY: 音频特征提取工具，专门用于人声分析和换气检测

import numpy as np
import librosa
import logging
from typing import Tuple, Dict, List, Optional
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter1d

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """音频特征提取器，专门针对人声分析优化"""
    
    def __init__(self, sample_rate: int = 22050):
        """初始化特征提取器
        
        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_fft = 2048
        self.win_length = 1024
        
        logger.debug(f"特征提取器初始化: sr={sample_rate}, hop={self.hop_length}")
    
    def extract_energy_features(self, audio: np.ndarray, 
                               window_size: float = 0.05,
                               hop_size: float = 0.01) -> Dict[str, np.ndarray]:
        """提取能量相关特征
        
        Args:
            audio: 音频数据
            window_size: 分析窗口大小(秒)
            hop_size: 跳跃步长(秒)
            
        Returns:
            包含各种能量特征的字典
        """
        window_samples = int(window_size * self.sample_rate)
        hop_samples = int(hop_size * self.sample_rate)
        
        # 短时能量
        energy = []
        rms_energy = []
        zero_crossing_rate = []
        
        for i in range(0, len(audio) - window_samples, hop_samples):
            window = audio[i:i + window_samples]
            
            # 能量
            energy.append(np.sum(window ** 2))
            
            # RMS能量
            rms_energy.append(np.sqrt(np.mean(window ** 2)))
            
            # 零交叉率
            zcr = np.sum(np.diff(np.sign(window)) != 0) / len(window)
            zero_crossing_rate.append(zcr)
        
        energy = np.array(energy)
        rms_energy = np.array(rms_energy)
        zero_crossing_rate = np.array(zero_crossing_rate)
        
        # 时间轴
        time_axis = np.arange(len(energy)) * hop_size
        
        return {
            'energy': energy,
            'rms_energy': rms_energy,
            'zero_crossing_rate': zero_crossing_rate,
            'time_axis': time_axis,
            'energy_derivative': np.gradient(energy),
            'energy_smooth': gaussian_filter1d(energy, sigma=2)
        }
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """提取频谱特征
        
        Args:
            audio: 音频数据
            
        Returns:
            包含频谱特征的字典
        """
        # 计算STFT
        stft = librosa.stft(audio, hop_length=self.hop_length, 
                           n_fft=self.n_fft, win_length=self.win_length)
        magnitude = np.abs(stft)
        
        # 频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # 频谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # 频谱滚降
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # MFCC特征
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length
        )
        
        # 色度特征
        chroma = librosa.feature.chroma_stft(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # 时间轴
        time_axis = librosa.frames_to_time(
            np.arange(magnitude.shape[1]), sr=self.sample_rate, hop_length=self.hop_length
        )
        
        return {
            'stft_magnitude': magnitude,
            'spectral_centroids': spectral_centroids,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'mfcc': mfcc,
            'chroma': chroma,
            'time_axis': time_axis
        }
    
    def extract_vocal_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """提取人声特定特征
        
        Args:
            audio: 音频数据
            
        Returns:
            包含人声特征的字典
        """
        # 基础频谱特征
        spectral_features = self.extract_spectral_features(audio)
        
        # 人声频率范围能量
        stft_mag = spectral_features['stft_magnitude']
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        
        # 人声主要频率范围 (80-4000 Hz)
        vocal_freq_mask = (freqs >= 80) & (freqs <= 4000)
        vocal_energy = np.sum(stft_mag[vocal_freq_mask, :], axis=0)
        
        # 基频范围能量 (200-1000 Hz)
        fundamental_freq_mask = (freqs >= 200) & (freqs <= 1000)
        fundamental_energy = np.sum(stft_mag[fundamental_freq_mask, :], axis=0)
        
        # 共振峰范围能量 (1000-3000 Hz)
        formant_freq_mask = (freqs >= 1000) & (freqs <= 3000)
        formant_energy = np.sum(stft_mag[formant_freq_mask, :], axis=0)
        
        # 人声清晰度指标
        vocal_clarity = fundamental_energy / (np.sum(stft_mag, axis=0) + 1e-8)
        
        # 谐波强度
        harmonic_strength = self._calculate_harmonic_strength(stft_mag, freqs)
        
        return {
            'vocal_energy': vocal_energy,
            'fundamental_energy': fundamental_energy,
            'formant_energy': formant_energy,
            'vocal_clarity': vocal_clarity,
            'harmonic_strength': harmonic_strength,
            'time_axis': spectral_features['time_axis']
        }
    
    def _calculate_harmonic_strength(self, stft_magnitude: np.ndarray, 
                                   freqs: np.ndarray) -> np.ndarray:
        """计算谐波强度
        
        Args:
            stft_magnitude: STFT幅度谱
            freqs: 频率数组
            
        Returns:
            谐波强度数组
        """
        harmonic_strength = np.zeros(stft_magnitude.shape[1])
        
        # 寻找基频和谐波
        for t in range(stft_magnitude.shape[1]):
            spectrum = stft_magnitude[:, t]
            
            # 寻找峰值
            peaks, _ = signal.find_peaks(spectrum, height=np.max(spectrum) * 0.1)
            
            if len(peaks) > 0:
                # 计算谐波比例
                peak_freqs = freqs[peaks]
                peak_magnitudes = spectrum[peaks]
                
                # 寻找可能的基频
                fundamental_candidates = peak_freqs[peak_freqs <= 500]
                
                if len(fundamental_candidates) > 0:
                    f0 = fundamental_candidates[np.argmax(peak_magnitudes[:len(fundamental_candidates)])]
                    
                    # 计算谐波强度
                    harmonic_sum = 0
                    total_sum = np.sum(spectrum)
                    
                    for h in range(1, 6):  # 前5个谐波
                        harmonic_freq = f0 * h
                        freq_idx = np.argmin(np.abs(freqs - harmonic_freq))
                        
                        if freq_idx < len(spectrum):
                            harmonic_sum += spectrum[freq_idx]
                    
                    harmonic_strength[t] = harmonic_sum / (total_sum + 1e-8)
        
        return harmonic_strength
    
    def detect_voice_activity(self, audio: np.ndarray, 
                             threshold: float = 0.02) -> np.ndarray:
        """检测语音活动
        
        Args:
            audio: 音频数据
            threshold: 检测阈值
            
        Returns:
            语音活动掩码 (True表示有语音)
        """
        # 提取能量特征
        energy_features = self.extract_energy_features(audio)
        energy = energy_features['energy_smooth']
        
        # 提取人声特征
        vocal_features = self.extract_vocal_features(audio)
        vocal_clarity = vocal_features['vocal_clarity']
        
        # 插值到相同长度
        if len(vocal_clarity) != len(energy):
            vocal_clarity = np.interp(
                np.linspace(0, 1, len(energy)),
                np.linspace(0, 1, len(vocal_clarity)),
                vocal_clarity
            )
        
        # 综合判断
        energy_normalized = energy / (np.max(energy) + 1e-8)
        voice_activity = (energy_normalized > threshold) & (vocal_clarity > 0.3)
        
        # 形态学处理，去除短暂的间断
        from scipy.ndimage import binary_closing, binary_opening
        voice_activity = binary_closing(voice_activity, structure=np.ones(5))
        voice_activity = binary_opening(voice_activity, structure=np.ones(3))
        
        return voice_activity
    
    def extract_breath_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """提取换气相关特征
        
        Args:
            audio: 音频数据
            
        Returns:
            包含换气特征的字典
        """
        # 能量特征
        energy_features = self.extract_energy_features(audio, window_size=0.05, hop_size=0.01)
        
        # 人声特征
        vocal_features = self.extract_vocal_features(audio)
        
        # 能量下降检测
        energy = energy_features['energy_smooth']
        energy_derivative = energy_features['energy_derivative']
        
        # 检测急剧的能量下降（可能的换气点）
        energy_drops = energy_derivative < -np.std(energy_derivative) * 2
        
        # 低能量区域
        energy_threshold = np.percentile(energy[energy > 0], 20)
        low_energy_regions = energy < energy_threshold
        
        # 人声清晰度下降
        vocal_clarity = vocal_features['vocal_clarity']
        if len(vocal_clarity) != len(energy):
            vocal_clarity = np.interp(
                np.linspace(0, 1, len(energy)),
                np.linspace(0, 1, len(vocal_clarity)),
                vocal_clarity
            )
        
        clarity_drops = vocal_clarity < np.mean(vocal_clarity) * 0.5
        
        return {
            'energy_drops': energy_drops,
            'low_energy_regions': low_energy_regions,
            'clarity_drops': clarity_drops,
            'energy': energy,
            'vocal_clarity': vocal_clarity,
            'time_axis': energy_features['time_axis']
        }

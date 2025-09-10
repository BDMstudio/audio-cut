#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/utils/audio_processor.py
# AI-SUMMARY: 音频处理工具，负责音频的加载、预处理和保存

import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import logging
from typing import Tuple, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioProcessor:
    """音频处理器，专门针对人声分割优化"""
    
    def __init__(self, sample_rate: int = 22050, channels: int = 1):
        """初始化音频处理器
        
        Args:
            sample_rate: 目标采样率
            channels: 目标声道数
        """
        self.sample_rate = sample_rate
        self.channels = channels
        
        logger.debug(f"音频处理器初始化: sr={sample_rate}, channels={channels}")
    
    def load_audio(self, file_path: str, normalize: bool = True) -> Tuple[np.ndarray, int]:
        """加载音频文件
        
        Args:
            file_path: 音频文件路径
            normalize: 是否归一化
            
        Returns:
            (音频数据, 采样率)
        """
        logger.debug(f"加载音频: {file_path}")
        
        # 使用librosa加载，强制单声道和目标采样率
        audio, sr = librosa.load(
            file_path, 
            sr=self.sample_rate, 
            mono=(self.channels == 1)
        )
        
        # 确保音频是float32类型
        audio = audio.astype(np.float32)
        
        # 归一化处理
        if normalize and np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        logger.info(f"音频加载完成: 时长={len(audio)/sr:.2f}s, 采样率={sr}Hz")
        
        return audio, sr
    
    def save_audio(self, 
                   audio: np.ndarray, 
                   file_path: str,
                   sample_rate: Optional[int] = None,
                   subtype: str = 'PCM_24') -> None:
        """保存音频文件
        
        Args:
            audio: 音频数据
            file_path: 保存路径
            sample_rate: 采样率（默认使用实例采样率）
            subtype: 音频子类型
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        # 确保输出目录存在
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存音频
        sf.write(file_path, audio, sample_rate, subtype=subtype)
        logger.debug(f"音频已保存: {file_path}")
    
    def apply_fade(self, 
                   audio: np.ndarray,
                   fade_in_duration: float = 0.01,
                   fade_out_duration: float = 0.01) -> np.ndarray:
        """应用淡入淡出效果
        
        Args:
            audio: 音频数据
            fade_in_duration: 淡入时长（秒）
            fade_out_duration: 淡出时长（秒）
            
        Returns:
            处理后的音频
        """
        audio_copy = audio.copy()
        
        # 计算样本数
        fade_in_samples = int(fade_in_duration * self.sample_rate)
        fade_out_samples = int(fade_out_duration * self.sample_rate)
        
        # 应用淡入
        if fade_in_samples > 0 and len(audio_copy) > fade_in_samples:
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            audio_copy[:fade_in_samples] *= fade_in_curve
        
        # 应用淡出
        if fade_out_samples > 0 and len(audio_copy) > fade_out_samples:
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            audio_copy[-fade_out_samples:] *= fade_out_curve
        
        return audio_copy
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """归一化音频到目标音量
        
        Args:
            audio: 音频数据
            target_db: 目标音量（dB）
            
        Returns:
            归一化后的音频
        """
        # 计算当前RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms > 0:
            # 计算目标RMS
            target_rms = 10 ** (target_db / 20)
            
            # 计算缩放因子
            scale_factor = target_rms / rms
            
            # 应用缩放，但限制最大增益
            scale_factor = min(scale_factor, 10.0)  # 最大增益20dB
            
            return audio * scale_factor
        
        return audio
    
    def split_audio(self, 
                    audio: np.ndarray,
                    split_points: list,
                    apply_fade: bool = True) -> list:
        """根据分割点切分音频
        
        Args:
            audio: 音频数据
            split_points: 分割点列表（秒）
            apply_fade: 是否应用淡入淡出
            
        Returns:
            音频片段列表
        """
        segments = []
        split_points = [0] + sorted(split_points) + [len(audio) / self.sample_rate]
        
        for i in range(len(split_points) - 1):
            start_time = split_points[i]
            end_time = split_points[i + 1]
            
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            segment = audio[start_sample:end_sample]
            
            if apply_fade and len(segment) > 0:
                segment = self.apply_fade(segment)
            
            segments.append(segment)
        
        return segments
    
    def export_segments(self,
                       segments: list,
                       output_dir: str,
                       filename_prefix: str = "segment",
                       normalize: bool = False) -> list:
        """导出音频片段
        
        Args:
            segments: 音频片段列表
            output_dir: 输出目录
            filename_prefix: 文件名前缀
            normalize: 是否归一化
            
        Returns:
            保存的文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        for i, segment in enumerate(segments):
            if normalize:
                segment = self.normalize_audio(segment)
            
            filename = f"{filename_prefix}_{i+1:03d}.wav"
            filepath = os.path.join(output_dir, filename)
            
            self.save_audio(segment, filepath)
            saved_files.append(filepath)
        
        logger.info(f"已导出 {len(segments)} 个音频片段到 {output_dir}")
        
        return saved_files


def time_to_sample(time_sec: float, sample_rate: int) -> int:
    """时间转换为样本索引
    
    Args:
        time_sec: 时间（秒）
        sample_rate: 采样率
        
    Returns:
        样本索引
    """
    return int(time_sec * sample_rate)


def sample_to_time(sample_idx: int, sample_rate: int) -> float:
    """样本索引转换为时间
    
    Args:
        sample_idx: 样本索引
        sample_rate: 采样率
        
    Returns:
        时间（秒）
    """
    return sample_idx / sample_rate


def map_time_between_domains(t_src_sec: float, sr_src: int, sr_dst: int,
                             latency_samples: int = 0) -> float:
    """
    跨采样率时间映射（vocal_prime.md核心方案）
    把 vocal 域时间映射回原混音域
    
    Args:
        t_src_sec: 源域时间（秒）
        sr_src: 源采样率
        sr_dst: 目标采样率
        latency_samples: 重采样延迟（样本数）
    
    Returns:
        目标域时间（秒）
    """
    # 先转换为源域样本索引
    src_samples = t_src_sec * sr_src
    
    # 映射到目标域（考虑延迟）
    dst_samples = (src_samples * sr_dst / sr_src) + latency_samples
    
    # 转换回秒
    return dst_samples / sr_dst


def find_zero_crossing(audio: np.ndarray, 
                       sample_rate: int,
                       time_sec: float,
                       window_ms: float = 10.0,
                       search_right_only: bool = True) -> float:
    """寻找最近的零交叉点
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        time_sec: 中心时间（秒）
        window_ms: 搜索窗口（毫秒）
        search_right_only: 是否只向右搜索
        
    Returns:
        零交叉点时间（秒）
    """
    center_sample = int(time_sec * sample_rate)
    window_samples = int(window_ms * 0.001 * sample_rate)
    
    # 确定搜索范围
    if search_right_only:
        start_sample = center_sample
        end_sample = min(center_sample + window_samples, len(audio) - 1)
    else:
        start_sample = max(0, center_sample - window_samples)
        end_sample = min(center_sample + window_samples, len(audio) - 1)
    
    # 在窗口内寻找零交叉
    min_abs = float('inf')
    best_sample = center_sample
    
    for i in range(start_sample, end_sample):
        if i > 0 and i < len(audio):
            # 检查符号变化
            if audio[i-1] * audio[i] <= 0:
                # 找到零交叉
                abs_val = abs(audio[i])
                if abs_val < min_abs:
                    min_abs = abs_val
                    best_sample = i
    
    return best_sample / sample_rate


def compute_rms(audio: np.ndarray, 
                frame_length: int = 2048,
                hop_length: int = 512) -> np.ndarray:
    """计算RMS能量
    
    Args:
        audio: 音频数据
        frame_length: 帧长度
        hop_length: 跳跃长度
        
    Returns:
        RMS能量数组
    """
    return librosa.feature.rms(y=audio, 
                               frame_length=frame_length,
                               hop_length=hop_length)[0]
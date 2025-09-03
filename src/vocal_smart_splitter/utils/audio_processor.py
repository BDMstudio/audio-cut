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
    
    def load_audio(self, file_path: Union[str, Path], 
                   normalize: bool = True) -> Tuple[np.ndarray, int]:
        """加载音频文件
        
        Args:
            file_path: 音频文件路径
            normalize: 是否标准化音频
            
        Returns:
            (音频数据, 采样率)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        
        try:
            # 使用librosa加载音频
            audio, sr = librosa.load(
                str(file_path),
                sr=self.sample_rate,
                mono=(self.channels == 1),
                dtype=np.float32
            )
            
            # 标准化
            if normalize and len(audio) > 0:
                audio = self._normalize_audio(audio)
            
            duration = len(audio) / sr
            logger.info(f"音频加载成功: {file_path.name}, 时长: {duration:.2f}秒, 采样率: {sr}")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"音频加载失败: {file_path}, 错误: {e}")
            raise
    
    def save_audio(self, audio: np.ndarray, sample_rate: int, 
                   output_path: Union[str, Path], 
                   quality: int = 192,
                   fade_in: float = 0.0,
                   fade_out: float = 0.0,
                   zero_processing: bool = False) -> bool:
        """保存音频文件
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            output_path: 输出路径
            quality: 音频质量 (kbps)
            fade_in: 渐入时长 (秒)
            fade_out: 渐出时长 (秒)
            zero_processing: 是否零处理模式（无缝分割用）
            
        Returns:
            是否保存成功
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 零处理模式：完全跳过音频处理
            if not zero_processing:
                # 应用渐入渐出
                if fade_in > 0 or fade_out > 0:
                    audio = self._apply_fades(audio, sample_rate, fade_in, fade_out)
                
                # 确保音频在合理范围内
                audio = np.clip(audio, -1.0, 1.0)
            
            # 根据文件扩展名选择保存方式
            ext = output_path.suffix.lower()
            
            if ext == '.mp3':
                self._save_as_mp3(audio, sample_rate, output_path, quality)
            elif ext in ['.wav', '.flac']:
                sf.write(str(output_path), audio, sample_rate)
            else:
                # 默认保存为wav
                output_path = output_path.with_suffix('.wav')
                sf.write(str(output_path), audio, sample_rate)
            
            duration = len(audio) / sample_rate
            logger.debug(f"音频保存成功: {output_path.name}, 时长: {duration:.2f}秒")
            
            return True
            
        except Exception as e:
            logger.error(f"音频保存失败: {output_path}, 错误: {e}")
            return False
    
    def _save_as_mp3(self, audio: np.ndarray, sample_rate: int,
                     output_path: Path, quality: int):
        """高质量MP3保存"""
        # 确保音频在合理范围内，使用软限制
        audio = np.clip(audio, -0.95, 0.95)

        # 转换为16位整数，使用更精确的缩放
        audio_int16 = (audio * 32767.0).astype(np.int16)

        # 使用pydub保存
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

        # 高质量MP3编码参数
        audio_segment.export(
            str(output_path),
            format="mp3",
            bitrate=f"{quality}k",
            parameters=[
                "-q:a", "0",  # 最高质量
                "-compression_level", "0"  # 最低压缩
            ]
        )
    
    def _normalize_audio(self, audio: np.ndarray,
                        target_level: float = -6.0) -> np.ndarray:
        """温和的音频标准化

        Args:
            audio: 音频数据
            target_level: 目标电平 (dB) - 更保守的目标

        Returns:
            标准化后的音频
        """
        if len(audio) == 0:
            return audio

        # 计算峰值而不是RMS，更保守
        peak = np.max(np.abs(audio))

        if peak > 0:
            # 计算目标增益，基于峰值
            target_peak = 10 ** (target_level / 20)
            gain = target_peak / peak

            # 更严格的增益限制，避免过度放大
            gain = np.clip(gain, 0.3, 3.0)

            audio = audio * gain

        # 软限制，避免硬切割
        audio = np.tanh(audio * 0.9) * 0.95

        return audio
    
    def _apply_fades(self, audio: np.ndarray, sample_rate: int,
                     fade_in: float, fade_out: float) -> np.ndarray:
        """应用渐入渐出效果
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            fade_in: 渐入时长 (秒)
            fade_out: 渐出时长 (秒)
            
        Returns:
            处理后的音频
        """
        audio = audio.copy()
        
        # 渐入
        if fade_in > 0:
            fade_in_samples = int(fade_in * sample_rate)
            fade_in_samples = min(fade_in_samples, len(audio) // 4)
            
            if fade_in_samples > 0:
                fade_curve = np.linspace(0, 1, fade_in_samples)
                audio[:fade_in_samples] *= fade_curve
        
        # 渐出
        if fade_out > 0:
            fade_out_samples = int(fade_out * sample_rate)
            fade_out_samples = min(fade_out_samples, len(audio) // 4)
            
            if fade_out_samples > 0:
                fade_curve = np.linspace(1, 0, fade_out_samples)
                audio[-fade_out_samples:] *= fade_curve
        
        return audio
    
    def trim_silence(self, audio: np.ndarray, 
                     threshold: float = 0.01,
                     frame_length: int = 2048,
                     hop_length: int = 512) -> np.ndarray:
        """去除音频首尾的静音
        
        Args:
            audio: 音频数据
            threshold: 静音阈值
            frame_length: 帧长度
            hop_length: 跳跃长度
            
        Returns:
            去除静音后的音频
        """
        try:
            # 使用librosa的trim功能
            trimmed_audio, _ = librosa.effects.trim(
                audio,
                top_db=20,  # 相对于峰值的dB阈值
                frame_length=frame_length,
                hop_length=hop_length
            )
            
            return trimmed_audio
            
        except Exception as e:
            logger.warning(f"静音去除失败: {e}")
            return audio
    
    def resample_audio(self, audio: np.ndarray, 
                      original_sr: int, 
                      target_sr: int) -> np.ndarray:
        """重采样音频
        
        Args:
            audio: 音频数据
            original_sr: 原始采样率
            target_sr: 目标采样率
            
        Returns:
            重采样后的音频
        """
        if original_sr == target_sr:
            return audio
        
        try:
            resampled_audio = librosa.resample(
                audio, 
                orig_sr=original_sr, 
                target_sr=target_sr,
                res_type='kaiser_best'
            )
            
            logger.debug(f"音频重采样: {original_sr} -> {target_sr} Hz")
            return resampled_audio
            
        except Exception as e:
            logger.error(f"音频重采样失败: {e}")
            return audio
    
    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """转换为单声道
        
        Args:
            audio: 音频数据 (可能是多声道)
            
        Returns:
            单声道音频
        """
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            # 如果是立体声，取平均值
            return np.mean(audio, axis=0)
        else:
            logger.warning(f"不支持的音频维度: {audio.ndim}")
            return audio
    
    def preprocess_audio(self, audio: np.ndarray, 
                        sample_rate: int,
                        normalize: bool = True,
                        trim_silence: bool = True,
                        target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """预处理音频
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            normalize: 是否标准化
            trim_silence: 是否去除静音
            target_sr: 目标采样率
            
        Returns:
            (处理后的音频, 采样率)
        """
        processed_audio = audio.copy()
        current_sr = sample_rate
        
        # 转换为单声道
        processed_audio = self.convert_to_mono(processed_audio)
        
        # 重采样
        if target_sr and target_sr != current_sr:
            processed_audio = self.resample_audio(processed_audio, current_sr, target_sr)
            current_sr = target_sr
        
        # 去除静音
        if trim_silence:
            processed_audio = self.trim_silence(processed_audio)
        
        # 标准化
        if normalize:
            processed_audio = self._normalize_audio(processed_audio)
        
        logger.debug(f"音频预处理完成: 长度={len(processed_audio)}, 采样率={current_sr}")
        
        return processed_audio, current_sr
    
    def get_audio_info(self, file_path: Union[str, Path]) -> dict:
        """获取音频文件信息
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            音频信息字典
        """
        try:
            file_path = Path(file_path)
            
            # 使用soundfile获取基本信息
            info = sf.info(str(file_path))
            
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype,
                'frames': info.frames,
                'file_size': file_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"获取音频信息失败: {file_path}, 错误: {e}")
            return {}

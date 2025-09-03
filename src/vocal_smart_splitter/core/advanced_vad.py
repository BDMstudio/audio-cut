#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/advanced_vad.py
# AI-SUMMARY: 基于先进VAD算法的精确人声检测器

"""
先进的语音活动检测器

使用多种先进的VAD算法来精确检测人声活动：
1. Silero VAD - 企业级预训练模型
2. pyannote.audio - 神经网络VAD
3. WebRTC VAD - 实时检测（备用）

重点：只在真正的人声停顿处分割，不考虑片段长度
"""

import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional
import librosa
from pathlib import Path

try:
    # 尝试导入 Silero VAD
    import torch
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False

try:
    # 尝试导入 pyannote.audio
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

from ..utils.config_manager import get_config


class AdvancedVAD:
    """先进的语音活动检测器"""
    
    def __init__(self, sample_rate: int = 16000):
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.vad_model = None
        self.vad_method = None
        
        # 初始化最佳可用的VAD方法
        self._initialize_vad()
        
        self.logger.info(f"先进VAD初始化完成，使用方法: {self.vad_method}")
    
    def _initialize_vad(self):
        """初始化最佳可用的VAD方法"""
        
        # 优先尝试 Silero VAD
        if SILERO_AVAILABLE:
            try:
                self.logger.info("尝试初始化 Silero VAD...")
                self.vad_model = load_silero_vad()
                self.vad_method = "silero"
                self.logger.info("✅ Silero VAD 初始化成功")
                return
            except Exception as e:
                self.logger.warning(f"Silero VAD 初始化失败: {e}")
        
        # 尝试 pyannote.audio
        if PYANNOTE_AVAILABLE:
            try:
                self.logger.info("尝试初始化 pyannote VAD...")
                # 使用预训练的VAD模型
                self.vad_model = VoiceActivityDetection(segmentation="pyannote/segmentation")
                self.vad_method = "pyannote"
                self.logger.info("✅ pyannote VAD 初始化成功")
                return
            except Exception as e:
                self.logger.warning(f"pyannote VAD 初始化失败: {e}")
        
        # 回退到 WebRTC VAD
        try:
            import webrtcvad
            aggressiveness = get_config('advanced_vad.webrtc_aggressiveness', 2)
            try:
                aggressiveness = int(aggressiveness)
            except Exception:
                aggressiveness = 2
            aggressiveness = min(max(aggressiveness, 0), 3)
            self.vad_model = webrtcvad.Vad(aggressiveness)
            self.vad_method = "webrtc"
            self.logger.info(f"✅ 使用 WebRTC VAD 作为备用方案 (aggr={aggressiveness})")
        except ImportError:
            self.logger.error("❌ 无法初始化任何VAD方法")
            self.vad_method = "none"
    
    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int) -> List[Dict]:
        """
        检测语音活动区间
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            
        Returns:
            语音活动区间列表，每个区间包含 start, end, confidence
        """
        if self.vad_method == "silero":
            return self._detect_with_silero(audio, sample_rate)
        elif self.vad_method == "pyannote":
            return self._detect_with_pyannote(audio, sample_rate)
        elif self.vad_method == "webrtc":
            return self._detect_with_webrtc(audio, sample_rate)
        else:
            self.logger.error("没有可用的VAD方法")
            return []
    
    def _detect_with_silero(self, audio: np.ndarray, sample_rate: int) -> List[Dict]:
        """使用 Silero VAD 检测语音活动"""
        try:
            # Silero VAD 需要 16kHz 采样率
            if sample_rate != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            else:
                audio_16k = audio
            
            # 转换为 torch tensor
            audio_tensor = torch.from_numpy(audio_16k).float()
            
            # 获取语音时间戳
            silero_min_speech = int(get_config('advanced_vad.silero_min_speech_ms', 200))
            silero_min_silence = int(get_config('advanced_vad.silero_min_silence_ms', 300))
            window_size = int(get_config('advanced_vad.silero_window_size_samples', 512))
            pad_ms = int(get_config('advanced_vad.silero_speech_pad_ms', 30))
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=16000,
                min_speech_duration_ms=silero_min_speech,
                min_silence_duration_ms=silero_min_silence,
                window_size_samples=window_size,
                speech_pad_ms=pad_ms
            )
            
            # 转换为我们的格式
            voice_segments = []
            for segment in speech_timestamps:
                start_time = segment['start'] / 16000  # 转换为秒
                end_time = segment['end'] / 16000
                
                # 如果原始采样率不是16kHz，需要调整时间
                if sample_rate != 16000:
                    start_time = start_time * sample_rate / 16000
                    end_time = end_time * sample_rate / 16000
                
                voice_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'confidence': 0.9,  # Silero VAD 通常很准确
                    'type': 'speech'
                })
            
            self.logger.info(f"Silero VAD 检测到 {len(voice_segments)} 个语音片段")
            return voice_segments
            
        except Exception as e:
            self.logger.error(f"Silero VAD 检测失败: {e}")
            return []
    
    def _detect_with_pyannote(self, audio: np.ndarray, sample_rate: int) -> List[Dict]:
        """使用 pyannote.audio 检测语音活动"""
        try:
            # pyannote 需要特定的音频格式
            # 这里需要更复杂的实现，暂时返回空列表
            self.logger.warning("pyannote VAD 实现待完善")
            return []
            
        except Exception as e:
            self.logger.error(f"pyannote VAD 检测失败: {e}")
            return []
    
    def _detect_with_webrtc(self, audio: np.ndarray, sample_rate: int) -> List[Dict]:
        """使用 WebRTC VAD 检测语音活动（改进版）"""
        try:
            # WebRTC VAD 需要特定采样率
            target_rate = 16000
            if sample_rate != target_rate:
                audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)
            else:
                audio_resampled = audio
            
            # 转换为16位整数
            audio_int16 = (audio_resampled * 32767).astype(np.int16)
            
            # 分帧检测（30ms帧）
            frame_duration = 0.03  # 30ms
            frame_size = int(target_rate * frame_duration)
            
            voice_frames = []
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size]
                is_speech = self.vad_model.is_speech(frame.tobytes(), target_rate)
                
                start_time = i / target_rate
                end_time = (i + frame_size) / target_rate
                
                # 调整回原始采样率的时间
                if sample_rate != target_rate:
                    start_time = start_time * sample_rate / target_rate
                    end_time = end_time * sample_rate / target_rate
                
                voice_frames.append({
                    'start': start_time,
                    'end': end_time,
                    'is_speech': is_speech,
                    'confidence': 0.7 if is_speech else 0.3
                })
            
            # 合并连续的语音帧
            voice_segments = self._merge_voice_frames(voice_frames)
            
            self.logger.info(f"WebRTC VAD 检测到 {len(voice_segments)} 个语音片段")
            return voice_segments
            
        except Exception as e:
            self.logger.error(f"WebRTC VAD 检测失败: {e}")
            return []
    
    def _merge_voice_frames(self, voice_frames: List[Dict]) -> List[Dict]:
        """合并连续的语音帧"""
        if not voice_frames:
            return []
        
        merged_segments = []
        current_segment = None
        
        for frame in voice_frames:
            if frame['is_speech']:
                if current_segment is None:
                    # 开始新的语音片段
                    current_segment = {
                        'start': frame['start'],
                        'end': frame['end'],
                        'confidence': frame['confidence'],
                        'type': 'speech'
                    }
                else:
                    # 扩展当前语音片段
                    current_segment['end'] = frame['end']
            else:
                if current_segment is not None:
                    # 结束当前语音片段
                    merged_segments.append(current_segment)
                    current_segment = None
        
        # 添加最后一个片段
        if current_segment is not None:
            merged_segments.append(current_segment)
        
        return merged_segments
    
    def find_silence_gaps(self, voice_segments: List[Dict], total_duration: float) -> List[Dict]:
        """
        找到语音片段之间的静音间隔
        
        Args:
            voice_segments: 语音片段列表
            total_duration: 音频总时长
            
        Returns:
            静音间隔列表
        """
        if not voice_segments:
            return [{'start': 0, 'end': total_duration, 'duration': total_duration, 'type': 'silence'}]
        
        silence_gaps = []
        
        # 开头的静音
        if voice_segments[0]['start'] > 0.1:  # 至少100ms的静音
            silence_gaps.append({
                'start': 0,
                'end': voice_segments[0]['start'],
                'duration': voice_segments[0]['start'],
                'type': 'silence'
            })
        
        # 中间的静音间隔
        for i in range(len(voice_segments) - 1):
            gap_start = voice_segments[i]['end']
            gap_end = voice_segments[i + 1]['start']
            gap_duration = gap_end - gap_start
            
            if gap_duration > 0.1:  # 至少100ms的静音
                silence_gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration': gap_duration,
                    'type': 'silence'
                })
        
        # 结尾的静音
        if voice_segments[-1]['end'] < total_duration - 0.1:
            silence_gaps.append({
                'start': voice_segments[-1]['end'],
                'end': total_duration,
                'duration': total_duration - voice_segments[-1]['end'],
                'type': 'silence'
            })
        
        # 按静音时长排序（最长的在前）
        silence_gaps.sort(key=lambda x: x['duration'], reverse=True)
        
        self.logger.info(f"检测到 {len(silence_gaps)} 个静音间隔")
        if silence_gaps:
            self.logger.info(f"最长静音间隔: {silence_gaps[0]['duration']:.2f}秒")
        
        return silence_gaps

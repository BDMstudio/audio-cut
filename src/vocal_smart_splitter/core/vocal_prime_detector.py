#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/vocal_prime_detector.py
# AI-SUMMARY: 基于vocal_prime.md方案的纯人声停顿检测器 - RMS能量包络+动态噪声地板+滞回检测

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import librosa
from scipy import signal

logger = logging.getLogger(__name__)

@dataclass
class SilencePlatform:
    """静音平台数据结构"""
    start_s: float
    end_s: float
    center_s: float
    flatness_db: float
    duration: float
    confidence: float = 0.0

@dataclass
class SpeechSegment:
    """人声段数据结构"""
    start_s: float
    end_s: float
    duration: float
    energy_mean: float = 0.0
    energy_std: float = 0.0

@dataclass
class VocalPrimePause:
    """纯人声停顿结构（与PureVocalPause兼容）"""
    start_time: float
    end_time: float
    duration: float
    pause_type: str  # 'true_pause', 'breath', 'uncertain'
    confidence: float
    features: Dict
    cut_point: float = 0.0  # 最佳切割点
    quality_grade: str = 'B'
    is_valid: bool = True

class VocalPrimeDetector:
    """基于vocal_prime.md方案的纯人声停顿检测器
    
    技术栈：
    1. RMS能量包络 (30ms帧/10ms跳) + EMA平滑 (120ms)
    2. 动态噪声地板 (滚动5%分位数)
    3. 滞回阈值 (down=floor+3dB, up=floor+6dB)
    4. 连续时长约束 (静音≥1.0s)
    5. 未来静默守卫 (切点后需静音≥1.0s)
    6. 平台平坦度检测 (≤6dB波动)
    7. 静音中心右偏 + 零交叉对齐
    """
    
    def __init__(self, sample_rate: int = 44100):
        """初始化检测器"""
        self.sr = sample_rate
        
        # 检测参数（优化后适合音乐场景）
        self.frame_ms = 30  # RMS帧长
        self.hop_ms = 10    # 跳跃长度
        self.smooth_ms = 120  # EMA平滑窗口
        self.floor_percentile = 0.05  # 噪声地板分位数
        self.db_above_floor_down = 3.0  # 下降阈值
        self.db_above_floor_up = 6.0    # 上升阈值
        self.min_silence_sec = 0.6      # 最小静音时长 (降低至0.6s适合音乐)
        self.plateau_flatness_db = 10.0  # 平台平坦度 (放宽至10dB适合真实音乐)
        self.lookahead_guard_ms = 600   # 未来静默守卫 (降低至0.6s)
        self.right_bias_ms = 80         # 右偏移量
        
        logger.info(f"[VocalPrime] 初始化检测器 - 采样率: {sample_rate}Hz")
    
    def moving_rms_db(self, x: np.ndarray, frame_ms: int = 30, hop_ms: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """计算移动RMS能量（dB）"""
        frame = int(self.sr * frame_ms / 1000)
        hop = int(self.sr * hop_ms / 1000)
        
        if frame <= 2:
            frame = 3
        if hop < 1:
            hop = 1
        
        n = (len(x) - frame) // hop + 1
        rms = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            seg = x[i*hop : i*hop + frame]
            if len(seg) == 0:
                break
            rms[i] = np.sqrt(np.mean(seg**2) + 1e-12)
        
        db = 20.0 * np.log10(rms + 1e-12)
        t_axis = np.arange(n) * (hop / self.sr)
        
        return db, t_axis
    
    def ema_smooth(self, x: np.ndarray, hop_ms: int = 10, smooth_ms: int = 120) -> np.ndarray:
        """指数移动平均平滑"""
        alpha = np.exp(-(hop_ms / smooth_ms))
        y = np.zeros_like(x, dtype=np.float32)
        acc = x[0] if len(x) > 0 else 0.0
        
        for i, v in enumerate(x):
            acc = alpha * acc + (1 - alpha) * v
            y[i] = acc
        
        return y
    
    def rolling_percentile_db(self, x: np.ndarray, hop_ms: int = 10, 
                              win_s: float = 30.0, p: float = 0.05) -> np.ndarray:
        """计算滚动分位数（动态噪声地板）"""
        win = int(win_s * 1000 / hop_ms)
        if win < 5:
            win = 5
        
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
    
    def detect_silence_and_speech(self, vocal_mono: np.ndarray) -> Tuple[List[SilencePlatform], List[SpeechSegment]]:
        """在纯人声轨上检测静音平台和人声段
        
        技术实现：
        1. RMS能量包络计算
        2. EMA平滑抑制抖动
        3. 动态噪声地板自适应
        4. 滞回状态机检测
        5. 平台平坦度验证
        6. 未来静默守卫
        """
        # 1) 能量包络 + 平滑
        rms_db, t_axis = self.moving_rms_db(vocal_mono, self.frame_ms, self.hop_ms)
        rms_db_smooth = self.ema_smooth(rms_db, self.hop_ms, self.smooth_ms)
        
        # 2) 动态噪声地板
        win_s = max(30.0, 10 * self.min_silence_sec)
        floor_db = self.rolling_percentile_db(rms_db_smooth, self.hop_ms, win_s, self.floor_percentile)
        th_down = floor_db + self.db_above_floor_down
        th_up = floor_db + self.db_above_floor_up
        
        # 3) 滞回 + 连续帧约束（状态机）
        state = 'speech'
        silence_start_idx = None
        min_sil_frames = int(self.min_silence_sec * 1000 / self.hop_ms)
        lookahead_frames = int(self.lookahead_guard_ms / self.hop_ms)
        
        silence_platforms: List[SilencePlatform] = []
        speech_segments: List[SpeechSegment] = []
        speech_start_idx = 0
        
        for i, val in enumerate(rms_db_smooth):
            if state == 'speech':
                if val < th_down[i]:
                    # 进入静音候选
                    state = 'maybe_silence'
                    silence_start_idx = i
                    
            elif state == 'maybe_silence':
                if val < th_down[i]:
                    # 仍在静音
                    if i - silence_start_idx + 1 >= min_sil_frames:
                        # 满足最小时长 → 检查平台质量
                        seg = rms_db_smooth[silence_start_idx:i+1]
                        flatness = np.max(seg) - np.min(seg)
                        
                        # 未来静默守卫
                        j_end = min(i + lookahead_frames, len(rms_db_smooth) - 1)
                        future_seg = rms_db_smooth[i:j_end+1]
                        future_th = th_down[i:j_end+1] if j_end > i else [th_down[i]]
                        future_floor = np.all(future_seg <= future_th)
                        
                        if flatness <= self.plateau_flatness_db and future_floor:
                            start_s = float(t_axis[silence_start_idx])
                            end_s = float(t_axis[i])
                            center = 0.5 * (start_s + end_s)
                            duration = end_s - start_s
                            
                            # 计算置信度
                            avg_below = np.mean(th_down[silence_start_idx:i+1] - seg)
                            confidence = min(1.0, avg_below / 10.0)  # 每10dB低于阈值得满分
                            
                            silence_platforms.append(SilencePlatform(
                                start_s, end_s, center, flatness, duration, confidence
                            ))
                            
                            # 记录前一段语音
                            if speech_start_idx < silence_start_idx:
                                speech_seg = rms_db_smooth[speech_start_idx:silence_start_idx]
                                speech_segments.append(SpeechSegment(
                                    start_s=float(t_axis[speech_start_idx]),
                                    end_s=float(t_axis[silence_start_idx-1]) if silence_start_idx > 0 else 0.0,
                                    duration=float(t_axis[silence_start_idx-1] - t_axis[speech_start_idx]) if silence_start_idx > 0 else 0.0,
                                    energy_mean=float(np.mean(speech_seg)),
                                    energy_std=float(np.std(speech_seg))
                                ))
                            
                            state = 'silence'
                else:
                    # 回到语音
                    state = 'speech'
                    
            elif state == 'silence':
                if val > th_up[i]:
                    # 语音重新上穿
                    speech_start_idx = i
                    state = 'speech'
        
        # 收尾：最后一段语音
        if speech_start_idx < len(t_axis) - 1 and state == 'speech':
            speech_seg = rms_db_smooth[speech_start_idx:]
            speech_segments.append(SpeechSegment(
                start_s=float(t_axis[speech_start_idx]),
                end_s=float(t_axis[-1]),
                duration=float(t_axis[-1] - t_axis[speech_start_idx]),
                energy_mean=float(np.mean(speech_seg)),
                energy_std=float(np.std(speech_seg))
            ))
        
        logger.info(f"[VocalPrime] 检测到 {len(silence_platforms)} 个静音平台, {len(speech_segments)} 个人声段")
        
        return silence_platforms, speech_segments
    
    def choose_split_points(self, silence_platforms: List[SilencePlatform], 
                           speech_segments: List[SpeechSegment],
                           vocal_mono: np.ndarray,
                           min_gap_s: float = 2.0) -> List[float]:
        """根据静音平台选择切点
        
        技术: 静音中心 + 右偏 + 零交叉对齐
        """
        cuts: List[float] = []
        
        # 处理中间静音
        for p in silence_platforms:
            # 只接受足够长的静音
            if p.duration < self.min_silence_sec:
                continue
            
            # 静音中心 + 右偏
            t = p.center_s + self.right_bias_ms / 1000.0
            t = min(t, p.end_s)  # 不超出平台
            
            # 零交叉对齐
            t = self.find_zero_crossing(vocal_mono, t)
            
            # 间隔约束
            if len(cuts) == 0 or (t - cuts[-1]) >= min_gap_s:
                cuts.append(t)
        
        # 处理头尾
        if len(speech_segments) > 0:
            # 头部静音
            first = speech_segments[0]
            if first.start_s >= self.min_silence_sec:
                t = max(0.0, first.start_s - 0.5)
                t = self.find_zero_crossing(vocal_mono, t)
                if len(cuts) == 0 or (cuts[0] - t) >= min_gap_s:
                    cuts = [t] + cuts
            
            # 尾部静音
            last = speech_segments[-1]
            audio_duration = len(vocal_mono) / self.sr
            tail_silence = audio_duration - last.end_s
            if tail_silence >= self.min_silence_sec:
                t = last.end_s + 0.5
                t = self.find_zero_crossing(vocal_mono, t)
                if len(cuts) == 0 or (t - cuts[-1]) >= min_gap_s:
                    cuts.append(t)
        
        return sorted(cuts)
    
    def find_zero_crossing(self, x: np.ndarray, t_center_s: float, window_ms: int = 10) -> float:
        """查找最近的零交叉点"""
        w = int(self.sr * window_ms / 1000)
        c = int(t_center_s * self.sr)
        a = max(1, c - w)
        b = min(len(x)-1, c + w)
        
        if b <= a:
            return t_center_s
        
        seg = x[a:b]
        
        # 找零交叉
        zero_crossings = np.where(np.diff(np.sign(seg)))[0]
        
        if len(zero_crossings) == 0:
            return t_center_s
        
        # 选最近的
        center_offset = (c - a)
        distances = np.abs(zero_crossings - center_offset)
        nearest = zero_crossings[np.argmin(distances)]
        
        return float((a + nearest) / self.sr)
    
    def detect_pure_vocal_pauses(self, vocal_track: np.ndarray, 
                                 original_audio: np.ndarray = None) -> List[VocalPrimePause]:
        """检测纯人声停顿（主接口，与v2.0兼容）
        
        Args:
            vocal_track: 纯人声音频
            original_audio: 原始混音（可选）
            
        Returns:
            检测到的停顿列表
        """
        logger.info("[VocalPrime] 开始纯人声停顿检测...")
        
        # 1. 检测静音平台和人声段
        silence_platforms, speech_segments = self.detect_silence_and_speech(vocal_track)
        
        # 2. 选择切点
        split_points = self.choose_split_points(silence_platforms, speech_segments, vocal_track)
        
        # 3. 转换为VocalPrimePause格式（与v2.0兼容）
        pauses = []
        
        for i, platform in enumerate(silence_platforms):
            # 找对应的切点
            cut_point = platform.center_s + self.right_bias_ms / 1000.0
            for sp in split_points:
                if abs(sp - cut_point) < 0.1:  # 10ms容差
                    cut_point = sp
                    break
            
            # 创建停顿对象
            pause = VocalPrimePause(
                start_time=platform.start_s,
                end_time=platform.end_s,
                duration=platform.duration,
                pause_type='true_pause' if platform.confidence > 0.7 else 'uncertain',
                confidence=platform.confidence,
                features={
                    'flatness_db': platform.flatness_db,
                    'center_s': platform.center_s,
                    'energy_drop': platform.confidence * 10,  # 转换为dB
                    'detection_method': 'vocal_prime_rms'
                },
                cut_point=cut_point,
                quality_grade='A' if platform.confidence > 0.8 else 'B',
                is_valid=True
            )
            pauses.append(pause)
        
        logger.info(f"[VocalPrime] 检测完成，找到 {len(pauses)} 个高质量停顿")
        
        return pauses
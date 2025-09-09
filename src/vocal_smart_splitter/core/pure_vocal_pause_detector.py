#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/pure_vocal_pause_detector.py
# AI-SUMMARY: 纯人声停顿检测器 - 基于MDX23/Demucs分离后的纯人声进行多维特征分析，解决高频换气误判问题

import numpy as np
import librosa
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from ..utils.config_manager import get_config

logger = logging.getLogger(__name__)

@dataclass
class VocalFeatures:
    """人声特征数据结构"""
    f0_contour: np.ndarray           # 基频轨迹
    f0_confidence: np.ndarray        # 基频置信度
    formant_energies: List[np.ndarray]  # 共振峰能量序列
    spectral_centroid: np.ndarray    # 频谱质心
    harmonic_ratio: np.ndarray       # 谐波比率
    zero_crossing_rate: np.ndarray   # 过零率
    rms_energy: np.ndarray           # RMS能量

@dataclass
class PureVocalPause:
    """纯人声停顿结构"""
    start_time: float
    end_time: float
    duration: float
    pause_type: str  # 'true_pause', 'breath', 'uncertain'
    confidence: float
    features: Dict  # 详细特征信息
    
class PureVocalPauseDetector:
    """基于纯人声的多维特征停顿检测器
    
    核心创新：
    1. F0连续性分析 - 检测基频突变识别真停顿
    2. 共振峰能量分布 - 区分换气vs静音
    3. 频谱质心追踪 - 识别高频衰减模式
    4. 谐波强度分析 - 评估发声质量
    """
    
    def __init__(self, sample_rate: int = 44100):
        """初始化纯人声停顿检测器
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        
        # 从配置加载参数
        self.min_pause_duration = get_config('pure_vocal_detection.min_pause_duration', 0.5)
        self.breath_duration_range = get_config('pure_vocal_detection.breath_duration_range', [0.1, 0.3])
        self.f0_weight = get_config('pure_vocal_detection.f0_weight', 0.3)
        self.formant_weight = get_config('pure_vocal_detection.formant_weight', 0.25)
        self.spectral_weight = get_config('pure_vocal_detection.spectral_weight', 0.25)
        self.duration_weight = get_config('pure_vocal_detection.duration_weight', 0.2)
        
        # 检测阈值
        self.energy_threshold_db = get_config('pure_vocal_detection.energy_threshold_db', -40)
        self.f0_drop_threshold = get_config('pure_vocal_detection.f0_drop_threshold', 0.7)
        self.breath_confidence_threshold = get_config('pure_vocal_detection.breath_filter_threshold', 0.3)
        self.pause_confidence_threshold = get_config('pure_vocal_detection.pause_confidence_threshold', 0.7)
        
        # 分析参数
        self.hop_length = int(sample_rate * 0.01)  # 10ms hop
        self.frame_length = int(sample_rate * 0.025)  # 25ms frame
        self.n_fft = 2048
        
        logger.info(f"纯人声停顿检测器初始化完成 (采样率: {sample_rate})")
    
    def detect_pure_vocal_pauses(self, vocal_audio: np.ndarray, 
                                original_audio: Optional[np.ndarray] = None) -> List[PureVocalPause]:
        """检测纯人声中的停顿
        
        Args:
            vocal_audio: 分离后的纯人声音频
            original_audio: 原始混音(可选，用于对比)
            
        Returns:
            检测到的停顿列表
        """
        logger.info("开始纯人声停顿检测...")
        
        # 1. 提取多维特征
        features = self._extract_vocal_features(vocal_audio)
        
        # 2. 检测候选停顿区域
        candidate_pauses = self._detect_candidate_pauses(features)
        
        # 3. 特征融合分析
        analyzed_pauses = self._analyze_pause_features(candidate_pauses, features, vocal_audio)
        
        # 4. 分类过滤
        filtered_pauses = self._classify_and_filter(analyzed_pauses)
        
        logger.info(f"检测完成: {len(filtered_pauses)}个高质量停顿点")
        return filtered_pauses
    
    def _extract_vocal_features(self, audio: np.ndarray) -> VocalFeatures:
        """提取人声多维特征
        
        Args:
            audio: 音频信号
            
        Returns:
            提取的特征集合
        """
        logger.debug("提取人声特征...")
        
        # 1. 基频(F0)提取 - 使用librosa的pyin算法
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'),  # 65Hz
            fmax=librosa.note_to_hz('C7'),  # 2093Hz
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # 2. 共振峰分析 - 使用LPC分析
        formant_energies = self._extract_formants(audio)
        
        # 3. 频谱质心
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # 4. 谐波分析
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_ratio = self._calculate_harmonic_ratio(harmonic, audio)
        
        # 5. 过零率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )[0]
        
        # 6. RMS能量
        rms_energy = librosa.feature.rms(
            y=audio, hop_length=self.hop_length
        )[0]
        
        return VocalFeatures(
            f0_contour=f0,
            f0_confidence=voiced_probs,
            formant_energies=formant_energies,
            spectral_centroid=spectral_centroid,
            harmonic_ratio=harmonic_ratio,
            zero_crossing_rate=zero_crossing_rate,
            rms_energy=rms_energy
        )
    
    def _extract_formants(self, audio: np.ndarray, n_formants: int = 3) -> List[np.ndarray]:
        """提取共振峰能量
        
        Args:
            audio: 音频信号
            n_formants: 要提取的共振峰数量
            
        Returns:
            共振峰能量序列
        """
        formants = []
        
        # 分帧处理
        frames = librosa.util.frame(audio, frame_length=self.frame_length, 
                                   hop_length=self.hop_length)
        
        for frame in frames.T:
            # LPC分析
            try:
                # 使用自相关方法估计LPC系数
                lpc_order = 2 + self.sample_rate // 1000  # 经验公式
                a = librosa.lpc(frame, order=min(lpc_order, len(frame) - 1))
                
                # 从LPC系数提取共振峰频率
                roots = np.roots(a)
                roots = roots[np.imag(roots) >= 0]  # 只保留正频率
                
                # 转换为频率
                angles = np.angle(roots)
                freqs = angles * self.sample_rate / (2 * np.pi)
                
                # 排序并选择前n个共振峰
                freqs = sorted(freqs[freqs > 0])[:n_formants]
                
                # 如果共振峰数量不足，填充零
                while len(freqs) < n_formants:
                    freqs.append(0)
                    
                formants.append(freqs)
            except:
                # LPC失败时填充零
                formants.append([0] * n_formants)
        
        # 转置得到每个共振峰的时间序列
        formants = np.array(formants).T
        return [formants[i] for i in range(n_formants)]
    
    def _calculate_harmonic_ratio(self, harmonic: np.ndarray, 
                                 original: np.ndarray) -> np.ndarray:
        """计算谐波比率
        
        Args:
            harmonic: 谐波成分
            original: 原始信号
            
        Returns:
            谐波比率时间序列
        """
        # 计算能量比
        harmonic_rms = librosa.feature.rms(y=harmonic, hop_length=self.hop_length)[0]
        original_rms = librosa.feature.rms(y=original, hop_length=self.hop_length)[0]
        
        # 避免除零
        ratio = np.zeros_like(harmonic_rms)
        non_zero = original_rms > 1e-10
        ratio[non_zero] = harmonic_rms[non_zero] / original_rms[non_zero]
        
        return ratio
    
    def _detect_candidate_pauses(self, features: VocalFeatures) -> List[Tuple[int, int]]:
        """检测候选停顿区域
        
        Args:
            features: 提取的特征
            
        Returns:
            候选停顿的帧索引区间列表
        """
        # 能量阈值检测
        energy_db = librosa.amplitude_to_db(features.rms_energy, ref=np.max)
        low_energy = energy_db < self.energy_threshold_db
        
        # F0不连续检测
        f0_missing = features.f0_confidence < self.f0_drop_threshold
        
        # 组合条件
        pause_frames = low_energy | f0_missing
        
        # 平滑处理
        pause_frames = gaussian_filter1d(pause_frames.astype(float), sigma=3) > 0.5
        
        # 查找连续区间
        candidates = []
        in_pause = False
        start_idx = 0
        
        for i, is_pause in enumerate(pause_frames):
            if is_pause and not in_pause:
                start_idx = i
                in_pause = True
            elif not is_pause and in_pause:
                # 转换为时间并检查最小时长
                duration = (i - start_idx) * self.hop_length / self.sample_rate
                if duration >= self.breath_duration_range[0]:  # 至少达到换气最小时长
                    candidates.append((start_idx, i))
                in_pause = False
        
        # 处理末尾
        if in_pause:
            duration = (len(pause_frames) - start_idx) * self.hop_length / self.sample_rate
            if duration >= self.breath_duration_range[0]:
                candidates.append((start_idx, len(pause_frames)))
        
        logger.debug(f"检测到{len(candidates)}个候选停顿区域")
        return candidates
    
    def _analyze_pause_features(self, candidates: List[Tuple[int, int]], 
                               features: VocalFeatures,
                               audio: np.ndarray) -> List[PureVocalPause]:
        """分析候选停顿的特征
        
        Args:
            candidates: 候选停顿区间
            features: 特征数据
            audio: 音频信号
            
        Returns:
            分析后的停顿列表
        """
        analyzed_pauses = []
        
        for start_idx, end_idx in candidates:
            # 时间信息
            start_time = start_idx * self.hop_length / self.sample_rate
            end_time = end_idx * self.hop_length / self.sample_rate
            duration = end_time - start_time
            
            # 提取区间特征
            pause_features = self._extract_pause_interval_features(
                features, start_idx, end_idx, audio
            )
            
            # 计算置信度
            confidence = self._calculate_pause_confidence(pause_features, duration)
            
            # 初步分类
            if duration <= self.breath_duration_range[1]:
                pause_type = 'breath'
            elif duration >= self.min_pause_duration:
                pause_type = 'true_pause'
            else:
                pause_type = 'uncertain'
            
            analyzed_pauses.append(PureVocalPause(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                pause_type=pause_type,
                confidence=confidence,
                features=pause_features
            ))
        
        return analyzed_pauses
    
    def _extract_pause_interval_features(self, features: VocalFeatures,
                                        start_idx: int, end_idx: int,
                                        audio: np.ndarray) -> Dict:
        """提取停顿区间的详细特征
        
        Args:
            features: 全局特征
            start_idx: 开始帧索引
            end_idx: 结束帧索引
            audio: 音频信号
            
        Returns:
            区间特征字典
        """
        # 前后文窗口(前后各0.5秒)
        context_frames = int(0.5 * self.sample_rate / self.hop_length)
        pre_start = max(0, start_idx - context_frames)
        post_end = min(len(features.rms_energy), end_idx + context_frames)
        
        # F0特征
        f0_drop_rate = 0.0
        if pre_start < start_idx:
            pre_f0 = np.nanmean(features.f0_contour[pre_start:start_idx])
            pause_f0 = np.nanmean(features.f0_contour[start_idx:end_idx])
            if not np.isnan(pre_f0) and not np.isnan(pause_f0) and pre_f0 > 0:
                f0_drop_rate = 1.0 - (pause_f0 / pre_f0)
        
        # 能量特征
        pre_energy = np.mean(features.rms_energy[pre_start:start_idx]) if pre_start < start_idx else 0
        pause_energy = np.mean(features.rms_energy[start_idx:end_idx])
        post_energy = np.mean(features.rms_energy[end_idx:post_end]) if end_idx < post_end else 0
        
        energy_drop = (pre_energy - pause_energy) / (pre_energy + 1e-10)
        energy_rise = (post_energy - pause_energy) / (pause_energy + 1e-10)
        
        # 频谱特征
        centroid_shift = 0.0
        if pre_start < start_idx:
            pre_centroid = np.mean(features.spectral_centroid[pre_start:start_idx])
            pause_centroid = np.mean(features.spectral_centroid[start_idx:end_idx])
            centroid_shift = abs(pre_centroid - pause_centroid) / (pre_centroid + 1e-10)
        
        # 谐波特征
        harmonic_drop = 0.0
        if pre_start < start_idx:
            pre_harmonic = np.mean(features.harmonic_ratio[pre_start:start_idx])
            pause_harmonic = np.mean(features.harmonic_ratio[start_idx:end_idx])
            harmonic_drop = (pre_harmonic - pause_harmonic) / (pre_harmonic + 1e-10)
        
        # 共振峰特征
        formant_stability = []
        for formant_track in features.formant_energies:
            if len(formant_track) > end_idx:
                pause_formant = formant_track[start_idx:end_idx]
                stability = 1.0 - (np.std(pause_formant) / (np.mean(pause_formant) + 1e-10))
                formant_stability.append(stability)
        
        return {
            'f0_drop_rate': f0_drop_rate,
            'energy_drop': energy_drop,
            'energy_rise': energy_rise,
            'centroid_shift': centroid_shift,
            'harmonic_drop': harmonic_drop,
            'formant_stability': np.mean(formant_stability) if formant_stability else 0.5,
            'pre_energy': pre_energy,
            'pause_energy': pause_energy,
            'post_energy': post_energy
        }
    
    def _calculate_pause_confidence(self, features: Dict, duration: float) -> float:
        """计算停顿置信度
        
        Args:
            features: 停顿特征
            duration: 停顿时长
            
        Returns:
            置信度分数(0-1)
        """
        # F0得分
        f0_score = min(1.0, features['f0_drop_rate'] / 0.5)  # 50%下降得满分
        
        # 能量得分
        energy_score = min(1.0, features['energy_drop'] / 0.7)  # 70%下降得满分
        
        # 频谱得分
        spectral_score = min(1.0, features['centroid_shift'] / 0.3)  # 30%偏移得满分
        
        # 时长得分
        if duration < self.breath_duration_range[1]:
            # 换气时长，低分
            duration_score = 0.3
        elif duration >= self.min_pause_duration:
            # 真停顿时长，高分
            duration_score = min(1.0, duration / 1.0)  # 1秒得满分
        else:
            # 中间时长，中等分
            duration_score = 0.5
        
        # 加权计算
        confidence = (
            self.f0_weight * f0_score +
            self.formant_weight * (1.0 - features.get('formant_stability', 0.5)) +
            self.spectral_weight * spectral_score +
            self.duration_weight * duration_score
        )
        
        # 能量作为额外加成
        confidence = confidence * (0.7 + 0.3 * energy_score)
        
        return min(1.0, confidence)
    
    def _classify_and_filter(self, pauses: List[PureVocalPause]) -> List[PureVocalPause]:
        """分类并过滤停顿
        
        Args:
            pauses: 候选停顿列表
            
        Returns:
            过滤后的高质量停顿
        """
        filtered = []
        
        for pause in pauses:
            # 根据置信度重新分类
            if pause.confidence >= self.pause_confidence_threshold:
                pause.pause_type = 'true_pause'
                filtered.append(pause)
                logger.debug(f"真停顿: {pause.start_time:.2f}-{pause.end_time:.2f}s, "
                           f"置信度: {pause.confidence:.3f}")
            elif pause.confidence <= self.breath_confidence_threshold:
                pause.pause_type = 'breath'
                # 过滤掉换气
                logger.debug(f"过滤换气: {pause.start_time:.2f}-{pause.end_time:.2f}s, "
                           f"置信度: {pause.confidence:.3f}")
            else:
                # 不确定的情况，根据时长决定
                if pause.duration >= self.min_pause_duration:
                    pause.pause_type = 'true_pause'
                    filtered.append(pause)
                    logger.debug(f"时长判定为停顿: {pause.start_time:.2f}-{pause.end_time:.2f}s")
                else:
                    logger.debug(f"过滤不确定: {pause.start_time:.2f}-{pause.end_time:.2f}s")
        
        # 合并相邻停顿
        filtered = self._merge_adjacent_pauses(filtered)
        
        logger.info(f"分类过滤完成: {len(pauses)}个候选 -> {len(filtered)}个高质量停顿")
        return filtered
    
    def _merge_adjacent_pauses(self, pauses: List[PureVocalPause], 
                              merge_threshold: float = 0.3) -> List[PureVocalPause]:
        """合并相邻的停顿
        
        Args:
            pauses: 停顿列表
            merge_threshold: 合并阈值(秒)
            
        Returns:
            合并后的停顿列表
        """
        if not pauses:
            return pauses
        
        # 按开始时间排序
        pauses = sorted(pauses, key=lambda p: p.start_time)
        
        merged = []
        current = pauses[0]
        
        for next_pause in pauses[1:]:
            gap = next_pause.start_time - current.end_time
            
            if gap <= merge_threshold:
                # 合并
                current = PureVocalPause(
                    start_time=current.start_time,
                    end_time=next_pause.end_time,
                    duration=next_pause.end_time - current.start_time,
                    pause_type='true_pause',
                    confidence=max(current.confidence, next_pause.confidence),
                    features={**current.features, **next_pause.features}
                )
            else:
                merged.append(current)
                current = next_pause
        
        merged.append(current)
        
        if len(merged) < len(pauses):
            logger.debug(f"合并相邻停顿: {len(pauses)} -> {len(merged)}")
        
        return merged
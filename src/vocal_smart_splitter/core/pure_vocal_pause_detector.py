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
    cut_point: float = 0.0  # 最佳切割点（新增）
    quality_grade: str = 'B'  # 质量等级（新增）
    is_valid: bool = True   # 是否有效（新增）
    
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
        
        # 🔥 关键修复：集成VocalPauseDetectorV2的能量谷检测能力
        from .vocal_pause_detector import VocalPauseDetectorV2
        self._cut_point_calculator = VocalPauseDetectorV2(sample_rate)
        
        logger.info(f"纯人声停顿检测器初始化完成 (采样率: {sample_rate}) - 已集成能量谷切点计算")
    
    def detect_pure_vocal_pauses(self, vocal_audio: np.ndarray, 
                                enable_mdd_enhancement: bool = False,
                                original_audio: Optional[np.ndarray] = None) -> List[PureVocalPause]:
        """检测纯人声中的停顿
        
        Args:
            vocal_audio: 分离后的纯人声音频
            original_audio: 原始混音(可选，用于对比)
            
        Returns:
            检测到的停顿列表
        """
        logger.info(f"开始纯人声停顿检测... (MDD增强: {enable_mdd_enhancement})")
        
        # 🔥 关键修复：启用相对能量谷检测
        enable_relative_mode = get_config('pure_vocal_detection.enable_relative_energy_mode', False)
        if enable_relative_mode:
            logger.info("使用相对能量谷检测模式...")
            peak_ratio = get_config('pure_vocal_detection.peak_relative_threshold_ratio', 0.1)
            rms_ratio = get_config('pure_vocal_detection.rms_relative_threshold_ratio', 0.05)
            # BPM/MDD 自适应倍率（在相对能量模式下启用）
            if get_config('pure_vocal_detection.relative_threshold_adaptation.enable', True):
                ref_audio = original_audio if original_audio is not None else vocal_audio
                try:
                    tempo_est, _ = librosa.beat.beat_track(y=ref_audio, sr=self.sample_rate)
                    # 兼容 ndarray/标量，统一为 float
                    tempo = float(np.squeeze(np.asarray(tempo_est))) if tempo_est is not None else 0.0
                except Exception:
                    tempo = 0.0
                bpm_cfg = get_config('vocal_pause_splitting.bpm_adaptive_settings', {})
                slow_thr = bpm_cfg.get('slow_bpm_threshold', 80)
                fast_thr = bpm_cfg.get('fast_bpm_threshold', 120)
                bpm_mul_slow = get_config('pure_vocal_detection.relative_threshold_adaptation.bpm.slow_multiplier', 1.10)
                bpm_mul_med = get_config('pure_vocal_detection.relative_threshold_adaptation.bpm.medium_multiplier', 1.00)
                bpm_mul_fast = get_config('pure_vocal_detection.relative_threshold_adaptation.bpm.fast_multiplier', 0.85)
                if tempo and tempo > 0:
                    if tempo < slow_thr:
                        mul_bpm = bpm_mul_slow; bpm_tag = 'slow'
                    elif tempo > fast_thr:
                        mul_bpm = bpm_mul_fast; bpm_tag = 'fast'
                    else:
                        mul_bpm = bpm_mul_med; bpm_tag = 'medium'
                else:
                    mul_bpm = bpm_mul_med; bpm_tag = 'unknown'

                # 估算全曲 MDD（简化版）
                def _mdd_score_simple(x, sr):
                    try:
                        rms = librosa.feature.rms(y=x, hop_length=512)[0]
                        flat = librosa.feature.spectral_flatness(y=x)[0]
                        onset_env = librosa.onset.onset_strength(y=x, sr=sr, hop_length=512)
                        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512)
                        dur = max(0.1, len(x)/sr)
                        onset_rate = len(onsets)/dur
                        def nz(v):
                            v = np.asarray(v); q10, q90 = np.quantile(v,0.1), np.quantile(v,0.9)
                            if q90-q10 < 1e-9: return 0.0
                            return float(np.clip((np.mean(v)-q10)/(q90-q10), 0, 1))
                        rms_s, flat_s = nz(rms), nz(flat)
                        onset_s = float(np.clip(onset_rate/10.0, 0, 1))
                        return float(np.clip(0.5*rms_s + 0.3*flat_s + 0.2*onset_s, 0, 1))
                    except Exception:
                        return 0.5
                mdd_s = _mdd_score_simple(ref_audio, self.sample_rate)
                mdd_base = get_config('pure_vocal_detection.relative_threshold_adaptation.mdd.base', 1.0)
                mdd_gain = get_config('pure_vocal_detection.relative_threshold_adaptation.mdd.gain', 0.2)
                mul_mdd = mdd_base + (0.1 - mdd_gain*mdd_s)
                clamp_min = get_config('pure_vocal_detection.relative_threshold_adaptation.clamp_min', 0.75)
                clamp_max = get_config('pure_vocal_detection.relative_threshold_adaptation.clamp_max', 1.25)
                mul = float(np.clip(mul_bpm*mul_mdd, clamp_min, clamp_max))
                peak_ratio *= mul; rms_ratio *= mul
                logger.info(f"相对阈值自适应：BPM={tempo:.1f}({bpm_tag}), MDD={mdd_s:.2f}, mul={mul:.2f} → peak={peak_ratio:.3f}, rms={rms_ratio:.3f}")
            
            # 使用相对能量谷检测
            filtered_pauses = self._detect_energy_valleys(vocal_audio, peak_ratio, rms_ratio)
        else:
            # 原有的多维特征检测流程
            # 1. 提取多维特征
            features = self._extract_vocal_features(vocal_audio)
            
            # 2. 检测候选停顿区域
            candidate_pauses = self._detect_candidate_pauses(features)
            
            # 3. 特征融合分析
            analyzed_pauses = self._analyze_pause_features(candidate_pauses, features, vocal_audio)
            
            # 4. 分类过滤
            filtered_pauses = self._classify_and_filter(analyzed_pauses)
            
        # 5. MDD增强处理
        if enable_mdd_enhancement and original_audio is not None:
            logger.info("应用MDD增强处理...")
            filtered_pauses = self._apply_mdd_enhancement(filtered_pauses, original_audio)
        
        # 🔥 关键修复：使用VocalPauseDetectorV2计算精确切点
        if filtered_pauses and vocal_audio is not None:
            filtered_pauses = self._calculate_precise_cut_points(filtered_pauses, vocal_audio)
        
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
        
        # 4. 谐波分析 - 基于纯人声信号，无需再分离
        harmonic_ratio = self._calculate_harmonic_ratio_direct(audio)
        
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
        """
        [v2.7 关键修复版] 检测候选停顿区域
        关键修复: 将能量和F0的判断逻辑从“或”改为“与”，确保相对能量阈值配置生效。
        """
        enable_relative_mode = get_config('pure_vocal_detection.enable_relative_energy_mode', False)

        if enable_relative_mode:
            # --- 相对能量模式 ---
            logger.info("启用相对能量谷检测模式...")
            peak_energy = np.max(features.rms_energy)
            avg_energy = np.mean(features.rms_energy)
            
            peak_ratio = get_config('pure_vocal_detection.peak_relative_threshold_ratio', 0.1)
            rms_ratio = get_config('pure_vocal_detection.rms_relative_threshold_ratio', 0.2)

            threshold_from_peak = peak_energy * peak_ratio
            threshold_from_rms = avg_energy * rms_ratio
            energy_threshold = min(threshold_from_peak, threshold_from_rms)

            logger.info(f"全局能量分析: 峰值={peak_energy:.4f}, 平均值={avg_energy:.4f}")
            logger.info(f"动态能量阈值: 基于峰值({peak_ratio*100}%) -> {threshold_from_peak:.4f}, "
                       f"基于RMS({rms_ratio*100}%) -> {threshold_from_rms:.4f}")
            logger.info(f"最终能量裁决阈值: {energy_threshold:.4f}")
            
            low_energy = features.rms_energy < energy_threshold
        else:
            # --- 传统绝对dB模式 ---
            logger.info("使用绝对dB能量谷检测模式...")
            energy_threshold_db = get_config('pure_vocal_detection.energy_threshold_db', -40)
            energy_db = librosa.amplitude_to_db(features.rms_energy, ref=np.max)
            low_energy = energy_db < energy_threshold_db

        # F0不连续检测
        f0_drop_threshold = get_config('pure_vocal_detection.f0_drop_threshold', 0.7)
        f0_missing = features.f0_confidence < f0_drop_threshold
        
        # 关键修复：使用“与”逻辑 (&)，必须同时满足两个条件
        pause_frames = low_energy & f0_missing
        
        # 平滑处理
        pause_frames = gaussian_filter1d(pause_frames.astype(float), sigma=3) > 0.5
        
        # 查找连续区间 (保持不变)
        candidates = []
        in_pause = False
        start_idx = 0
        
        min_duration_s = get_config('pure_vocal_detection.breath_duration_range', [0.1, 0.3])[0]

        for i, is_pause in enumerate(pause_frames):
            if is_pause and not in_pause:
                start_idx = i
                in_pause = True
            elif not is_pause and in_pause:
                duration = (i - start_idx) * self.hop_length / self.sample_rate
                if duration >= min_duration_s:
                    candidates.append((start_idx, i))
                in_pause = False
        
        if in_pause:
            duration = (len(pause_frames) - start_idx) * self.hop_length / self.sample_rate
            if duration >= min_duration_s:
                candidates.append((start_idx, len(pause_frames)))

        logger.info(f"找到 {len(candidates)} 个候选停顿区域 (基于'与'逻辑)")
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
    
    def _calculate_harmonic_ratio_direct(self, audio: np.ndarray) -> np.ndarray:
        """直接计算纯人声的谐波比率
        
        Args:
            audio: 纯人声音频信号
            
        Returns:
            谐波比率序列
        """
        # 计算短时谱
        stft = librosa.stft(audio, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # 计算谐波比率：前1/3频段能量 vs 后2/3频段能量
        n_bins = magnitude.shape[0]
        low_freq_energy = np.sum(magnitude[:n_bins//3, :], axis=0)
        high_freq_energy = np.sum(magnitude[n_bins//3:, :], axis=0)
        
        # 谐波比率：低频能量占比（人声主要在低频）
        total_energy = low_freq_energy + high_freq_energy
        harmonic_ratio = low_freq_energy / (total_energy + 1e-10)
        
        return harmonic_ratio
    
    def _extract_formants(self, audio: np.ndarray) -> List[np.ndarray]:
        """提取共振峰特征
        
        Args:
            audio: 音频信号
            
        Returns:
            共振峰能量序列列表
        """
        # 使用线性预测编码(LPC)分析共振峰
        frame_length = int(0.025 * self.sample_rate)  # 25ms窗口
        hop_length = self.hop_length
        
        formant_tracks = [[] for _ in range(3)]  # F1, F2, F3
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            
            if len(frame) < frame_length:
                break
                
            # 预加重
            frame = np.append(frame[0], frame[1:] - 0.95 * frame[:-1])
            
            # LPC分析
            try:
                # 使用librosa的LPC
                lpc_coeffs = librosa.lpc(frame, order=12)
                
                # 从LPC系数计算频率响应
                w, h = signal.freqz(1, lpc_coeffs, worN=512, fs=self.sample_rate)
                
                # 找峰值作为共振峰
                magnitude = np.abs(h)
                peaks, _ = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.1)
                
                # 取前3个峰值的频率
                peak_freqs = w[peaks] if len(peaks) > 0 else []
                peak_mags = magnitude[peaks] if len(peaks) > 0 else []
                
                # 排序并分配给F1, F2, F3
                if len(peak_freqs) > 0:
                    sorted_indices = np.argsort(peak_freqs)
                    for j in range(min(3, len(sorted_indices))):
                        if j < len(peak_mags):
                            formant_tracks[j].append(peak_mags[sorted_indices[j]])
                        else:
                            formant_tracks[j].append(0.0)
                else:
                    for j in range(3):
                        formant_tracks[j].append(0.0)
                        
            except Exception as e:
                # LPC分析失败时填充零值
                for j in range(3):
                    formant_tracks[j].append(0.0)
        
        return [np.array(track) for track in formant_tracks]
    
    def _calculate_precise_cut_points(self, pure_vocal_pauses: List[PureVocalPause], 
                                    vocal_audio: np.ndarray) -> List[PureVocalPause]:
        """使用VocalPauseDetectorV2计算精确切点
        
        Args:
            pure_vocal_pauses: 纯人声停顿列表
            vocal_audio: 纯人声音频数据
            
        Returns:
            包含精确切点的停顿列表
        """
        logger.info(f"🔥 使用能量谷算法计算 {len(pure_vocal_pauses)} 个停顿的精确切点...")
        
        # 转换为VocalPause格式以使用能量谷计算
        from .vocal_pause_detector import VocalPause
        vocal_pauses = []
        
        for i, pure_pause in enumerate(pure_vocal_pauses):
            vocal_pause = VocalPause(
                start_time=pure_pause.start_time,
                end_time=pure_pause.end_time, 
                duration=pure_pause.duration,
                position_type='middle',  # 默认中间停顿
                confidence=pure_pause.confidence,
                cut_point=(pure_pause.start_time + pure_pause.end_time) / 2  # 临时切点
            )
            vocal_pauses.append(vocal_pause)
        
        # 🔥 关键修复：调用VocalPauseDetectorV2的能量谷切点计算，传入vocal_audio作为waveform
        try:
            vocal_pauses = self._cut_point_calculator._calculate_cut_points(
                vocal_pauses, 
                bpm_features=None,  # 纯人声模式不使用BPM对齐
                waveform=vocal_audio  # 关键：传递纯人声音频数据用于能量谷检测
            )
            logger.info("✅ 能量谷切点计算成功")
        except Exception as e:
            logger.error(f"❌ 能量谷切点计算失败: {e}") 
            logger.info("使用停顿中心作为兜底切点")
            for vocal_pause in vocal_pauses:
                vocal_pause.cut_point = (vocal_pause.start_time + vocal_pause.end_time) / 2
        
        # 将结果映射回PureVocalPause
        for i, (pure_pause, vocal_pause) in enumerate(zip(pure_vocal_pauses, vocal_pauses)):
            pure_pause.cut_point = vocal_pause.cut_point
            pure_pause.quality_grade = 'A' if hasattr(vocal_pause, 'cut_point') and vocal_pause.cut_point != (vocal_pause.start_time + vocal_pause.end_time) / 2 else 'B'
            logger.debug(f"停顿 {i+1}: [{pure_pause.start_time:.3f}s, {pure_pause.end_time:.3f}s] -> 切点 {pure_pause.cut_point:.3f}s ({pure_pause.quality_grade})")
        
        return pure_vocal_pauses

    def _detect_energy_valleys(self, vocal_audio: np.ndarray, peak_ratio: float, rms_ratio: float) -> List[PureVocalPause]:
        """
        🔥 相对能量谷检测 - 解决长音频分割不足问题
        
        Args:
            vocal_audio: 纯人声音频
            peak_ratio: 峰值能量比率阈值
            rms_ratio: RMS能量比率阈值
            
        Returns:
            检测到的能量谷停顿列表
        """
        logger.info(f"🔥 相对能量谷检测: peak_ratio={peak_ratio}, rms_ratio={rms_ratio}")
        
        # 1. 计算RMS能量包络
        frame_length = int(self.sample_rate * 0.025)  # 25ms
        hop_length = int(self.sample_rate * 0.01)     # 10ms
        rms_energy = librosa.feature.rms(y=vocal_audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 2. 计算动态阈值
        peak_energy = np.max(rms_energy)
        avg_energy = np.mean(rms_energy)
        peak_threshold = peak_energy * peak_ratio
        rms_threshold = avg_energy * rms_ratio
        
        # 🔥 关键修复：使用更宽松的阈值(取较小值)
        energy_threshold = min(peak_threshold, rms_threshold)
        logger.info(f"峰值能量: {peak_energy:.6f}, 平均能量: {avg_energy:.6f}")
        logger.info(f"能量谷阈值: {energy_threshold:.6f} (peak:{peak_threshold:.6f}, rms:{rms_threshold:.6f})")
        
        # 3. 找到低于阈值的区域
        low_energy_mask = rms_energy < energy_threshold
        time_frames = librosa.frames_to_time(np.arange(len(rms_energy)), sr=self.sample_rate, hop_length=hop_length)
        
        # 4. 将连续的低能量区域合并
        pauses = []
        in_pause = False
        pause_start = 0.0
        
        # 🔥 关键修复：对于能量谷检测，使用更短的最小停顿时长
        min_pause_duration = 0.2  # 200ms，适合音乐中的短暂停顿
        
        for i, (is_low, time) in enumerate(zip(low_energy_mask, time_frames)):
            if is_low and not in_pause:
                # 开始新的停顿
                pause_start = time
                in_pause = True
            elif not is_low and in_pause:
                # 结束当前停顿
                pause_end = time
                duration = pause_end - pause_start
                
                if duration >= min_pause_duration:
                    # 计算停顿的平均能量作为置信度
                    start_frame = max(0, int(pause_start * self.sample_rate / hop_length))
                    end_frame = min(len(rms_energy), int(pause_end * self.sample_rate / hop_length))
                    
                    if start_frame < end_frame:
                        pause_energy = np.mean(rms_energy[start_frame:end_frame])
                        confidence = 1.0 - (pause_energy / energy_threshold)  # 越低能量置信度越高
                        confidence = max(0.1, min(0.95, confidence))
                        
                        pause = PureVocalPause(
                            start_time=pause_start,
                            end_time=pause_end,
                            duration=duration,
                            pause_type='energy_valley',
                            confidence=confidence,
                            features={'energy': pause_energy, 'threshold': energy_threshold},
                            cut_point=(pause_start + pause_end) / 2
                        )
                        pauses.append(pause)
                        logger.debug(f"能量谷停顿: {pause_start:.3f}-{pause_end:.3f}s (时长:{duration:.3f}s, 置信度:{confidence:.3f})")
                
                in_pause = False
        
        # 处理文件末尾的停顿
        if in_pause:
            pause_end = time_frames[-1]
            duration = pause_end - pause_start
            if duration >= min_pause_duration:
                confidence = 0.8  # 末尾停顿给予较高置信度
                pause = PureVocalPause(
                    start_time=pause_start,
                    end_time=pause_end,
                    duration=duration,
                    pause_type='energy_valley',
                    confidence=confidence,
                    features={'energy': 0.0, 'threshold': energy_threshold},
                    cut_point=(pause_start + pause_end) / 2
                )
                pauses.append(pause)
        
        logger.info(f"🔥 能量谷检测完成: 发现{len(pauses)}个能量谷停顿")
        return pauses

    def _apply_mdd_enhancement(self, pauses: List[PureVocalPause], original_audio: np.ndarray) -> List[PureVocalPause]:
        """
        🔥 MDD (音乐动态密度) 增强处理
        
        Args:
            pauses: 原始停顿列表
            original_audio: 原始混音音频
            
        Returns:
            MDD增强后的停顿列表
        """
        logger.info("🔥 开始MDD增强处理...")
        
        if not pauses:
            return pauses
            
        # 1. 计算音乐动态密度
        frame_length = int(self.sample_rate * 0.1)  # 100ms窗口
        hop_length = int(self.sample_rate * 0.05)   # 50ms跳跃
        
        # RMS能量密度
        rms_energy = librosa.feature.rms(y=original_audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 频谱平坦度
        spectral_flatness = librosa.feature.spectral_flatness(y=original_audio, hop_length=hop_length)[0]
        
        # 音符起始检测
        onset_frames = librosa.onset.onset_detect(y=original_audio, sr=self.sample_rate, hop_length=hop_length)
        onset_strength = librosa.onset.onset_strength(y=original_audio, sr=self.sample_rate, hop_length=hop_length)
        
        # 时间轴
        time_frames = librosa.frames_to_time(np.arange(len(rms_energy)), sr=self.sample_rate, hop_length=hop_length)
        
        # 2. 计算MDD指标权重
        energy_weight = get_config('musical_dynamic_density.energy_weight', 0.7)
        spectral_weight = get_config('musical_dynamic_density.spectral_weight', 0.3)
        onset_weight = get_config('musical_dynamic_density.onset_weight', 0.2)
        
        # 3. 为每个停顿计算MDD评分
        enhanced_pauses = []
        threshold_multiplier = get_config('musical_dynamic_density.threshold_multiplier', 0.3)
        max_multiplier = get_config('musical_dynamic_density.max_multiplier', 1.4)
        min_multiplier = get_config('musical_dynamic_density.min_multiplier', 0.6)
        
        for pause in pauses:
            # 找到停顿对应的时间窗口
            start_frame = np.argmin(np.abs(time_frames - pause.start_time))
            end_frame = np.argmin(np.abs(time_frames - pause.end_time))
            
            if start_frame >= end_frame or start_frame >= len(rms_energy):
                enhanced_pauses.append(pause)
                continue
                
            # 计算停顿周围的MDD
            window_start = max(0, start_frame - 10)  # 扩展窗口
            window_end = min(len(rms_energy), end_frame + 10)
            
            # RMS能量密度
            local_rms = np.mean(rms_energy[window_start:window_end])
            energy_score = local_rms / np.max(rms_energy) if np.max(rms_energy) > 0 else 0.0
            
            # 频谱平坦度 (越平坦密度越低)
            local_flatness = np.mean(spectral_flatness[window_start:window_end])
            spectral_score = 1.0 - local_flatness  # 反转，密度越高分数越高
            
            # 音符起始密度
            onset_count = np.sum((onset_frames >= window_start) & (onset_frames < window_end))
            onset_score = min(1.0, onset_count / 5.0)  # 归一化到0-1
            
            # 综合MDD评分
            mdd_score = (energy_score * energy_weight + 
                        spectral_score * spectral_weight + 
                        onset_score * onset_weight)
            
            # 根据MDD调整停顿置信度
            confidence_multiplier = 1.0 + (mdd_score * threshold_multiplier)
            confidence_multiplier = max(min_multiplier, min(max_multiplier, confidence_multiplier))
            
            # 创建增强的停顿
            enhanced_pause = PureVocalPause(
                start_time=pause.start_time,
                end_time=pause.end_time,
                duration=pause.duration,
                pause_type=f"{pause.pause_type}_mdd",
                confidence=pause.confidence * confidence_multiplier,
                features={**pause.features, 'mdd_score': mdd_score, 'confidence_multiplier': confidence_multiplier},
                cut_point=pause.cut_point,
                quality_grade=pause.quality_grade
            )
            enhanced_pauses.append(enhanced_pause)
            
            logger.debug(f"MDD增强 - 停顿{pause.start_time:.2f}s: MDD={mdd_score:.3f}, 置信度倍数={confidence_multiplier:.3f}")
        
        logger.info(f"🔥 MDD增强完成: {len(enhanced_pauses)}个停顿已优化")
        return enhanced_pauses

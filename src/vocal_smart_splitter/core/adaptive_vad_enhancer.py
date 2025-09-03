#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/adaptive_vad_enhancer.py
# AI-SUMMARY: 编曲复杂度自适应VAD增强器，解决流行音乐后半部分编曲复杂化问题

import numpy as np
import librosa
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

@dataclass
class BPMFeatures:
    """BPM分析特征"""
    main_bpm: float              # 主要BPM
    bpm_category: str            # BPM类别 (slow/medium/fast)
    beat_strength: float         # 节拍强度 (0-1)
    bpm_confidence: float        # BPM检测置信度 (0-1) 
    tempo_variance: float        # 节拍变化程度 (0-1)
    adaptive_factors: Dict = None # BPM自适应因子
    beat_positions: np.ndarray = None  # 节拍位置数组

class BPMAnalyzer:
    """BPM节拍分析器 - 提取音乐节拍特征用于自适应VAD调整"""
    
    def __init__(self, sample_rate: int = 44100):
        """初始化BPM分析器
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        
        # BPM分类阈值
        self.bpm_categories = {
            'slow': (50, 80),       # 慢歌
            'medium': (80, 120),    # 中速
            'fast': (120, 160),     # 快歌  
            'very_fast': (160, 200) # 极快
        }
        
        logger.info(f"BPM分析器初始化完成 (采样率: {sample_rate})")
    
    def extract_bpm_features(self, audio: np.ndarray) -> BPMFeatures:
        """提取BPM特征
        
        Args:
            audio: 音频数据
            
        Returns:
            BPM特征对象
        """
        logger.info("开始分析音频BPM特征...")
        
        try:
            # 1. 基础节拍检测
            tempo, beats = librosa.beat.beat_track(
                y=audio, 
                sr=self.sample_rate,
                hop_length=512,
                start_bpm=120.0,
                tightness=100
            )
            
            # 2. 节拍稳定性计算
            beat_stability = self._calculate_beat_stability(beats, len(audio))
            
            # 3. 节拍变化分析
            tempo_variance = self._calculate_tempo_variance(audio)
            
            # 4. 音乐类型分类
            music_category = self._classify_music_by_bpm(tempo)
            
            # 5. 计算自适应因子
            adaptive_factors = self._calculate_bpm_adaptive_factors(tempo, beat_stability, tempo_variance)
            
            bpm_features = BPMFeatures(
                main_bpm=tempo,
                bpm_category=music_category,
                beat_strength=beat_stability,
                bpm_confidence=0.8,  # 默认置信度
                tempo_variance=tempo_variance,
                adaptive_factors=adaptive_factors,  # 添加自适应因子
                beat_positions=beats  # 添加节拍位置
            )
            
            logger.info(f"BPM分析完成: {float(tempo):.1f} BPM, 类型: {music_category}, 稳定性: {float(beat_stability):.3f}")
            return bpm_features
            
        except Exception as e:
            logger.error(f"BPM特征提取失败: {e}")
            # 返回默认特征
            return self._get_default_bpm_features()
    
    def _calculate_beat_stability(self, beats: np.ndarray, audio_length: int) -> float:
        """计算节拍稳定性
        
        Args:
            beats: 节拍位置数组
            audio_length: 音频总长度
            
        Returns:
            节拍稳定性 (0-1, 1表示非常稳定)
        """
        if len(beats) < 3:
            return 0.5  # 默认中等稳定性
        
        # 计算相邻节拍间隔
        beat_intervals = np.diff(beats)
        
        if len(beat_intervals) < 2:
            return 0.5
        
        # 稳定性 = 1 - (标准差 / 均值)，值越大越稳定
        mean_interval = np.mean(beat_intervals)
        std_interval = np.std(beat_intervals)
        
        if mean_interval == 0:
            return 0.5
            
        stability = 1.0 - (std_interval / mean_interval)
        return np.clip(stability, 0.0, 1.0)
    
    def _calculate_tempo_variance(self, audio: np.ndarray) -> float:
        """计算节拍变化程度
        
        Args:
            audio: 音频数据
            
        Returns:
            节拍变化程度 (值越大变化越大)
        """
        try:
            # 计算短时节拍变化
            hop_length = 512
            frame_length = 2048
            
            # 使用onset strength作为节拍变化的基础
            onset_envelope = librosa.onset.onset_strength(
                y=audio, 
                sr=self.sample_rate,
                hop_length=hop_length,
                aggregate=np.median
            )
            
            # 计算动态tempo变化
            tempo_curve = librosa.beat.tempo(
                onset_envelope=onset_envelope,
                sr=self.sample_rate,
                hop_length=hop_length,
                aggregate=None  # 不聚合，保持时间变化
            )
            
            if len(tempo_curve) > 1:
                # 确保数组操作的兼容性
                tempo_array = np.asarray(tempo_curve, dtype=np.float64)
                variance = float(np.std(tempo_array)) / (float(np.mean(tempo_array)) + 1e-8)
                return float(np.clip(variance, 0.0, 1.0))
            else:
                return 0.1  # 默认低变化
                
        except Exception as e:
            logger.warning(f"节拍变化分析失败: {e}")
            return 0.1
    
    def _classify_music_by_bpm(self, bpm: float) -> str:
        """根据BPM分类音乐类型
        
        Args:
            bpm: 节拍速度
            
        Returns:
            音乐类型标签
        """
        for category, (min_bpm, max_bpm) in self.bpm_categories.items():
            if min_bpm <= bpm < max_bpm:
                return category
        
        # 超出范围的处理
        if bpm < 50:
            return 'very_slow'
        else:
            return 'extreme_fast'
    
    def _calculate_bpm_adaptive_factors(self, bpm: float, stability: float, variance: float) -> Dict:
        """计算BPM自适应调整因子
        
        Args:
            bpm: 节拍速度
            stability: 节拍稳定性
            variance: 节拍变化程度
            
        Returns:
            自适应因子字典
        """
        # 基于BPM的基础调整（参数化配置）
        from ..utils.config_manager import get_config
        
        if bpm < 70:  # 慢歌
            pause_multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.pause_duration_multipliers.slow_song_multiplier', 1.5)
            base_factors = {
                'threshold_modifier': -0.05,    # 降低阈值，更敏感
                'min_pause_modifier': pause_multiplier,      # 使用配置的慢歌乘数
                'min_speech_modifier': 1.2,     # 允许更长语音
                'sensitivity': 'high'
            }
        elif bpm < 100:  # 中速
            pause_multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.pause_duration_multipliers.medium_song_multiplier', 1.0)
            base_factors = {
                'threshold_modifier': 0.0,      # 基准阈值
                'min_pause_modifier': pause_multiplier,      # 使用配置的中速歌乘数
                'min_speech_modifier': 1.0,     # 标准语音
                'sensitivity': 'medium'
            }
        elif bpm < 140:  # 快歌
            pause_multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.pause_duration_multipliers.fast_song_multiplier', 0.7)
            base_factors = {
                'threshold_modifier': 0.1,      # 提高阈值，更保守
                'min_pause_modifier': pause_multiplier,      # 使用配置的快歌乘数
                'min_speech_modifier': 0.8,     # 缩短最小语音
                'sensitivity': 'low'
            }
        else:  # 极快
            pause_multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.pause_duration_multipliers.fast_song_multiplier', 0.7)
            base_factors = {
                'threshold_modifier': 0.15,
                'min_pause_modifier': pause_multiplier,      # 使用快歌配置
                'min_speech_modifier': 0.6,
                'sensitivity': 'very_low'
            }
        
        # 稳定性调整：不稳定时更保守
        stability_adjustment = (1.0 - stability) * 0.1  # 最多增加0.1的阈值
        base_factors['threshold_modifier'] += stability_adjustment
        
        # 变化度调整：变化大时需要更高置信度
        variance_adjustment = variance * 0.05
        base_factors['threshold_modifier'] += variance_adjustment
        
        # 添加具体数值
        base_factors.update({
            'bpm_value': bpm,
            'stability_score': stability,
            'variance_score': variance,
            'recommended_window_size': self._calculate_analysis_window_size(bpm),
            'beat_sync_important': bpm > 100  # 快歌需要节拍对齐
        })
        
        return base_factors
    
    def _calculate_analysis_window_size(self, bpm: float) -> float:
        """根据BPM计算最佳分析窗口大小
        
        Args:
            bpm: 节拍速度
            
        Returns:
            分析窗口大小（秒）
        """
        # 快歌用较短窗口，慢歌用较长窗口
        if bpm < 70:
            return 12.0  # 慢歌：12秒窗口
        elif bpm < 120:
            return 10.0  # 中速：10秒窗口
        else:
            return 8.0   # 快歌：8秒窗口
    
    def _get_default_bpm_features(self) -> BPMFeatures:
        """获取默认BPM特征（当分析失败时使用）
        
        Returns:
            默认BPM特征
        """
        # 默认自适应因子
        default_adaptive_factors = {
            'threshold_modifier': 0.0,
            'min_pause_modifier': 1.0,
            'min_speech_modifier': 1.0,
            'sensitivity': 'medium',
            'bpm_value': 110.0,
            'stability_score': 0.6,
            'variance_score': 0.2,
            'recommended_window_size': 10.0,
            'beat_sync_important': False
        }
        
        return BPMFeatures(
            main_bpm=110.0,  # 默认中速
            bpm_category='medium',
            beat_strength=0.6,
            bpm_confidence=0.5,
            tempo_variance=0.2,
            adaptive_factors=default_adaptive_factors,
            beat_positions=np.array([])
        )

@dataclass
class ArrangementComplexitySegment:
    """编曲复杂度片段"""
    start_time: float           # 开始时间（秒）
    end_time: float             # 结束时间（秒）
    complexity_score: float     # 复杂度评分 (0-1)
    spectral_density: float     # 频谱密度
    harmonic_content: float     # 谐波内容
    bpm_influence: float        # BPM影响因子 (0-1)
    beat_alignment: float       # 节拍对齐度 (0-1)
    recommended_threshold: float # 推荐VAD阈值
    recommended_min_pause: float # 推荐最小停顿时长
    instrument_count: int = 0    # 检测到的乐器数量
    arrangement_density: float = 0.0  # 编曲密度评分

class InstrumentComplexityAnalyzer:
    """乐器复杂度分析器 - 检测编曲中的乐器数量和复杂度"""
    
    def __init__(self, sample_rate: int = 44100):
        """初始化乐器复杂度分析器
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        
        # 乐器频段定义（Hz）
        self.instrument_bands = {
            'bass': (40, 250),           # 贝斯
            'kick_drum': (50, 120),      # 底鼓
            'snare_drum': (150, 300),    # 军鼓
            'guitar_low': (80, 800),     # 吉他低频
            'guitar_mid': (800, 3000),   # 吉他中频
            'vocal_main': (200, 2000),   # 人声主频段
            'vocal_formant': (1000, 4000), # 人声共振峰
            'cymbals': (3000, 12000),    # 镲片
            'piano_low': (80, 500),      # 钢琴低音
            'piano_mid': (500, 2000),    # 钢琴中音
            'piano_high': (2000, 8000),  # 钢琴高音
            'strings': (200, 4000),      # 弦乐
            'brass': (100, 3000),        # 铜管
            'synth_lead': (200, 8000),   # 合成器主音
            'synth_pad': (50, 2000)      # 合成器铺底
        }
        
        logger.info("乐器复杂度分析器初始化完成")
    
    def analyze_instrument_complexity(self, audio: np.ndarray) -> Dict:
        """分析音频中的乐器复杂度
        
        Args:
            audio: 音频数据
            
        Returns:
            乐器复杂度分析结果
        """
        logger.info("开始乐器复杂度分析...")
        
        try:
            # 1. 频段能量分析
            band_energies = self._analyze_frequency_bands(audio)
            
            # 2. 乐器数量估算
            instrument_count = self._estimate_instrument_count(band_energies, audio)
            
            # 3. 编曲密度分析
            arrangement_density = self._calculate_arrangement_density(audio)
            
            # 4. 音色复杂度分析
            timbre_complexity = self._analyze_timbre_complexity(audio)
            
            # 5. 谐波层次分析
            harmonic_layers = self._analyze_harmonic_layers(audio)
            
            # 6. 时序复杂度分析
            temporal_complexity = self._analyze_temporal_complexity(audio)
            
            # 7. 乐器分离置信度
            separation_confidence = self._calculate_separation_confidence(band_energies)
            
            # 8. 人声干扰评估
            vocal_interference = self._assess_vocal_interference(band_energies, audio)
            
            complexity_result = {
                'instrument_count': instrument_count,
                'arrangement_density': arrangement_density,
                'timbre_complexity': timbre_complexity,
                'harmonic_layers': harmonic_layers,
                'temporal_complexity': temporal_complexity,
                'separation_confidence': separation_confidence,
                'vocal_interference': vocal_interference,
                'band_energies': band_energies,
                'overall_complexity': self._calculate_overall_complexity(
                    instrument_count, arrangement_density, timbre_complexity,
                    harmonic_layers, temporal_complexity, vocal_interference
                )
            }
            
            logger.info(f"乐器复杂度分析完成: {instrument_count}种乐器, 总复杂度: {complexity_result['overall_complexity']:.3f}")
            
            return complexity_result
            
        except Exception as e:
            logger.error(f"乐器复杂度分析失败: {e}")
            return self._get_default_complexity_result()
    
    def _analyze_frequency_bands(self, audio: np.ndarray) -> Dict[str, float]:
        """分析各频段的能量分布"""
        # 计算短时傅里叶变换
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        power_spectrum = magnitude ** 2
        
        # 频率轴
        freq_axis = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
        
        band_energies = {}
        
        for band_name, (low_freq, high_freq) in self.instrument_bands.items():
            # 找到频率范围对应的索引
            low_idx = np.argmin(np.abs(freq_axis - low_freq))
            high_idx = np.argmin(np.abs(freq_axis - high_freq))
            
            # 计算该频段的平均能量
            band_power = np.mean(power_spectrum[low_idx:high_idx, :])
            band_energies[band_name] = float(band_power)
        
        return band_energies
    
    def _estimate_instrument_count(self, band_energies: Dict[str, float], audio: np.ndarray) -> int:
        """估算活跃乐器数量"""
        total_energy = sum(band_energies.values())
        if total_energy == 0:
            return 1
        
        # 乐器组合逻辑检测
        instrument_evidence = {
            'bass': band_energies['bass'] > 0.08 * total_energy,
            'drums': (band_energies['kick_drum'] > 0.04 * total_energy or 
                     band_energies['snare_drum'] > 0.04 * total_energy or
                     band_energies['cymbals'] > 0.02 * total_energy),
            'guitar': (band_energies['guitar_low'] > 0.06 * total_energy or
                      band_energies['guitar_mid'] > 0.06 * total_energy),
            'piano': (band_energies['piano_low'] > 0.05 * total_energy and
                     band_energies['piano_mid'] > 0.05 * total_energy),
            'strings': band_energies['strings'] > 0.08 * total_energy,
            'brass': band_energies['brass'] > 0.06 * total_energy,
            'synth': (band_energies['synth_lead'] > 0.05 * total_energy or
                     band_energies['synth_pad'] > 0.08 * total_energy),
            'vocal': (band_energies['vocal_main'] > 0.12 * total_energy and
                     band_energies['vocal_formant'] > 0.08 * total_energy)
        }
        
        active_instruments = sum(evidence for evidence in instrument_evidence.values())
        return int(np.clip(active_instruments, 1, 8))
    
    def _calculate_arrangement_density(self, audio: np.ndarray) -> float:
        """计算编曲密度"""
        try:
            # 使用频谱质心分析
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            centroid_std = np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-8)
            
            # 频谱展开度
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            rolloff_variation = np.std(spectral_rolloff) / (np.mean(spectral_rolloff) + 1e-8)
            
            density_score = min(centroid_std * 0.001 + rolloff_variation * 0.0001, 1.0)
            return float(np.clip(density_score, 0.0, 1.0))
        except:
            return 0.5
    
    def _analyze_timbre_complexity(self, audio: np.ndarray) -> float:
        """分析音色复杂度"""
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            mfcc_variance = np.mean(np.var(mfccs, axis=1))
            complexity = min(mfcc_variance / 100.0, 1.0)
            return float(np.clip(complexity, 0.0, 1.0))
        except:
            return 0.5
    
    def _analyze_harmonic_layers(self, audio: np.ndarray) -> float:
        """分析谐波层次"""
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            chroma_threshold = 0.3 * np.max(chroma, axis=0, keepdims=True)
            active_pitches_per_frame = np.sum(chroma > chroma_threshold, axis=0)
            avg_active_pitches = np.mean(active_pitches_per_frame)
            harmonic_complexity = min(avg_active_pitches / 6.0, 1.0)
            return float(np.clip(harmonic_complexity, 0.0, 1.0))
        except:
            return 0.5
    
    def _analyze_temporal_complexity(self, audio: np.ndarray) -> float:
        """分析时序复杂度"""
        try:
            rms_energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            energy_variance = np.std(rms_energy) / (np.mean(rms_energy) + 1e-8)
            temporal_complexity = min(energy_variance * 2.0, 1.0)
            return float(np.clip(temporal_complexity, 0.0, 1.0))
        except:
            return 0.5
    
    def _calculate_separation_confidence(self, band_energies: Dict[str, float]) -> float:
        """计算乐器分离置信度"""
        total_energy = sum(band_energies.values())
        if total_energy == 0:
            return 0.5
        
        vocal_energy = band_energies['vocal_main'] + band_energies['vocal_formant']
        competing_energy = (band_energies['guitar_mid'] + band_energies['piano_mid'] + 
                          band_energies['strings'] + band_energies['synth_lead'])
        
        vocal_ratio = vocal_energy / total_energy
        competing_ratio = competing_energy / total_energy
        
        if vocal_ratio > competing_ratio * 1.5:
            return 0.8
        elif vocal_ratio > competing_ratio:
            return 0.6
        else:
            return 0.3
    
    def _assess_vocal_interference(self, band_energies: Dict[str, float], audio: np.ndarray) -> float:
        """评估人声检测的干扰程度"""
        total_energy = sum(band_energies.values())
        if total_energy == 0:
            return 0.5
        
        interference_sources = {
            'guitar_interference': band_energies['guitar_mid'] / total_energy,
            'piano_interference': band_energies['piano_mid'] / total_energy,
            'strings_interference': band_energies['strings'] / total_energy,
            'synth_interference': band_energies['synth_lead'] / total_energy,
            'brass_interference': band_energies['brass'] / total_energy
        }
        
        total_interference = sum(interference_sources.values())
        return float(np.clip(total_interference, 0.0, 1.0))
    
    def _calculate_overall_complexity(self, instrument_count: int, arrangement_density: float,
                                   timbre_complexity: float, harmonic_layers: float,
                                   temporal_complexity: float, vocal_interference: float) -> float:
        """计算总体复杂度评分"""
        normalized_instruments = min(instrument_count / 8.0, 1.0)
        
        overall_score = (
            0.25 * normalized_instruments +
            0.20 * arrangement_density +
            0.15 * timbre_complexity +
            0.15 * harmonic_layers +
            0.10 * temporal_complexity +
            0.15 * vocal_interference
        )
        
        return float(np.clip(overall_score, 0.0, 1.0))
    
    def _get_default_complexity_result(self) -> Dict:
        """获取默认复杂度分析结果"""
        return {
            'instrument_count': 3,
            'arrangement_density': 0.5,
            'timbre_complexity': 0.5,
            'harmonic_layers': 0.5,
            'temporal_complexity': 0.5,
            'separation_confidence': 0.5,
            'vocal_interference': 0.5,
            'band_energies': {},
            'overall_complexity': 0.5
        }

class AdaptiveVADEnhancer:
    """BPM感知的编曲复杂度自适应VAD增强器
    
    解决流行音乐中常见的问题：
    1. 前半部分编曲简单，VAD过敏感 → 产生超短片段  
    2. 后半部分编曲复杂，VAD不敏感 → 漏检真实停顿
    3. 不同BPM的音乐需要不同的检测策略
    """
    
    def __init__(self, sample_rate: int = 44100):
        """初始化自适应增强器
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        
        # 初始化BPM分析器
        self.bpm_analyzer = BPMAnalyzer(sample_rate)
        
        # 初始化乐器复杂度分析器
        self.instrument_analyzer = InstrumentComplexityAnalyzer(sample_rate)
        
        # 编曲复杂度分析参数（将根据BPM动态调整）
        self.default_analysis_window = 10.0  # 默认窗口大小
        self.complexity_threshold = 0.6      # 复杂度阈值
        
        # VAD自适应阈值范围（扩展以适应多乐器环境）
        self.min_vad_threshold = 0.20  # 简单编曲最低阈值（降低）
        self.max_vad_threshold = 0.80  # 复杂编曲最高阈值（提高）
        self.base_threshold = 0.35     # 基准阈值
        
        # 新增：BPM感知权重
        self.bpm_weight = 0.35         # BPM指标在复杂度评分中的权重
        self.beat_alignment_weight = 0.1  # 节拍对齐的权重
        
        logger.info("BPM感知的编曲复杂度自适应VAD增强器初始化完成")
    
    def analyze_bpm(self, audio: np.ndarray) -> BPMFeatures:
        """分析音频的BPM特征（对外接口）
        
        Args:
            audio: 音频数据
            
        Returns:
            BPM特征对象
        """
        return self.bpm_analyzer.extract_bpm_features(audio)
    
    def generate_adaptive_thresholds(self, bpm_features: BPMFeatures, 
                                   complexity_scores: List[float]) -> Dict:
        """生成BPM自适应阈值
        
        Args:
            bpm_features: BPM特征
            complexity_scores: 复杂度评分列表
            
        Returns:
            自适应阈值字典
        """
        # 根据BPM类别调整基础阈值
        if bpm_features.bpm_category == 'slow':
            base_threshold = self.base_threshold * 0.8  # 慢歌更敏感
            bpm_factor = 0.7
        elif bpm_features.bpm_category == 'fast':
            base_threshold = self.base_threshold * 1.2  # 快歌更保守
            bpm_factor = 1.3
        else:
            base_threshold = self.base_threshold
            bpm_factor = 1.0
        
        # 根据复杂度生成分段阈值（增强多乐器适应性）
        segment_thresholds = []
        for score in complexity_scores:
            # 多乐器环境下的增强调整
            if bpm_features.bpm_category in ['fast', 'very_fast']:
                # 快歌环境：更保守的阈值调整
                instrument_boost = score * 0.35  # 从0.2增加到0.35
            elif score > 0.6:  # 高复杂度环境
                instrument_boost = score * 0.45  # 显著提升阈值
            else:
                instrument_boost = score * 0.25  # 轻度调整
            
            adaptive_threshold = base_threshold + instrument_boost
            adaptive_threshold = np.clip(adaptive_threshold, self.min_vad_threshold, self.max_vad_threshold)
            segment_thresholds.append(adaptive_threshold)
        
        return {
            'base_threshold': base_threshold,
            'segment_thresholds': segment_thresholds,
            'bpm_factor': bpm_factor,
            'bpm_category': bpm_features.bpm_category
        }
    
    def analyze_arrangement_complexity(self, audio: np.ndarray) -> Tuple[List[ArrangementComplexitySegment], BPMFeatures]:
        """分析音频的编曲复杂度变化（集成BPM分析）
        
        Args:
            audio: 音频数据
            
        Returns:
            (编曲复杂度片段列表, BPM特征)
        """
        logger.info("开始BPM感知的编曲复杂度分析...")
        
        try:
            # 1. 首先提取整体BPM特征
            bpm_features = self.bpm_analyzer.extract_bpm_features(audio)
            
            # 2. 根据BPM调整分析窗口大小
            analysis_window = bpm_features.adaptive_factors['recommended_window_size']
            
            # 3. 计算音频总长度
            total_duration = len(audio) / self.sample_rate
            segments = []
            
            # 4. 全局乐器复杂度分析
            instrument_complexity = self.instrument_analyzer.analyze_instrument_complexity(audio)
            logger.info(f"检测到 {instrument_complexity['instrument_count']} 种乐器，总复杂度: {instrument_complexity['overall_complexity']:.3f}")
            
            # 5. 按窗口分析复杂度
            window_samples = int(analysis_window * self.sample_rate)
            hop_samples = window_samples // 2  # 50%重叠
            
            # 5. 节拍对齐的分析窗口（如果BPM稳定且需要节拍同步）
            beat_positions = bpm_features.beat_positions
            use_beat_alignment = (
                bpm_features.adaptive_factors['beat_sync_important'] and 
                len(beat_positions) > 0
            )
            
            for i in range(0, len(audio) - window_samples, hop_samples):
                start_sample = i
                end_sample = min(i + window_samples, len(audio))
                segment_audio = audio[start_sample:end_sample]
                
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                
                # 6. 计算传统复杂度指标
                complexity_metrics = self._calculate_complexity_metrics(segment_audio)
                
                # 7. 计算BPM影响因子和节拍对齐度
                bpm_influence = self._calculate_bpm_influence_factor(
                    bpm_features, start_time, end_time, total_duration
                )
                beat_alignment = self._calculate_beat_alignment(
                    beat_positions, start_sample, end_sample
                ) if use_beat_alignment else 0.5
                
                # 8. 综合复杂度评分（集成BPM）
                complexity_score = self._calculate_enhanced_complexity(
                    complexity_metrics, bpm_features, bpm_influence, beat_alignment
                )
                
                # 9. 多维自适应阈值和停顿时长
                adaptive_params = self._calculate_multi_dimensional_adaptive_params(
                    complexity_score, bpm_features, start_time / total_duration
                )
                
                segment = ArrangementComplexitySegment(
                    start_time=start_time,
                    end_time=end_time,
                    complexity_score=complexity_score,
                    spectral_density=complexity_metrics['spectral_density'],
                    harmonic_content=complexity_metrics['harmonic_content'],
                    bpm_influence=bpm_influence,
                    beat_alignment=beat_alignment,
                    recommended_threshold=adaptive_params['voice_threshold'],
                    recommended_min_pause=adaptive_params['min_pause_duration']
                )
                
                segments.append(segment)
            
            # 10. 平滑复杂度变化（避免阈值突变）
            segments = self._smooth_complexity_transitions(segments)
            
            logger.info(f"BPM感知复杂度分析完成，共分析 {len(segments)} 个片段")
            self._log_enhanced_complexity_summary(segments, bpm_features)
            
            return segments, bpm_features
            
        except Exception as e:
            logger.error(f"BPM感知复杂度分析失败: {e}")
            return [], self.bpm_analyzer._get_default_bpm_features()
    
    def _calculate_complexity_metrics(self, audio_segment: np.ndarray) -> Dict[str, float]:
        """计算编曲复杂度指标
        
        Args:
            audio_segment: 音频片段
            
        Returns:
            复杂度指标字典
        """
        metrics = {}
        
        # 1. 频谱密度 - 衡量频率成分的丰富程度
        stft = librosa.stft(audio_segment, hop_length=512, n_fft=2048)
        magnitude = np.abs(stft)
        
        # 计算每个频率bin的活跃度
        freq_activity = np.mean(magnitude > 0.01 * np.max(magnitude), axis=1)
        metrics['spectral_density'] = np.mean(freq_activity)
        
        # 2. 谐波内容 - 检测多乐器叠加
        try:
            # 使用chromagram检测和声复杂度
            chroma = librosa.feature.chroma_stft(S=magnitude**2, sr=self.sample_rate)
            # 计算同时活跃的音高类别数量
            active_pitches = np.mean(np.sum(chroma > 0.3 * np.max(chroma, axis=0), axis=0))
            metrics['harmonic_content'] = min(active_pitches / 6.0, 1.0)  # 归一化到0-1
        except:
            metrics['harmonic_content'] = 0.5  # 默认中等复杂度
        
        # 3. 动态范围 - 衡量音量变化的剧烈程度
        rms = librosa.feature.rms(y=audio_segment, hop_length=512)[0]
        if len(rms) > 1:
            dynamic_range = np.std(rms) / (np.mean(rms) + 1e-8)
            metrics['dynamic_range'] = min(dynamic_range * 2.0, 1.0)
        else:
            metrics['dynamic_range'] = 0.3
        
        # 4. 频谱质心变化 - 衡量音色变化
        try:
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_segment, sr=self.sample_rate, hop_length=512
            )[0]
            if len(spectral_centroids) > 1:
                centroid_variation = np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-8)
                metrics['spectral_variation'] = min(centroid_variation * 0.001, 1.0)
            else:
                metrics['spectral_variation'] = 0.3
        except:
            metrics['spectral_variation'] = 0.3
        
        # 5. 零交叉率变化 - 衡量瞬态内容
        zcr = librosa.feature.zero_crossing_rate(audio_segment, hop_length=512)[0]
        if len(zcr) > 1:
            zcr_variation = np.std(zcr) / (np.mean(zcr) + 1e-8)
            metrics['transient_content'] = min(zcr_variation * 10.0, 1.0)
        else:
            metrics['transient_content'] = 0.3
        
        return metrics
    
    def _calculate_bpm_influence_factor(
        self, bpm_features: BPMFeatures, start_time: float, end_time: float, total_duration: float
    ) -> float:
        """计算BPM在当前时间段的影响因子
        
        Args:
            bpm_features: BPM特征
            start_time: 片段开始时间
            end_time: 片段结束时间
            total_duration: 音频总时长
            
        Returns:
            BPM影响因子 (0-1)
        """
        # 基础BPM影响 - 快歌影响更大
        bpm_base_influence = min(bpm_features.main_bpm / 160.0, 1.0)
        
        # 时间位置影响 - 后半部分通常编曲更复杂
        time_position = (start_time + end_time) / (2 * total_duration)
        time_influence = 0.5 + 0.5 * time_position  # 0.5-1.0
        
        # 节拍稳定性影响 - 不稳定的节拍增加复杂度
        stability_influence = 1.0 - bpm_features.beat_strength
        
        # 综合影响因子
        influence = (
            0.5 * bpm_base_influence +
            0.3 * time_influence +
            0.2 * stability_influence
        )
        
        return np.clip(influence, 0.0, 1.0)
    
    def _calculate_beat_alignment(
        self, beat_positions: np.ndarray, start_sample: int, end_sample: int
    ) -> float:
        """计算当前片段与节拍的对齐度
        
        Args:
            beat_positions: 节拍位置数组（样本点）
            start_sample: 片段开始样本点
            end_sample: 片段结束样本点
            
        Returns:
            节拍对齐度 (0-1, 1表示完全对齐)
        """
        if len(beat_positions) == 0:
            return 0.5  # 默认中等对齐度
        
        # 找到片段内的节拍点
        beats_in_segment = beat_positions[
            (beat_positions >= start_sample) & (beat_positions <= end_sample)
        ]
        
        if len(beats_in_segment) == 0:
            return 0.3  # 无节拍点，对齐度较低
        
        # 计算节拍分布的均匀性
        segment_length = end_sample - start_sample
        expected_beats = len(beats_in_segment)
        
        if expected_beats < 2:
            return 0.5
        
        # 计算实际节拍间隔的一致性
        actual_intervals = np.diff(beats_in_segment)
        interval_consistency = 1.0 - (np.std(actual_intervals) / (np.mean(actual_intervals) + 1e-8))
        
        return np.clip(interval_consistency, 0.0, 1.0)
    
    def _calculate_enhanced_complexity(
        self, 
        traditional_metrics: Dict[str, float], 
        bpm_features: BPMFeatures,
        bpm_influence: float,
        beat_alignment: float
    ) -> float:
        """计算增强的复杂度评分（集成BPM）
        
        Args:
            traditional_metrics: 传统复杂度指标
            bpm_features: BPM特征
            bpm_influence: BPM影响因子
            beat_alignment: 节拍对齐度
            
        Returns:
            增强的复杂度评分 (0-1)
        """
        # 新权重分配（BPM感知）
        weights = {
            'bpm_factor': 0.25,           # BPM因子最重要
            'beat_alignment': 0.1,        # 节拍对齐度
            'spectral_density': 0.2,      # 频谱密度（从0.3降权）
            'harmonic_content': 0.15,     # 和声复杂度（从0.25降权）
            'dynamic_range': 0.15,        # 动态范围
            'spectral_variation': 0.1,    # 频谱变化
            'transient_content': 0.05     # 瞬态内容
        }
        
        # 计算综合复杂度
        complexity_score = 0.0
        
        # BPM相关因子
        complexity_score += weights['bpm_factor'] * bpm_influence
        complexity_score += weights['beat_alignment'] * (1.0 - beat_alignment)  # 对齐度低=复杂度高
        
        # 传统指标
        for metric, weight in weights.items():
            if metric in ['bpm_factor', 'beat_alignment']:
                continue
            complexity_score += traditional_metrics.get(metric, 0.5) * weight
        
        return np.clip(complexity_score, 0.0, 1.0)
    
    def _calculate_multi_dimensional_adaptive_params(
        self, complexity_score: float, bpm_features: BPMFeatures, time_position: float
    ) -> Dict[str, float]:
        """多维自适应阈值生成算法
        
        Args:
            complexity_score: 综合复杂度评分
            bpm_features: BPM特征
            time_position: 时间位置比例 (0-1)
            
        Returns:
            自适应参数字典
        """
        # 获取BPM基础调整因子
        bpm_factors = bpm_features.adaptive_factors
        
        # 1. 基础阈值计算
        base_threshold = self.base_threshold
        
        # 2. BPM调整
        bpm_adjustment = bpm_factors['threshold_modifier']
        
        # 3. 复杂度调整
        complexity_adjustment = (complexity_score - 0.5) * 0.3
        
        # 4. 时间位置调整（后半部分更保守）
        time_adjustment = 0.15 * time_position if time_position > 0.6 else 0
        
        # 5. 综合阈值
        final_threshold = base_threshold + bpm_adjustment + complexity_adjustment + time_adjustment
        final_threshold = np.clip(final_threshold, self.min_vad_threshold, self.max_vad_threshold)
        
        # 6. 自适应停顿时长
        base_min_pause = 1.0
        pause_adjustment = (
            base_min_pause * 
            bpm_factors['min_pause_modifier'] * 
            (1.0 + 0.3 * complexity_score)  # 复杂度高时需要更长停顿
        )
        
        # 7. 自适应语音时长
        base_min_speech = 0.4
        speech_adjustment = (
            base_min_speech * 
            bpm_factors['min_speech_modifier'] * 
            (1.0 + 0.2 * complexity_score)
        )
        
        return {
            'voice_threshold': round(float(final_threshold), 3),
            'min_pause_duration': round(float(pause_adjustment), 2),
            'min_speech_duration': round(float(speech_adjustment), 2),
            'complexity_context': {
                'bpm_category': bpm_features.bpm_category,
                'bpm_value': bpm_features.main_bpm,
                'complexity_score': complexity_score,
                'time_position': time_position
            }
        }
    
    def get_enhanced_adaptive_vad_params(
        self, 
        segments: List[ArrangementComplexitySegment], 
        bpm_features: BPMFeatures,
        current_time: float
    ) -> Dict[str, float]:
        """获取当前时间点的BPM感知自适应VAD参数
        
        Args:
            segments: 编曲复杂度片段列表
            bpm_features: BPM特征
            current_time: 当前时间（秒）
            
        Returns:
            增强的自适应VAD参数字典
        """
        # 找到当前时间对应的复杂度片段
        current_segment = None
        for segment in segments:
            if segment.start_time <= current_time <= segment.end_time:
                current_segment = segment
                break
        
        if current_segment is None:
            # 如果没找到对应片段，使用BPM基础参数
            base_factors = bpm_features.adaptive_factors
            return {
                'voice_threshold': self.base_threshold + base_factors['threshold_modifier'],
                'min_silence_duration_ms': int(1000 * base_factors['min_pause_modifier']),
                'min_speech_duration_ms': int(400 * base_factors['min_speech_modifier']),
                'bpm_context': {
                    'bpm_value': bpm_features.main_bpm,
                    'music_category': bpm_features.bpm_category,
                    'complexity_score': 0.5
                }
            }
        
        # 使用片段的推荐参数
        return {
            'voice_threshold': current_segment.recommended_threshold,
            'min_silence_duration_ms': int(current_segment.recommended_min_pause * 1000),
            'min_speech_duration_ms': int(400 * bpm_features.adaptive_factors['min_speech_modifier']),
            'bpm_context': {
                'bpm_value': bpm_features.main_bpm,
                'music_category': bpm_features.bpm_category,
                'complexity_score': current_segment.complexity_score,
                'bpm_influence': current_segment.bpm_influence,
                'beat_alignment': current_segment.beat_alignment
            },
            'window_size_samples': int(bpm_features.adaptive_factors['recommended_window_size'] * self.sample_rate / 100)
        }
    
    def _calculate_overall_complexity(self, metrics: Dict[str, float]) -> float:
        """计算综合复杂度评分
        
        Args:
            metrics: 复杂度指标
            
        Returns:
            综合复杂度评分 (0-1)
        """
        # 权重设计（基于流行音乐特点）
        weights = {
            'spectral_density': 0.3,    # 频谱密度最重要
            'harmonic_content': 0.25,   # 和声复杂度次重要
            'dynamic_range': 0.2,       # 动态范围
            'spectral_variation': 0.15, # 频谱变化
            'transient_content': 0.1    # 瞬态内容
        }
        
        complexity_score = 0.0
        for metric, weight in weights.items():
            complexity_score += metrics.get(metric, 0.5) * weight
        
        return min(max(complexity_score, 0.0), 1.0)
    
    def _calculate_adaptive_threshold(self, complexity_score: float) -> float:
        """根据复杂度计算自适应VAD阈值
        
        Args:
            complexity_score: 复杂度评分 (0-1)
            
        Returns:
            推荐的VAD阈值
        """
        # 复杂度越高，VAD阈值越高（更保守）
        # 复杂度越低，VAD阈值越低（更敏感）
        
        if complexity_score < 0.3:
            # 简单编曲：使用较低阈值，但不能太低（避免超短片段）
            threshold = self.min_vad_threshold + (complexity_score / 0.3) * 0.1
        elif complexity_score > 0.7:
            # 复杂编曲：使用较高阈值
            threshold = self.base_threshold + ((complexity_score - 0.7) / 0.3) * (
                self.max_vad_threshold - self.base_threshold
            )
        else:
            # 中等编曲：线性插值
            threshold = self.min_vad_threshold + complexity_score * (
                self.base_threshold - self.min_vad_threshold
            )
        
        return round(float(threshold), 3)
    
    def _smooth_complexity_transitions(
        self, segments: List[ArrangementComplexitySegment]
    ) -> List[ArrangementComplexitySegment]:
        """平滑复杂度变化，避免阈值突变
        
        Args:
            segments: 原始复杂度片段
            
        Returns:
            平滑后的复杂度片段
        """
        if len(segments) < 3:
            return segments
        
        # 使用移动平均平滑阈值
        window_size = 3
        for i in range(len(segments)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(segments), i + window_size // 2 + 1)
            
            # 计算窗口内的平均阈值
            avg_threshold = np.mean([
                seg.recommended_threshold for seg in segments[start_idx:end_idx]
            ])
            
            # 更新阈值（保持一定的原始特性）
            segments[i].recommended_threshold = (
                0.7 * segments[i].recommended_threshold + 0.3 * avg_threshold
            )
        
        return segments
    
    def _log_enhanced_complexity_summary(
        self, segments: List[ArrangementComplexitySegment], bpm_features: BPMFeatures
    ):
        """记录增强版复杂度分析摘要（包含BPM信息）"""
        if not segments:
            return
        
        complexity_scores = [seg.complexity_score for seg in segments]
        thresholds = [seg.recommended_threshold for seg in segments]
        min_pauses = [seg.recommended_min_pause for seg in segments]
        bpm_influences = [seg.bpm_influence for seg in segments]
        beat_alignments = [seg.beat_alignment for seg in segments]
        
        logger.info("=== BPM感知复杂度分析摘要 ===")
        logger.info(f"🎵 音乐特征: {float(bpm_features.main_bpm):.1f} BPM ({bpm_features.bpm_category})")
        logger.info(f"🎼 节拍稳定性: {float(bpm_features.beat_strength):.3f}")
        logger.info(f"📊 复杂度范围: {float(min(complexity_scores)):.3f} - {float(max(complexity_scores)):.3f}")
        logger.info(f"📊 平均复杂度: {float(np.mean(complexity_scores)):.3f}")
        logger.info(f"⚙️  VAD阈值范围: {float(min(thresholds)):.3f} - {float(max(thresholds)):.3f}")
        logger.info(f"⏱️  停顿时长范围: {float(min(min_pauses)):.2f}s - {float(max(min_pauses)):.2f}s")
        logger.info(f"🎵 BPM影响因子: {float(np.mean(bpm_influences)):.3f}")
        logger.info(f"🎼 节拍对齐度: {float(np.mean(beat_alignments)):.3f}")
        
        # BPM自适应效果分析
        bpm_factors = bpm_features.adaptive_factors
        logger.info(f"🔧 BPM自适应调整: 阈值{bpm_factors['threshold_modifier']:+.3f}, 停顿×{bpm_factors['min_pause_modifier']:.2f}")
        
        # 检测编曲复杂度趋势
        first_half = complexity_scores[:len(complexity_scores)//2]
        second_half = complexity_scores[len(complexity_scores)//2:]
        
        if np.mean(second_half) > np.mean(first_half) + 0.1:
            logger.info("📈 检测到编曲复杂度递增趋势（典型流行音乐模式）+ BPM自适应调整")
        elif np.mean(first_half) > np.mean(second_half) + 0.1:
            logger.info("📉 检测到编曲复杂度递减趋势 + BPM自适应调整")
        else:
            logger.info("➡️  编曲复杂度相对稳定 + BPM基准调整")
    
    def _log_complexity_summary(self, segments: List[ArrangementComplexitySegment]):
        """记录复杂度分析摘要"""
        if not segments:
            return
        
        complexity_scores = [seg.complexity_score for seg in segments]
        thresholds = [seg.recommended_threshold for seg in segments]
        
        logger.info("=== 编曲复杂度分析摘要 ===")
        logger.info(f"复杂度范围: {min(complexity_scores):.3f} - {max(complexity_scores):.3f}")
        logger.info(f"平均复杂度: {np.mean(complexity_scores):.3f}")
        logger.info(f"VAD阈值范围: {min(thresholds):.3f} - {max(thresholds):.3f}")
        
        # 检测编曲复杂度趋势
        first_half = complexity_scores[:len(complexity_scores)//2]
        second_half = complexity_scores[len(complexity_scores)//2:]
        
        if np.mean(second_half) > np.mean(first_half) + 0.1:
            logger.info("📈 检测到编曲复杂度递增趋势（典型流行音乐模式）")
        elif np.mean(first_half) > np.mean(second_half) + 0.1:
            logger.info("📉 检测到编曲复杂度递减趋势")
        else:
            logger.info("➡️  编曲复杂度相对稳定")
    
    def get_adaptive_vad_params(
        self, 
        segments: List[ArrangementComplexitySegment], 
        current_time: float
    ) -> Dict[str, float]:
        """获取当前时间点的自适应VAD参数
        
        Args:
            segments: 编曲复杂度片段列表
            current_time: 当前时间（秒）
            
        Returns:
            自适应VAD参数字典
        """
        # 找到当前时间对应的复杂度片段
        current_segment = None
        for segment in segments:
            if segment.start_time <= current_time <= segment.end_time:
                current_segment = segment
                break
        
        if current_segment is None:
            # 如果没找到对应片段，使用默认参数
            return {
                'voice_threshold': self.base_threshold,
                'min_silence_duration_ms': 1000,
                'min_speech_duration_ms': 400
            }
        
        # 根据复杂度调整参数
        complexity = current_segment.complexity_score
        threshold = current_segment.recommended_threshold
        
        # 复杂度高时，增加最小静音和语音时长要求
        min_silence_ms = 800 + int(complexity * 600)  # 800-1400ms
        min_speech_ms = 300 + int(complexity * 400)   # 300-700ms
        
        return {
            'voice_threshold': threshold,
            'min_silence_duration_ms': min_silence_ms,
            'min_speech_duration_ms': min_speech_ms,
            'complexity_score': complexity
        }
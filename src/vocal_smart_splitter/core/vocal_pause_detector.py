#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/vocal_pause_detector.py
# AI-SUMMARY: 人声停顿检测器 - 使用Silero VAD直接在原始音频上检测人声停顿

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..utils.config_manager import get_config

logger = logging.getLogger(__name__)

# 尝试导入自适应增强器
try:
    from .adaptive_vad_enhancer import AdaptiveVADEnhancer
    ADAPTIVE_VAD_AVAILABLE = True
    logger.info("✅ 自适应VAD增强器可用")
except ImportError as e:
    logger.warning(f"⚠️  自适应VAD增强器不可用: {e}")
    ADAPTIVE_VAD_AVAILABLE = False

@dataclass
class VocalPause:
    """人声停顿数据结构"""
    start_time: float        # 停顿开始时间（秒）
    end_time: float          # 停顿结束时间（秒）  
    duration: float          # 停顿时长（秒）
    position_type: str       # 位置类型：'head', 'middle', 'tail'
    confidence: float        # 置信度 (0-1)
    cut_point: float         # 切割点时间（秒）

class VocalPauseDetectorV2:
    """改进的人声停顿检测器 - 直接在原始音频上使用Silero VAD"""
    
    def __init__(self, sample_rate: int = 44100):
        """初始化人声停顿检测器
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        
        # 配置参数
        self.min_pause_duration = get_config('vocal_pause_splitting.min_pause_duration', 1.0)
        self.voice_threshold = get_config('vocal_pause_splitting.voice_threshold', 0.3)
        self.min_confidence = get_config('vocal_pause_splitting.min_confidence', 0.5)
        self.head_offset = get_config('vocal_pause_splitting.head_offset', -0.5)
        self.tail_offset = get_config('vocal_pause_splitting.tail_offset', 0.5)
        
        # BPM感知自适应增强器
        self.enable_bpm_adaptation = get_config('vocal_pause_splitting.enable_bpm_adaptation', True)
        self.adaptive_enhancer = None
        
        if self.enable_bpm_adaptation and ADAPTIVE_VAD_AVAILABLE:
            try:
                self.adaptive_enhancer = AdaptiveVADEnhancer(sample_rate)
                logger.info("🎵 BPM自适应增强器已启用")
            except Exception as e:
                logger.warning(f"BPM自适应增强器初始化失败: {e}")
                self.enable_bpm_adaptation = False
        else:
            logger.info("使用固定阈值VAD模式")
        
        # 初始化Silero VAD
        self._init_silero_vad()
        
        logger.info("人声停顿检测器初始化完成 (采样率: {})".format(sample_rate))
    
    def _init_silero_vad(self):
        """初始化Silero VAD"""
        try:
            import torch
            torch.set_num_threads(1)
            
            # 下载并加载Silero VAD模型
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            self.vad_model = model
            self.vad_utils = utils
            (self.get_speech_timestamps,
             self.save_audio, self.read_audio,
             self.VADIterator, self.collect_chunks) = utils
            
            logger.info("✅ Silero VAD模型加载成功")
            
        except Exception as e:
            logger.error(f"Silero VAD初始化失败: {e}")
            self.vad_model = None
    
    def detect_vocal_pauses(self, original_audio: np.ndarray) -> List[VocalPause]:
        """检测人声停顿（集成BPM感知自适应增强）
        
        Args:
            original_audio: 原始音频（包含背景音乐）
            
        Returns:
            检测到的人声停顿列表
        """
        logger.info("开始BPM感知的人声停顿检测...")
        
        try:
            if self.vad_model is None:
                logger.error("Silero VAD模型未加载")
                return []
            
            # 存储分析结果用于自适应调整
            complexity_segments = None
            bpm_features = None
            
            # 1. BPM感知复杂度分析（如果启用）
            if self.enable_bpm_adaptation and self.adaptive_enhancer:
                logger.info("执行BPM感知的编曲复杂度分析...")
                complexity_segments, bpm_features = self.adaptive_enhancer.analyze_arrangement_complexity(original_audio)
                
                if complexity_segments and bpm_features:
                    logger.info(f"🎵 音频分析完成: {float(bpm_features.main_bpm):.1f} BPM ({bpm_features.bpm_category})")
                    # 🆕 存储乐器复杂度分析结果用于多乐器环境优化
                    if hasattr(self.adaptive_enhancer, 'last_instrument_analysis'):
                        self.last_complexity_analysis = self.adaptive_enhancer.last_instrument_analysis
                else:
                    logger.warning("复杂度分析失败，使用固定阈值模式")
                    self.enable_bpm_adaptation = False
            
            # 2. 自适应VAD检测语音时间戳
            speech_timestamps = self._detect_adaptive_speech_timestamps(
                original_audio, complexity_segments, bpm_features
            )
            
            # 3. 计算停顿区域（语音片段之间的间隙）
            pause_segments = self._calculate_pause_segments(speech_timestamps, len(original_audio))
            
            # 4. 自适应过滤有效停顿
            valid_pauses = self._filter_adaptive_pauses(pause_segments, complexity_segments, bpm_features)
            
            # 5. 分类停顿位置（头部/中间/尾部）
            vocal_pauses = self._classify_pause_positions(valid_pauses, speech_timestamps, len(original_audio))
            
            # 6. 计算切割点
            vocal_pauses = self._calculate_cut_points(vocal_pauses, bpm_features)
            
            # 7. BPM感知的停顿优化（如果启用）
            if self.enable_bpm_adaptation and bpm_features:
                vocal_pauses = self._optimize_pauses_with_bpm(vocal_pauses, bpm_features)
            
            logger.info(f"检测到 {len(vocal_pauses)} 个有效人声停顿")
            if self.enable_bpm_adaptation and bpm_features:
                logger.info(f"🎵 BPM自适应优化完成 ({bpm_features.bpm_category}音乐)")
            
            return vocal_pauses
            
        except Exception as e:
            logger.error(f"BPM感知人声停顿检测失败: {e}")
            return []
    
    def _detect_speech_timestamps(self, audio: np.ndarray) -> List[Dict]:
        """使用Silero VAD检测语音时间戳
        
        Args:
            audio: 音频数据
            
        Returns:
            语音时间戳列表 [{'start': int, 'end': int}] (样本索引)
        """
        try:
            import torch
            import librosa
            
            # Silero VAD只支持16000Hz，需要重采样
            target_sr = 16000
            if self.sample_rate != target_sr:
                audio_resampled = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr)
            else:
                audio_resampled = audio
            
            # 转换为torch tensor
            audio_tensor = torch.from_numpy(audio_resampled).float()
            
            # 使用Silero VAD检测语音时间戳
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.vad_model,
                sampling_rate=target_sr,
                threshold=self.voice_threshold,
                min_speech_duration_ms=250,  # 降低至250ms检测更短语音片段
                min_silence_duration_ms=int(self.min_pause_duration * 1000),  # 最小静音时长现为400ms
                window_size_samples=512,
                speech_pad_ms=10  # 减少填充提高精度
            )
            
            # 将时间戳映射回原始采样率
            if self.sample_rate != target_sr:
                scale_factor = self.sample_rate / target_sr
                for ts in speech_timestamps:
                    ts['start'] = int(ts['start'] * scale_factor)
                    ts['end'] = int(ts['end'] * scale_factor)
            
            logger.info(f"Silero VAD检测结果: {len(speech_timestamps)} 个语音片段")
            
            # 详细调试信息
            for i, ts in enumerate(speech_timestamps[:10]):  # 只显示前10个
                start_sec = ts['start'] / self.sample_rate
                end_sec = ts['end'] / self.sample_rate
                duration = end_sec - start_sec
                logger.info(f"  语音片段{i+1}: {start_sec:.2f}s - {end_sec:.2f}s (时长: {duration:.2f}s)")
            
            if len(speech_timestamps) > 10:
                logger.info(f"  ... 还有 {len(speech_timestamps)-10} 个语音片段")
            
            return speech_timestamps
            
        except Exception as e:
            logger.error(f"Silero VAD检测失败: {e}")
            return []
    
    def _detect_adaptive_speech_timestamps(
        self, audio: np.ndarray, complexity_segments=None, bpm_features=None
    ) -> List[Dict]:
        """自适应VAD检测语音时间戳（集成BPM感知）
        
        Args:
            audio: 音频数据
            complexity_segments: 编曲复杂度片段（可选）
            bpm_features: BPM特征（可选）
            
        Returns:
            语音时间戳列表
        """
        if not self.enable_bpm_adaptation or not complexity_segments or not bpm_features:
            # 使用固定阈值的原始方法
            return self._detect_speech_timestamps(audio)
        
        try:
            import torch
            import librosa
            
            # 重采样到Silero VAD支持的采样率
            target_sr = 16000
            if self.sample_rate != target_sr:
                audio_resampled = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr)
            else:
                audio_resampled = audio
            
            # 转换为torch tensor
            audio_tensor = torch.from_numpy(audio_resampled).float()
            
            # 使用分段自适应检测
            all_speech_timestamps = []
            
            # 按复杂度片段进行分段检测
            for segment in complexity_segments:
                # 计算当前片段的样本范围
                start_sample = int(segment.start_time * target_sr)
                end_sample = int(min(segment.end_time * target_sr, len(audio_resampled)))
                
                if end_sample <= start_sample:
                    continue
                
                segment_audio = audio_tensor[start_sample:end_sample]
                
                # 获取当前片段的自适应参数
                adaptive_params = self.adaptive_enhancer.get_enhanced_adaptive_vad_params(
                    complexity_segments, bpm_features, (segment.start_time + segment.end_time) / 2
                )
                
                # 使用自适应参数进行VAD检测
                segment_timestamps = self.get_speech_timestamps(
                    segment_audio,
                    self.vad_model,
                    sampling_rate=target_sr,
                    threshold=adaptive_params['voice_threshold'],
                    min_speech_duration_ms=adaptive_params['min_speech_duration_ms'],
                    min_silence_duration_ms=adaptive_params['min_silence_duration_ms'],
                    window_size_samples=512,
                    speech_pad_ms=30
                )
                
                # 将片段时间戳映射回全局时间
                for ts in segment_timestamps:
                    ts['start'] += start_sample
                    ts['end'] += start_sample
                
                all_speech_timestamps.extend(segment_timestamps)
            
            # 合并重叠的时间戳
            all_speech_timestamps = self._merge_overlapping_timestamps(all_speech_timestamps)
            
            # 映射回原始采样率
            if self.sample_rate != target_sr:
                scale_factor = self.sample_rate / target_sr
                for ts in all_speech_timestamps:
                    ts['start'] = int(ts['start'] * scale_factor)
                    ts['end'] = int(ts['end'] * scale_factor)
            
            logger.info(f"🎵 自适应VAD检测完成: {len(all_speech_timestamps)} 个语音片段")
            return all_speech_timestamps
            
        except Exception as e:
            logger.error(f"自适应VAD检测失败: {e}，回退到固定阈值模式")
            return self._detect_speech_timestamps(audio)
    
    def _merge_overlapping_timestamps(self, timestamps: List[Dict]) -> List[Dict]:
        """合并重叠的时间戳"""
        if not timestamps:
            return []
        
        # 按开始时间排序
        timestamps = sorted(timestamps, key=lambda x: x['start'])
        merged = [timestamps[0]]
        
        for current in timestamps[1:]:
            last = merged[-1]
            
            # 如果当前片段与上一个片段重叠或相邻，则合并
            if current['start'] <= last['end'] + 1000:  # 1000样本的容忍度
                last['end'] = max(last['end'], current['end'])
            else:
                merged.append(current)
        
        return merged
    
    def _calculate_pause_segments(self, speech_timestamps: List[Dict], audio_length: int) -> List[Dict]:
        """计算停顿区域
        
        Args:
            speech_timestamps: 语音时间戳
            audio_length: 音频总长度（样本数）
            
        Returns:
            停顿区域列表 [{'start': int, 'end': int}]
        """
        pause_segments = []
        
        if not speech_timestamps:
            # 没有检测到语音，整个音频都是停顿
            pause_segments.append({
                'start': 0,
                'end': audio_length
            })
            return pause_segments
        
        # 头部停顿（音频开始到第一个语音片段）
        if speech_timestamps[0]['start'] > 0:
            pause_segments.append({
                'start': 0,
                'end': speech_timestamps[0]['start']
            })
        
        # 中间停顿（语音片段之间）
        for i in range(len(speech_timestamps) - 1):
            current_end = speech_timestamps[i]['end']
            next_start = speech_timestamps[i + 1]['start']
            
            if next_start > current_end:
                pause_segments.append({
                    'start': current_end,
                    'end': next_start
                })
        
        # 尾部停顿（最后一个语音片段到音频结束）
        if speech_timestamps[-1]['end'] < audio_length:
            pause_segments.append({
                'start': speech_timestamps[-1]['end'],
                'end': audio_length
            })
        
        return pause_segments
    
    def _filter_valid_pauses(self, pause_segments: List[Dict]) -> List[Dict]:
        """过滤有效停顿
        
        Args:
            pause_segments: 停顿区域列表
            
        Returns:
            有效停顿列表
        """
        valid_pauses = []
        min_pause_samples = int(self.min_pause_duration * self.sample_rate)
        
        for pause in pause_segments:
            duration_samples = pause['end'] - pause['start']
            duration_seconds = duration_samples / self.sample_rate
            
            if duration_samples >= min_pause_samples:
                valid_pauses.append({
                    **pause,
                    'duration': duration_seconds
                })
        
        logger.debug(f"过滤后保留 {len(valid_pauses)} 个有效停顿")
        return valid_pauses
    
    def _classify_pause_positions(self, pause_segments: List[Dict], 
                                speech_timestamps: List[Dict], 
                                audio_length: int) -> List[VocalPause]:
        """分类停顿位置
        
        Args:
            pause_segments: 有效停顿列表
            speech_timestamps: 语音时间戳
            audio_length: 音频总长度
            
        Returns:
            分类后的人声停顿列表
        """
        vocal_pauses = []
        
        for pause in pause_segments:
            start_time = pause['start'] / self.sample_rate
            end_time = pause['end'] / self.sample_rate
            duration = pause['duration']
            
            # 判断停顿位置类型
            if pause['start'] == 0:
                # 头部停顿
                position_type = 'head'
            elif pause['end'] == audio_length:
                # 尾部停顿
                position_type = 'tail'
            else:
                # 中间停顿
                position_type = 'middle'
            
            # 计算置信度（基于停顿时长，越长置信度越高）
            confidence = min(1.0, duration / (self.min_pause_duration * 2))
            
            vocal_pause = VocalPause(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                position_type=position_type,
                confidence=confidence,
                cut_point=0.0  # 稍后计算
            )
            
            vocal_pauses.append(vocal_pause)
        
        return vocal_pauses
    
    def _calculate_cut_points(self, vocal_pauses: List[VocalPause], bpm_features: Optional['BPMFeatures'] = None) -> List[VocalPause]:
        """计算精确的切割点位置（BPM自适应）
        
        Args:
            vocal_pauses: 人声停顿列表
            bpm_features: BPM分析特征（用于自适应偏移）
            
        Returns:
            包含切割点的停顿列表
        """
        # 获取BPM自适应偏移（如果启用）
        if self.enable_bpm_adaptation and bpm_features:
            adaptive_head_offset, adaptive_tail_offset = self._get_adaptive_offsets(bpm_features)
        else:
            adaptive_head_offset, adaptive_tail_offset = self.head_offset, self.tail_offset
        
        for pause in vocal_pauses:
            if pause.position_type == 'head':
                # 头部停顿：使用自适应偏移
                pause.cut_point = pause.end_time + adaptive_head_offset
            elif pause.position_type == 'tail':
                # 尾部停顿：使用自适应偏移
                pause.cut_point = pause.start_time + adaptive_tail_offset
            else:  # middle
                # 中间停顿：在停顿中心点切割
                pause.cut_point = (pause.start_time + pause.end_time) / 2
            
            # 确保切割点在有效范围内
            pause.cut_point = max(0, pause.cut_point)
        
        return vocal_pauses
    
    def _filter_adaptive_pauses(self, pause_segments: List[Dict], 
                              complexity_segments: List,
                              bpm_features: 'BPMFeatures') -> List[Dict]:
        """基于BPM特征自适应过滤停顿
        
        Args:
            pause_segments: 停顿区域列表
            bpm_features: BPM分析特征
            
        Returns:
            自适应过滤后的停顿列表
        """
        if not hasattr(self, 'adaptive_enhancer') or not self.adaptive_enhancer:
            return self._filter_valid_pauses(pause_segments)
        
        valid_pauses = []
        
        # 基于BPM和乐器复杂度动态调整最小停顿时长
        if bpm_features.bpm_category == 'slow':
            # 慢歌：允许更短的停顿（歌手有更多时间换气）
            multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.pause_duration_multipliers.slow_song_multiplier', 1.5)
            min_pause_duration = max(0.6, self.min_pause_duration * multiplier)
        elif bpm_features.bpm_category == 'fast':
            # 快歌：需要更长的停顿才认为是真正的停顿
            multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.pause_duration_multipliers.fast_song_multiplier', 0.7)
            min_pause_duration = self.min_pause_duration * multiplier
        else:
            # 中等速度：使用可配置的标准乘数
            multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.pause_duration_multipliers.medium_song_multiplier', 1.0)
            min_pause_duration = self.min_pause_duration * multiplier
            
        # 🆕 多乐器环境增强：根据乐器数量和复杂度进一步调整
        if hasattr(self, 'last_complexity_analysis') and self.last_complexity_analysis:
            complexity = self.last_complexity_analysis.get('total_complexity', 0.0)
            instrument_count = self.last_complexity_analysis.get('instrument_count', 1)
            
            # 乐器越多，需要更长的停顿来确保是真实的人声停顿
            if instrument_count >= 4:  # 4种以上乐器
                base_factor = get_config('vocal_pause_splitting.bpm_adaptive_settings.complexity_multipliers.instrument_4_plus_base', 1.4)
                step_factor = get_config('vocal_pause_splitting.bpm_adaptive_settings.complexity_multipliers.instrument_4_plus_step', 0.1)
                instrument_factor = base_factor + (instrument_count - 4) * step_factor
            elif instrument_count >= 3:  # 3种乐器
                base_factor = get_config('vocal_pause_splitting.bpm_adaptive_settings.complexity_multipliers.instrument_3_base', 1.2)
                complexity_factor = get_config('vocal_pause_splitting.bpm_adaptive_settings.complexity_multipliers.instrument_3_complexity_factor', 0.4)
                instrument_factor = base_factor + (complexity - 0.5) * complexity_factor
            elif instrument_count >= 2:  # 2种乐器
                base_factor = get_config('vocal_pause_splitting.bpm_adaptive_settings.complexity_multipliers.instrument_2_base', 1.1)
                complexity_factor = get_config('vocal_pause_splitting.bpm_adaptive_settings.complexity_multipliers.instrument_2_complexity_factor', 0.2)
                instrument_factor = base_factor + (complexity - 0.3) * complexity_factor
            else:
                instrument_factor = 1.0
                
            min_pause_duration = min_pause_duration * instrument_factor
            logger.info(f"🎸 多乐器调整: {instrument_count}种乐器, 复杂度{float(complexity):.3f}, 系数×{instrument_factor:.2f}")
        
        # 确保不会过度调整
        min_pause_duration = np.clip(min_pause_duration, 0.5, 3.0)
        
        min_pause_samples = int(min_pause_duration * self.sample_rate)
        
        for pause in pause_segments:
            duration_samples = pause['end'] - pause['start']
            duration_seconds = duration_samples / self.sample_rate
            
            # BPM感知的停顿验证
            if duration_samples >= min_pause_samples:
                # 根据节拍强度调整置信度
                confidence = 0.8  # 基础置信度
                
                # 如果停顿时长与节拍周期对齐，提高置信度
                beat_duration = 60.0 / bpm_features.main_bpm if bpm_features.main_bpm > 0 else 1.0
                if abs(duration_seconds % beat_duration) < 0.1 or \
                   abs(duration_seconds % (beat_duration * 2)) < 0.1:
                    confidence += 0.1
                
                valid_pauses.append({
                    **pause,
                    'duration': duration_seconds,
                    'confidence': confidence,
                    'bpm_aligned': abs(duration_seconds % beat_duration) < 0.1
                })
        
        logger.debug(f"BPM自适应过滤后保留 {len(valid_pauses)} 个有效停顿 (BPM: {float(bpm_features.main_bpm):.1f})")
        return valid_pauses
    
    def _get_adaptive_offsets(self, bpm_features: 'BPMFeatures') -> Tuple[float, float]:
        """根据BPM获取动态偏移乘数
        
        Args:
            bpm_features: BPM分析特征
            
        Returns:
            Tuple[head_offset, tail_offset]: 调整后的偏移值
        """
        if bpm_features.bpm_category == 'slow':
            # 慢歌：使用更长的偏移，给歌手更多的停顿缓冲时间
            multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.offset_multipliers.slow_song_offset_multiplier', 1.6)
        elif bpm_features.bpm_category == 'fast':
            # 快歌：使用更短的偏移，保持紧凑的节奏感
            multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.offset_multipliers.fast_song_offset_multiplier', 0.6)
        else:
            # 中速歌：使用标准偏移
            multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.offset_multipliers.medium_song_offset_multiplier', 1.0)
        
        adaptive_head_offset = self.head_offset * multiplier
        adaptive_tail_offset = self.tail_offset * multiplier
        
        logger.debug(f"BPM自适应偏移: {bpm_features.bpm_category}歌, 乘数×{multiplier:.1f}, 偏移({adaptive_head_offset:.2f}s, +{adaptive_tail_offset:.2f}s)")
        
        return adaptive_head_offset, adaptive_tail_offset
    
    def _optimize_pauses_with_bpm(self, vocal_pauses: List[VocalPause], 
                                 bpm_features: 'BPMFeatures') -> List[VocalPause]:
        """使用BPM信息优化停顿切点
        
        Args:
            vocal_pauses: 人声停顿列表
            bpm_features: BPM分析特征
            
        Returns:
            BPM优化后的停顿列表
        """
        if not hasattr(self, 'adaptive_enhancer') or not self.adaptive_enhancer:
            return vocal_pauses
        
        beat_duration = 60.0 / bpm_features.main_bpm if bpm_features.main_bpm > 0 else 1.0
        
        for pause in vocal_pauses:
            original_cut_point = pause.cut_point
            
            # 尝试将切点对齐到最近的节拍点
            if bpm_features.beat_strength > 0.6:  # 节拍较强时才对齐
                # 找到最近的节拍点
                beat_times = []
                current_beat = 0
                while current_beat < pause.end_time + 2:  # 搜索范围扩展
                    beat_times.append(current_beat)
                    current_beat += beat_duration
                
                # 找到最接近当前切点的节拍点
                if beat_times:
                    closest_beat = min(beat_times, key=lambda x: abs(x - original_cut_point))
                    
                    # 如果节拍点在停顿范围内且距离不太远，使用节拍点
                    if (pause.start_time <= closest_beat <= pause.end_time and 
                        abs(closest_beat - original_cut_point) < 0.3):
                        pause.cut_point = closest_beat
                        pause.confidence += 0.05  # 节拍对齐提高置信度
                        logger.debug(f"停顿切点对齐到节拍: {original_cut_point:.2f}s -> {closest_beat:.2f}s")
        
        return vocal_pauses

    def generate_pause_report(self, vocal_pauses: List[VocalPause]) -> Dict:
        """生成停顿检测报告
        
        Args:
            vocal_pauses: 人声停顿列表
            
        Returns:
            报告字典
        """
        if not vocal_pauses:
            return {
                'total_pauses': 0,
                'avg_confidence': 0.0,
                'total_pause_duration': 0.0,
                'pause_types': {'head': 0, 'middle': 0, 'tail': 0}
            }
        
        # 统计停顿类型
        pause_types = {'head': 0, 'middle': 0, 'tail': 0}
        total_duration = 0.0
        total_confidence = 0.0
        
        for pause in vocal_pauses:
            pause_types[pause.position_type] += 1
            total_duration += pause.duration
            total_confidence += pause.confidence
        
        return {
            'total_pauses': len(vocal_pauses),
            'avg_confidence': total_confidence / len(vocal_pauses),
            'total_pause_duration': total_duration,
            'pause_types': pause_types
        }
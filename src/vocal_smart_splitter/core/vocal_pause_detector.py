#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/vocal_pause_detector.py
# AI-SUMMARY: 人声停顿检测器 - 使用Silero VAD直接在原始音频上检测人声停顿

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..utils.config_manager import get_config
from ..utils.adaptive_parameter_calculator import create_adaptive_calculator, AdaptiveParameters

logger = logging.getLogger(__name__)

# 尝试导入自适应增强器
try:
    from .adaptive_vad_enhancer import AdaptiveVADEnhancer
    ADAPTIVE_VAD_AVAILABLE = True
    logger.info("自适应VAD增强器可用")
except ImportError as e:
    logger.warning(f"自适应VAD增强器不可用: {e}")
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
        """初始化人声停顿检测器 (v1.2.0 - BPM自适应增强)

        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate

        # 🆕 v1.2.0: BPM自适应参数计算器
        self.adaptive_calculator = create_adaptive_calculator()

        # 🔄 动态参数（将被AdaptiveParameterCalculator覆盖）
        self.current_adaptive_params: Optional[AdaptiveParameters] = None

        # 静态配置参数（不受BPM影响）
        self.min_confidence = get_config('vocal_pause_splitting.min_confidence', 0.5)
        self.head_offset = get_config('vocal_pause_splitting.head_offset', -0.5)
        self.tail_offset = get_config('vocal_pause_splitting.tail_offset', 0.5)

        # ❌ 以下参数已迁移到动态计算（保留作为fallback）
        self.fallback_min_pause_duration = get_config('vocal_pause_splitting.min_pause_duration', 1.0)
        self.fallback_voice_threshold = get_config('vocal_pause_splitting.voice_threshold', 0.3)

        # 🔄 初始化时使用fallback值（将在检测时被动态参数覆盖）
        self.min_pause_duration = self.fallback_min_pause_duration
        self.voice_threshold = self.fallback_voice_threshold

        # BPM感知自适应增强器
        self.enable_bpm_adaptation = get_config('vocal_pause_splitting.enable_bpm_adaptation', True)
        self.adaptive_enhancer = None

        if self.enable_bpm_adaptation and ADAPTIVE_VAD_AVAILABLE:
            try:
                self.adaptive_enhancer = AdaptiveVADEnhancer(sample_rate)
                logger.info("BPM自适应增强器已启用")
            except Exception as e:
                logger.warning(f"BPM自适应增强器初始化失败: {e}")
                self.enable_bpm_adaptation = False
        else:
            logger.info("使用固定阈值VAD模式")

        # 初始化Silero VAD
        self._init_silero_vad()

        logger.info(f"人声停顿检测器初始化完成 (采样率: {sample_rate}, BPM自适应: {'开启' if self.enable_bpm_adaptation else '关闭'})")

    def apply_adaptive_parameters(self, bpm: float, complexity: float, instrument_count: int):
        """应用BPM自适应参数 (v1.2.0)

        Args:
            bpm: 检测到的BPM值
            complexity: 编曲复杂度 (0-1)
            instrument_count: 乐器数量
        """
        try:
            # 使用AdaptiveParameterCalculator计算参数
            self.current_adaptive_params = self.adaptive_calculator.calculate_all_parameters(
                bpm, complexity, instrument_count
            )

            # 应用动态参数到配置系统
            override_params = self.adaptive_calculator.get_static_override_parameters(
                self.current_adaptive_params
            )
            self.adaptive_calculator.apply_dynamic_parameters(
                self.current_adaptive_params, override_params
            )

            # 更新实例变量（用于直接访问）
            self.min_pause_duration = self.current_adaptive_params.min_pause_duration
            self.voice_threshold = self.current_adaptive_params.vad_threshold

            logger.info("=== BPM自适应参数已应用 ===")
            logger.info(f"BPM: {self.current_adaptive_params.bpm_value} ({self.current_adaptive_params.category})")
            logger.info(f"停顿时长: {self.current_adaptive_params.min_pause_duration:.3f}s")
            logger.info(f"VAD阈值: {self.current_adaptive_params.vad_threshold:.3f}")
            logger.info(f"补偿系数: {self.current_adaptive_params.compensation_factor:.3f}")

            return True

        except Exception as e:
            logger.error(f"应用BPM自适应参数失败: {e}")
            # 使用fallback参数
            self.min_pause_duration = self.fallback_min_pause_duration
            self.voice_threshold = self.fallback_voice_threshold
            return False

    def get_current_parameters_info(self) -> Dict:
        """获取当前参数信息（用于调试和监控）"""
        if self.current_adaptive_params:
            return {
                'mode': 'adaptive',
                'bpm': self.current_adaptive_params.bpm_value,
                'category': self.current_adaptive_params.category,
                'min_pause_duration': self.current_adaptive_params.min_pause_duration,
                'vad_threshold': self.current_adaptive_params.vad_threshold,
                'compensation_factor': self.current_adaptive_params.compensation_factor,
                'complexity_score': self.current_adaptive_params.complexity_score,
                'instrument_count': self.current_adaptive_params.instrument_count
            }
        else:
            return {
                'mode': 'fallback',
                'min_pause_duration': self.fallback_min_pause_duration,
                'vad_threshold': self.fallback_voice_threshold
            }

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

            logger.info("Silero VAD模型加载成功")

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

            # 4.b 可选：在“长语音段”内按网格扫描 valley 生成合成停顿（默认关闭，零破坏）
            try:
                enable_voiced_valley = bool(get_config('vocal_pause_splitting.voiced_valley_fallback.enable', False))
            except Exception:
                enable_voiced_valley = False
            if enable_voiced_valley and original_audio is not None and len(original_audio) > 0 and speech_timestamps:
                try:
                    sr = self.sample_rate
                    local_rms_ms = int(get_config('vocal_pause_splitting.local_rms_window_ms', 25))
                    floor_pct = float(get_config('vocal_pause_splitting.silence_floor_percentile', 5))
                    min_gap_s = float(get_config('vocal_pause_splitting.voiced_valley_fallback.min_gap_s', 6.0))
                    synth_window_s = float(get_config('vocal_pause_splitting.voiced_valley_fallback.window_s', 0.30))
                    half_win = max(0.05, synth_window_s / 2.0)
                    added = 0
                    for ts in speech_timestamps:
                        seg_start_s = ts['start'] / sr
                        seg_end_s = ts['end'] / sr
                        seg_dur = seg_end_s - seg_start_s
                        if seg_dur >= max(min_gap_s * 1.2, min_gap_s + 1.0):
                            # 网格中心从 half-step 开始，覆盖左侧半个间隔，避免漏掉段首的明显谷
                            step = float(min_gap_s)
                            center = seg_start_s + step * 0.5
                            while center <= (seg_end_s - step * 0.5):
                                l_idx = max(ts['start'], int((center - 0.50) * sr))
                                r_idx = min(ts['end'],   int((center + 0.50) * sr))
                                if r_idx - l_idx <= int(0.10 * sr):
                                    center += step
                                    continue
                                v_idx = self._select_valley_cut_point(original_audio, l_idx, r_idx, sr, local_rms_ms, -1, floor_pct)
                                if v_idx is not None:
                                    cp = v_idx / float(sr)
                                    s = max(seg_start_s, cp - half_win)
                                    e = min(seg_end_s,   cp + half_win)
                                    if (e - s) >= 0.08:
                                        # 标记为强制 valley，以在切点阶段跳过“中心右偏+宽半径零交叉”的漂移
                                        valid_pauses.append({'start': int(s * sr), 'end': int(e * sr), 'duration': (e - s), 'force_valley': True})
                                        added += 1
                                center += step
                    if added > 0:
                        logger.info(f"voiced_valley_fallback: [32m+{added}[0m synthetic pauses added within voiced segments")
                except Exception as _e:
                    logger.warning(f"voiced_valley_fallback failed: {_e}")

            # 5. 分类停顿位置（头部/中间/尾部）
            vocal_pauses = self._classify_pause_positions(valid_pauses, speech_timestamps, len(original_audio))

            # 6. 计算切割点（静音平台中心+右偏+零交叉吸附）
            vocal_pauses = self._calculate_cut_points(vocal_pauses, bpm_features, original_audio)

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
            # ✅ --- 核心修复：从config实时读取所有VAD参数 ---
            logger.debug("实时从config加载VAD参数...")
            
            # 从 'advanced_vad' 部分读取，这是我们新的“黄金参数”存放地
            vad_threshold = get_config('advanced_vad.silero_prob_threshold_down', 0.35)
            min_speech_ms = get_config('advanced_vad.silero_min_speech_ms', 250)
            min_silence_ms = get_config('advanced_vad.silero_min_silence_ms', 700)
            window_size = get_config('advanced_vad.silero_window_size_samples', 512)
            pad_ms = get_config('advanced_vad.silero_speech_pad_ms', 150)

            logger.info(f"应用VAD参数: threshold={vad_threshold}, min_speech={min_speech_ms}ms, min_silence={min_silence_ms}ms")
            
            # 使用Silero VAD检测语音时间戳
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.vad_model,
                sampling_rate=target_sr,
                threshold=vad_threshold,           # ✅ 使用实时读取的阈值
                min_speech_duration_ms=min_speech_ms, # ✅ 使用实时读取的参数
                min_silence_duration_ms=min_silence_ms, # ✅ 使用实时读取的参数
                window_size_samples=window_size,     # ✅ 使用实时读取的参数
                speech_pad_ms=pad_ms                 # ✅ 使用实时读取的参数
            )

            # 将时间戳映射回原始采样率（使用正确的跨域映射）
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

            # 映射回原始采样率（使用正确的跨域映射）
            if self.sample_rate != target_sr:
                from ..utils.audio_processor import map_time_between_domains
                # 获取重采样延迟（如果配置中有）
                latency_samples = int(get_config('time_mapping.latency_samples', 0))
                
                for ts in all_speech_timestamps:
                    # 转换为秒
                    start_sec = ts['start'] / target_sr
                    end_sec = ts['end'] / target_sr
                    
                    # 映射到原始采样率域
                    start_sec_mapped = map_time_between_domains(
                        start_sec, target_sr, self.sample_rate, latency_samples
                    )
                    end_sec_mapped = map_time_between_domains(
                        end_sec, target_sr, self.sample_rate, latency_samples
                    )
                    
                    # 转换回样本
                    ts['start'] = int(start_sec_mapped * self.sample_rate)
                    ts['end'] = int(end_sec_mapped * self.sample_rate)

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
            # 透传合成停顿的“强制 valley”标记（不修改数据结构，动态属性即可）
            try:
                if isinstance(pause, dict) and pause.get('force_valley', False):
                    setattr(vocal_pause, 'force_valley', True)
            except Exception:
                pass

            vocal_pauses.append(vocal_pause)

        return vocal_pauses

    def _compute_rms_envelope(self, waveform: np.ndarray, win_samples: int) -> np.ndarray:
        """计算简易滑动RMS包络（same对齐）。"""
        if win_samples <= 1:
            return np.abs(waveform).astype(np.float32)
        kernel = np.ones(int(win_samples), dtype=np.float32) / float(max(1, int(win_samples)))
        return np.sqrt(np.convolve((waveform.astype(np.float32) ** 2), kernel, mode='same'))

    def _future_silence_guard(self, rms: np.ndarray, start_idx: int, guard_samples: int, floor_val: float,
                               allowance: float = 1.2, ratio: float = 0.7) -> bool:
        """未来静默守卫：在 [start_idx, start_idx+guard] 内多数样本低于 floor×allowance。"""
        end_idx = min(start_idx + max(0, int(guard_samples)), len(rms))
        if end_idx <= start_idx:
            return False
        window = rms[start_idx:end_idx]
        if window.size == 0:
            return False
        below = np.sum(window <= (floor_val * allowance))
        return (below / float(window.size)) >= ratio

    def _select_valley_cut_point(self, waveform: np.ndarray, left_idx: int, right_idx: int,
                                 sample_rate: int, local_rms_ms: int, guard_ms: int, floor_percentile: float) -> Optional[int]:
        """在[left_idx, right_idx]内选择RMS谷值切点，带未来静默守卫；若失败返回None。"""
        left_idx = max(0, int(left_idx)); right_idx = min(len(waveform), int(right_idx))
        if right_idx - left_idx <= 8:
            return None
        win_samples = max(1, int(local_rms_ms / 1000.0 * sample_rate))
        guard_samples = max(1, int(guard_ms / 1000.0 * sample_rate)) if guard_ms and guard_ms > 0 else 0
        segment = waveform[left_idx:right_idx]
        rms = self._compute_rms_envelope(segment, win_samples)
        floor_val = np.percentile(np.abs(segment), floor_percentile)
        # 候选按RMS升序尝试，优先更“静”的点
        order = np.argsort(rms)
        max_try = min(200, len(order))
        # 边界保护：至少距左右边界各20ms，避免贴边切
        margin_samples = max(1, int(0.02 * sample_rate))
        # 谷宽/坡度约束参数
        try:
            min_valley_ms = int(get_config('vocal_pause_splitting.min_valley_width_ms', 120))
        except Exception:
            min_valley_ms = 120
        valley_half = max(1, int((min_valley_ms / 1000.0) * sample_rate / 2))
        edge_band = max(1, int(0.01 * sample_rate))  # 10ms 边带用于评估两侧上坡
        slope_ratio = 1.15  # 两侧应当明显高于谷底

        candidates = []
        for k in range(max_try):
            j = int(order[k])
            if j < margin_samples or j > (len(rms) - margin_samples):
                continue
            # 谷宽与坡度约束（仅对 valley 路径）：确保 j 周围 min_valley_ms 范围内为“真谷”
            left_edge = max(0, j - valley_half)
            right_edge = min(len(rms) - 1, j + valley_half)
            if right_edge - left_edge < 3:
                continue
            window = rms[left_edge:right_edge + 1]
            wmin = float(np.min(window))
            if float(rms[j]) > (wmin + 1e-12):
                # j 必须是该窗口内的（或近似）最小点
                continue
            # 两侧上坡：两侧边带平均应显著高于谷底
            left_band_vals = rms[left_edge:min(len(rms), left_edge + edge_band)]
            right_band_vals = rms[max(0, right_edge - edge_band):right_edge]
            if left_band_vals.size == 0 or right_band_vals.size == 0:
                continue
            left_mean = float(np.mean(left_band_vals))
            right_mean = float(np.mean(right_band_vals))
            if not (left_mean >= slope_ratio * float(rms[j]) and right_mean >= slope_ratio * float(rms[j])):
                continue

            if guard_samples > 0:
                if self._future_silence_guard(rms, j, guard_samples, floor_val):
                    return left_idx + j
            else:
                # 在 valley 强制模式下暂不启用未来守卫，先收集候选谷
                candidates.append(j)
        if guard_samples == 0 and len(candidates) > 0:
            # 使用简单谱特征为谷打分：flatness + 0.3*centroid_norm + 0.3*unvoiced
            # 气声/摩擦音：平坦度高、质心偏高、无基音（自相关峰低）
            def _score(j_idx: int) -> float:
                half = max(1, int(0.02 * sample_rate))  # 20ms 半窗
                s0 = max(0, j_idx - half)
                s1 = min(len(segment), j_idx + half)
                w = segment[s0:s1]
                if w.size < 8:
                    return -1e9
                win = np.hanning(w.size)
                spec = np.abs(np.fft.rfft(w * win)) + 1e-12
                geo = np.exp(np.mean(np.log(spec)))
                arith = np.mean(spec) + 1e-12
                flat = float(geo / arith)
                freqs = np.fft.rfftfreq(w.size, d=1.0 / float(sample_rate))
                centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
                centroid_norm = centroid / (0.5 * float(sample_rate) + 1e-12)
                # 简化 voicing：自相关法，排除0滞后，取≤20ms范围内的峰
                w_zm = w - float(np.mean(w))
                ac = np.correlate(w_zm, w_zm, mode='full')
                ac = ac[ac.size // 2:]
                if ac.size > 1 and ac[0] > 0:
                    maxlag = min(len(ac) - 1, int(0.02 * sample_rate))
                    peak = float(np.max(ac[1:maxlag + 1] / (ac[0] + 1e-12))) if maxlag >= 1 else 0.0
                else:
                    peak = 0.0
                unvoiced = 1.0 - max(0.0, min(1.0, peak))
                return float(flat + 0.3 * centroid_norm + 0.3 * unvoiced)
            best_j = max(candidates, key=_score)
            return left_idx + int(best_j)
        # 兜底：选择满足边界保护的RMS最小点；若均不满足则夹紧到边界内
        if order.size > 0:
            for jj in order:
                j = int(jj)
                if j >= margin_samples and j <= (len(rms) - margin_samples):
                    return left_idx + j
            j = int(order[0])
            j = max(margin_samples, min(len(rms) - margin_samples, j))
            return left_idx + j
        return None

    def _calculate_cut_points(self, vocal_pauses: List[VocalPause], bpm_features: Optional['BPMFeatures'] = None, waveform: Optional[np.ndarray] = None) -> List[VocalPause]:
            """
            计算精确的切割点位置（强制能量谷检测）
            """
            # 读取切点精修配置
            max_shift_s = float(get_config('vocal_pause_splitting.max_shift_from_silence_center', 0.08))
            backoff_ms = int(get_config('vocal_pause_splitting.boundary_backoff_ms', 180))
            backoff_s = backoff_ms / 1000.0
            local_rms_ms = int(get_config('vocal_pause_splitting.local_rms_window_ms', 25))
            floor_pct = float(get_config('vocal_pause_splitting.silence_floor_percentile', 5))
            guard_ms = int(get_config('vocal_pause_splitting.lookahead_guard_ms', 120))

            logger.info(f"计算 {len(vocal_pauses)} 个停顿的切割点 (强制能量谷检测模式)...")

            for i, pause in enumerate(vocal_pauses):
                # ✅ --- 关键修复：恢复 left 和 right 变量的定义 ---
                # 默认搜索范围是整个停顿区域，并向内收缩一个边界缓冲
                left = pause.start_time + backoff_s
                right = pause.end_time - backoff_s
                # 如果收缩后范围无效，则使用原始停顿范围
                if right <= left:
                    left, right = pause.start_time, pause.end_time
                # ✅ --- 修复结束 ---

                selected_idx: Optional[int] = None

                # 全面采用能量谷检测逻辑
                if waveform is not None and len(waveform) > 0:
                    l_idx = max(0, int(left * self.sample_rate))
                    r_idx = min(len(waveform), int(right * self.sample_rate))

                    if r_idx > l_idx:
                        # 强制使用能量谷检测
                        valley_idx = self._select_valley_cut_point(
                            waveform, l_idx, r_idx, self.sample_rate,
                            local_rms_ms, guard_ms, floor_pct
                        )

                        if valley_idx is not None:
                            selected_idx = valley_idx
                            logger.debug(f"停顿 {i+1}: 强制使用 valley 切点 idx={selected_idx}")
                        else:
                            # 如果找不到能量谷（极少见），回退到停顿中心
                            selected_idx = int((pause.start_time + pause.end_time) / 2 * self.sample_rate)
                            logger.warning(f"停顿 {i+1}: 未找到能量谷，回退到中心点")
                    else:
                        # 如果搜索范围无效，也回退到中心点
                        selected_idx = int((pause.start_time + pause.end_time) / 2 * self.sample_rate)
                else:
                    # 如果没有波形数据，同样回退到中心点
                    selected_idx = int((pause.start_time + pause.end_time) / 2 * self.sample_rate)

                # 将最终选择的样本索引转换为时间
                pause.cut_point = selected_idx / self.sample_rate
                logger.info(f"停顿 {i+1} ({pause.position_type}): {pause.start_time:.2f}s-{pause.end_time:.2f}s → 切点: {pause.cut_point:.2f}s")

            return vocal_pauses
    
    # Removed dead code that was unreachable after return statement
    
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
        # 关键修复：当未启用BPM自适应、未初始化增强器，或bpm特征不可用时，回退到固定阈值过滤
        if (not getattr(self, 'enable_bpm_adaptation', False)) or (not hasattr(self, 'adaptive_enhancer')) or (not self.adaptive_enhancer) or (bpm_features is None):
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

        # 🆕 第一遍：计算所有中间停顿的平均时长（排除头尾停顿）
        middle_pause_durations = []
        # 从最后一个停顿推断音频总长度
        total_audio_length = pause_segments[-1]['end'] if pause_segments else 0

        for i, pause in enumerate(pause_segments):
            duration_samples = pause['end'] - pause['start']
            duration_seconds = duration_samples / self.sample_rate

            # 只统计中间停顿，排除头尾停顿
            is_head = (i == 0 and pause['start'] == 0)
            is_tail = (i == len(pause_segments) - 1 and pause['end'] >= total_audio_length * 0.95)  # 允许5%的误差

            if duration_samples >= min_pause_samples and not is_head and not is_tail:
                middle_pause_durations.append(duration_seconds)

        logger.info(f"中间停顿统计: 总停顿{len(pause_segments)}个, 中间停顿{len(middle_pause_durations)}个")

        if not middle_pause_durations:
            logger.warning("没有找到符合要求的中间停顿，回退到所有停顿")
            # 回退策略：使用所有符合最小时长的停顿
            all_pause_durations = []
            for pause in pause_segments:
                duration_samples = pause['end'] - pause['start']
                duration_seconds = duration_samples / self.sample_rate
                if duration_samples >= min_pause_samples:
                    all_pause_durations.append(duration_seconds)
            if not all_pause_durations:
                return []
            middle_pause_durations = all_pause_durations

        # 计算中间停顿时长统计
        average_pause_duration = np.mean(middle_pause_durations)
        median_pause_duration = np.median(middle_pause_durations)
        std_pause_duration = np.std(middle_pause_durations)

        logger.info(f"停顿时长统计: 平均={average_pause_duration:.3f}s, 中位={median_pause_duration:.3f}s, 标准差={std_pause_duration:.3f}s")

        # 动态阈值：使用平均值和中位数的较大者作为基准
        duration_threshold = max(average_pause_duration, median_pause_duration)

        # 对于变化较大的停顿分布，适当降低阈值
        if std_pause_duration > average_pause_duration * 0.5:
            duration_threshold = average_pause_duration * 0.8  # 降低20%
            logger.info(f"检测到高变异性停顿分布，降低阈值至 {duration_threshold:.3f}s")

        # 🆕 第二遍：基于平均值筛选分割点
        valid_pauses = []

        for pause in pause_segments:
            duration_samples = pause['end'] - pause['start']
            duration_seconds = duration_samples / self.sample_rate

            # 基础时长检查
            if duration_samples < min_pause_samples:
                continue

            # 边界停顿放宽：头/尾仅需满足最小停顿时长；中间停顿需 ≥ 动态阈值
            is_head = (pause.get('start', 0) == 0)
            is_tail = (pause.get('end', 0) >= total_audio_length * 0.95)

            if is_head or is_tail:
                duration_ratio = duration_seconds / max(average_pause_duration, 1e-6)
                valid_pauses.append({
                    **pause,
                    'duration': duration_seconds,
                    'confidence': 0.75,  # 边界停顿基础置信度
                    'bpm_aligned': False,
                    'duration_ratio': duration_ratio
                })
                logger.debug(f"边界停顿保留: {duration_seconds:.3f}s (head={is_head}, tail={is_tail})")
                continue

            # 🎯 中间停顿：只选择时长≥阈值的停顿作为分割点
            if duration_seconds >= duration_threshold:
                # 根据节拍强度调整置信度
                confidence = 0.8  # 基础置信度
                beat_duration = 60.0 / bpm_features.main_bpm if (bpm_features and getattr(bpm_features, 'main_bpm', 0) > 0) else 1.0
                if abs(duration_seconds % beat_duration) < 0.1 or abs(duration_seconds % (beat_duration * 2)) < 0.1:
                    confidence += 0.1

                duration_ratio = duration_seconds / max(average_pause_duration, 1e-6)
                if duration_ratio >= 1.5:
                    confidence += 0.1

                valid_pauses.append({
                    **pause,
                    'duration': duration_seconds,
                    'confidence': confidence,
                    'bpm_aligned': abs(duration_seconds % beat_duration) < 0.1,
                    'duration_ratio': duration_ratio
                })
                logger.debug(f"选择停顿: {duration_seconds:.3f}s (比例: {duration_ratio:.2f}x)")
            else:
                logger.debug(f"跳过短停顿: {duration_seconds:.3f}s < {duration_threshold:.3f}s")

        logger.info(f"平均值筛选完成: {len(middle_pause_durations)}个候选 → {len(valid_pauses)}个分割点 (阈值: {duration_threshold:.3f}s)")
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
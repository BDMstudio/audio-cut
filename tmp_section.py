#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/pure_vocal_pause_detector.py
# AI-SUMMARY: ������ͣ�ټ���� - ����MDX23/Demucs�����Ĵ��������ж�ά���������������Ƶ������������

import numpy as np
import librosa
import logging
from typing import List, Dict, Tuple, Optional, Sequence
from dataclasses import dataclass
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from audio_cut.analysis import TrackFeatureCache

from ..utils.config_manager import get_config

logger = logging.getLogger(__name__)

@dataclass
class VocalFeatures:
    """�����������ݽṹ"""
    f0_contour: np.ndarray           # ��Ƶ�켣
    f0_confidence: np.ndarray        # ��Ƶ���Ŷ�
    formant_energies: List[np.ndarray]  # �������������
    spectral_centroid: np.ndarray    # Ƶ������
    harmonic_ratio: np.ndarray       # г������
    zero_crossing_rate: np.ndarray   # ������
    rms_energy: np.ndarray           # RMS����

@dataclass
class PureVocalPause:
    """������ͣ�ٽṹ"""
    start_time: float
    end_time: float
    duration: float
    pause_type: str  # 'true_pause', 'breath', 'uncertain'
    confidence: float
    features: Dict  # ��ϸ������Ϣ
    cut_point: float = 0.0  # ����и�㣨������
    quality_grade: str = 'B'  # �����ȼ���������
    is_valid: bool = True   # �Ƿ���Ч��������
    
class PureVocalPauseDetector:
    """���ڴ������Ķ�ά����ͣ�ټ����
    
    ���Ĵ��£�
    1. F0�����Է��� - ����Ƶͻ��ʶ����ͣ��
    2. ����������ֲ� - ���ֻ���vs����
    3. Ƶ������׷�� - ʶ���Ƶ˥��ģʽ
    4. г��ǿ�ȷ��� - ������������
    """
    
    def __init__(self, sample_rate: int = 44100):
        """��ʼ��������ͣ�ټ����
        
        Args:
            sample_rate: ������
        """
        self.sample_rate = sample_rate
        
        # �����ü��ز���
        self.min_pause_duration = get_config('pure_vocal_detection.min_pause_duration', 0.5)
        self.breath_duration_range = get_config('pure_vocal_detection.breath_duration_range', [0.1, 0.3])
        self.f0_weight = get_config('pure_vocal_detection.f0_weight', 0.3)
        self.formant_weight = get_config('pure_vocal_detection.formant_weight', 0.25)
        self.spectral_weight = get_config('pure_vocal_detection.spectral_weight', 0.25)
        self.duration_weight = get_config('pure_vocal_detection.duration_weight', 0.2)

        self._last_feature_cache: Optional[TrackFeatureCache] = None
        self._last_focus_windows: List[Tuple[float, float]] = []
        
        # �����ֵ
        self.energy_threshold_db = get_config('pure_vocal_detection.energy_threshold_db', -40)
        self.f0_drop_threshold = get_config('pure_vocal_detection.f0_drop_threshold', 0.7)
        self.breath_confidence_threshold = get_config('pure_vocal_detection.breath_filter_threshold', 0.3)
        self.pause_confidence_threshold = get_config('pure_vocal_detection.pause_confidence_threshold', 0.7)
        
        # ��������
        self.hop_length = int(sample_rate * 0.01)  # 10ms hop
        self.frame_length = int(sample_rate * 0.025)  # 25ms frame
        self.n_fft = 2048
        
        # ?? �ؼ��޸�������VocalPauseDetectorV2�������ȼ������
        from .vocal_pause_detector import VocalPauseDetectorV2
        self._cut_point_calculator = VocalPauseDetectorV2(sample_rate)
        
        logger.info(f"������ͣ�ټ������ʼ����� (������: {sample_rate}) - �Ѽ����������е����")
    
    def detect_pure_vocal_pauses(self, vocal_audio: np.ndarray,
                                enable_mdd_enhancement: bool = False,
                                original_audio: Optional[np.ndarray] = None,
                                feature_cache: Optional[TrackFeatureCache] = None,
                                vad_segments: Optional[List[Dict[str, float]]] = None) -> List[PureVocalPause]:
        """��ⴿ�����е�ͣ��
        
        Args:
            vocal_audio: �����Ĵ�������Ƶ
            original_audio: ԭʼ����(��ѡ�����ڶԱ�)
            feature_cache: ����������棬�ɸ���BPM/MDD��ȫ��ָ��
            
        Returns:
            ��⵽��ͣ���б�
        """
        logger.info(f"��ʼ������ͣ�ټ��... (MDD��ǿ: {enable_mdd_enhancement})")

        if isinstance(feature_cache, TrackFeatureCache) and feature_cache.sr == self.sample_rate and feature_cache.frame_count() > 0:
            cache = feature_cache
        else:
            cache = None
        if cache is not None:
            self._last_feature_cache = cache
        feature_cache = cache

        focus_windows: Optional[List[Tuple[float, float]]] = None
        if vad_segments:
            pad_cfg = get_config('advanced_vad.focus_window_pad_s', 0.2)
            min_width_cfg = get_config('advanced_vad.focus_window_min_width_s', 0.0)
            focus_windows = self._focus_windows_from_vad_segments(
                vad_segments,
                pad_s=float(pad_cfg),
                min_width_s=float(min_width_cfg),
            )
        elif cache is not None:
            focus_windows = self._compute_focus_windows(vocal_audio)
        self._last_focus_windows = list(focus_windows or [])

        # ?? �ؼ��޸���������������ȼ��
        enable_relative_mode = get_config('pure_vocal_detection.enable_relative_energy_mode', False)
        if enable_relative_mode:
            logger.info("ʹ����������ȼ��ģʽ...")
            peak_ratio = get_config('pure_vocal_detection.peak_relative_threshold_ratio', 0.1)
            rms_ratio = get_config('pure_vocal_detection.rms_relative_threshold_ratio', 0.05)
            # BPM/MDD ����Ӧ���ʣ����������ģʽ�����ã�
            if get_config('pure_vocal_detection.relative_threshold_adaptation.enable', True):
                ref_audio = original_audio if original_audio is not None else vocal_audio
                bpm_cfg = get_config('vocal_pause_splitting.bpm_adaptive_settings', {})
                slow_thr = bpm_cfg.get('slow_bpm_threshold', 80)
                fast_thr = bpm_cfg.get('fast_bpm_threshold', 120)
                bpm_mul_slow = get_config('pure_vocal_detection.relative_threshold_adaptation.bpm.slow_multiplier', 1.10)
                bpm_mul_med = get_config('pure_vocal_detection.relative_threshold_adaptation.bpm.medium_multiplier', 1.00)
                bpm_mul_fast = get_config('pure_vocal_detection.relative_threshold_adaptation.bpm.fast_multiplier', 0.85)

                tempo = 0.0
                bpm_tag = 'unknown'
                if feature_cache and feature_cache.bpm_features is not None:
                    bpm_info = feature_cache.bpm_features
                    tempo = float(getattr(bpm_info, 'main_bpm', 0.0) or 0.0)
                    bpm_tag = getattr(bpm_info, 'bpm_category', 'unknown') or 'unknown'
                else:
                    try:
                        tempo_est, _ = librosa.beat.beat_track(y=ref_audio, sr=self.sample_rate)
                        tempo = float(np.squeeze(np.asarray(tempo_est))) if tempo_est is not None else 0.0
                    except Exception:
                        tempo = 0.0

                if tempo > 0:
                    if tempo < slow_thr:
                        mul_bpm = bpm_mul_slow
                        if bpm_tag == 'unknown':
                            bpm_tag = 'slow'
                    elif tempo > fast_thr:
                        mul_bpm = bpm_mul_fast
                        if bpm_tag == 'unknown':
                            bpm_tag = 'fast'
                    else:
                        mul_bpm = bpm_mul_med
                        if bpm_tag == 'unknown':
                            bpm_tag = 'medium'
                else:
                    mul_bpm = bpm_mul_med

                def _mdd_score_simple(x, sr):
                    try:
                        rms = librosa.feature.rms(y=x, hop_length=512)[0]
                        flat = librosa.feature.spectral_flatness(y=x)[0]
                        onset_env = librosa.onset.onset_strength(y=x, sr=sr, hop_length=512)
                        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512)
                        dur = max(0.1, len(x) / sr)
                        onset_rate = len(onsets) / dur

                        def nz(v):
                            v = np.asarray(v)
                            q10, q90 = np.quantile(v, 0.1), np.quantile(v, 0.9)
                            if q90 - q10 < 1e-9:
                                return 0.0
                            return float(np.clip((np.mean(v) - q10) / (q90 - q10), 0, 1))

                        rms_s, flat_s = nz(rms), nz(flat)
                        onset_s = float(np.clip(onset_rate / 10.0, 0, 1))
                        return float(np.clip(0.5 * rms_s + 0.3 * flat_s + 0.2 * onset_s, 0, 1))
                    except Exception:
                        return 0.5

                if feature_cache and feature_cache.global_mdd is not None:
                    mdd_s = float(np.clip(feature_cache.global_mdd, 0.0, 1.0))
                else:
                    mdd_s = _mdd_score_simple(ref_audio, self.sample_rate)

                mdd_base = get_config('pure_vocal_detection.relative_threshold_adaptation.mdd.base', 1.0)
                mdd_gain = get_config('pure_vocal_detection.relative_threshold_adaptation.mdd.gain', 0.2)
                mul_mdd = mdd_base + (0.1 - mdd_gain * mdd_s)
                clamp_min = get_config('pure_vocal_detection.relative_threshold_adaptation.clamp_min', 0.75)
                clamp_max = get_config('pure_vocal_detection.relative_threshold_adaptation.clamp_max', 1.25)
                mul = float(np.clip(mul_bpm * mul_mdd, clamp_min, clamp_max))
                peak_ratio *= mul
                rms_ratio *= mul
                logger.info(
                    "�����ֵ����Ӧ��BPM=%.1f(%s), MDD=%.2f, mul=%.2f �� peak=%.3f, rms=%.3f",
                    tempo,
                    bpm_tag,
                    mdd_s,
                    mul,
                    peak_ratio,
                    rms_ratio,
                )
            
            # ʹ����������ȼ��
            try:
                if get_config('pure_vocal_detection.pause_stats_adaptation.enable', True):
                    mul_pause, vpp_log = self._estimate_vpp_multiplier(vocal_audio, focus_windows)
                    clamp_min = get_config('pure_vocal_detection.pause_stats_adaptation.clamp_min', 0.75)
                    clamp_max = get_config('pure_vocal_detection.pause_stats_adaptation.clamp_max', 1.25)
                    mul_pause = float(np.clip(mul_pause, clamp_min, clamp_max))
                    peak_ratio *= mul_pause; rms_ratio *= mul_pause
                    logger.info(f"VPP����Ӧ��{vpp_log}, mul_pause={mul_pause:.2f} �� peak={peak_ratio:.3f}, rms={rms_ratio:.3f}")
            except Exception as e:
                logger.warning(f"VPP����Ӧʧ�ܣ����ԣ���{e}")
            filtered_pauses = self._detect_energy_valleys(vocal_audio, peak_ratio, rms_ratio, focus_windows=focus_windows)
            # VPP�����ϲ�����ͣ�����ɸ���ޣ���ֹ��ѡ��ը
            try:
                filtered_pauses = self._compress_pauses(filtered_pauses)
            except Exception:
                pass
            # VPP����޶���total_valley = ����ʱ�� / segment_min_duration
            try:
                duration_s = float(len(vocal_audio)) / float(self.sample_rate)
                filtered_pauses = self._apply_total_valley_cap(filtered_pauses, duration_s)
            except Exception as e:
                logger.warning(f"VPP����޶�Ӧ��ʧ�ܣ����ԣ���{e}")
        else:
            # ԭ�еĶ�ά�����������
            # 1. ��ȡ��ά����
            features = self._extract_vocal_features(vocal_audio)
            
            # 2. ����ѡͣ������
            candidate_pauses = self._detect_candidate_pauses(features)
            
            # 3. �����ںϷ���
            analyzed_pauses = self._analyze_pause_features(candidate_pauses, features, vocal_audio)
            
            # 4. �������
            filtered_pauses = self._classify_and_filter(analyzed_pauses)
            
        # 5. MDD��ǿ����
        if enable_mdd_enhancement and (original_audio is not None or feature_cache is not None):
            logger.info("Ӧ��MDD��ǿ����...")
            filtered_pauses = self._apply_mdd_enhancement(filtered_pauses, original_audio, feature_cache=feature_cache, focus_windows=focus_windows)
        
        # ?? �ؼ��޸���ʹ��VocalPauseDetectorV2���㾫ȷ�е�
        if filtered_pauses and vocal_audio is not None:
            filtered_pauses = self._calculate_precise_cut_points(filtered_pauses, vocal_audio, feature_cache=feature_cache)
        
        logger.info(f"������: {len(filtered_pauses)}��������ͣ�ٵ�")
        return filtered_pauses
    

    def _focus_windows_from_vad_segments(
        self,
        segments: Sequence[Dict[str, float]],
        *,
        pad_s: float = 0.2,
        min_width_s: float = 0.0,
    ) -> List[Tuple[float, float]]:
        """Build focus windows directly from Silero VAD timeline."""
        if not segments:
            return []
        pad = max(0.0, float(pad_s))
        min_width = max(0.0, float(min_width_s))
        merge_gap = float(get_config('advanced_vad.focus_merge_gap_s', 0.12))
        windows: List[Tuple[float, float]] = []
        track_end = 0.0
        for seg in segments:
            try:
                start = float(seg.get('start', seg.get('start_time', 0.0)))
                end = float(seg.get('end', seg.get('end_time', start)))
            except Exception:
                continue
            if end <= start:
                continue
            track_end = max(track_end, end + pad)
            left = max(0.0, start - pad)
            right = max(left, end + pad)
            windows.append((left, right))
        if not windows:
            return []
        if track_end <= 0.0:
            track_end = max(end for _, end in windows)
        clipped = [(start, min(track_end, end)) for start, end in windows]
        clipped.sort(key=lambda item: item[0])
        merged: List[Tuple[float, float]] = []
        for start, end in clipped:
            if not merged:
                merged.append((start, end))
                continue
            if start - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return self._merge_windows(merged, min_width=min_width)

    def _compute_focus_windows(self, vocal_audio: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """���� Silero VAD ������Ƭ�����ɽ��㴰�ڣ�����������ɨ�跶Χ��"""
        enabled = bool(get_config('advanced_vad.focus_window_enable', True))
        if not enabled or vocal_audio is None or not hasattr(vocal_audio, 'size') or vocal_audio.size == 0:
            return []
        pad_s = float(get_config('advanced_vad.focus_window_pad_s', 0.2))
        try:
            segments = self._cut_point_calculator._detect_speech_timestamps(vocal_audio)
        except Exception as exc:
            logger.warning(f'���㴰�ڼ���ʧ�ܣ�����ȫ��ɨ��: {exc}')
            return []
        if not segments:
            return []
        duration = len(vocal_audio) / float(self.sample_rate) if self.sample_rate > 0 else 0.0
        windows: List[Tuple[float, float]] = []
        for seg in segments:
            start = float(seg.get('start', 0)) / float(self.sample_rate)
            end = float(seg.get('end', 0)) / float(self.sample_rate)
            left = max(0.0, start - pad_s)
            right = min(duration, start + pad_s)
            windows.append((left, right))
            tail_left = max(0.0, end - pad_s)
            tail_right = min(duration, end + pad_s)
            windows.append((tail_left, tail_right))
        return self._merge_windows(windows)

    @staticmethod
    def _merge_windows(windows: List[Tuple[float, float]], min_width: float = 0.0) -> List[Tuple[float, float]]:
        if not windows:
            return []
        merged: List[Tuple[float, float]] = []
        for start, end in sorted(windows, key=lambda w: w[0]):
            if end <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        if min_width > 0.0:
            merged = [(s, e) for s, e in merged if (e - s) >= min_width]
        return merged

    def _extract_vocal_features(self, audio: np.ndarray) -> VocalFeatures:
        """��ȡ������ά����
        
        Args:
            audio: ��Ƶ�ź�
            
        Returns:
            ��ȡ����������
        """
        logger.debug("��ȡ��������...")
        
        # 1. ��Ƶ(F0)��ȡ - ʹ��librosa��pyin�㷨
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'),  # 65Hz
            fmax=librosa.note_to_hz('C7'),  # 2093Hz
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # 2. �������� - ʹ��LPC����
        formant_energies = self._extract_formants(audio)
        
        # 3. Ƶ������
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # 4. г������ - ���ڴ������źţ������ٷ���
        harmonic_ratio = self._calculate_harmonic_ratio_direct(audio)
        
        # 5. ������
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )[0]
        
        # 6. RMS����
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


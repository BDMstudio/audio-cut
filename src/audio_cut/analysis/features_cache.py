#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/analysis/features_cache.py
# AI-SUMMARY: 构建并缓存整轨 BPM / MDD 等全局特征，供各检测器复用，避免重复计算。

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import librosa
import numpy as np

from src.vocal_smart_splitter.utils.config_manager import get_config
from src.vocal_smart_splitter.core.adaptive_vad_enhancer import BPMAnalyzer, BPMFeatures

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _ensure_mono(wave: np.ndarray) -> np.ndarray:
    if wave.ndim == 1:
        return wave
    if wave.ndim == 2:
        return np.mean(wave, axis=0)
    return wave.reshape(-1)


@dataclass
class TrackFeatureCache:
    sr: int
    hop_length: int
    hop_s: float
    duration_s: float
    rms_series: np.ndarray
    spectral_flatness: np.ndarray
    onset_envelope: np.ndarray
    onset_strength: np.ndarray
    onset_frames: np.ndarray
    rms_max: float
    onset_max: float
    bpm_features: Optional[BPMFeatures]
    tempo_curve: Optional[np.ndarray]
    beat_times: np.ndarray
    global_mdd: float
    mdd_series: np.ndarray

    def frame_count(self) -> int:
        return len(self.rms_series)

    def frame_index(self, t: float) -> int:
        if self.hop_s <= 0:
            return 0
        idx = int(round(t / self.hop_s))
        return int(np.clip(idx, 0, max(self.frame_count() - 1, 0)))

    def frame_slice(self, start_time: float, end_time: float, pad_frames: int = 0) -> slice:
        start_idx = self.frame_index(start_time) - pad_frames
        end_idx = self.frame_index(end_time) + pad_frames + 1
        start_idx = max(0, start_idx)
        end_idx = min(self.frame_count(), max(start_idx + 1, end_idx))
        return slice(start_idx, end_idx)

    def count_onsets(self, frame_slice: slice) -> int:
        if self.onset_frames.size == 0:
            return 0
        start = frame_slice.start
        end = frame_slice.stop
        mask = (self.onset_frames >= start) & (self.onset_frames < end)
        return int(np.sum(mask))

    def window_stats(self, start_time: float, end_time: float, pad_frames: int = 0) -> Dict[str, np.ndarray]:
        sl = self.frame_slice(start_time, end_time, pad_frames=pad_frames)
        return {
            'rms': self.rms_series[sl],
            'spectral_flatness': self.spectral_flatness[sl],
            'onset_strength': self.onset_strength[sl],
            'mdd': self.mdd_series[sl],
            'slice': sl,
        }


def _compute_mdd_series(rms: np.ndarray, flatness: np.ndarray, onset_strength: np.ndarray) -> np.ndarray:
    energy_weight = get_config('musical_dynamic_density.energy_weight', 0.5)
    spectral_weight = get_config('musical_dynamic_density.spectral_weight', 0.3)
    onset_weight = get_config('musical_dynamic_density.onset_weight', 0.2)

    rms_norm = rms / (np.max(rms) + _EPS)
    flat_norm = 1.0 - np.clip(flatness, 0.0, 1.0)
    onset_norm = onset_strength / (np.max(onset_strength) + _EPS)

    mdd_series = (
        energy_weight * rms_norm
        + spectral_weight * flat_norm
        + onset_weight * onset_norm
    )
    return np.clip(mdd_series, 0.0, 1.0)


def build_feature_cache(
    mix_wave: np.ndarray,
    vocal_wave: Optional[np.ndarray],
    sr: int,
    *,
    hop_s: float = 0.05,
) -> TrackFeatureCache:
    """构建全局特征缓存。"""

    mix_wave = _ensure_mono(mix_wave)
    if mix_wave is None or mix_wave.size == 0:
        raise ValueError('mix_wave is empty, cannot build feature cache')

    _ = vocal_wave  # vocal 波形暂未使用，保留参数以兼容后续扩展

    hop_length = max(1, int(round(sr * hop_s)))
    frame_length = max(hop_length * 2, int(round(sr * 0.1)))

    rms_series = librosa.feature.rms(y=mix_wave, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=mix_wave, hop_length=hop_length)[0]
    onset_envelope = librosa.onset.onset_strength(y=mix_wave, sr=sr, hop_length=hop_length)
    onset_strength = onset_envelope.copy()
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr, hop_length=hop_length)

    bpm_analyzer = BPMAnalyzer(sr)
    bpm_features = bpm_analyzer.extract_bpm_features(mix_wave)

    tempo_curve = librosa.beat.tempo(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        aggregate=None,
    )
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    mdd_series = _compute_mdd_series(rms_series, spectral_flatness, onset_strength)
    global_mdd = float(np.mean(mdd_series))

    duration_s = len(mix_wave) / float(sr)

    return TrackFeatureCache(
        sr=sr,
        hop_length=hop_length,
        hop_s=hop_s,
        duration_s=duration_s,
        rms_series=rms_series,
        spectral_flatness=spectral_flatness,
        onset_envelope=onset_envelope,
        onset_strength=onset_strength,
        onset_frames=onset_frames,
        rms_max=float(np.max(rms_series) if rms_series.size else 0.0),
        onset_max=float(np.max(onset_strength) if onset_strength.size else 0.0),
        bpm_features=bpm_features,
        tempo_curve=tempo_curve,
        beat_times=beat_times,
        global_mdd=global_mdd,
        mdd_series=mdd_series,
    )

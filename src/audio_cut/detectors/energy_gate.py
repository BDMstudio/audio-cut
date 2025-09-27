#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/detectors/energy_gate.py
# AI-SUMMARY: Energy-based lightweight VAD that emits speech activity ranges without external models.

from __future__ import annotations

from typing import Dict, List

import numpy as np
import librosa

_EPS = 1e-12


def _ensure_mono(wave: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio into mono."""

    if wave.ndim == 1:
        return wave
    if wave.ndim == 2:
        return np.mean(wave, axis=0)
    return wave.reshape(-1)


def detect_activity_segments(
    wave: np.ndarray,
    sr: int,
    *,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    floor_percentile: float = 15.0,
    threshold_db: float = -45.0,
    boost_db: float = 6.0,
    min_speech_ms: float = 200.0,
    min_silence_ms: float = 120.0,
    hang_ms: float = 120.0,
    smoothing_frames: int = 3,
) -> List[Dict[str, int]]:
    """Return speech activity segments as sample ranges."""

    if wave is None or wave.size == 0 or sr <= 0:
        return []

    mono = _ensure_mono(np.asarray(wave, dtype=np.float32))

    frame_length = max(1, int(round(frame_ms / 1000.0 * sr)))
    hop_length = max(1, int(round(hop_ms / 1000.0 * sr)))
    if frame_length < hop_length:
        frame_length = hop_length * 2

    rms = librosa.feature.rms(y=mono, frame_length=frame_length, hop_length=hop_length, center=False)[0]
    rms_db = 20.0 * np.log10(rms + _EPS)

    noise_floor_db = float(np.percentile(rms_db, floor_percentile)) if 0.0 < floor_percentile < 100.0 else float(np.min(rms_db))
    effective_threshold = max(threshold_db, noise_floor_db + boost_db)

    active = rms_db >= effective_threshold
    if smoothing_frames > 1:
        kernel = np.ones(int(smoothing_frames), dtype=np.float32)
        kernel /= float(np.sum(kernel))
        smoothed = np.convolve(active.astype(np.float32), kernel, mode='same')
        active = smoothed >= 0.5

    frame_duration = hop_length / float(sr)
    min_speech_frames = max(1, int(round((min_speech_ms / 1000.0) / frame_duration)))
    bridge_frames = max(
        int(round((min_silence_ms / 1000.0) / frame_duration)),
        int(round((hang_ms / 1000.0) / frame_duration)),
    )

    segments: List[Dict[str, int]] = []
    start_idx = None
    last_active = None
    silent_stretch = 0

    for idx, flag in enumerate(active):
        if flag:
            if start_idx is None:
                start_idx = idx
            last_active = idx
            silent_stretch = 0
        else:
            if start_idx is None:
                continue
            silent_stretch += 1
            if silent_stretch <= bridge_frames:
                continue
            if last_active is not None and (last_active - start_idx + 1) >= min_speech_frames:
                start_sample = start_idx * hop_length
                end_sample = min(len(mono), (last_active + 1) * hop_length + frame_length)
                if end_sample > start_sample:
                    segments.append({'start': int(start_sample), 'end': int(end_sample)})
            start_idx = None
            last_active = None
            silent_stretch = 0

    if start_idx is not None and last_active is not None and (last_active - start_idx + 1) >= min_speech_frames:
        start_sample = start_idx * hop_length
        end_sample = min(len(mono), (last_active + 1) * hop_length + frame_length)
        if end_sample > start_sample:
            segments.append({'start': int(start_sample), 'end': int(end_sample)})

    return segments


def detect_speech_timestamps(wave: np.ndarray, sr: int, **kwargs) -> List[Dict[str, int]]:
    """Alias compatible with Silero style speech timestamps."""

    return detect_activity_segments(wave, sr, **kwargs)

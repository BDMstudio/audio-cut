#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/cutting/beat_candidates.py
# AI-SUMMARY: Generates weak VPBD beat candidates in high-energy regions with vocal-risk metadata.

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from audio_cut.analysis.chorus_regions import detect_chorus_regions
from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate


def generate_beat_candidates(
    *,
    beat_times: Iterable[float],
    rms_series: Iterable[float],
    hop_s: float,
    duration_s: float,
    sample_rate: int,
    vocal_track: Optional[np.ndarray],
    bars_per_cut: int = 2,
    base_score: float = 0.3,
    energy_threshold: Optional[float] = None,
    min_consecutive_bars: int = 4,
    guard_win_ms: float = 80.0,
) -> List[CutCandidate]:
    """Create weak beat-aligned candidates only inside continuous high-energy bars."""

    beats = np.asarray(list(beat_times), dtype=np.float32)
    rms = np.asarray(list(rms_series), dtype=np.float32)
    duration_s = float(duration_s)
    if beats.size < 2 or rms.size == 0 or duration_s <= 0.0:
        return []

    bar_times = _bar_times_from_beats(beats, duration_s)
    if len(bar_times) < 2:
        return []

    bar_energies = _bar_energies_from_rms(rms, float(hop_s), bar_times)
    if not bar_energies:
        return []

    threshold = float(energy_threshold) if energy_threshold is not None else float(np.mean(bar_energies))
    high_energy_bars = detect_chorus_regions(
        bar_energies,
        threshold,
        min_consecutive_bars=min_consecutive_bars,
    )
    if not high_energy_bars:
        return []

    candidates: List[CutCandidate] = []
    for region in _contiguous_bar_regions(sorted(high_energy_bars)):
        for offset, bar_idx in enumerate(region):
            if offset % max(1, int(bars_per_cut)) != 0:
                continue
            if bar_idx >= len(bar_times) - 1:
                continue
            t = float(bar_times[bar_idx])
            if t <= 0.0 or t >= duration_s:
                continue
            candidates.append(
                CutCandidate(
                    t=t,
                    score=float(base_score),
                    source=CandidateSource.BEAT,
                    reasons=['high_energy_beat'],
                    features={'vocal_cut_risk': _vocal_cut_risk(vocal_track, sample_rate, t, guard_win_ms)},
                    meta={'bar_index': int(bar_idx), 'bars_per_cut': int(bars_per_cut)},
                )
            )
    return candidates


def _bar_times_from_beats(beat_times: np.ndarray, duration_s: float, beats_per_bar: int = 4) -> List[float]:
    beats = sorted(float(beat) for beat in beat_times if 0.0 <= float(beat) <= duration_s)
    if not beats:
        return [0.0, duration_s]
    if beats[0] > 1e-6:
        beats.insert(0, 0.0)
    bar_times = [beats[idx] for idx in range(0, len(beats), max(1, int(beats_per_bar)))]
    if bar_times[-1] < duration_s:
        bar_times.append(duration_s)
    return bar_times


def _bar_energies_from_rms(rms_series: np.ndarray, hop_s: float, bar_times: List[float]) -> List[float]:
    if hop_s <= 0.0:
        return []
    energies: List[float] = []
    for start_s, end_s in zip(bar_times, bar_times[1:]):
        start_idx = max(0, int(np.floor(start_s / hop_s)))
        end_idx = min(len(rms_series), max(start_idx + 1, int(np.ceil(end_s / hop_s))))
        if start_idx >= len(rms_series):
            energies.append(0.0)
        else:
            energies.append(float(np.mean(rms_series[start_idx:end_idx])))
    return energies


def _contiguous_bar_regions(bar_indices: List[int]) -> List[List[int]]:
    if not bar_indices:
        return []
    regions: List[List[int]] = []
    current = [bar_indices[0]]
    for bar_idx in bar_indices[1:]:
        if bar_idx == current[-1] + 1:
            current.append(bar_idx)
        else:
            regions.append(current)
            current = [bar_idx]
    regions.append(current)
    return regions


def _vocal_cut_risk(
    vocal_track: Optional[np.ndarray],
    sample_rate: int,
    time_s: float,
    guard_win_ms: float,
) -> float:
    if vocal_track is None or sample_rate <= 0:
        return 0.0
    audio = np.asarray(vocal_track, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    mono = audio if audio.ndim == 1 else np.mean(audio, axis=-1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak <= 1e-9:
        return 0.0

    center = int(round(float(time_s) * sample_rate))
    half_win = max(1, int(round(sample_rate * float(guard_win_ms) / 1000.0)))
    start = max(0, center - half_win)
    end = min(len(mono), center + half_win)
    if start >= end:
        return 0.0
    window_rms = float(np.sqrt(np.mean(np.square(mono[start:end], dtype=np.float64))))
    return max(0.0, min(1.0, window_rms / peak))

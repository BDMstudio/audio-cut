#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/analysis/beat_analyzer.py
# AI-SUMMARY: Unified beat/energy analysis with TrackFeatureCache integration.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set, TYPE_CHECKING

import librosa
import numpy as np

if TYPE_CHECKING:
    from .features_cache import TrackFeatureCache

logger = logging.getLogger(__name__)


@dataclass
class BeatAnalysisResult:
    """Result from beat and energy analysis.

    Attributes:
        tempo: Detected BPM
        beat_times: Beat timestamps in seconds (from librosa)
        bar_times: Bar boundary timestamps in seconds (grouped beats)
        bar_duration: Duration of one bar in seconds
        bar_energies: Average RMS energy per bar
        high_energy_bars: Set of bar indices above energy threshold
        energy_threshold: The computed energy threshold value
    """
    tempo: float
    beat_times: np.ndarray
    bar_times: np.ndarray
    bar_duration: float
    bar_energies: List[float]
    high_energy_bars: Set[int]
    energy_threshold: float
    # Additional metadata
    num_beats: int = field(default=0)
    num_bars: int = field(default=0)

    def __post_init__(self):
        self.num_beats = len(self.beat_times) if self.beat_times is not None else 0
        self.num_bars = len(self.bar_times) - 1 if self.bar_times is not None and len(self.bar_times) > 1 else 0


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is mono (1D array)."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=0)
    return audio.reshape(-1)


def _generate_bar_boundaries(
    beat_times: np.ndarray,
    audio_duration: float,
    time_signature: int = 4,
) -> np.ndarray:
    """Generate bar boundaries from beat times.

    Groups beats according to time signature to create bar boundaries.

    Args:
        beat_times: Array of beat timestamps in seconds
        audio_duration: Total audio duration in seconds
        time_signature: Beats per bar (e.g., 4 for 4/4 time)

    Returns:
        Array of bar boundary timestamps in seconds
    """
    if len(beat_times) < time_signature:
        # Not enough beats detected - fallback to estimated bar duration
        if len(beat_times) >= 2:
            avg_beat_interval = float(np.mean(np.diff(beat_times)))
            bar_duration = avg_beat_interval * time_signature
        else:
            # Assume 120 BPM as fallback
            bar_duration = 60.0 / 120.0 * time_signature
        return np.arange(0, audio_duration + bar_duration, bar_duration)

    # Group beats into bars
    bar_times_list: List[float] = []
    for i in range(0, len(beat_times), time_signature):
        bar_times_list.append(float(beat_times[i]))

    # Add audio end as final boundary
    bar_times_list.append(float(audio_duration))

    return np.array(bar_times_list)


def _compute_bar_energies(
    audio: np.ndarray,
    sr: int,
    bar_times: np.ndarray,
    hop_length: int = 512,
) -> List[float]:
    """Compute average RMS energy for each bar.

    Args:
        audio: Audio signal (mono)
        sr: Sample rate
        bar_times: Bar boundary timestamps in seconds
        hop_length: Hop length for RMS computation

    Returns:
        List of average RMS energy per bar
    """
    # Compute RMS energy curve
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    bar_energies: List[float] = []
    for i in range(len(bar_times) - 1):
        bar_start = bar_times[i]
        bar_end = bar_times[i + 1]

        # Find RMS frames within this bar
        mask = (rms_times >= bar_start) & (rms_times < bar_end)
        if np.any(mask):
            bar_energies.append(float(np.mean(rms[mask])))
        else:
            bar_energies.append(0.0)

    return bar_energies


def analyze_beats(
    audio: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    time_signature: int = 4,
    energy_percentile: float = 70.0,
    feature_cache: Optional['TrackFeatureCache'] = None,
) -> BeatAnalysisResult:
    """Unified beat and energy analysis entry point.

    Analyzes audio for beat/bar detection and energy-based segment classification.
    Integrates with TrackFeatureCache to avoid redundant computation.

    Priority:
        1. Use beat_times from feature_cache if available
        2. Use BPM from feature_cache.bpm_features if available
        3. Fall back to librosa.beat.beat_track() for fresh computation

    Args:
        audio: Audio signal (mono or stereo, will be converted to mono)
        sr: Sample rate in Hz
        hop_length: Hop length for analysis (default: 512)
        time_signature: Beats per bar (default: 4 for 4/4 time)
        energy_percentile: Percentile threshold for high-energy detection (default: 70)
        feature_cache: Optional TrackFeatureCache with pre-computed features

    Returns:
        BeatAnalysisResult with tempo, beats, bars, and energy data
    """
    audio = _ensure_mono(audio)
    audio_duration = len(audio) / float(sr)

    # 1. Get beat data (priority: cache > compute)
    beat_times: Optional[np.ndarray] = None
    tempo: float = 0.0

    if feature_cache is not None:
        # Try to reuse cached beat_times
        if feature_cache.beat_times is not None and len(feature_cache.beat_times) > 0:
            beat_times = feature_cache.beat_times
            logger.debug("[BeatAnalyzer] Reusing %d cached beat times", len(beat_times))

        # Try to reuse cached BPM
        if feature_cache.bpm_features is not None:
            tempo = float(feature_cache.bpm_features.main_bpm)
            logger.debug("[BeatAnalyzer] Reusing cached BPM: %.1f", tempo)

    # Fall back to librosa beat detection if needed
    if beat_times is None:
        detected_tempo, beat_frames = librosa.beat.beat_track(
            y=audio, sr=sr, hop_length=hop_length
        )
        # Handle librosa returning array vs scalar
        tempo = float(detected_tempo) if not hasattr(detected_tempo, '__len__') else float(detected_tempo[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        logger.debug("[BeatAnalyzer] Detected %d beats at %.1f BPM", len(beat_times), tempo)
    elif tempo == 0.0 and len(beat_times) >= 2:
        # Estimate tempo from beat intervals
        avg_beat_interval = float(np.mean(np.diff(beat_times)))
        tempo = 60.0 / avg_beat_interval if avg_beat_interval > 0 else 120.0

    # Handle edge case: no beats detected
    if tempo == 0.0:
        tempo = 120.0  # Default fallback
        logger.warning("[BeatAnalyzer] No tempo detected, using default 120 BPM")

    # 2. Calculate bar duration
    bar_duration = 60.0 / tempo * time_signature

    # 3. Generate bar boundaries
    bar_times = _generate_bar_boundaries(beat_times, audio_duration, time_signature)

    # 4. Compute bar energies
    bar_energies = _compute_bar_energies(audio, sr, bar_times, hop_length)

    # 5. Identify high-energy bars
    if bar_energies:
        energy_threshold = float(np.percentile(bar_energies, energy_percentile))
    else:
        energy_threshold = 0.0

    high_energy_bars: Set[int] = {
        i for i, energy in enumerate(bar_energies) if energy >= energy_threshold
    }

    logger.info(
        "[BeatAnalyzer] Analysis complete: BPM=%.1f, %d beats, %d bars, %d high-energy (P%.0f=%.4f)",
        tempo, len(beat_times), len(bar_times) - 1, len(high_energy_bars),
        energy_percentile, energy_threshold
    )

    return BeatAnalysisResult(
        tempo=tempo,
        beat_times=beat_times,
        bar_times=bar_times,
        bar_duration=bar_duration,
        bar_energies=bar_energies,
        high_energy_bars=high_energy_bars,
        energy_threshold=energy_threshold,
    )


class BeatAnalyzer:
    """Stateful beat analyzer for reusable analysis across multiple calls.

    This class wraps the analyze_beats function and can cache results
    for repeated analysis on the same audio.

    Usage:
        analyzer = BeatAnalyzer(sample_rate=44100)
        result = analyzer.analyze(audio, feature_cache=cache)
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        time_signature: int = 4,
        energy_percentile: float = 70.0,
    ):
        """Initialize the beat analyzer.

        Args:
            sample_rate: Default sample rate for analysis
            hop_length: Default hop length for analysis
            time_signature: Default time signature (beats per bar)
            energy_percentile: Default percentile for high-energy detection
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.time_signature = time_signature
        self.energy_percentile = energy_percentile
        self._last_result: Optional[BeatAnalysisResult] = None

    def analyze(
        self,
        audio: np.ndarray,
        *,
        sr: Optional[int] = None,
        hop_length: Optional[int] = None,
        time_signature: Optional[int] = None,
        energy_percentile: Optional[float] = None,
        feature_cache: Optional['TrackFeatureCache'] = None,
    ) -> BeatAnalysisResult:
        """Analyze audio for beats and energy.

        Args:
            audio: Audio signal
            sr: Sample rate (uses instance default if None)
            hop_length: Hop length (uses instance default if None)
            time_signature: Time signature (uses instance default if None)
            energy_percentile: Energy percentile (uses instance default if None)
            feature_cache: Optional pre-computed feature cache

        Returns:
            BeatAnalysisResult with analysis data
        """
        result = analyze_beats(
            audio=audio,
            sr=sr or self.sample_rate,
            hop_length=hop_length or self.hop_length,
            time_signature=time_signature or self.time_signature,
            energy_percentile=energy_percentile or self.energy_percentile,
            feature_cache=feature_cache,
        )
        self._last_result = result
        return result

    @property
    def last_result(self) -> Optional[BeatAnalysisResult]:
        """Get the result from the last analysis call."""
        return self._last_result

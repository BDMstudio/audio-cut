#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/strategies/base.py
# AI-SUMMARY: Abstract base class for segmentation strategies in hybrid_mdd mode.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class SegmentationContext:
    """Context data passed to segmentation strategies.
    
    Attributes:
        audio: Original audio array (mono, normalized)
        sample_rate: Sample rate in Hz
        tempo: Detected BPM
        beat_times: Beat timestamps in seconds (from librosa)
        bar_times: Bar timestamps in seconds (grouped beats)
        bar_duration: Duration of one bar in seconds
        mdd_cut_points_samples: MDD-detected cut points (sample indices)
        energy_threshold: RMS energy threshold for high-energy detection
        bar_energies: List of average RMS energy per bar
        config: Strategy-specific configuration dict
    """
    audio: np.ndarray
    sample_rate: int
    tempo: float
    beat_times: np.ndarray
    bar_times: np.ndarray
    bar_duration: float
    mdd_cut_points_samples: List[int]
    energy_threshold: float
    bar_energies: List[float]
    config: Dict[str, Any]


@dataclass
class SegmentationResult:
    """Result from a segmentation strategy.
    
    Attributes:
        cut_points_samples: Final cut point positions (sample indices)
        lib_flags: Per-segment flag indicating if segment ends at beat boundary
        metadata: Optional additional data for debugging/logging
    """
    cut_points_samples: List[int]
    lib_flags: List[bool]
    metadata: Optional[Dict[str, Any]] = None


class SegmentationStrategy(ABC):
    """Abstract base class for hybrid_mdd segmentation strategies.
    
    Subclasses implement different cut point generation algorithms:
    - MddStartStrategy (Plan A): MDD starts + beat ends
    - BeatOnlyStrategy (Plan B): Pure beat alignment
    - HybridSnapStrategy (Plan C): MDD snapped to nearest beats
    """
    
    @abstractmethod
    def generate_cut_points(
        self,
        context: SegmentationContext,
    ) -> SegmentationResult:
        """Generate cut points and lib flags based on the strategy.
        
        Args:
            context: SegmentationContext with all necessary data
            
        Returns:
            SegmentationResult with cut points and lib flags
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier for logging/config."""
        pass


def identify_high_energy_bars(
    bar_energies: List[float],
    energy_threshold: float,
) -> Set[int]:
    """Identify bar indices whose energy meets or exceeds the threshold."""
    return {i for i, energy in enumerate(bar_energies) if energy >= energy_threshold}


def deduplicate_and_convert_cuts(
    cut_with_flags: List[Tuple[float, bool]],
    sample_rate: int,
    audio_len: int,
    *,
    time_tolerance_s: float = 0.1,
) -> Tuple[List[int], List[bool]]:
    """Deduplicate cut times, convert to samples, and align lib flags to segments."""
    if sample_rate <= 0 or audio_len < 0:
        return [0, max(0, audio_len)], []

    audio_duration = audio_len / float(sample_rate)
    if not cut_with_flags:
        cut_with_flags = [(0.0, False), (audio_duration, False)]

    unique: List[Tuple[float, bool]] = []
    seen = set()
    for t, flag in cut_with_flags:
        if t in seen:
            continue
        seen.add(t)
        unique.append((float(t), bool(flag)))

    unique.sort(key=lambda x: x[0])
    if not unique or unique[0][0] != 0.0:
        unique.insert(0, (0.0, False))
    if unique[-1][0] != audio_duration:
        unique.append((audio_duration, False))

    cut_points_samples: List[int] = []
    lib_flags: List[bool] = []
    for i, (t, is_lib) in enumerate(unique):
        sample_idx = int(t * sample_rate)
        sample_idx = max(0, min(sample_idx, audio_len))
        cut_points_samples.append(sample_idx)
        if i > 0:
            lib_flags.append(is_lib)

    if cut_points_samples[0] != 0:
        cut_points_samples.insert(0, 0)
        lib_flags.insert(0, False)
    if cut_points_samples[-1] != audio_len:
        cut_points_samples.append(audio_len)

    cut_points_samples = sorted(set(cut_points_samples))

    num_segments = len(cut_points_samples) - 1
    if len(lib_flags) != num_segments:
        time_to_lib = {t: is_lib for t, is_lib in unique}
        lib_flags = []
        for i in range(num_segments):
            end_time = cut_points_samples[i + 1] / float(sample_rate)
            is_lib = any(
                abs(end_time - t) < time_tolerance_s and flag
                for t, flag in time_to_lib.items()
            )
            lib_flags.append(is_lib)

    return cut_points_samples, lib_flags

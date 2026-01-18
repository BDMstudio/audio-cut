#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/strategies/base.py
# AI-SUMMARY: Abstract base class for segmentation strategies in hybrid_mdd mode.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

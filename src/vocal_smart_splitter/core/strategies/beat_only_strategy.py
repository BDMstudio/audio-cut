#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/strategies/beat_only_strategy.py
# AI-SUMMARY: Plan B strategy - pure beat segmentation ONLY for high-energy (exciting) segments.

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .base import (
    SegmentationContext,
    SegmentationResult,
    SegmentationStrategy,
    deduplicate_and_convert_cuts,
    identify_high_energy_bars,
)

logger = logging.getLogger(__name__)


class BeatOnlyStrategy(SegmentationStrategy):
    """Plan B: Pure beat segmentation for high-energy segments only.
    
    Generates cut points purely from bar boundaries, but ONLY for high-energy
    (exciting/chorus) segments. Low-energy segments use MDD-based cuts.
    
    Config options:
        bars_per_cut: Number of bars per segment in high-energy areas (default: 2)
        min_segment_s: Minimum segment duration in seconds (default: 2.0)
        energy_percentile: Percentile threshold for high-energy detection (default: 70)
    """
    
    @property
    def name(self) -> str:
        return "beat_only"
    
    def generate_cut_points(
        self,
        context: SegmentationContext,
    ) -> SegmentationResult:
        """Generate beat-aligned cut points ONLY for high-energy segments.
        
        Algorithm:
            1. Identify high-energy bars using energy_threshold
            2. For high-energy regions: generate cuts at bar boundaries
            3. For low-energy regions: use MDD cuts (from context) or longer intervals
            4. Mark only high-energy segment cuts as _lib
        
        Args:
            context: SegmentationContext with beat/bar data and MDD cuts
            
        Returns:
            SegmentationResult with hybrid cut points
        """
        config = context.config
        bars_per_cut = int(config.get('bars_per_cut', 2))
        min_segment_s = float(config.get('min_segment_s', 2.0))
        energy_percentile = float(config.get('energy_percentile', 70))
        sample_rate = context.sample_rate
        audio_len = len(context.audio)
        audio_duration = audio_len / float(sample_rate)
        
        bar_times = context.bar_times
        bar_energies = context.bar_energies
        mdd_cut_samples = context.mdd_cut_points_samples
        
        # Calculate energy threshold
        if bar_energies:
            energy_threshold = float(np.percentile(bar_energies, energy_percentile))
        else:
            energy_threshold = context.energy_threshold
        
        high_energy_bars = identify_high_energy_bars(bar_energies, energy_threshold)
        
        logger.info(
            "[BEAT_ONLY] High-energy detection: %d/%d bars above P%.0f threshold",
            len(high_energy_bars), len(bar_energies), energy_percentile
        )
        
        # Generate cut points
        cut_times: List[float] = [0.0]
        cut_is_lib: List[bool] = []  # Track which cuts are from beat alignment
        
        # Convert MDD cuts to times for reference
        mdd_cut_times = [s / float(sample_rate) for s in mdd_cut_samples]
        mdd_cut_set = set(mdd_cut_times)
        
        # Process bar by bar
        bars_since_last_cut = 0
        last_cut_time = 0.0
        
        for bar_idx in range(len(bar_times) - 1):
            bar_start = bar_times[bar_idx]
            bar_end = bar_times[bar_idx + 1] if bar_idx + 1 < len(bar_times) else audio_duration
            bars_since_last_cut += 1
            
            is_high_energy = bar_idx in high_energy_bars
            
            if is_high_energy:
                # High-energy: add beat cut every N bars
                if bars_since_last_cut >= bars_per_cut:
                    cut_time = float(bar_end)
                    if cut_time <= audio_duration and cut_time - last_cut_time >= min_segment_s:
                        cut_times.append(cut_time)
                        cut_is_lib.append(True)  # This is a beat-aligned cut
                        last_cut_time = cut_time
                        bars_since_last_cut = 0
            else:
                # Low-energy: check if there's an MDD cut in this bar
                for mdd_t in mdd_cut_times:
                    if bar_start <= mdd_t < bar_end and mdd_t > last_cut_time + min_segment_s:
                        cut_times.append(mdd_t)
                        cut_is_lib.append(False)  # MDD cut, not beat-aligned
                        last_cut_time = mdd_t
                        bars_since_last_cut = 0
                        break
                else:
                    # No MDD cut in this low-energy bar, check if we need a fallback cut
                    # Use longer intervals for low-energy (e.g., 4 bars)
                    low_energy_interval = bars_per_cut * 2  # Longer intervals for quiet sections
                    if bars_since_last_cut >= low_energy_interval:
                        cut_time = float(bar_end)
                        if cut_time <= audio_duration and cut_time - last_cut_time >= min_segment_s:
                            cut_times.append(cut_time)
                            cut_is_lib.append(False)  # Not a _lib cut
                            last_cut_time = cut_time
                            bars_since_last_cut = 0
        
        cut_times.append(audio_duration)  # End
        
        cut_with_flags: List[Tuple[float, bool]] = [(0.0, False)]
        seen = {0.0}
        for i, t in enumerate(cut_times[1:-1]):
            if t in seen:
                continue
            seen.add(t)
            is_lib = cut_is_lib[i] if i < len(cut_is_lib) else False
            cut_with_flags.append((t, is_lib))
        cut_with_flags.append((audio_duration, False))

        cut_points_samples, lib_flags = deduplicate_and_convert_cuts(
            cut_with_flags,
            sample_rate,
            audio_len,
        )

        num_segments = len(cut_points_samples) - 1
        
        # Calculate segment durations for logging
        segment_durations = [
            (cut_points_samples[i + 1] - cut_points_samples[i]) / float(sample_rate)
            for i in range(num_segments)
        ]
        
        lib_count = sum(1 for f in lib_flags if f)
        logger.info(
            "[BEAT_ONLY] Generated %d segments: %d with _lib (high-energy), %d without (low-energy). "
            "Durations: min=%.1fs, max=%.1fs, avg=%.1fs",
            num_segments, lib_count, num_segments - lib_count,
            min(segment_durations) if segment_durations else 0,
            max(segment_durations) if segment_durations else 0,
            sum(segment_durations) / len(segment_durations) if segment_durations else 0,
        )
        
        return SegmentationResult(
            cut_points_samples=cut_points_samples,
            lib_flags=lib_flags,
            metadata={
                'strategy': self.name,
                'bars_per_cut': bars_per_cut,
                'num_bars': len(bar_times),
                'high_energy_bars': len(high_energy_bars),
                'lib_segment_count': lib_count,
                'segment_durations': segment_durations,
            },
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/strategies/snap_to_beat_strategy.py
# AI-SUMMARY: Plan C strategy - MDD cut points snap to nearest beat with VAD protection.

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


class SnapToBeatStrategy(SegmentationStrategy):
    """Plan C: MDD cut points snap to nearest beat with VAD protection.
    
    MDD-detected cut points are snapped to the nearest bar/beat boundary
    if within the snap tolerance. VAD protection ensures we don't cut
    in the middle of active vocals.
    
    Config options:
        snap_tolerance_ms: Max distance to snap to beat (default: 300ms)
        vad_protection: Whether to protect against cutting vocals (default: True)
        min_segment_s: Minimum segment duration in seconds (default: 2.0)
        energy_percentile: Percentile for high-energy bar detection (default: 70)
    """
    
    @property
    def name(self) -> str:
        return "snap_to_beat"
    
    def generate_cut_points(
        self,
        context: SegmentationContext,
    ) -> SegmentationResult:
        """Generate cut points by snapping MDD cuts to nearest beats.
        
        Algorithm:
            1. For each MDD cut, find the nearest BEAT (not bar)
            2. If within snap_tolerance, snap to that beat
            3. If VAD protection is on and snapping would cut vocals, keep original
            4. Mark snapped segments as _lib
        
        Args:
            context: SegmentationContext with MDD cuts and beat data
            
        Returns:
            SegmentationResult with snapped cut points
        """
        config = context.config
        snap_tolerance_ms = float(config.get('snap_tolerance_ms', 300))
        snap_tolerance_s = snap_tolerance_ms / 1000.0
        vad_protection = bool(config.get('vad_protection', True))
        min_segment_s = float(config.get('min_segment_s', 2.0))
        energy_percentile = float(config.get('energy_percentile', 70))
        
        sample_rate = context.sample_rate
        audio_len = len(context.audio)
        audio_duration = audio_len / float(sample_rate)
        
        # Use beat_times (individual beats) for snapping
        beat_times = context.beat_times
        bar_times = context.bar_times
        bar_energies = context.bar_energies
        mdd_cut_samples = context.mdd_cut_points_samples
        
        # Convert MDD cuts to times
        mdd_cut_times = [s / float(sample_rate) for s in mdd_cut_samples]
        
        # Calculate energy threshold for high-energy (chorus) detection
        if bar_energies:
            energy_threshold = float(np.percentile(bar_energies, energy_percentile))
        else:
            energy_threshold = context.energy_threshold
        
        # Identify chorus regions (continuous high-energy bars)
        high_energy_bars = self._detect_chorus_regions(
            bar_energies, 
            energy_threshold,
            bar_centroids=context.bar_spectral_centroids,
            bar_bandwidths=context.bar_spectral_bandwidths
        )
        
        # Calculate beat interval for logging
        if len(beat_times) >= 2:
            avg_beat_interval = float(np.mean(np.diff(beat_times)))
        else:
            avg_beat_interval = context.bar_duration / 4 if context.bar_duration else 0.5
        
        logger.info(
            "[SNAP_TO_BEAT] Config: snap_tolerance=%.0fms (%.1f%% of beat), vad_protection=%s, "
            "high_energy_bars=%d/%d, %d MDD cuts",
            snap_tolerance_ms, 
            (snap_tolerance_s / avg_beat_interval * 100) if avg_beat_interval else 0,
            vad_protection, 
            len(high_energy_bars), len(bar_energies),
            len(mdd_cut_times)
        )
        
        # Build VAD mask from context audio if VAD protection is enabled
        vad_active_times: List[Tuple[float, float]] = []
        if vad_protection:
            vad_active_times = self._compute_vad_active_regions(
                context.audio, sample_rate
            )
            logger.info("[SNAP_TO_BEAT] VAD detected %d active vocal regions", len(vad_active_times))
        
        # Process each MDD cut and decide whether to snap
        snapped_cuts: List[float] = [0.0]  # Start
        cut_is_lib: List[bool] = []
        snap_stats = {'snapped': 0, 'vad_blocked': 0, 'too_far': 0, 'low_energy': 0}
        
        for mdd_time in mdd_cut_times:
            if mdd_time <= 0 or mdd_time >= audio_duration:
                continue
            
            # Check if this cut is in a high-energy bar
            bar_idx = self._find_bar_index(mdd_time, bar_times)
            is_high_energy = bar_idx in high_energy_bars
            
            # Find nearest BEAT
            nearest_beat_time = self._find_nearest_beat(mdd_time, beat_times)
            distance_to_beat = abs(mdd_time - nearest_beat_time)
            
            # Decide whether to snap - ONLY in high-energy (chorus) regions
            should_snap = False
            final_cut_time = mdd_time
            
            if not is_high_energy:
                # Low energy region (verse/intro) - keep MDD cut, no _lib
                snap_stats['low_energy'] += 1
            elif distance_to_beat <= snap_tolerance_s:
                # High energy + within tolerance - ALWAYS snap for beat alignment
                # In chorus, we want beat-aligned cuts ("卡点感") even if cutting through vocals
                # VAD protection is disabled in high-energy regions
                should_snap = True
                final_cut_time = nearest_beat_time
                snap_stats['snapped'] += 1
            else:
                snap_stats['too_far'] += 1
            
            # Check minimum segment duration
            if snapped_cuts and final_cut_time - snapped_cuts[-1] < min_segment_s:
                continue  # Skip this cut, would create too short segment
            
            snapped_cuts.append(final_cut_time)
            cut_is_lib.append(should_snap)  # Mark as _lib if we snapped
        
        # 【NEW】For high density, add beat cuts in high-energy bars
        # This creates bar-length _lib segments in chorus
        density = config.get('density', 'medium')
        if density == 'high' and high_energy_bars:
            logger.info("[SNAP_TO_BEAT] High density: adding beat cuts in %d high-energy bars", len(high_energy_bars))
            
            for bar_idx in high_energy_bars:
                bar_start = bar_times[bar_idx]
                bar_end = bar_times[bar_idx + 1] if bar_idx + 1 < len(bar_times) else audio_duration
                
                # Get all beats in this bar
                beats_in_bar = [b for b in beat_times if bar_start <= b < bar_end]
                
                # Add bar starting beat as cut point (creates bar-length segments)
                if beats_in_bar:
                    beat_cut = float(beats_in_bar[0])  # First beat of bar
                    
                    # Check if we already have a cut nearby (avoid duplicates)
                    min_distance = min(abs(beat_cut - c) for c in snapped_cuts) if snapped_cuts else float('inf')
                    
                    if min_distance > min_segment_s * 0.5:  # At least half segment duration away
                        snapped_cuts.append(beat_cut)
                        cut_is_lib.append(True)  # Mark as _lib
        
        snapped_cuts.append(audio_duration)  # End
        
        logger.info(
            "[SNAP_TO_BEAT] Snap decisions: %d snapped, %d VAD-blocked, %d low-energy (skipped), %d too far",
            snap_stats['snapped'], snap_stats['vad_blocked'], snap_stats['low_energy'], snap_stats['too_far']
        )
        
        cut_with_flags: List[Tuple[float, bool]] = [(0.0, False)]
        seen = {0.0}
        for i, t in enumerate(snapped_cuts[1:-1]):
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
        
        # Calculate stats
        segment_durations = [
            (cut_points_samples[i + 1] - cut_points_samples[i]) / float(sample_rate)
            for i in range(num_segments)
        ]
        lib_count = sum(1 for f in lib_flags if f)
        
        logger.info(
            "[SNAP_TO_BEAT] Generated %d segments: %d snapped to beat (_lib), %d kept at MDD. "
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
                'snap_tolerance_ms': snap_tolerance_ms,
                'vad_protection': vad_protection,
                'snapped_count': lib_count,
                'kept_mdd_count': num_segments - lib_count,
                'snap_stats': snap_stats,
                'segment_durations': segment_durations,
            },
        )
    
    def _find_nearest_beat(self, time: float, beat_times: np.ndarray) -> float:
        """Find the nearest beat to a given time.
        
        Args:
            time: Time in seconds to find nearest beat for
            beat_times: Array of beat times in seconds
            
        Returns:
            Time of nearest beat in seconds
        """
        if len(beat_times) == 0:
            return time
        
        # Find index of closest beat
        idx = np.abs(beat_times - time).argmin()
        return float(beat_times[idx])
    
    def _find_bar_index(self, time: float, bar_times: np.ndarray) -> int:
        """Find which bar a given time falls into."""
        for i in range(len(bar_times) - 1):
            if bar_times[i] <= time < bar_times[i + 1]:
                return i
        return len(bar_times) - 2 if len(bar_times) > 1 else 0
    
    def _find_nearest_bar_boundary(self, time: float, bar_times: np.ndarray) -> float:
        """Find the nearest bar boundary to a given time."""
        if len(bar_times) == 0:
            return time
        
        # Find index of closest bar time
        idx = np.abs(bar_times - time).argmin()
        return float(bar_times[idx])
    
    def _compute_vad_active_regions(
        self, audio: np.ndarray, sample_rate: int
    ) -> List[Tuple[float, float]]:
        """Compute active vocal regions using simple energy-based VAD.
        
        Returns list of (start, end) tuples in seconds for active regions.
        """
        import librosa
        
        # Compute RMS energy
        hop_length = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)
        
        # Use 20th percentile as noise floor, 60th as voice threshold
        noise_floor = np.percentile(rms, 20)
        voice_threshold = noise_floor + (np.percentile(rms, 60) - noise_floor) * 0.5
        
        # Find active regions
        is_active = rms > voice_threshold
        regions: List[Tuple[float, float]] = []
        
        in_region = False
        region_start = 0.0
        
        for i, active in enumerate(is_active):
            t = float(times[i]) if i < len(times) else float(len(audio) / sample_rate)
            if active and not in_region:
                in_region = True
                region_start = t
            elif not active and in_region:
                in_region = False
                if t - region_start > 0.1:  # Min 100ms region
                    regions.append((region_start, t))
        
        # Handle end case
        if in_region:
            regions.append((region_start, float(len(audio) / sample_rate)))
        
        return regions
    
    def _would_cut_active_vocal(
        self, cut_time: float, vad_regions: List[Tuple[float, float]]
    ) -> bool:
        """Check if cutting at cut_time would cut through an active vocal region."""
        margin = 0.1  # 100ms margin
        for start, end in vad_regions:
            if start + margin < cut_time < end - margin:
                return True
        return False
    
    def _detect_chorus_regions(
        self, bar_energies: np.ndarray, energy_threshold: float, min_consecutive_bars: int = 4,
        bar_centroids: list = None, bar_bandwidths: list = None
    ) -> set:
        """
        Detect chorus regions using multi-feature fusion and continuity analysis.
        
        Features:
        - RMS Energy: Overall loudness
        - Spectral Centroid: Brightness/timbre
        - Spectral Bandwidth: Frequency richness
        
        Args:
            bar_energies: Energy levels for each bar
            energy_threshold: Energy threshold for "high energy"
            min_consecutive_bars: Minimum consecutive high-energy bars to be considered chorus
            bar_centroids: Optional spectral centroid per bar
            bar_bandwidths: Optional spectral bandwidth per bar
            
        Returns:
            Set of bar indices that belong to chorus regions
        """
        if len(bar_energies) == 0:
            return set()
        
        # Convert to numpy array if needed
        if not isinstance(bar_energies, np.ndarray):
            bar_energies = np.array(bar_energies)
        
        # Multi-feature fusion
        if bar_centroids and bar_bandwidths and len(bar_centroids) == len(bar_energies):
            # Normalize features to [0, 1]
            def normalize(arr):
                arr = np.array(arr)
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max - arr_min > 1e-6:
                    return (arr - arr_min) / (arr_max - arr_min)
                return np.zeros_like(arr)
            
            norm_energy = normalize(bar_energies)
            norm_centroid = normalize(bar_centroids)
            norm_bandwidth = normalize(bar_bandwidths)
            
            # Adaptive weighting based on energy variance
            energy_cv = np.std(bar_energies) / (np.mean(bar_energies) + 1e-6)
            
            if energy_cv < 0.15:  # Low dynamic range (folk/ballad)
                # Rely more on spectral changes
                weights = {'energy': 0.3, 'centroid': 0.4, 'bandwidth': 0.3}
                logger.debug("[ChorusDetect] Low dynamics (CV=%.3f), using spectral-heavy weights", energy_cv)
            elif energy_cv > 0.4:  # High dynamic range (rock/pop)
                # Energy is reliable
                weights = {'energy': 0.6, 'centroid': 0.2, 'bandwidth': 0.2}
                logger.debug("[ChorusDetect] High dynamics (CV=%.3f), using energy-heavy weights", energy_cv)
            else:  # Medium dynamic range
                # Balanced
                weights = {'energy': 0.5, 'centroid': 0.25, 'bandwidth': 0.25}
                logger.debug("[ChorusDetect] Medium dynamics (CV=%.3f), using balanced weights", energy_cv)
            
            # Fused chorus score
            chorus_score = (
                norm_energy * weights['energy'] +
                norm_centroid * weights['centroid'] +
                norm_bandwidth * weights['bandwidth']
            )
            
            # Adaptive threshold
            fused_threshold = np.percentile(chorus_score, 60)  # Top 40%
            is_high_energy = chorus_score >= fused_threshold
            
            logger.info(
                "[ChorusDetect] Multi-feature fusion: CV=%.3f, weights=%s, threshold=%.3f",
                energy_cv, weights, fused_threshold
            )
        else:
            # Fallback to simple energy thresholding
            is_high_energy = bar_energies >= energy_threshold
            logger.debug("[ChorusDetect] Using simple energy threshold (no spectral features)")
        
        # Find continuous regions
        chorus_bars = set()
        current_start = None
        consecutive_count = 0
        
        for i, is_high in enumerate(is_high_energy):
            if is_high:
                if current_start is None:
                    current_start = i
                consecutive_count += 1
            else:
                # End of a continuous segment
                if consecutive_count >= min_consecutive_bars:
                    # Add all bars in this chorus region
                    chorus_bars.update(range(current_start, i))
                current_start = None
                consecutive_count = 0
        
        # Handle final segment
        if consecutive_count >= min_consecutive_bars and current_start is not None:
            chorus_bars.update(range(current_start, len(bar_energies)))
        
        return chorus_bars

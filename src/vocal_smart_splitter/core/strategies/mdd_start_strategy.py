#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/strategies/mdd_start_strategy.py
# AI-SUMMARY: Plan A strategy - MDD cuts plus beat-aligned ends in high-energy bars.

from __future__ import annotations

import bisect
import logging
from typing import List

from .base import (
    SegmentationContext,
    SegmentationResult,
    SegmentationStrategy,
    identify_high_energy_bars,
)

logger = logging.getLogger(__name__)


class MddStartStrategy(SegmentationStrategy):
    """Plan A: MDD start + beat end alignment for high-energy bars."""

    @property
    def name(self) -> str:
        return "mdd_start"

    def generate_cut_points(
        self,
        context: SegmentationContext,
    ) -> SegmentationResult:
        config = context.config
        enable_beat_cuts = bool(config.get('enable_beat_cuts', True))
        bars_per_cut = int(config.get('bars_per_cut', 2))
        min_segment_s = float(config.get('min_segment_s', 2.0))
        snap_to_pause_ms = float(config.get('snap_to_pause_ms', 300))
        energy_percentile = float(config.get('energy_percentile', 70))

        sample_rate = context.sample_rate
        audio_len = len(context.audio)
        audio_duration = audio_len / float(sample_rate) if sample_rate else 0.0

        mdd_cut_points_samples = list(context.mdd_cut_points_samples or [])
        if not mdd_cut_points_samples:
            mdd_cut_points_samples = [0, audio_len]
        else:
            if mdd_cut_points_samples[0] != 0:
                mdd_cut_points_samples.insert(0, 0)
            if mdd_cut_points_samples[-1] != audio_len:
                mdd_cut_points_samples.append(audio_len)
            mdd_cut_points_samples = sorted(set(mdd_cut_points_samples))

        if not enable_beat_cuts:
            lib_flags = [False] * max(0, len(mdd_cut_points_samples) - 1)
            return SegmentationResult(
                cut_points_samples=mdd_cut_points_samples,
                lib_flags=lib_flags,
                metadata={
                    'strategy': self.name,
                    'enable_beat_cuts': False,
                    'energy_percentile': energy_percentile,
                },
            )

        bar_times = context.bar_times
        bar_energies = context.bar_energies
        energy_threshold = context.energy_threshold

        high_energy_bars = identify_high_energy_bars(bar_energies, energy_threshold)

        logger.info(
            "[MDD_START] High-energy bars: %d/%d (P%.0f threshold=%.4f)",
            len(high_energy_bars),
            len(bar_energies),
            energy_percentile,
            energy_threshold,
        )

        beat_cut_times: List[float] = []
        bars_since_last_cut = 0
        for bar_idx in range(len(bar_times) - 1):
            bars_since_last_cut += 1
            if bar_idx in high_energy_bars and bars_since_last_cut >= bars_per_cut:
                bar_end_time = float(bar_times[bar_idx + 1])
                if 0.0 < bar_end_time < audio_duration:
                    beat_cut_times.append(bar_end_time)
                bars_since_last_cut = 0

        min_segment_samples = int(min_segment_s * sample_rate)
        snap_samples = int(snap_to_pause_ms / 1000.0 * sample_rate)

        all_cuts_sorted = sorted(mdd_cut_points_samples)
        cut_sources = {cut: False for cut in all_cuts_sorted}

        added_beat_cuts = 0
        skipped_too_close = 0
        skipped_duplicates = 0

        for beat_time in sorted(beat_cut_times):
            beat_sample = int(beat_time * sample_rate)
            if beat_sample <= 0 or beat_sample >= audio_len:
                continue

            if any(abs(beat_sample - mdd_sample) <= snap_samples for mdd_sample in mdd_cut_points_samples):
                skipped_duplicates += 1
                continue

            prev_cut = 0
            next_cut = audio_len
            for existing_cut in all_cuts_sorted:
                if existing_cut < beat_sample:
                    prev_cut = existing_cut
                elif existing_cut > beat_sample:
                    next_cut = existing_cut
                    break

            duration_before = beat_sample - prev_cut
            duration_after = next_cut - beat_sample

            if duration_before >= min_segment_samples and duration_after >= min_segment_samples:
                bisect.insort(all_cuts_sorted, beat_sample)
                cut_sources[beat_sample] = True
                added_beat_cuts += 1
            else:
                skipped_too_close += 1

        final_cut_points = sorted(set(all_cuts_sorted))
        if final_cut_points[0] != 0:
            final_cut_points.insert(0, 0)
        if final_cut_points[-1] != audio_len:
            final_cut_points.append(audio_len)
        final_cut_points = sorted(set(final_cut_points))

        lib_flags: List[bool] = []
        for i in range(len(final_cut_points) - 1):
            end_cut = final_cut_points[i + 1]
            lib_flags.append(bool(cut_sources.get(end_cut, False)))

        logger.info(
            "[MDD_START] Beat cuts: %d added, %d skipped (short), %d duplicates",
            added_beat_cuts,
            skipped_too_close,
            skipped_duplicates,
        )

        return SegmentationResult(
            cut_points_samples=final_cut_points,
            lib_flags=lib_flags,
            metadata={
                'strategy': self.name,
                'added_beat_cuts': added_beat_cuts,
                'skipped_too_close': skipped_too_close,
                'skipped_duplicates': skipped_duplicates,
                'num_mdd_cuts': len(mdd_cut_points_samples),
                'num_high_energy_bars': len(high_energy_bars),
            },
        )

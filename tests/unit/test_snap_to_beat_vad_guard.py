#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_snap_to_beat_vad_guard.py
# AI-SUMMARY: Verifies hybrid_mdd beat snapping respects vocal-track quiet guards.

import numpy as np

from vocal_smart_splitter.core.strategies.base import SegmentationContext
from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from vocal_smart_splitter.core.strategies.beat_only_strategy import BeatOnlyStrategy
from vocal_smart_splitter.core.strategies.snap_to_beat_strategy import SnapToBeatStrategy


def _context_with_vocal_burst_at_beat(
    *,
    mdd_time_s: float = 5.19,
    snap_tolerance_ms: float = 240,
    density: str = 'medium',
    burst_start_s: float = 4.92,
    burst_end_s: float = 5.08,
) -> SegmentationContext:
    sample_rate = 1000
    duration_s = 12.0
    audio = np.zeros(int(sample_rate * duration_s), dtype=np.float32)
    vocal = np.zeros_like(audio)
    vocal[int(burst_start_s * sample_rate):int(burst_end_s * sample_rate)] = 0.8

    context = SegmentationContext(
        audio=audio,
        sample_rate=sample_rate,
        tempo=120.0,
        beat_times=np.arange(0.0, duration_s + 0.001, 0.5),
        bar_times=np.arange(0.0, duration_s + 0.001, 2.0),
        bar_duration=2.0,
        mdd_cut_points_samples=[int(mdd_time_s * sample_rate)],
        energy_threshold=0.5,
        bar_energies=[1.0] * 6,
        config={
            'density': density,
            'energy_percentile': 50,
            'min_segment_s': 2.0,
            'snap_tolerance_ms': snap_tolerance_ms,
            'vad_protection': True,
            'guard_win_ms': 80,
        },
    )
    context.vocal_track = vocal
    return context


def test_snap_to_beat_keeps_mdd_cut_when_nearest_beat_has_active_vocal() -> None:
    result = SnapToBeatStrategy().generate_cut_points(_context_with_vocal_burst_at_beat())

    cut_times = [sample / 1000.0 for sample in result.cut_points_samples]

    assert 5.19 in cut_times
    assert 5.0 not in cut_times
    assert result.metadata is not None
    assert result.metadata['snap_stats']['vad_blocked'] == 1
    assert result.lib_flags == [False, False]


def test_high_density_inserted_beat_cuts_respect_vocal_guard() -> None:
    context = _context_with_vocal_burst_at_beat(
        mdd_time_s=8.8,
        snap_tolerance_ms=100,
        density='high',
        burst_start_s=1.92,
        burst_end_s=2.08,
    )

    result = SnapToBeatStrategy().generate_cut_points(context)
    cut_times = [sample / 1000.0 for sample in result.cut_points_samples]

    assert 2.0 not in cut_times


def test_chorus_force_snap_restores_legacy_snap_even_when_vocal_is_active() -> None:
    context = _context_with_vocal_burst_at_beat()
    context.config['chorus_force_snap'] = True

    result = SnapToBeatStrategy().generate_cut_points(context)
    cut_times = [sample / 1000.0 for sample in result.cut_points_samples]

    assert 5.0 in cut_times
    assert 5.19 not in cut_times
    assert result.metadata is not None
    assert result.metadata['snap_stats']['vad_blocked'] == 0


def test_snap_tolerance_is_clamped_to_fraction_of_beat_interval() -> None:
    context = _context_with_vocal_burst_at_beat(
        mdd_time_s=5.24,
        snap_tolerance_ms=500,
        burst_start_s=0.0,
        burst_end_s=0.0,
    )

    result = SnapToBeatStrategy().generate_cut_points(context)
    cut_times = [sample / 1000.0 for sample in result.cut_points_samples]

    assert 5.24 in cut_times
    assert 5.0 not in cut_times
    assert result.metadata is not None
    assert result.metadata['snap_tolerance_ms'] == 200.0


def test_beat_only_bar_cuts_respect_vocal_guard() -> None:
    context = _context_with_vocal_burst_at_beat(
        mdd_time_s=8.8,
        density='medium',
        burst_start_s=3.92,
        burst_end_s=4.08,
    )
    context.config.update({
        'bars_per_cut': 2,
        'vad_protection': True,
        'guard_win_ms': 80,
        'guard_db': 2.5,
    })

    result = BeatOnlyStrategy().generate_cut_points(context)
    cut_times = [sample / 1000.0 for sample in result.cut_points_samples]

    assert 4.0 not in cut_times


def test_hybrid_lib_flags_remap_to_guarded_cut_boundaries() -> None:
    raw_cut_points = [0, 1000, 2000, 3000]
    raw_lib_flags = [True, False, False]
    guarded_cut_points = [0, 1080, 3000]

    remapped = SeamlessSplitter._remap_lib_flags_to_refined_cuts(
        raw_cut_points,
        raw_lib_flags,
        guarded_cut_points,
    )

    assert remapped == [True, False]

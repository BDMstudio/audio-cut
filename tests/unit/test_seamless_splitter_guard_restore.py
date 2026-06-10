#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_seamless_splitter_guard_restore.py
# AI-SUMMARY: Tests VPBD ASR guard restoration when local guard moves cuts into ASR word intervals.

from audio_cut.cutting.refine import CutAdjustment
from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter


def test_restore_guard_point_when_guard_moves_cut_into_word_interval() -> None:
    splitter = SeamlessSplitter(sample_rate=1000)
    points, adjustments = splitter._restore_guard_points_outside_lyrics_words(
        [0, 2000, 10000],
        [
            CutAdjustment(
                raw_time=1.5,
                guard_time=2.0,
                final_time=2.0,
                score=0.8,
                guard_shift_ms=500.0,
                final_shift_ms=500.0,
            )
        ],
        [(1.9, 2.2)],
        sample_count=10000,
        min_gap_s=0.5,
    )

    assert points == [0, 1500, 10000]
    assert adjustments is not None
    assert adjustments[0].final_time == 1.5
    assert adjustments[0].guard_shift_ms == 0.0


def test_restore_guard_point_keeps_cut_when_raw_is_also_inside_word_interval() -> None:
    splitter = SeamlessSplitter(sample_rate=1000)
    points, adjustments = splitter._restore_guard_points_outside_lyrics_words(
        [0, 2100, 10000],
        [
            CutAdjustment(
                raw_time=2.0,
                guard_time=2.1,
                final_time=2.1,
                score=0.8,
                guard_shift_ms=100.0,
                final_shift_ms=100.0,
            )
        ],
        [(1.9, 2.2)],
        sample_count=10000,
        min_gap_s=0.5,
    )

    assert points == [0, 2100, 10000]
    assert adjustments is None

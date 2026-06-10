#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_lyrics_cut_protection.py
# AI-SUMMARY: Tests ASR lyric boundaries as soft priors and word intervals as VPBD guard constraints.

from __future__ import annotations

from audio_cut.cutting.refine import CutAdjustment
from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter


def test_lyrics_soft_prior_collects_sentence_and_vad_boundaries_not_words() -> None:
    lyrics_alignment = {
        "timeline": {
            "words": [
                {"start_s": 19.9, "end_s": 20.38, "text": "人"},
                {"start_s": 23.53, "end_s": 24.86, "text": "笑"},
            ],
            "sentences": [
                {"start_s": 19.32, "end_s": 23.53, "text": "情人为我添伤痕，"},
            ],
            "vad_regions": [
                {"start_s": 19.32, "end_s": 28.54, "kind": "singing"},
            ],
        }
    }

    assert SeamlessSplitter._collect_lyrics_boundary_times(lyrics_alignment) == [19.32, 23.53, 28.54]


def test_lyrics_word_intervals_are_collected_for_guard_constraints() -> None:
    lyrics_alignment = {
        "timeline": {
            "words": [
                {"start_s": 19.9, "end_s": 20.38, "text": "人"},
                {"start_s": 23.53, "end_s": 24.86, "text": "笑"},
            ]
        }
    }

    assert SeamlessSplitter._collect_lyrics_word_intervals(lyrics_alignment) == [
        (19.9, 20.38),
        (23.53, 24.86),
    ]


def test_guard_restore_reverts_only_when_raw_time_was_outside_word_interval() -> None:
    splitter = SeamlessSplitter(sample_rate=1000)

    points, adjustments = splitter._restore_guard_points_outside_lyrics_words(
        [0, 20030, 30000],
        [
            CutAdjustment(
                raw_time=19.47,
                guard_time=20.03,
                final_time=20.03,
                score=1.0,
                guard_shift_ms=560.0,
                final_shift_ms=560.0,
            )
        ],
        [(19.9, 20.38)],
        sample_count=30000,
        min_gap_s=1.0,
    )

    assert points == [0, 19470, 30000]
    assert adjustments is not None
    assert adjustments[0].final_time == 19.47
    assert adjustments[0].final_shift_ms == 0.0

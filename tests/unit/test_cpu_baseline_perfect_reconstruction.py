#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_cpu_baseline_perfect_reconstruction.py
# AI-SUMMARY: Verifies sample-level split segments can reconstruct the original CPU audio buffer exactly.

import numpy as np

from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter


def test_split_at_sample_level_reconstructs_original_audio_exactly() -> None:
    """Splitting by final sample cut points must preserve every original sample."""

    splitter = SeamlessSplitter(sample_rate=1000)
    audio = np.linspace(-1.0, 1.0, 1001, dtype=np.float32)
    cut_points = [0, 123, 456, 789, len(audio)]

    segments, flags, debug = splitter._split_at_sample_level(
        audio,
        cut_points,
        segment_flags=[True, False, True, False],
    )

    reconstructed = np.concatenate(segments)

    np.testing.assert_array_equal(reconstructed, audio)
    assert flags == [True, False, True, False]
    assert debug is None

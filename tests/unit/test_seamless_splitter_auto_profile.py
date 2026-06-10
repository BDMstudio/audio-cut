#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_seamless_splitter_auto_profile.py
# AI-SUMMARY: Tests SeamlessSplitter applies smart_cut auto/manual profiles from feature cache.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from vocal_smart_splitter.utils.config_manager import get_config, reset_runtime_config, set_runtime_config


def _cache() -> SimpleNamespace:
    return SimpleNamespace(
        bpm_features=SimpleNamespace(main_bpm=142.0),
        global_mdd=0.58,
        rms_series=np.asarray([0.40, 0.52, 0.47], dtype=np.float32),
        vocal_coverage_ratio=0.82,
    )


def test_seamless_splitter_applies_auto_profile_runtime_overrides() -> None:
    set_runtime_config(
        {
            "smart_cut.profile": "auto",
            "smart_cut.cut_style": "rhythmic",
            "smart_cut.target_duration_s": [4.0, 10.0],
        }
    )
    try:
        meta = SeamlessSplitter(sample_rate=44100)._apply_smart_cut_runtime(_cache())

        assert meta["style"] == "rap"
        assert get_config("global_planner.target_min_s") == 4.0
        assert get_config("global_planner.hard_max_s") == 15.0
        assert get_config("segment_layout.soft_max_s") == 10.0
        assert get_config("quality_control.segment_max_duration") == 15.0
        assert get_config("phrase_boundary.weights.beat_affinity") >= 0.18
    finally:
        reset_runtime_config()


def test_seamless_splitter_manual_profile_takes_priority_over_auto_estimate() -> None:
    set_runtime_config({"smart_cut.profile": "ballad", "smart_cut.target_duration_s": [5.0, 12.0]})
    try:
        meta = SeamlessSplitter(sample_rate=44100)._apply_smart_cut_runtime(_cache())

        assert meta is None
        assert get_config("meta.profile") == "ballad"
        assert get_config("pure_vocal_detection.min_pause_duration") == 0.6
        assert get_config("phrase_boundary.weights.acoustic_pause") == 0.40
        assert get_config("phrase_boundary.weights.beat_affinity") == 0.05
    finally:
        reset_runtime_config()


def test_seamless_splitter_derives_vocal_coverage_for_auto_profile() -> None:
    cache = SimpleNamespace(
        bpm_features=SimpleNamespace(main_bpm=142.0),
        global_mdd=0.58,
        rms_series=np.asarray([0.40, 0.52, 0.47], dtype=np.float32),
    )
    vocal = np.ones(44100, dtype=np.float32)
    set_runtime_config({"smart_cut.profile": "auto"})
    try:
        meta = SeamlessSplitter(sample_rate=44100)._apply_smart_cut_runtime(cache, vocal_track=vocal)

        assert meta["features"]["vocal_coverage_ratio"] > 0.9
        assert meta["style"] == "rap"
    finally:
        reset_runtime_config()

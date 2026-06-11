#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_seamless_splitter_intent_runtime.py
# AI-SUMMARY: Verifies SeamlessSplitter applies v2.8 intent overrides after profile weights.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from vocal_smart_splitter.utils.config_manager import get_config, reset_runtime_config, set_runtime_config


def _cache() -> SimpleNamespace:
    return SimpleNamespace(
        bpm_features=SimpleNamespace(main_bpm=108.0),
        global_mdd=0.38,
        rms_series=np.asarray([0.2, 0.42, 0.31], dtype=np.float32),
        vocal_coverage_ratio=0.56,
        beat_times=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
    )


def test_splitter_applies_alignment_after_auto_profile() -> None:
    reset_runtime_config()
    set_runtime_config({"smart_cut.alignment": "beat", "smart_cut.segments": "many"})
    try:
        splitter = SeamlessSplitter(sample_rate=44100)
        meta = splitter._apply_smart_cut_runtime(_cache(), vocal_track=np.zeros(44100, dtype=np.float32))

        assert get_config("phrase_boundary.weights.beat_affinity") == 0.32
        assert get_config("vpbd.beat_candidates.base_score") == 0.65
        assert get_config("global_planner.beat_conflict_weight") == 0.3
        assert get_config("global_planner.target_min_s") == 3.0
        assert splitter._last_intent_meta["alignment"] == 1.0
        assert splitter._last_intent_meta["target_duration_s"] == [3.0, 8.0]
        assert meta["alignment"] == {"value": 1.0, "raw": "beat"}
    finally:
        reset_runtime_config()


def test_splitter_balanced_alignment_keeps_existing_weight_overrides_empty() -> None:
    reset_runtime_config()
    set_runtime_config({"smart_cut.alignment": "balanced"})
    try:
        splitter = SeamlessSplitter(sample_rate=44100)
        splitter._apply_smart_cut_runtime(_cache(), vocal_track=np.zeros(44100, dtype=np.float32))

        assert splitter._last_intent_meta["alignment"] == 0.5
        assert splitter._last_intent_meta["applied_overrides"] == []
    finally:
        reset_runtime_config()

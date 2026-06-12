#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_seamless_splitter_intent_runtime.py
# AI-SUMMARY: Verifies SeamlessSplitter applies v2.8 intent overrides after profile weights.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from audio_cut.analysis.features_cache import TrackFeatureCache
from audio_cut.cutting.segment_layout_refiner import LayoutConfig, Segment, refine_layout
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


def _track_feature_cache(*, sample_rate: int, duration_s: float, hop_s: float = 0.05) -> TrackFeatureCache:
    frame_count = int(duration_s / hop_s) + 1
    ones = np.ones(frame_count, dtype=np.float32)
    zeros = np.zeros(frame_count, dtype=np.float32)
    return TrackFeatureCache(
        sr=sample_rate,
        hop_length=max(1, int(round(sample_rate * hop_s))),
        hop_s=hop_s,
        duration_s=duration_s,
        rms_series=ones,
        spectral_flatness=zeros.copy(),
        onset_envelope=zeros.copy(),
        onset_strength=zeros.copy(),
        onset_frames=np.array([], dtype=np.int64),
        rms_max=1.0,
        onset_max=0.0,
        bpm_features=None,
        tempo_curve=None,
        beat_times=np.array([], dtype=np.float32),
        global_mdd=0.0,
        mdd_series=zeros.copy(),
    )


def test_layout_feature_cache_uses_vocal_rms_for_secondary_splits() -> None:
    sample_rate = 1000
    mix_cache = _track_feature_cache(sample_rate=sample_rate, duration_s=30.0)
    vocal = np.full(sample_rate * 30, 0.4, dtype=np.float32)
    center = int(20.0 * sample_rate)
    half_width = int(2.0 * sample_rate)
    for idx in range(center - half_width, center + half_width):
        distance = abs(idx - center) / float(half_width)
        vocal[idx] = 0.4 * distance

    layout_features = SeamlessSplitter._build_vocal_layout_feature_cache(mix_cache, vocal)
    result = refine_layout(
        [Segment(0.0, 30.0, "human"), Segment(30.0, 35.0, "music")],
        [],
        config=LayoutConfig(enable=True, soft_max_s=15.0, min_gap_s=1.0),
        sample_rate=sample_rate,
        features=layout_features,
    )

    boundaries = [seg.end for seg in result.segments[:-1]]
    assert any(abs(boundary - 20.0) <= 0.25 for boundary in boundaries)


def test_local_boundary_refine_does_not_create_micro_segment() -> None:
    reset_runtime_config()
    set_runtime_config({"segment_layout.micro_merge_s": 2.0})
    try:
        sample_rate = 1000
        splitter = SeamlessSplitter(sample_rate=sample_rate)
        vocal = np.full(sample_rate * 18, 0.5, dtype=np.float32)
        vocal[12480:12520] = 0.001
        boundaries = [0, 12000, 14000, 18000]

        refined = splitter._refine_boundaries_local_valley(
            boundaries,
            vocal,
            {"search_radius_ms": 1000, "window_ms": 20, "min_drop_db": 3.0},
            min_gap_s=1.0,
        )

        assert refined == boundaries
    finally:
        reset_runtime_config()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_alignment_overrides.py
# AI-SUMMARY: Verifies v2.8 intent alignment and segment-density derivation helpers.

from __future__ import annotations

import pytest

from audio_cut.config.auto_profile import (
    ALIGNMENT_STOPS,
    BEAT_POLE,
    LYRIC_POLE,
    derive_alignment_overrides,
    derive_smart_cut_overrides,
    resolve_alignment,
    resolve_smart_cut_intent,
)


BASE_WEIGHTS = {
    "acoustic_pause": 0.35,
    "asr_gap": 0.20,
    "sentence_end": 0.15,
    "beat_affinity": 0.08,
    "mdd_affinity": 0.10,
    "breath": 0.12,
    "inside_word_penalty": 0.80,
    "singing_penalty": 0.50,
}


def test_resolve_alignment_accepts_stops_float_and_none() -> None:
    assert ALIGNMENT_STOPS["beat_lean"] == 0.75
    assert resolve_alignment(None) == 0.5
    assert resolve_alignment("balanced") == 0.5
    assert resolve_alignment("beat_lean") == 0.75
    assert resolve_alignment("0.8") == 0.8
    assert resolve_alignment(1.2) == 1.0
    assert resolve_alignment(-0.1) == 0.0


def test_resolve_alignment_rejects_unknown_stop() -> None:
    with pytest.raises(ValueError, match="smart_cut.alignment"):
        resolve_alignment("chorus")


def test_alignment_midpoint_is_identity_override() -> None:
    assert derive_alignment_overrides(0.5, BASE_WEIGHTS) == {}


def test_alignment_endpoints_match_poles() -> None:
    lyric = derive_alignment_overrides(0.0, BASE_WEIGHTS)
    beat = derive_alignment_overrides(1.0, BASE_WEIGHTS)

    assert lyric["phrase_boundary.weights.acoustic_pause"] == LYRIC_POLE["acoustic_pause"]
    assert lyric["phrase_boundary.weights.asr_gap"] == LYRIC_POLE["asr_gap"]
    assert lyric["vpbd.beat_candidates.base_score"] == 0.0
    assert lyric["global_planner.beat_conflict_weight"] == 0.0

    assert beat["phrase_boundary.weights.beat_affinity"] == BEAT_POLE["beat_affinity"]
    assert beat["phrase_boundary.weights.inside_word_penalty"] == BEAT_POLE["inside_word_penalty"]
    assert beat["vpbd.beat_candidates.base_score"] == 0.65
    assert beat["global_planner.beat_conflict_weight"] == 0.3


def test_alignment_weight_monotonicity() -> None:
    samples = [derive_alignment_overrides(a, BASE_WEIGHTS) for a in (0.0, 0.25, 0.75, 1.0)]

    beat_values = [item["phrase_boundary.weights.beat_affinity"] for item in samples]
    asr_values = [item["phrase_boundary.weights.asr_gap"] for item in samples]

    assert beat_values == sorted(beat_values)
    assert asr_values == sorted(asr_values, reverse=True)


def test_segment_density_stops_drive_duration_derivation() -> None:
    overrides = derive_smart_cut_overrides({"segments": "many"})

    assert overrides["global_planner.target_min_s"] == 3.0
    assert overrides["global_planner.target_max_s"] == 8.0
    assert overrides["segment_layout.soft_min_s"] == 3.0
    assert overrides["segment_layout.soft_max_s"] == 8.0


def test_target_duration_wins_over_segments_with_warning() -> None:
    with pytest.warns(UserWarning, match="target_duration_s"):
        overrides = derive_smart_cut_overrides(
            {"segments": "many", "target_duration_s": [6.0, 14.0]},
            explicit_keys={"smart_cut.segments", "smart_cut.target_duration_s"},
        )

    assert overrides["global_planner.target_min_s"] == 6.0
    assert overrides["global_planner.target_max_s"] == 14.0


def test_cut_style_maps_to_axes_and_warns() -> None:
    with pytest.warns(DeprecationWarning, match="cut_style"):
        rhythmic = resolve_smart_cut_intent({"cut_style": "rhythmic"})
    with pytest.warns(DeprecationWarning, match="cut_style"):
        dense = resolve_smart_cut_intent({"cut_style": "dense"})
    with pytest.warns(DeprecationWarning, match="alignment"):
        explicit = resolve_smart_cut_intent({"cut_style": "rhythmic", "alignment": "lyric"})

    assert rhythmic["alignment"] == 0.7
    assert dense["target_duration_s"] == [3.0, 8.0]
    assert explicit["alignment"] == 0.0

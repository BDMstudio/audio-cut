#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_auto_profile.py
# AI-SUMMARY: Tests v2.7 AutoProfile style estimation and profile-weight interpolation.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from audio_cut.config.auto_profile import (
    build_auto_profile_overrides,
    estimate_style,
)


def _cache(
    *,
    bpm: float,
    mdd: float,
    rms: list[float],
    vocal_coverage: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        bpm_features=SimpleNamespace(main_bpm=bpm),
        global_mdd=mdd,
        rms_series=np.asarray(rms, dtype=np.float32),
        vocal_coverage_ratio=vocal_coverage,
    )


def test_estimate_style_classifies_typical_feature_vectors() -> None:
    assert estimate_style(_cache(bpm=72, mdd=0.26, rms=[0.35, 0.36, 0.34], vocal_coverage=0.42)).profile == "ballad"
    assert estimate_style(_cache(bpm=108, mdd=0.38, rms=[0.20, 0.42, 0.31], vocal_coverage=0.56)).profile == "pop"
    assert estimate_style(_cache(bpm=128, mdd=0.25, rms=[0.05, 0.90, 0.08, 0.95], vocal_coverage=0.38)).profile == "edm"
    assert estimate_style(_cache(bpm=142, mdd=0.58, rms=[0.40, 0.52, 0.47], vocal_coverage=0.82)).profile == "rap"


def test_estimate_style_low_confidence_falls_back_to_pop() -> None:
    estimate = estimate_style(_cache(bpm=0, mdd=0.0, rms=[], vocal_coverage=0.0))

    assert estimate.profile == "pop"
    assert estimate.confidence < 0.6
    assert estimate.fallback_reason == "low_confidence"


def test_auto_profile_interpolates_between_neighboring_profile_anchors() -> None:
    estimate = estimate_style(_cache(bpm=95, mdd=0.35, rms=[0.2, 0.4, 0.3], vocal_coverage=0.5))

    overrides = build_auto_profile_overrides(estimate)

    assert overrides["meta.auto_profile"]["style"] == "pop"
    assert overrides["meta.auto_profile"]["anchor_weights"] == {"ballad": 0.3, "pop": 0.7}
    assert 0.24 < overrides["pure_vocal_detection.peak_relative_threshold_ratio"] < 0.26


def test_style_weight_overrides_shift_with_cut_style_and_profile() -> None:
    rap = estimate_style(_cache(bpm=142, mdd=0.58, rms=[0.4, 0.52, 0.47], vocal_coverage=0.82))
    ballad = estimate_style(_cache(bpm=72, mdd=0.26, rms=[0.35, 0.36, 0.34], vocal_coverage=0.42))

    rap_weights = build_auto_profile_overrides(rap, cut_style="rhythmic")
    ballad_weights = build_auto_profile_overrides(ballad, cut_style="natural")

    assert rap_weights["phrase_boundary.weights.beat_affinity"] > ballad_weights["phrase_boundary.weights.beat_affinity"]
    assert rap_weights["phrase_boundary.weights.breath"] > ballad_weights["phrase_boundary.weights.breath"]
    assert ballad_weights["phrase_boundary.weights.acoustic_pause"] > rap_weights["phrase_boundary.weights.acoustic_pause"]
    assert ballad_weights["phrase_boundary.weights.sentence_end"] > rap_weights["phrase_boundary.weights.sentence_end"]

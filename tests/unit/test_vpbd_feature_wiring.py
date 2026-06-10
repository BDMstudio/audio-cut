#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_vpbd_feature_wiring.py
# AI-SUMMARY: Tests VPBD wires cached MDD/RMS features into boundary scoring.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate
from audio_cut.lyrics.models import LyricsTimeline
from vocal_smart_splitter.core.vocal_phrase_boundary_detector import VocalPhraseBoundaryDetector


def test_vpbd_score_candidates_uses_cached_mdd_valleys() -> None:
    detector = VocalPhraseBoundaryDetector(sample_rate=1000)
    cache = SimpleNamespace(
        beat_times=np.array([], dtype=np.float32),
        mdd_series=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        rms_series=np.zeros(3, dtype=np.float32),
        hop_s=1.0,
    )

    scored = detector._score_candidates(
        candidates=[CutCandidate(1.0, 0.5, CandidateSource.ACOUSTIC_PAUSE)],
        timeline=LyricsTimeline(duration_s=3.0, source="none"),
        feature_cache=cache,
    )

    assert scored[0].features["mdd_affinity"] > 0.0


def test_vpbd_score_candidates_uses_cached_rms_for_vocal_risk() -> None:
    detector = VocalPhraseBoundaryDetector(sample_rate=1000)
    rms = np.full(100, 0.1, dtype=np.float32)
    rms[39:43] = 1.0
    cache = SimpleNamespace(
        beat_times=np.array([], dtype=np.float32),
        mdd_series=np.zeros(100, dtype=np.float32),
        rms_series=rms,
        hop_s=0.05,
    )

    scored = detector._score_candidates(
        candidates=[CutCandidate(2.0, 0.5, CandidateSource.ACOUSTIC_PAUSE)],
        timeline=LyricsTimeline(duration_s=5.0, source="none"),
        feature_cache=cache,
    )

    assert scored[0].features["vocal_cut_risk"] > 0.8

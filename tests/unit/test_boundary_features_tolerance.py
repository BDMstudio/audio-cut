#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_boundary_features_tolerance.py
# AI-SUMMARY: Tests VPBD boundary feature tolerances and risk features used by scoring.

from __future__ import annotations

import numpy as np

from audio_cut.analysis.boundary_features import BoundaryFeatureExtractor
from audio_cut.lyrics.models import LyricsTimeline, Sentence, Word


def _timeline() -> LyricsTimeline:
    return LyricsTimeline(
        words=[Word("line", 1.0, 2.0, confidence=0.95)],
        sentences=[Sentence("line.", 1.0, 2.0, confidence=1.0)],
        duration_s=4.0,
        source="fake",
    )


def test_sentence_end_tolerance_survives_asr_timestamp_jitter() -> None:
    extractor = BoundaryFeatureExtractor(timeline=_timeline())

    features = extractor.extract(1.85)

    assert features.sentence_end > 0.0


def test_inside_word_penalty_softens_near_word_edges() -> None:
    extractor = BoundaryFeatureExtractor(timeline=_timeline(), word_edge_tolerance_ms=60.0)

    center = extractor.extract(1.50)
    near_edge = extractor.extract(1.98)

    assert center.inside_word_penalty == 1.0
    assert 0.0 < near_edge.inside_word_penalty < center.inside_word_penalty


def test_vocal_cut_risk_uses_rms_percentile_window() -> None:
    rms = np.full(100, 0.1, dtype=np.float32)
    rms[39:43] = 1.0
    extractor = BoundaryFeatureExtractor(
        timeline=LyricsTimeline(duration_s=5.0, source="none"),
        rms_series=rms,
        hop_s=0.05,
    )

    quiet = extractor.extract(0.25)
    loud = extractor.extract(2.0)

    assert quiet.vocal_cut_risk < 0.2
    assert loud.vocal_cut_risk > 0.8


def test_beat_conflict_marks_boundaries_far_from_beats() -> None:
    extractor = BoundaryFeatureExtractor(
        timeline=LyricsTimeline(duration_s=5.0, source="none"),
        beat_times=[1.0, 2.0],
        affinity_tolerance_s=0.12,
    )

    on_beat = extractor.extract(1.0)
    off_beat = extractor.extract(1.35)

    assert on_beat.beat_conflict == 0.0
    assert off_beat.beat_conflict > 0.8

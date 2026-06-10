#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_boundary_data_models.py
# AI-SUMMARY: Tests VPBD candidate and boundary feature containers.

from audio_cut.analysis.boundary_features import BoundaryFeatures
from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate


def test_cut_candidate_clamps_score_and_serializes_source() -> None:
    candidate = CutCandidate(
        t=1.25,
        score=1.5,
        source="lyrics_gap",
        reasons=["word gap"],
        features={"asr_gap": 0.9},
        meta={"word_left": "hello"},
    )

    assert candidate.score == 1.0
    assert candidate.source == CandidateSource.LYRICS_GAP
    assert candidate.to_dict()["source"] == "lyrics_gap"


def test_boundary_features_clamp_to_unit_range() -> None:
    features = BoundaryFeatures(
        acoustic_pause=2.0,
        asr_gap=-1.0,
        sentence_end=0.5,
    )

    assert features.acoustic_pause == 1.0
    assert features.asr_gap == 0.0
    assert features.to_dict()["sentence_end"] == 0.5

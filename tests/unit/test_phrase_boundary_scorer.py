#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_phrase_boundary_scorer.py
# AI-SUMMARY: Tests VPBD phrase boundary scoring and candidate debug serialization.

import json

from audio_cut.analysis.boundary_features import BoundaryFeatures
from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate
from audio_cut.cutting.phrase_boundary_scorer import (
    PhraseBoundaryScorer,
    write_candidate_debug_json,
)


def test_phrase_boundary_scorer_clamps_positive_and_penalty_weights() -> None:
    scorer = PhraseBoundaryScorer(
        weights={
            "acoustic_pause": 0.5,
            "asr_gap": 0.4,
            "sentence_end": 0.3,
            "inside_word_penalty": 1.0,
            "singing_penalty": 1.0,
        }
    )

    good = scorer.score(
        BoundaryFeatures(acoustic_pause=1.0, asr_gap=1.0, sentence_end=1.0)
    )
    bad = scorer.score(
        BoundaryFeatures(acoustic_pause=1.0, inside_word_penalty=1.0, singing_penalty=1.0)
    )

    assert good == 1.0
    assert bad == 0.0


def test_phrase_boundary_scorer_reads_weights_from_config_mapping() -> None:
    scorer = PhraseBoundaryScorer.from_config(
        {"weights": {"asr_gap": 0.25, "inside_word_penalty": 0.5}}
    )

    assert scorer.weights["asr_gap"] == 0.25
    assert scorer.weights["inside_word_penalty"] == 0.5


def test_score_candidate_attaches_features_and_reasons() -> None:
    scorer = PhraseBoundaryScorer(weights={"asr_gap": 0.5, "sentence_end": 0.5})
    candidate = CutCandidate(1.2, 0.0, CandidateSource.LYRICS_GAP)
    features = BoundaryFeatures(asr_gap=1.0, sentence_end=1.0)

    scored = scorer.score_candidate(candidate, features)

    assert scored.score == 1.0
    assert scored.features == features.to_dict()
    assert "vpbd_score" in scored.reasons


def test_phrase_boundary_scorer_applies_breath_weight() -> None:
    scorer = PhraseBoundaryScorer(weights={"breath": 0.2})

    assert scorer.score(BoundaryFeatures(breath=1.0)) == 0.2


def test_write_candidate_debug_json(tmp_path) -> None:
    candidate = CutCandidate(
        1.2,
        0.8,
        CandidateSource.LYRICS_GAP,
        reasons=["asr_gap"],
        features={"asr_gap": 1.0},
    )
    path = tmp_path / "candidates.json"

    write_candidate_debug_json([candidate], path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["candidates"][0]["source"] == "lyrics_gap"
    assert payload["candidates"][0]["features"]["asr_gap"] == 1.0

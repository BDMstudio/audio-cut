#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_boundary_features.py
# AI-SUMMARY: Tests VPBD candidate generation and boundary feature extraction.

from pathlib import Path

from audio_cut.analysis.boundary_features import BoundaryFeatureExtractor
from audio_cut.cutting.candidate_adapters import adapt_legacy_acoustic_candidates
from audio_cut.cutting.cut_candidate import CandidateSource
from audio_cut.lyrics.candidates import LyricsBoundaryCandidateGenerator
from audio_cut.lyrics.models import LyricsTimeline, Sentence, VadRegion, Word


def _timeline() -> LyricsTimeline:
    return LyricsTimeline(
        words=[
            Word("hello", 0.50, 0.90, confidence=0.95),
            Word("world", 1.40, 1.80, confidence=0.94),
            Word("again", 3.00, 3.50, confidence=0.40),
        ],
        sentences=[
            Sentence("hello world!", 0.50, 1.80, confidence=0.90),
        ],
        vad_regions=[
            VadRegion(0.45, 1.90, confidence=0.92, kind="singing"),
            VadRegion(2.80, 3.70, confidence=0.40, kind="singing"),
        ],
        duration_s=5.0,
        source="fake",
    )


def test_adapt_legacy_acoustic_candidates_preserves_source_score_and_meta() -> None:
    candidates = adapt_legacy_acoustic_candidates(
        [(1.25, 0.7, {"rms_valley_db": -45.0})],
        source=CandidateSource.ACOUSTIC_PAUSE,
    )

    assert candidates[0].source == CandidateSource.ACOUSTIC_PAUSE
    assert candidates[0].score == 0.7
    assert candidates[0].meta["rms_valley_db"] == -45.0
    assert "legacy_acoustic" in candidates[0].reasons


def test_lyrics_candidate_generator_emits_word_gap_sentence_and_mvad_boundaries() -> None:
    generator = LyricsBoundaryCandidateGenerator(min_word_gap_s=0.3)

    candidates = generator.generate(_timeline())
    sources = [candidate.source for candidate in candidates]

    assert CandidateSource.LYRICS_GAP in sources
    assert CandidateSource.SENTENCE_END in sources
    assert CandidateSource.MVAD_BOUNDARY in sources
    sentence_candidate = next(c for c in candidates if c.source == CandidateSource.SENTENCE_END)
    assert "punctuation_end" in sentence_candidate.reasons


def test_boundary_feature_extractor_penalizes_high_confidence_word_and_singing() -> None:
    extractor = BoundaryFeatureExtractor(
        timeline=_timeline(),
        beat_times=[1.0, 2.0],
        mdd_times=[1.6],
    )

    inside_high_conf = extractor.extract(0.70)
    low_conf_region = extractor.extract(3.20)
    gap_boundary = extractor.extract(1.15)

    assert inside_high_conf.inside_word_penalty == 1.0
    assert inside_high_conf.singing_penalty == 1.0
    assert 0.0 < low_conf_region.inside_word_penalty < 1.0
    assert 0.0 < low_conf_region.singing_penalty < 1.0
    assert gap_boundary.asr_gap > 0.0


def test_boundary_feature_extractor_scores_sentence_and_affinities() -> None:
    extractor = BoundaryFeatureExtractor(
        timeline=_timeline(),
        beat_times=[1.80],
        mdd_times=[1.78],
        affinity_tolerance_s=0.05,
    )

    features = extractor.extract(1.80)

    assert features.sentence_end > 0.0
    assert features.beat_affinity == 1.0
    assert features.mdd_affinity > 0.0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_candidate_pool_fusion.py
# AI-SUMMARY: Tests VPBD fuses ASR lyrics boundaries into the unified candidate pool.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate
from audio_cut.lyrics.models import LyricsTimeline, Sentence
from vocal_smart_splitter.core.vocal_phrase_boundary_detector import VocalPhraseBoundaryDetector
from vocal_smart_splitter.utils.config_manager import reset_runtime_config, set_runtime_config


class _NoPauseDetector:
    def detect_pure_vocal_pauses(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def test_asr_sentence_end_candidate_is_selected_when_acoustic_misses(tmp_path, monkeypatch) -> None:
    class _SentenceProvider:
        name = "sentence_provider"

        def align(self, request) -> LyricsTimeline:  # type: ignore[no-untyped-def]
            return LyricsTimeline(
                duration_s=request.duration_s,
                source="fake",
                sentences=[Sentence(text="phrase.", start_s=0.5, end_s=6.0, confidence=1.0)],
            )

    monkeypatch.setattr(
        "vocal_smart_splitter.core.vocal_phrase_boundary_detector.build_lyrics_provider",
        lambda cfg: _SentenceProvider(),
    )
    set_runtime_config(
        {
            "lyrics_alignment.enabled": True,
            "lyrics_alignment.provider": "cli",
            "lyrics_alignment.strict": True,
            "global_planner.hard_min_s": 2.0,
            "global_planner.hard_max_s": 8.0,
            "global_planner.target_min_s": 5.0,
            "global_planner.target_max_s": 7.0,
            "global_planner.rescue_enabled": True,
        }
    )
    try:
        sample_rate = 44100
        duration_s = 12.0
        vocal = np.zeros(int(sample_rate * duration_s), dtype=np.float32)

        result = VocalPhraseBoundaryDetector(sample_rate=sample_rate).detect(
            mode="vpbd_asr",
            vocal_track=vocal,
            original_audio=vocal.copy(),
            pure_vocal_detector=_NoPauseDetector(),
            feature_cache=None,
            vad_segments=[],
            input_path=str(Path(tmp_path) / "sample.wav"),
            output_dir=str(Path(tmp_path) / "out"),
        )
    finally:
        reset_runtime_config()

    sources = {candidate.source for candidate in result.selected_candidates}
    assert CandidateSource.SENTENCE_END in sources
    assert result.boundary_detection["candidate_counts"]["acoustic"] == 0
    assert result.boundary_detection["candidate_counts"]["lyrics"] == 1
    assert result.boundary_detection["candidate_counts"]["merged"] == 1


def test_candidate_pool_merge_dedupes_nearby_candidates_and_tracks_sources() -> None:
    detector = VocalPhraseBoundaryDetector(sample_rate=44100)
    acoustic = CutCandidate(
        t=6.04,
        score=0.4,
        source=CandidateSource.ACOUSTIC_PAUSE,
        reasons=["legacy_acoustic"],
        meta={"pause_type": "true_pause"},
    )
    sentence = CutCandidate(
        t=6.0,
        score=0.85,
        source=CandidateSource.SENTENCE_END,
        reasons=["sentence_end", "punctuation_end"],
        meta={"text": "phrase."},
    )

    merged = detector._merge_candidate_pool([acoustic], [sentence], tolerance_s=0.12)

    assert len(merged) == 1
    assert merged[0].t == 6.0
    assert merged[0].source == CandidateSource.SENTENCE_END
    assert set(merged[0].meta["sources"]) == {"acoustic_pause", "sentence_end"}
    assert merged[0].meta["source_count"] == 2

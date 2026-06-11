#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_candidate_pool_fusion.py
# AI-SUMMARY: Tests VPBD fuses ASR lyrics boundaries into the unified candidate pool.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate
from audio_cut.lyrics.models import LyricsTimeline, Sentence
from vocal_smart_splitter.core.vocal_phrase_boundary_detector import VocalPhraseBoundaryDetector
from vocal_smart_splitter.utils.config_manager import reset_runtime_config, set_runtime_config


class _NoPauseDetector:
    def detect_pure_vocal_pauses(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class _DebugPauseDetector:
    def detect_pure_vocal_pauses(self, *args: Any, **kwargs: Any) -> list[Any]:
        return [
            SimpleNamespace(
                start_time=5.94,
                end_time=6.06,
                cut_point=6.0,
                confidence=1.0,
                duration=0.12,
                pause_type="breath",
            )
        ]


def _debug_feature_cache() -> SimpleNamespace:
    return SimpleNamespace(
        beat_times=np.arange(0.0, 12.001, 0.5, dtype=np.float32),
        rms_series=np.array(
            [0.05] * 40
            + [0.90] * 40
            + [0.92] * 40
            + [0.94] * 40
            + [0.91] * 40
            + [0.05] * 40,
            dtype=np.float32,
        ),
        hop_s=0.05,
        duration_s=12.0,
        mdd_series=np.full(240, 0.5, dtype=np.float32),
    )


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


def test_legacy_candidate_pool_keeps_lyrics_out_of_planner(tmp_path, monkeypatch) -> None:
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
            "vpbd.candidate_pool": "legacy",
            "lyrics_alignment.enabled": True,
            "lyrics_alignment.provider": "cli",
            "lyrics_alignment.strict": True,
            "global_planner.hard_min_s": 2.0,
            "global_planner.hard_max_s": 8.0,
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

    assert result.selected_candidates == []
    assert result.boundary_detection["candidate_pool"] == "legacy"
    assert result.boundary_detection["candidate_counts"]["lyrics"] == 1
    assert result.boundary_detection["candidate_counts"]["merged"] == 0


class _BreathMddDetector:
    def detect_pure_vocal_pauses(self, *args: Any, **kwargs: Any) -> list[Any]:
        return [
            SimpleNamespace(
                start_time=5.8,
                end_time=6.2,
                cut_point=6.0,
                confidence=0.8,
                duration=0.4,
                pause_type="breath_mdd",
            )
        ]


def test_legacy_candidate_pool_preserves_v26_acoustic_candidates_with_pause_type(tmp_path) -> None:
    set_runtime_config(
        {
            "vpbd.candidate_pool": "legacy",
            "vpbd.breath_score_scale": 0.0,
            "lyrics_alignment.enabled": False,
            "global_planner.hard_min_s": 2.0,
            "global_planner.hard_max_s": 8.0,
            "global_planner.target_min_s": 5.0,
            "global_planner.target_max_s": 7.0,
        }
    )
    try:
        sample_rate = 44100
        duration_s = 12.0
        vocal = np.zeros(int(sample_rate * duration_s), dtype=np.float32)

        result = VocalPhraseBoundaryDetector(sample_rate=sample_rate).detect(
            mode="vpbd_acoustic",
            vocal_track=vocal,
            original_audio=vocal.copy(),
            pure_vocal_detector=_BreathMddDetector(),
            feature_cache=None,
            vad_segments=[],
            input_path=str(Path(tmp_path) / "sample.wav"),
            output_dir=str(Path(tmp_path) / "out"),
        )
    finally:
        reset_runtime_config()

    assert result.boundary_detection["candidate_pool"] == "legacy"
    assert result.boundary_detection["candidate_counts"]["acoustic"] == 1
    assert result.boundary_detection["candidate_counts"]["merged"] == 1
    assert result.selected_candidates[0].source == CandidateSource.ACOUSTIC_PAUSE

def test_candidate_debug_json_records_sources_and_features(tmp_path, monkeypatch) -> None:
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
            "vpbd.candidate_debug_json": True,
            "lyrics_alignment.enabled": True,
            "lyrics_alignment.provider": "cli",
            "lyrics_alignment.strict": True,
            "global_planner.hard_min_s": 2.0,
            "global_planner.hard_max_s": 8.0,
            "global_planner.target_min_s": 5.0,
            "global_planner.target_max_s": 7.0,
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
            pure_vocal_detector=_DebugPauseDetector(),
            feature_cache=_debug_feature_cache(),
            vad_segments=[],
            input_path=str(Path(tmp_path) / "sample.wav"),
            output_dir=str(Path(tmp_path) / "out"),
        )
    finally:
        reset_runtime_config()

    debug_path = Path(result.boundary_detection["candidate_debug_path"])
    payload = __import__("json").loads(debug_path.read_text(encoding="utf-8"))
    all_sources = {
        source
        for item in payload["candidates"]
        for source in [item["source"], *item.get("meta", {}).get("sources", [])]
    }
    assert debug_path.exists()
    assert {"sentence_end", "breath", "beat"}.issubset(all_sources)
    assert all("features" in item for item in payload["candidates"])
    assert any(item["features"].get("breath", 0.0) > 0.0 for item in payload["candidates"])
    assert any("vocal_cut_risk" in item["features"] for item in payload["candidates"])

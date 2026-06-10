#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/integration/test_pipeline_vpbd_asr_fake_provider.py
# AI-SUMMARY: Integration test for vpbd_asr using a fixture-backed lyrics provider.

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audio_cut.api import _build_manifest
from vocal_smart_splitter.core.enhanced_vocal_separator import SeparationResult
from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from vocal_smart_splitter.core.vocal_phrase_boundary_detector import VocalPhraseBoundaryDetector
from vocal_smart_splitter.utils.config_manager import reset_runtime_config, set_runtime_config


class _FakeSeparator:
    def separate_for_detection(self, original_audio, gpu_context=None):
        return SeparationResult(
            vocal_track=np.asarray(original_audio, dtype=np.float32),
            instrumental_track=np.zeros_like(original_audio, dtype=np.float32),
            separation_confidence=1.0,
            backend_used="fake",
            processing_time=0.0,
            quality_metrics={},
            feature_cache=None,
            vad_segments=[],
            gpu_meta={"gpu_pipeline_used": False},
        )


class _FakePauseDetector:
    def detect_pure_vocal_pauses(self, *args, **kwargs):
        return [
            SimpleNamespace(
                start_time=5.3,
                end_time=5.7,
                cut_point=5.5,
                confidence=0.6,
                duration=0.4,
            )
        ]


class _PriorityPauseDetector:
    def detect_pure_vocal_pauses(self, *args, **kwargs):
        return [
            SimpleNamespace(
                start_time=4.34,
                end_time=4.46,
                cut_point=4.4,
                confidence=1.0,
                duration=0.12,
                pause_type="breath",
            ),
            SimpleNamespace(
                start_time=5.3,
                end_time=5.7,
                cut_point=5.5,
                confidence=1.0,
                duration=0.4,
                pause_type="true_pause",
            ),
        ]


def _priority_feature_cache() -> SimpleNamespace:
    return SimpleNamespace(
        beat_times=np.arange(0.0, 8.001, 0.5, dtype=np.float32),
        rms_series=np.full(160, 0.8, dtype=np.float32),
        hop_s=0.05,
        duration_s=8.0,
        mdd_series=np.full(160, 0.5, dtype=np.float32),
    )


@pytest.fixture(autouse=True)
def _runtime_config():
    fixture = "tests/fixtures/lyrics/simple_song_timeline.json"
    set_runtime_config(
        {
            "gpu_pipeline.enable": False,
            "segment_layout.enable": False,
            "quality_control.enforce_quiet_cut.enable": False,
            "quality_control.local_boundary_refine.enable": False,
            "quality_control.pure_music_min_duration": 0.0,
            "quality_control.min_split_gap": 1.0,
            "global_planner.hard_min_s": 1.0,
            "global_planner.hard_max_s": 6.0,
            "global_planner.target_min_s": 2.0,
            "global_planner.target_max_s": 5.0,
            "lyrics_alignment.enabled": True,
            "lyrics_alignment.provider": "fake",
            "lyrics_alignment.fixture_path": fixture,
            "lyrics_alignment.strict": True,
        }
    )
    try:
        yield
    finally:
        reset_runtime_config()


def test_vpbd_asr_fake_provider_adds_lyrics_metadata(tmp_path) -> None:
    input_path = tmp_path / "song.wav"
    sf.write(input_path, np.zeros(44100 * 8, dtype=np.float32), 44100, subtype="PCM_16")
    splitter = SeamlessSplitter(sample_rate=44100)
    splitter.separator = _FakeSeparator()
    splitter.pure_vocal_detector = _FakePauseDetector()

    result = splitter.split_audio_seamlessly(
        str(input_path),
        str(tmp_path / "out"),
        mode="vpbd_asr",
        export_plan=("none",),
    )

    assert result["success"] is True
    assert result["boundary_detection"]["actual_mode"] == "vpbd_asr"
    assert result["lyrics_alignment"]["provider"] == "fake"
    assert result["lyrics_alignment"]["word_count"] == 3
    counts = result["boundary_detection"]["candidate_counts"]
    assert counts["acoustic"] > 0
    assert counts["lyrics"] > 0
    assert counts["merged"] <= counts["acoustic"] + counts["lyrics"]
    assert result["lyrics_cut_protection_applied"] is False

    words = result["lyrics_alignment"]["timeline"]["words"]
    selected = result["boundary_detection"]["selected"]
    assert selected
    selected_sources = {cut["source"] for cut in selected}
    assert selected_sources & {"lyrics_gap", "sentence_end", "mvad_boundary"}
    for cut in selected:
        t = float(cut["t"])
        assert not any(
            float(word["start_s"]) < t < float(word["end_s"])
            and float(word.get("confidence") or 0.0) >= 0.8
            for word in words
        )

    manifest = _build_manifest(
        result=result,
        input_path=input_path,
        export_dir=tmp_path / "out",
        mode="vpbd_asr",
        sample_rate=44100,
        channels=1,
        layout_cfg={},
    )
    assert any(segment.get("lyrics") for segment in manifest["segments"])


def test_vpbd_asr_fake_provider_prioritizes_pause_breath_sentence_then_beat(tmp_path) -> None:
    fixture_path = tmp_path / "priority_timeline.json"
    fixture_path.write_text(
        """{
  "duration_s": 8.0,
  "source": "fake",
  "words": [
    {"text": "lead", "start_s": 0.50, "end_s": 0.90, "confidence": 0.95},
    {"text": "hold", "start_s": 1.00, "end_s": 2.30, "confidence": 0.93},
    {"text": "line", "start_s": 3.80, "end_s": 4.40, "confidence": 0.91}
  ],
  "sentences": [
    {"text": "lead hold", "start_s": 0.50, "end_s": 2.30, "confidence": 0.94},
    {"text": "line", "start_s": 3.80, "end_s": 4.40, "confidence": 0.91}
  ],
  "vad_regions": [
    {"start_s": 0.45, "end_s": 2.35, "confidence": 0.90, "kind": "singing"},
    {"start_s": 3.75, "end_s": 4.40, "confidence": 0.87, "kind": "singing"}
  ]
}
""",
        encoding="utf-8",
    )
    set_runtime_config(
        {
            "lyrics_alignment.fixture_path": str(fixture_path),
            "vpbd.candidate_pool": "unified",
            "vpbd.breath_score_scale": 0.6,
            "vpbd.beat_candidates.enable": True,
            "vpbd.beat_candidates.bars_per_cut": 1,
            "vpbd.beat_candidates.base_score": 0.3,
            "global_planner.hard_min_s": 1.0,
            "global_planner.hard_max_s": 6.0,
            "global_planner.target_min_s": 2.0,
            "global_planner.target_max_s": 5.0,
            "global_planner.vocal_risk_weight": 0.0,
            "global_planner.beat_conflict_weight": 0.0,
        }
    )
    sample_rate = 16000
    audio = np.zeros(sample_rate * 8, dtype=np.float32)

    result = VocalPhraseBoundaryDetector(sample_rate=sample_rate).detect(
        mode="vpbd_asr",
        vocal_track=audio,
        original_audio=audio.copy(),
        pure_vocal_detector=_PriorityPauseDetector(),
        feature_cache=_priority_feature_cache(),
        vad_segments=[],
        input_path=str(tmp_path / "sample.wav"),
        output_dir=str(tmp_path / "out"),
    )

    all_candidates = (
        result.boundary_detection["selected"]
        + result.boundary_detection["suppressed"]
    )
    long_pause = _candidate_at(all_candidates, 5.5, source="acoustic_pause")
    breath_sentence = _candidate_with_sources(all_candidates, {"breath", "sentence_end"})
    beat = max(
        (candidate for candidate in all_candidates if _has_source(candidate, "beat")),
        key=lambda candidate: candidate["score"],
    )

    assert result.boundary_detection["candidate_counts"]["beat"] > 0
    assert long_pause["score"] > breath_sentence["score"] > beat["score"]
    selected_sources = {candidate["source"] for candidate in result.boundary_detection["selected"]}
    assert "acoustic_pause" in selected_sources
    assert any(_has_source(candidate, "breath") for candidate in result.boundary_detection["selected"])


def _candidate_at(candidates, t: float, *, source: str):
    for candidate in candidates:
        if candidate["source"] == source and abs(float(candidate["t"]) - t) <= 0.02:
            return candidate
    raise AssertionError(f"candidate not found: {source}@{t}")


def _candidate_with_sources(candidates, sources: set[str]):
    for candidate in candidates:
        if sources.issubset(set(candidate.get("meta", {}).get("sources", []))):
            return candidate
    raise AssertionError(f"candidate with sources not found: {sorted(sources)}")


def _has_source(candidate, source: str) -> bool:
    return candidate["source"] == source or source in candidate.get("meta", {}).get("sources", [])


def test_vpbd_asr_layout_soft_priors_ignore_word_boundaries() -> None:
    boundaries = SeamlessSplitter._collect_lyrics_boundary_times(
        {
            "timeline": {
                "words": [
                    {"text": "a", "start_s": 1.0, "end_s": 1.4},
                    {"text": "b", "start_s": 2.0, "end_s": 2.4},
                ],
                "sentences": [
                    {"text": "a b", "start_s": 1.0, "end_s": 2.4},
                ],
                "vad_regions": [
                    {"start_s": 0.9, "end_s": 2.5, "kind": "singing"},
                ],
            }
        }
    )

    assert boundaries == [0.9, 2.4, 2.5]

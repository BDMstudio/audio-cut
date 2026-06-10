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

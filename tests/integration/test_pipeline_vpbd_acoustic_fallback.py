#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/integration/test_pipeline_vpbd_acoustic_fallback.py
# AI-SUMMARY: Integration test for vpbd_acoustic and vpbd_asr disabled-ASR fallback without model dependencies.

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate
from audio_cut.cutting.global_cut_planner import GlobalCutPlanResult
from audio_cut.exceptions import FireRedProviderError
from audio_cut.lyrics import providers
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


class _FakeVpbdDetectorWithZeroScoreFallback:
    def detect(self, *args, **kwargs):
        bad = CutCandidate(
            t=4.0,
            score=0.0,
            source=CandidateSource.ACOUSTIC_PAUSE,
            reasons=["inside_word"],
        )
        good = CutCandidate(
            t=10.0,
            score=0.6,
            source=CandidateSource.ACOUSTIC_PAUSE,
            reasons=["acoustic_pause"],
        )
        plan = GlobalCutPlanResult(
            cut_times=[0.0, 20.0],
            selected_candidates=[],
            suppressed_candidates=[bad, good],
            rescue_points=[],
            feasible=False,
            metadata={"planner": "rescue", "selected_count": 0, "suppressed_count": 2},
        )
        return SimpleNamespace(
            selected_candidates=[],
            planner_result=plan,
            boundary_detection={
                "mode": "vpbd_acoustic",
                "actual_mode": "vpbd_acoustic",
                "candidate_counts": {"total": 2, "selected": 0, "suppressed": 2},
                "planner": dict(plan.metadata),
                "selected": [],
                "suppressed": [bad.to_dict(), good.to_dict()],
            },
            lyrics_alignment={"enabled": False},
        )


class _TimeoutCliProvider:
    name = "firered_cli"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def is_available(self) -> bool:
        return True

    def align(self, request):
        raise FireRedProviderError("FireRed CLI timed out after 0.01s")


class _FakePauseDetector:
    def detect_pure_vocal_pauses(self, *args, **kwargs):
        return [
            SimpleNamespace(
                start_time=3.8,
                end_time=4.2,
                cut_point=4.0,
                confidence=0.9,
                duration=0.4,
            )
        ]


@pytest.fixture(autouse=True)
def _runtime_config():
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
            "lyrics_alignment.enabled": False,
            "lyrics_alignment.strict": False,
        }
    )
    try:
        yield
    finally:
        reset_runtime_config()


def _write_audio(path, duration_s: float = 8.0, sample_rate: int = 44100) -> None:
    audio = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def test_vpbd_acoustic_uses_detector_metadata(tmp_path) -> None:
    input_path = tmp_path / "song.wav"
    _write_audio(input_path)
    splitter = SeamlessSplitter(sample_rate=44100)
    splitter.separator = _FakeSeparator()
    splitter.pure_vocal_detector = _FakePauseDetector()

    result = splitter.split_audio_seamlessly(
        str(input_path),
        str(tmp_path / "out"),
        mode="vpbd_acoustic",
        export_plan=("none",),
    )

    assert result["success"] is True
    assert result["method"] == "pure_vocal_split_vpbd_acoustic"
    assert result["boundary_detection"]["mode"] == "vpbd_acoustic"
    assert result["boundary_detection"]["planner"]["selected_count"] >= 1
    assert result["lyrics_alignment"]["enabled"] is False


def test_vpbd_rescue_falls_back_to_scored_candidates_not_uniform_cuts(tmp_path) -> None:
    input_path = tmp_path / "song.wav"
    _write_audio(input_path, duration_s=20.0)
    splitter = SeamlessSplitter(sample_rate=44100)
    splitter.separator = _FakeSeparator()
    splitter.pure_vocal_detector = _FakePauseDetector()

    result = splitter.split_audio_seamlessly(
        str(input_path),
        str(tmp_path / "out"),
        mode="vpbd_acoustic",
        export_plan=("none",),
    )

    assert result["success"] is True
    assert result["boundary_detection"]["planner"]["planner"] == "rescue"
    assert result["num_segments"] == 2
    assert result["cut_points_sec"] == pytest.approx([0.0, 4.0, 20.0])


def test_vpbd_asr_disabled_alignment_falls_back_to_acoustic(tmp_path) -> None:
    input_path = tmp_path / "song.wav"
    _write_audio(input_path)
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
    assert result["boundary_detection"]["mode"] == "vpbd_asr"
    assert result["boundary_detection"]["actual_mode"] == "vpbd_acoustic"
    assert result["lyrics_alignment"]["fallback_reason"] == "lyrics_alignment_disabled"

def test_vpbd_asr_unavailable_provider_falls_back_when_not_strict(tmp_path) -> None:
    set_runtime_config(
        {
            "lyrics_alignment.enabled": True,
            "lyrics_alignment.provider": "auto",
            "lyrics_alignment.strict": False,
            "fire_red.provider_order": ["sidecar", "cli", "in_process", "null"],
            "fire_red.endpoint": None,
            "fire_red.cli.executable": None,
        }
    )
    input_path = tmp_path / "song.wav"
    _write_audio(input_path)
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
    assert result["boundary_detection"]["actual_mode"] == "vpbd_acoustic"
    assert result["lyrics_alignment"]["fallback_reason"] == "lyrics_alignment_unavailable"
    assert result["lyrics_alignment"]["provider"] == "null"

def test_vpbd_asr_firered_timeout_falls_back_when_not_strict(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(providers, "FireRedCliProvider", _TimeoutCliProvider)
    set_runtime_config(
        {
            "lyrics_alignment.enabled": True,
            "lyrics_alignment.provider": "cli",
            "lyrics_alignment.strict": False,
            "fire_red.cli.executable": "fake-worker",
            "fire_red.cli.timeout_s": 0.01,
        }
    )
    input_path = tmp_path / "song.wav"
    _write_audio(input_path)
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
    assert result["boundary_detection"]["actual_mode"] == "vpbd_acoustic"
    assert "timed out" in result["lyrics_alignment"]["fallback_reason"]
    assert result["lyrics_alignment"]["provider"] == "firered_cli"

def test_vpbd_rescue_ignores_zero_score_suppressed_candidates(tmp_path) -> None:
    input_path = tmp_path / "song.wav"
    _write_audio(input_path, duration_s=20.0)
    splitter = SeamlessSplitter(sample_rate=44100)
    splitter.separator = _FakeSeparator()
    splitter.vpbd_detector = _FakeVpbdDetectorWithZeroScoreFallback()

    result = splitter.split_audio_seamlessly(
        str(input_path),
        str(tmp_path / "out"),
        mode="vpbd_acoustic",
        export_plan=("none",),
    )

    assert result["success"] is True
    assert result["cut_points_sec"] == pytest.approx([0.0, 10.0, 20.0])

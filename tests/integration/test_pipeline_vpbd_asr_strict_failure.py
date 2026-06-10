#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/integration/test_pipeline_vpbd_asr_strict_failure.py
# AI-SUMMARY: Integration test for strict vpbd_asr provider failure behavior.

import numpy as np
import pytest
import soundfile as sf

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


class _TimeoutCliProvider:
    name = "firered_cli"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def is_available(self) -> bool:
        return True

    def align(self, request):
        raise FireRedProviderError("FireRed CLI timed out after 0.01s")


@pytest.fixture(autouse=True)
def _runtime_config():
    set_runtime_config(
        {
            "gpu_pipeline.enable": False,
            "lyrics_alignment.enabled": True,
            "lyrics_alignment.provider": "disabled",
            "lyrics_alignment.strict": True,
        }
    )
    try:
        yield
    finally:
        reset_runtime_config()


def test_vpbd_asr_strict_provider_failure_returns_failed_result(tmp_path) -> None:
    input_path = tmp_path / "song.wav"
    sf.write(input_path, np.zeros(44100 * 8, dtype=np.float32), 44100, subtype="PCM_16")
    splitter = SeamlessSplitter(sample_rate=44100)
    splitter.separator = _FakeSeparator()

    result = splitter.split_audio_seamlessly(
        str(input_path),
        str(tmp_path / "out"),
        mode="vpbd_asr",
        export_plan=("none",),
    )

    assert result["success"] is False
    assert "lyrics alignment disabled" in result["error"]

def test_vpbd_asr_strict_firered_timeout_returns_failed_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(providers, "FireRedCliProvider", _TimeoutCliProvider)
    set_runtime_config(
        {
            "lyrics_alignment.enabled": True,
            "lyrics_alignment.provider": "cli",
            "lyrics_alignment.strict": True,
            "fire_red.cli.executable": "fake-worker",
            "fire_red.cli.timeout_s": 0.01,
        }
    )
    input_path = tmp_path / "song.wav"
    sf.write(input_path, np.zeros(44100 * 8, dtype=np.float32), 44100, subtype="PCM_16")
    splitter = SeamlessSplitter(sample_rate=44100)
    splitter.separator = _FakeSeparator()

    result = splitter.split_audio_seamlessly(
        str(input_path),
        str(tmp_path / "out"),
        mode="vpbd_asr",
        export_plan=("none",),
    )

    assert result["success"] is False
    assert "timed out" in result["error"]

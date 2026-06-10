#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/integration/test_firered_cli_provider_real.py
# AI-SUMMARY: Optional real FireRed CLI provider smoke test gated by environment and GPU marker.

from __future__ import annotations

import os
from pathlib import Path

import pytest

from audio_cut.lyrics.firered_cli_provider import FireRedCliProvider
from audio_cut.lyrics.providers import LyricsProviderRequest


@pytest.mark.firered
@pytest.mark.gpu
def test_firered_cli_provider_real_worker() -> None:
    executable = os.environ.get("FIRERED_CLI_WORKER")
    audio_path = os.environ.get("FIRERED_TEST_WAV")
    if not executable or not audio_path:
        pytest.skip("set FIRERED_CLI_WORKER and FIRERED_TEST_WAV to run real FireRed CLI test")

    provider = FireRedCliProvider(executable=executable, timeout_s=180.0)
    timeline = provider.align(
        LyricsProviderRequest(
            vocal_path=Path(audio_path),
            sample_rate=16000,
            strict=True,
        )
    )

    assert timeline.words or timeline.sentences or timeline.vad_regions

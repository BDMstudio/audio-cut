#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_vocal_phrase_boundary_detector.py
# AI-SUMMARY: Tests VPBD ASR detector prepares readable vocal audio for lyrics providers.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_cut.lyrics.models import LyricsTimeline
from vocal_smart_splitter.core.vocal_phrase_boundary_detector import VocalPhraseBoundaryDetector
from vocal_smart_splitter.utils.config_manager import reset_runtime_config, set_runtime_config


class _NoPauseDetector:
    def detect_pure_vocal_pauses(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def test_vpbd_asr_writes_readable_vocal_copy_before_provider_call(tmp_path, monkeypatch) -> None:
    seen: dict[str, Any] = {}

    class _RecordingProvider:
        name = "recording_provider"

        def align(self, request) -> LyricsTimeline:  # type: ignore[no-untyped-def]
            seen["path"] = Path(request.vocal_path)
            info = sf.info(str(request.vocal_path))
            seen["samplerate"] = int(info.samplerate)
            seen["channels"] = int(info.channels)
            seen["subtype"] = str(info.subtype)
            return LyricsTimeline(duration_s=request.duration_s, source="recording")

    monkeypatch.setattr(
        "vocal_smart_splitter.core.vocal_phrase_boundary_detector.build_lyrics_provider",
        lambda cfg: _RecordingProvider(),
    )
    set_runtime_config(
        {
            "lyrics_alignment.enabled": True,
            "lyrics_alignment.provider": "cli",
            "lyrics_alignment.strict": True,
            "global_planner.hard_min_s": 1.0,
            "global_planner.hard_max_s": 6.0,
            "global_planner.target_min_s": 2.0,
            "global_planner.target_max_s": 5.0,
        }
    )
    try:
        sample_rate = 44100
        vocal = np.sin(np.linspace(0.0, 6.28, sample_rate, dtype=np.float32))

        result = VocalPhraseBoundaryDetector(sample_rate=sample_rate).detect(
            mode="vpbd_asr",
            vocal_track=vocal,
            original_audio=vocal.copy(),
            pure_vocal_detector=_NoPauseDetector(),
            feature_cache=None,
            vad_segments=[],
            input_path=str(tmp_path / "song.wav"),
            output_dir=str(tmp_path / "out"),
        )
    finally:
        reset_runtime_config()

    assert result.lyrics_alignment["provider"] == "recording_provider"
    assert seen["path"].exists()
    assert seen["path"].name == "song_vocal_for_asr.wav"
    assert seen["samplerate"] == 16000
    assert seen["channels"] == 1
    assert seen["subtype"] == "PCM_16"


def test_src_namespace_runtime_config_enables_vpbd_asr_provider(tmp_path, monkeypatch) -> None:
    from src.vocal_smart_splitter.core.vocal_phrase_boundary_detector import (
        VocalPhraseBoundaryDetector as SrcVocalPhraseBoundaryDetector,
    )
    from src.vocal_smart_splitter.utils.config_manager import (
        reset_runtime_config as reset_src_runtime_config,
        set_runtime_config as set_src_runtime_config,
    )

    seen: dict[str, Any] = {}

    class _RecordingProvider:
        name = "src_recording_provider"

        def align(self, request) -> LyricsTimeline:  # type: ignore[no-untyped-def]
            seen["called"] = True
            return LyricsTimeline(duration_s=request.duration_s, source="recording")

    monkeypatch.setattr(
        "src.vocal_smart_splitter.core.vocal_phrase_boundary_detector.build_lyrics_provider",
        lambda cfg: _RecordingProvider(),
    )
    set_src_runtime_config(
        {
            "lyrics_alignment.enabled": True,
            "lyrics_alignment.provider": "cli",
            "lyrics_alignment.strict": True,
            "global_planner.hard_min_s": 1.0,
            "global_planner.hard_max_s": 6.0,
            "global_planner.target_min_s": 2.0,
            "global_planner.target_max_s": 5.0,
        }
    )
    try:
        sample_rate = 44100
        vocal = np.sin(np.linspace(0.0, 6.28, sample_rate, dtype=np.float32))

        result = SrcVocalPhraseBoundaryDetector(sample_rate=sample_rate).detect(
            mode="vpbd_asr",
            vocal_track=vocal,
            original_audio=vocal.copy(),
            pure_vocal_detector=_NoPauseDetector(),
            feature_cache=None,
            vad_segments=[],
            input_path=str(tmp_path / "song.wav"),
            output_dir=str(tmp_path / "out"),
        )
    finally:
        reset_src_runtime_config()

    assert seen["called"] is True
    assert result.boundary_detection["actual_mode"] == "vpbd_asr"
    assert result.lyrics_alignment["provider"] == "src_recording_provider"

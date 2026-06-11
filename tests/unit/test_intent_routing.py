#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_intent_routing.py
# AI-SUMMARY: Verifies v2.8 intent routing across CLI and public API entry points.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

import audio_cut.api as api_module
from run_splitter import (
    apply_intent_runtime_overrides,
    build_parser,
    resolve_effective_mode,
)


def test_cli_intent_flags_route_to_unified_engine_when_mode_is_omitted() -> None:
    parser = build_parser()
    args = parser.parse_args(["input/song.mp3", "--segments", "many", "--align", "beat_lean"])
    overrides: Dict[str, Any] = {}

    apply_intent_runtime_overrides(args, overrides)

    assert resolve_effective_mode(args) == "vpbd_asr"
    assert overrides["smart_cut.segments"] == "many"
    assert overrides["smart_cut.alignment"] == "beat_lean"
    assert overrides["lyrics_alignment.enabled"] is True
    assert overrides["lyrics_alignment.provider"] == "auto"
    assert overrides["lyrics_alignment.strict"] is False


def test_cli_explicit_mode_wins_over_intent_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["input/song.mp3", "--mode", "hybrid_mdd", "--segments", "few", "--align", "beat"]
    )

    assert resolve_effective_mode(args) == "hybrid_mdd"


def test_cli_without_intent_keeps_legacy_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["input/song.mp3"])

    assert resolve_effective_mode(args) == "v2.2_mdd"


def test_public_api_intent_parameters_route_to_unified_engine(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    sf.write(input_path, np.zeros(44100 * 2, dtype=np.float32), 44100, subtype="PCM_16")
    calls: List[Dict[str, Any]] = []

    class FakeSplitter:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate

        def split_audio_seamlessly(
            self,
            input_file: str,
            output_dir: str,
            *,
            mode: str,
            export_plan: Any = None,
        ) -> Dict[str, Any]:
            calls.append({"mode": mode, "export_plan": export_plan, "sample_rate": self.sample_rate})
            return {
                "success": True,
                "method": f"pure_vocal_split_{mode}",
                "export_plan": ["music_segments"],
                "cut_points_sec": [0.0, 2.0],
                "cut_points_samples": [0, 88200],
                "segment_labels": ["human"],
                "segment_durations": [2.0],
                "segment_vocal_flags": [True],
            }

    monkeypatch.setattr(api_module, "SeamlessSplitter", FakeSplitter)

    manifest = api_module.separate_and_segment(
        input_uri=str(input_path),
        export_dir=str(tmp_path / "out"),
        segments="medium",
        alignment=0.75,
    )

    assert calls == [{"mode": "vpbd_asr", "export_plan": None, "sample_rate": 44100}]
    assert manifest["version"] == "vpbd_asr"
    assert manifest["intent"] == {
        "target_duration_s": [5.0, 12.0],
        "segments": "medium",
        "alignment": 0.75,
        "alignment_raw": 0.75,
        "lyrics": "auto",
        "profile": "auto",
    }
    assert "qa_report" in manifest


def test_public_api_without_intent_keeps_legacy_default(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    sf.write(input_path, np.zeros(44100 * 2, dtype=np.float32), 44100, subtype="PCM_16")
    calls: List[str] = []

    class FakeSplitter:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate

        def split_audio_seamlessly(
            self,
            input_file: str,
            output_dir: str,
            *,
            mode: str,
            export_plan: Any = None,
        ) -> Dict[str, Any]:
            calls.append(mode)
            return {
                "success": True,
                "method": f"pure_vocal_split_{mode}",
                "export_plan": [],
                "cut_points_sec": [0.0, 2.0],
                "cut_points_samples": [0, 88200],
                "segment_labels": ["human"],
                "segment_durations": [2.0],
                "segment_vocal_flags": [True],
            }

    monkeypatch.setattr(api_module, "SeamlessSplitter", FakeSplitter)

    manifest = api_module.separate_and_segment(
        input_uri=str(input_path),
        export_dir=str(tmp_path / "out"),
    )

    assert calls == ["v2.2_mdd"]
    assert manifest["version"] == "v2.2_mdd"
    assert "intent" not in manifest

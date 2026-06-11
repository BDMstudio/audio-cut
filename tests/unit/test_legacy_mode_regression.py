#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_legacy_mode_regression.py
# AI-SUMMARY: Guards legacy mode CLI, naming, hybrid suffix and librosa Manifest contracts.

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

import audio_cut.api as api_module
import vocal_smart_splitter.core.utils.segment_exporter as segment_exporter_module
from audio_cut.api import _build_manifest
from run_splitter import build_parser, resolve_effective_mode
from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from vocal_smart_splitter.core.utils.segment_exporter import SegmentExporter
from vocal_smart_splitter.utils.config_manager import reset_runtime_config, set_runtime_config


def test_legacy_cli_requires_no_new_asr_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(["input/song.mp3"])

    assert args.mode is None
    assert resolve_effective_mode(args) == "v2.2_mdd"
    assert args.lyrics_provider is None
    assert args.firered_endpoint is None
    assert args.asr_chunk_s is None
    assert args.asr_overlap_s is None
    assert args.asr_strict is False
    assert args.lyrics_fixture is None


def test_segment_exporter_preserves_duration_suffix_and_hybrid_lib_marker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    exported_bases: List[str] = []

    def fake_export_audio(
        audio: np.ndarray,
        sample_rate: int,
        output_base: Path,
        export_format: str,
        options: Dict[str, Any],
    ) -> str:
        exported_bases.append(output_base.name)
        return f"{output_base}.{export_format}"

    monkeypatch.setattr(segment_exporter_module, "export_audio", fake_export_audio)
    exporter = SegmentExporter(sample_rate=44100)

    files = exporter.export_segments(
        [np.zeros(44100, dtype=np.float32), np.zeros(44100, dtype=np.float32)],
        str(tmp_path),
        segment_is_vocal=[True, False],
        export_format="wav",
        export_options={},
        lib_flags=[True, False],
        duration_map={0: 2.0, 1: 3.5},
    )

    assert exported_bases == ["segment_001_human_lib_2.0", "segment_002_music_3.5"]
    assert Path(files[0]).name == "segment_001_human_lib_2.0.wav"
    assert Path(files[1]).name == "segment_002_music_3.5.wav"


def test_librosa_onset_manifest_keeps_smart_segmentation_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    sf.write(input_path, np.zeros(44100 * 4, dtype=np.float32), 44100, subtype="PCM_16")
    result = {
        "success": True,
        "method": "smart_segment_v2",
        "export_plan": [],
        "cut_points_sec": [0.0, 2.0, 4.0],
        "cut_points_samples": [0, 88200, 176400],
        "segment_labels": ["human", "music"],
        "segment_durations": [2.0, 2.0],
        "segment_vocal_flags": [True, False],
        "bpm": 120.0,
        "bar_duration_s": 2.0,
        "density": "medium",
        "silence_boundaries": [2.0],
    }

    manifest = _build_manifest(
        result=result,
        input_path=input_path,
        export_dir=tmp_path,
        mode="librosa_onset",
        sample_rate=44100,
        channels=1,
        layout_cfg={},
    )

    assert manifest["version"] == "librosa_onset"
    assert manifest["smart_segmentation"] == {
        "method": "smart_segment_v2",
        "bpm": 120.0,
        "bar_duration_s": 2.0,
        "density": "medium",
        "silence_boundaries": [2.0],
    }

def test_public_api_default_v22_manifest_shape_is_unchanged(monkeypatch, tmp_path: Path) -> None:
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
                "method": "pure_vocal_split_v2.2_mdd",
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
        export_manifest=True,
    )

    assert calls == [{"mode": "v2.2_mdd", "export_plan": None, "sample_rate": 44100}]
    assert manifest["version"] == "v2.2_mdd"
    assert manifest["segments"][0]["label"] == "human"
    assert "lyrics_alignment" not in manifest
    assert "boundary_detection" not in manifest
    assert Path(manifest["manifest_path"]).name == "SegmentManifest.json"

def test_legacy_mode_warns_but_ignores_lyrics_alignment(monkeypatch, caplog) -> None:
    called = []

    def fake_process(self, input_path, output_dir, mode, export_plan=None):
        called.append(mode)
        return {"success": True, "method": f"pure_vocal_split_{mode}"}

    monkeypatch.setattr(SeamlessSplitter, "_process_pure_vocal_split", fake_process)
    set_runtime_config(
        {
            "lyrics_alignment.enabled": True,
            "lyrics_alignment.provider": "fake",
        }
    )
    try:
        splitter = SeamlessSplitter(sample_rate=44100)
        with caplog.at_level(logging.WARNING):
            result = splitter.split_audio_seamlessly("in.wav", "out", mode="v2.2_mdd", export_plan=("none",))
    finally:
        reset_runtime_config()

    assert result["success"] is True
    assert called == ["v2.2_mdd"]
    assert "lyrics_alignment is configured but ignored for mode v2.2_mdd" in caplog.text

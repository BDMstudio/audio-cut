#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_api_manifest.py
# AI-SUMMARY: Tests base Manifest compatibility when optional VPBD ASR fields are absent.

from pathlib import Path

import numpy as np
import soundfile as sf

from audio_cut.api import _build_manifest


def test_manifest_keeps_legacy_shape_without_vpbd_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    sf.write(input_path, np.zeros(44100 * 2, dtype=np.float32), 44100, subtype="PCM_16")
    result = {
        "success": True,
        "export_plan": [],
        "cut_points_sec": [0.0, 2.0],
        "cut_points_samples": [0, 88200],
        "segment_labels": ["human"],
        "segment_durations": [2.0],
        "segment_vocal_flags": [True],
    }

    manifest = _build_manifest(
        result=result,
        input_path=input_path,
        export_dir=tmp_path,
        mode="v2.2_mdd",
        sample_rate=44100,
        channels=1,
        layout_cfg={},
    )

    assert manifest["cuts"]["final"] == [0.0, 2.0]
    assert "lyrics_alignment" not in manifest
    assert "boundary_detection" not in manifest
    assert "lyrics" not in manifest["segments"][0]


def test_manifest_includes_optional_auto_profile_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    sf.write(input_path, np.zeros(44100 * 2, dtype=np.float32), 44100, subtype="PCM_16")
    result = {
        "success": True,
        "export_plan": [],
        "cut_points_sec": [0.0, 2.0],
        "cut_points_samples": [0, 88200],
        "segment_labels": ["human"],
        "segment_durations": [2.0],
        "segment_vocal_flags": [True],
        "auto_profile": {
            "style": "pop",
            "confidence": 0.7,
            "bpm": 108.0,
            "mdd": 0.38,
            "applied_overrides": ["global_planner.target_min_s"],
        },
    }

    manifest = _build_manifest(
        result=result,
        input_path=input_path,
        export_dir=tmp_path,
        mode="vpbd_asr",
        sample_rate=44100,
        channels=1,
        layout_cfg={},
    )

    assert manifest["auto_profile"]["style"] == "pop"
    assert manifest["auto_profile"]["applied_overrides"] == ["global_planner.target_min_s"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_qa_report.py
# AI-SUMMARY: Tests derived VPBD ASR QA metrics from Manifest-compatible data.

from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from audio_cut.api import _build_manifest
from audio_cut.qa_report import build_qa_report


def test_qa_report_derives_segment_lyrics_boundary_and_guard_metrics() -> None:
    manifest = {
        "audio": {"duration": 20.0},
        "segments": [
            {"duration": 6.0},
            {"duration": 10.0},
            {"duration": 16.0},
        ],
        "cuts": {
            "final": [
                {"t": 0.0},
                {"t": 1.2, "score": 0.4, "guard_shift_ms": 10.0, "source": "breath"},
                {"t": 8.0, "score": 0.8, "guard_shift_ms": 30.0, "source": "beat", "features": {"beat_affinity": 1.0}},
                {"t": 20.0},
            ]
        },
        "lyrics_alignment": {
            "fallback_reason": "timeout",
            "timeline": {
                "duration_s": 20.0,
                "words": [
                    {"text": "hello", "start_s": 1.0, "end_s": 1.5, "confidence": 0.8},
                    {"text": "world", "start_s": 6.0, "end_s": 7.0, "confidence": None},
                ],
                "vad_regions": [
                    {"start_s": 1.0, "end_s": 2.0, "confidence": 0.9, "kind": "singing"},
                ],
            },
        },
    }

    report = build_qa_report(manifest)

    assert report["segments_count"] == 3
    assert report["median_segment_s"] == 10.0
    assert report["segment_5_15_pass_rate"] == pytest.approx(2 / 3)
    assert report["cut_inside_word_rate"] == 0.5
    assert report["cut_inside_singing_rate"] == 0.5
    assert report["avg_boundary_score"] == 0.6
    assert report["lyrics_coverage_ratio"] == 0.075
    assert report["asr_avg_confidence"] == 0.8
    assert report["guard_shift_p50_ms"] == 20.0
    assert report["guard_shift_p95_ms"] == 29.0
    assert report["fallback_reason"] == "timeout"
    assert report["breath_cut_ratio"] == 0.5
    assert report["beat_aligned_ratio"] == 0.5


def test_manifest_includes_qa_report(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    sf.write(input_path, np.zeros(44100 * 8, dtype=np.float32), 44100, subtype="PCM_16")
    result = {
        "success": True,
        "export_plan": [],
        "cut_points_sec": [0.0, 8.0],
        "cut_points_samples": [0, 352800],
        "segment_labels": ["human"],
        "segment_durations": [8.0],
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

    assert manifest["qa_report"]["segments_count"] == 1
    assert manifest["qa_report"]["segment_5_15_pass_rate"] == 1.0

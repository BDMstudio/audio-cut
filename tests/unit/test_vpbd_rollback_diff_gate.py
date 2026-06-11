#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_vpbd_rollback_diff_gate.py
# AI-SUMMARY: Tests VPBD rollback diff gate manifest comparisons.

from scripts.vpbd_rollback_diff_gate import diff_vpbd_rollback_manifests


def test_vpbd_rollback_diff_accepts_matching_output_with_legacy_marker() -> None:
    baseline = _manifest(cuts=[0, 1000])
    current = _manifest(cuts=[0, 1000])
    current["boundary_detection"]["candidate_pool"] = "legacy"

    assert diff_vpbd_rollback_manifests(baseline, current) == []


def test_vpbd_rollback_diff_rejects_auto_profile_and_cut_changes() -> None:
    baseline = _manifest(cuts=[0, 1000])
    current = _manifest(cuts=[0, 900])
    current["boundary_detection"]["candidate_pool"] = "unified"
    current["auto_profile"] = {"style": "pop"}

    issues = diff_vpbd_rollback_manifests(baseline, current)

    assert "current rollback manifest unexpectedly includes auto_profile" in issues
    assert "current boundary_detection.candidate_pool is not legacy" in issues
    assert "cut samples differ" in issues


def test_vpbd_rollback_diff_rejects_output_filename_changes() -> None:
    baseline = _manifest(cuts=[0, 1000])
    current = _manifest(cuts=[0, 1000])
    current["boundary_detection"]["candidate_pool"] = "legacy"
    current["segments"][0]["mix_path"] = "segment_001_music_1.1.wav"

    issues = diff_vpbd_rollback_manifests(baseline, current)

    assert "segment 1 output differs: ('music', 'segment_001_music_1.0.wav') -> ('music', 'segment_001_music_1.1.wav')" in issues


def _manifest(cuts: list[int]) -> dict:
    return {
        "version": "vpbd_asr",
        "success": True,
        "job": {"source": "/tmp/song.wav"},
        "export_plan": ["music_segments"],
        "audio": {"sr": 44100, "channels": 1, "duration": 1.0, "hash": "sha256:test"},
        "layout_cfg": {"applied": False},
        "cuts": {"final": [0.0, 1.0], "samples": cuts, "suppressed": []},
        "segments": [
            {
                "id": "0001",
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "label": "music",
                "mix_path": "segment_001_music_1.0.wav",
            }
        ],
        "artifacts": {
            "music_segments": ["segment_001_music_1.0.wav"],
            "all": ["segment_001_music_1.0.wav"],
            "output_dir": "/tmp/out",
        },
        "guard": {"shift_stats": {}, "adjustments": [], "precision_ok": True, "threshold_ms": {}},
        "separation": {"backend": "test", "confidence": None},
        "timings_ms": {"total": 1},
        "stats": {"num_segments": 1},
        "boundary_detection": {
            "mode": "vpbd_asr",
            "actual_mode": "vpbd_asr",
            "candidate_counts": {},
            "planner": {},
            "selected": [],
            "suppressed": [],
        },
        "lyrics_alignment": {"enabled": True, "provider": "fake"},
        "qa_report": {},
    }

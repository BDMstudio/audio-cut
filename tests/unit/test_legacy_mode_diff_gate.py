#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_legacy_mode_diff_gate.py
# AI-SUMMARY: Tests legacy mode diff gate Manifest and filename contract checks.

from scripts.legacy_mode_diff_gate import diff_manifests


def test_diff_manifests_accepts_legacy_shape_with_cut_time_changes() -> None:
    baseline = _manifest(cuts=[0.0, 5.0, 10.0], duration="5.0")
    current = _manifest(cuts=[0.0, 5.2, 10.0], duration="5.2")

    assert diff_manifests("hybrid_mdd", baseline, current) == []


def test_diff_manifests_rejects_new_vpbd_fields_in_legacy_mode() -> None:
    baseline = _manifest(cuts=[0.0, 5.0])
    current = _manifest(cuts=[0.0, 5.0])
    current["boundary_detection"] = {"candidate_counts": {}}

    issues = diff_manifests("v2.2_mdd", baseline, current)

    assert [issue.message for issue in issues] == [
        "top-level keys differ: missing=[] extra=['boundary_detection']",
        "legacy mode unexpectedly includes boundary_detection",
    ]


def test_diff_manifests_rejects_segment_key_and_filename_drift() -> None:
    baseline = _manifest(cuts=[0.0, 5.0])
    current = _manifest(cuts=[0.0, 5.0])
    current["segments"][0]["extra"] = True
    current["segments"][0]["mix_path"] = "unexpected-name.wav"

    messages = [issue.message for issue in diff_manifests("librosa_onset", baseline, current)]

    assert "segment 1 keys differ: missing=[] extra=['extra']" in messages
    assert "segment 1 mix_path has unexpected name: unexpected-name.wav" in messages


def _manifest(cuts: list[float], duration: str = "5.0") -> dict:
    return {
        "version": "hybrid_mdd",
        "success": True,
        "job": {"source": "/tmp/song.wav"},
        "export_plan": ["music_segments"],
        "audio": {"sr": 44100, "channels": 1, "duration": cuts[-1], "hash": "sha256:test"},
        "layout_cfg": {"applied": False},
        "cuts": {"final": cuts, "samples": [], "suppressed": []},
        "segments": [
            {
                "id": "0001",
                "start": cuts[0],
                "end": cuts[-1],
                "duration": cuts[-1] - cuts[0],
                "label": "music",
                "mix_path": f"segment_001_music_{duration}.wav",
            }
        ],
        "artifacts": {
            "music_segments": [f"segment_001_music_{duration}.wav"],
            "all": [f"segment_001_music_{duration}.wav"],
            "output_dir": "/tmp/out",
        },
        "guard": {"shift_stats": {}, "adjustments": [], "precision_ok": True, "threshold_ms": {}},
        "separation": {"backend": "test", "confidence": None},
        "timings_ms": {"total": 1},
        "stats": {"num_segments": 1},
        "qa_report": {},
    }

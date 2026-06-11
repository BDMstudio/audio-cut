#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/contracts/test_agent_intent_contract.py
# AI-SUMMARY: Contract smoke test for agent-facing v2.8 intent Manifest fields.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf

import audio_cut.api as api_module


def test_agent_intent_manifest_contract(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "fixture.wav"
    sf.write(input_path, np.zeros(44100 * 4, dtype=np.float32), 44100, subtype="PCM_16")

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
            return {
                "success": True,
                "method": f"pure_vocal_split_{mode}",
                "export_plan": ["music_segments"],
                "cut_points_sec": [0.0, 2.0, 4.0],
                "cut_points_samples": [0, 88200, 176400],
                "segment_labels": ["human", "music"],
                "segment_durations": [2.0, 2.0],
                "segment_vocal_flags": [True, False],
                "lyrics_alignment": {
                    "provider": "fake",
                    "timeline": {
                        "duration_s": 4.0,
                        "words": [
                            {"text": "hello", "start_s": 0.2, "end_s": 0.8, "confidence": 0.9},
                            {"text": "world", "start_s": 1.0, "end_s": 1.6, "confidence": 0.9},
                        ],
                        "sentences": [
                            {"text": "hello world", "start_s": 0.2, "end_s": 1.6},
                        ],
                    },
                },
                "boundary_detection": {
                    "mode": mode,
                    "selected": [
                        {"t": 2.0, "source": "beat", "features": {"breath": 0.0}, "meta": {}},
                    ],
                    "planner": {},
                },
            }

    monkeypatch.setattr(api_module, "SeamlessSplitter", FakeSplitter)

    manifest = api_module.separate_and_segment(
        input_uri=str(input_path),
        export_dir=str(tmp_path / "out"),
        segments="medium",
        alignment=0.75,
        export_manifest=True,
    )

    assert manifest["version"] == "vpbd_asr"
    assert manifest["audio"]["hash"].startswith("sha256:")
    assert manifest["cuts"]["samples"] == [0, 88200, 176400]
    assert manifest["intent"]["target_duration_s"] == [5.0, 12.0]
    assert manifest["intent"]["alignment"] == 0.75
    assert manifest["intent"]["lyrics"] == "auto"
    assert manifest["segments"][0]["lyrics"]["text"] == "hello world"
    assert "lyrics" not in manifest["segments"][1] or manifest["segments"][1]["lyrics"] is None
    assert "qa_report" in manifest
    assert "beat_aligned_ratio" in manifest["qa_report"]
    assert "breath_cut_ratio" in manifest["qa_report"]
    assert Path(manifest["manifest_path"]).name == "SegmentManifest.json"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_fireredasr2s_worker.py
# AI-SUMMARY: Tests the FireRedASR2S CLI worker adapter without requiring real model weights.

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import fireredasr2s_worker as worker
from scripts.fireredasr2s_worker import convert_result_jsonl


def test_fireredasr2s_worker_converts_result_jsonl_to_timeline(tmp_path: Path) -> None:
    result_jsonl = tmp_path / "result.jsonl"
    result_jsonl.write_text(
        '{"uttid":"song","dur_s":2.32,'
        '"sentences":[{"start_ms":310,"end_ms":1840,"text":"你好世界。","asr_confidence":0.875}],'
        '"vad_segments_ms":[[310,1840]],'
        '"words":[{"start_ms":490,"end_ms":690,"text":"你"},'
        '{"start_ms":690,"end_ms":1090,"text":"好"}]}\n',
        encoding="utf-8",
    )

    timeline = convert_result_jsonl(result_jsonl)

    assert timeline["duration_s"] == 2.32
    assert timeline["source"] == "fireredasr2s_cli"
    assert timeline["sentences"] == [
        {"start_ms": 310, "end_ms": 1840, "text": "你好世界。", "confidence": 0.875}
    ]
    assert timeline["vad_regions"] == [
        {"start_ms": 310, "end_ms": 1840, "kind": "singing", "confidence": None}
    ]
    assert timeline["words"][0] == {
        "start_ms": 490,
        "end_ms": 690,
        "text": "你",
        "confidence": None,
    }

def test_fireredasr2s_worker_invokes_official_cli_as_module(tmp_path, monkeypatch) -> None:
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "timeline.json"
    firered_root = tmp_path / "FireRedASR2S"
    firered_root.mkdir()
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"")
    request_path.write_text(json.dumps({"audio_path": str(audio_path)}), encoding="utf-8")

    captured = {}

    def fake_run(cmd, cwd, capture_output, text, check):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.jsonl").write_text(
            '{"uttid":"song","dur_s":1.0,"words":[],"sentences":[],"vad_segments_ms":[]}\n',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(worker.subprocess, "run", fake_run)

    code = worker.main([
        "--input-json",
        str(request_path),
        "--output",
        str(output_path),
        "--firered-root",
        str(firered_root),
        "--python",
        "/fake/python",
    ])

    assert code == 0
    assert captured["cmd"][:3] == [
        "/fake/python",
        "-m",
        "fireredasr2s.fireredasr2s_cli",
    ]
    assert captured["cwd"] == str(firered_root)
    assert json.loads(output_path.read_text(encoding="utf-8"))["duration_s"] == 1.0

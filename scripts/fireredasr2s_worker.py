#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/fireredasr2s_worker.py
# AI-SUMMARY: Adapts the official FireRedASR2S CLI output to audio-cut lyrics_timeline JSON.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def convert_result_jsonl(result_jsonl: Path | str) -> Dict[str, Any]:
    """Convert FireRedASR2S result.jsonl into the audio-cut worker output protocol."""

    path = Path(result_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"FireRedASR2S result.jsonl not found: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"FireRedASR2S result.jsonl is empty: {path}")
    row = rows[0]
    return {
        "duration_s": _optional_float(row.get("dur_s")),
        "source": "fireredasr2s_cli",
        "warnings": [],
        "meta": {
            "uttid": row.get("uttid"),
            "text": row.get("text", ""),
            "wav_path": row.get("wav_path"),
        },
        "words": [_word_payload(item) for item in row.get("words", []) or []],
        "sentences": [_sentence_payload(item) for item in row.get("sentences", []) or []],
        "vad_regions": [_vad_payload(item) for item in row.get("vad_segments_ms", []) or []],
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Run the official FireRedASR2S CLI and write lyrics_timeline JSON."""

    parser = argparse.ArgumentParser(description="FireRedASR2S worker adapter for audio-cut")
    parser.add_argument("--input-json", default="-", help="JSON request path or '-' for stdin")
    parser.add_argument("--output", required=True, help="lyrics_timeline.json output path")
    parser.add_argument("--firered-root", default="/home/ubuntu/asr_test/FireRedASR2S")
    parser.add_argument("--python", default="/home/ubuntu/asr_test/venv/bin/python")
    parser.add_argument("--asr-use-gpu", default="1")
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args(argv)

    request = _load_request(args.input_json)
    audio_path = request.get("audio_path")
    if not audio_path:
        raise ValueError("worker request requires audio_path")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fireredasr2s_worker_") as tmp_dir:
        outdir = Path(tmp_dir) / "firered_out"
        cmd = [
            args.python,
            "-m",
            "fireredasr2s.fireredasr2s_cli",
            "--wav_path",
            str(audio_path),
            "--outdir",
            str(outdir),
            "--write_textgrid",
            "0",
            "--write_srt",
            "0",
            "--save_segment",
            "0",
            "--asr_use_gpu",
            str(args.asr_use_gpu),
        ]
        cmd.extend(str(item) for item in args.extra_arg)
        completed = subprocess.run(
            cmd,
            cwd=str(args.firered_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stderr or completed.stdout)
            return completed.returncode
        timeline = convert_result_jsonl(outdir / "result.jsonl")
        output_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _load_request(input_json: str) -> Dict[str, Any]:
    if input_json == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(input_json).read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("worker request must be a JSON object")
    return parsed


def _word_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start_ms": item.get("start_ms"),
        "end_ms": item.get("end_ms"),
        "text": str(item.get("text", "")),
        "confidence": _first_present(item, "confidence", "asr_confidence"),
    }


def _sentence_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start_ms": item.get("start_ms"),
        "end_ms": item.get("end_ms"),
        "text": str(item.get("text", "")),
        "confidence": _first_present(item, "confidence", "asr_confidence"),
    }


def _vad_payload(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return {
            "start_ms": item.get("start_ms"),
            "end_ms": item.get("end_ms"),
            "kind": str(item.get("kind", "singing")),
            "confidence": _first_present(item, "confidence", "vad_confidence"),
        }
    start_ms, end_ms = item
    return {"start_ms": start_ms, "end_ms": end_ms, "kind": "singing", "confidence": None}


def _first_present(item: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())

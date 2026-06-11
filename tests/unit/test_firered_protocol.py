#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_firered_protocol.py
# AI-SUMMARY: Verifies FireRed worker JSON protocol normalization into lyrics timelines.

from pathlib import Path

from audio_cut.lyrics.firered_protocol import build_worker_request, parse_worker_response
from audio_cut.lyrics.providers import LyricsProviderRequest


def test_build_worker_request_uses_standard_json_input_protocol() -> None:
    request = LyricsProviderRequest(
        vocal_path=Path("/tmp/vocal.wav"),
        duration_s=12.5,
        sample_rate=16000,
        strict=True,
        meta={"chunk_id": "0003", "global_t0_s": 42.0},
    )

    payload = build_worker_request(request)

    assert payload == {
        "audio_path": "/tmp/vocal.wav",
        "duration_s": 12.5,
        "sample_rate": 16000,
        "strict": True,
        "meta": {"chunk_id": "0003", "global_t0_s": 42.0},
    }


def test_parse_worker_response_converts_local_ms_to_global_seconds() -> None:
    payload = {
        "words": [
            {"text": "你", "start_ms": 100, "end_ms": 320},
            {"text": "好", "start_ms": 500, "end_ms": 800, "confidence": 0.9},
        ],
        "sentences": [
            {"text": "你好", "start_ms": 100, "end_ms": 800},
        ],
        "mvad": [
            {"start_ms": 50, "end_ms": 900, "kind": "singing"},
        ],
    }

    timeline = parse_worker_response(
        payload,
        duration_s=20.0,
        global_t0_s=10.0,
        source="firered_cli",
        strict=True,
    )

    assert timeline.source == "firered_cli"
    assert timeline.duration_s == 20.0
    assert timeline.words[0].start_s == 10.1
    assert timeline.words[0].end_s == 10.32
    assert timeline.words[0].confidence is None
    assert timeline.words[1].confidence == 0.9
    assert timeline.sentences[0].start_s == 10.1
    assert timeline.vad_regions[0].start_s == 10.05


def test_parse_worker_response_clamps_minor_ms_rounding_past_duration_in_strict_mode() -> None:
    duration_s = 75.86575
    payload = {
        "duration_s": duration_s,
        "sentences": [
            {"text": "tail", "start_ms": 69130, "end_ms": 75866},
        ],
    }

    timeline = parse_worker_response(
        payload,
        duration_s=duration_s,
        global_t0_s=0.0,
        source="firered_cli",
        strict=True,
    )

    assert timeline.sentences[0].end_s == duration_s
    assert timeline.warnings == [
        "Sentence[0]: end_s clamped to timeline duration after minor rounding overshoot"
    ]


def test_parse_worker_response_allows_missing_mvad() -> None:
    timeline = parse_worker_response(
        {"words": [{"text": "hello", "start_ms": 0, "end_ms": 250}]},
        duration_s=1.0,
        global_t0_s=0.0,
        source="firered_sidecar",
        strict=True,
    )

    assert [word.text for word in timeline.words] == ["hello"]
    assert timeline.vad_regions == []

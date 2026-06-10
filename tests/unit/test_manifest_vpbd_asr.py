#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_manifest_vpbd_asr.py
# AI-SUMMARY: Tests VPBD ASR Manifest optional fields and segment lyrics attachment.

from pathlib import Path

import soundfile as sf
import numpy as np

from audio_cut.api import _build_manifest
from audio_cut.lyrics.models import LyricsTimeline, Word
from audio_cut.lyrics.segment_attach import attach_lyrics_to_segments


def test_attach_lyrics_to_segments_joins_english_with_spaces_and_chinese_without_spaces() -> None:
    timeline = LyricsTimeline(
        words=[
            Word("hello", 0.1, 0.4, 0.9),
            Word("world", 0.6, 0.9, 0.9),
            Word("你", 2.1, 2.3, 0.9),
            Word("好", 2.35, 2.55, 0.9),
        ],
        duration_s=4.0,
        source="test",
    )
    segments = [
        {"id": "0001", "start": 0.0, "end": 1.0},
        {"id": "0002", "start": 2.0, "end": 3.0},
    ]

    attached = attach_lyrics_to_segments(segments, timeline)

    assert attached[0]["lyrics"]["text"] == "hello world"
    assert attached[1]["lyrics"]["text"] == "你好"
    assert len(attached[0]["lyrics"]["words"]) == 2


def test_attach_lyrics_to_segments_excludes_low_overlap_words_and_allows_empty() -> None:
    timeline = LyricsTimeline(
        words=[
            Word("keep", 0.2, 0.6, 0.9),
            Word("drop", 0.9, 1.9, 0.9),
        ],
        duration_s=2.0,
        source="test",
    )
    segments = [
        {"id": "0001", "start": 0.0, "end": 1.0},
        {"id": "0002", "start": 1.0, "end": 2.0},
        {"id": "0003", "start": 2.0, "end": 3.0},
    ]

    attached = attach_lyrics_to_segments(segments, timeline, min_word_overlap_ratio=0.5)

    assert attached[0]["lyrics"]["text"] == "keep"
    assert attached[1]["lyrics"]["text"] == "drop"
    assert attached[2]["lyrics"] is None


def test_manifest_includes_vpbd_optional_fields_and_segment_lyrics(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    sf.write(input_path, np.zeros(44100 * 4, dtype=np.float32), 44100, subtype="PCM_16")
    timeline = LyricsTimeline(
        words=[Word("hello", 0.2, 0.6, 0.9), Word("world", 1.0, 1.4, 0.9)],
        duration_s=4.0,
        source="fake",
    )
    result = {
        "success": True,
        "export_plan": [],
        "cut_points_sec": [0.0, 2.0, 4.0],
        "cut_points_samples": [0, 88200, 176400],
        "segment_labels": ["human", "music"],
        "segment_durations": [2.0, 2.0],
        "segment_vocal_flags": [True, False],
        "boundary_detection": {
            "mode": "vpbd_asr",
            "selected": [
                {
                    "t": 2.0,
                    "score": 0.8,
                    "source": "lyrics_gap",
                    "features": {"asr_gap": 1.0},
                    "reasons": ["vpbd_score"],
                }
            ],
            "planner": {
                "guard_shift_ms_by_raw_time": {2.0: 25.0},
            },
        },
        "lyrics_alignment": {
            "provider": "fake",
            "timeline": timeline.to_dict(),
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

    assert manifest["lyrics_alignment"]["provider"] == "fake"
    assert manifest["boundary_detection"]["mode"] == "vpbd_asr"
    assert manifest["cuts"]["final"][1]["features"] == {"asr_gap": 1.0}
    assert manifest["cuts"]["final"][1]["guard_shift_ms"] == 25.0
    assert manifest["segments"][0]["lyrics"]["text"] == "hello world"
    assert manifest["segments"][1]["lyrics"] is None

def test_manifest_uses_guard_shifted_segment_bounds_for_lyrics_and_cut_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "song.wav"
    sf.write(input_path, np.zeros(44100 * 4, dtype=np.float32), 44100, subtype="PCM_16")
    timeline = LyricsTimeline(
        words=[Word("after", 2.05, 2.15, 0.95)],
        duration_s=4.0,
        source="fake",
    )
    result = {
        "success": True,
        "export_plan": [],
        "cut_points_sec": [0.0, 2.2, 4.0],
        "cut_points_samples": [0, 97020, 176400],
        "segment_labels": ["human", "music"],
        "segment_durations": [2.2, 1.8],
        "segment_vocal_flags": [True, False],
        "boundary_detection": {
            "mode": "vpbd_asr",
            "selected": [
                {
                    "t": 2.0,
                    "score": 0.8,
                    "source": "lyrics_gap",
                    "features": {"asr_gap": 1.0},
                    "reasons": ["vpbd_score"],
                }
            ],
            "planner": {
                "guard_shift_ms_by_raw_time": {2.0: 200.0},
                "final_time_by_raw_time": {2.0: 2.2},
            },
        },
        "lyrics_alignment": {
            "provider": "fake",
            "timeline": timeline.to_dict(),
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

    assert manifest["cuts"]["final"][1]["t"] == 2.2
    assert manifest["cuts"]["final"][1]["source"] == "lyrics_gap"
    assert manifest["cuts"]["final"][1]["guard_shift_ms"] == 200.0
    assert manifest["segments"][0]["lyrics"]["text"] == "after"
    assert manifest["segments"][1]["lyrics"] is None

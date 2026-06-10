#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_run_splitter_cli.py
# AI-SUMMARY: Verifies run_splitter VPBD ASR CLI arguments and runtime config overrides.

from run_splitter import apply_asr_runtime_overrides, build_parser


def test_run_splitter_accepts_vpbd_asr_lyrics_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "input/song.mp3",
            "--mode",
            "vpbd_asr",
            "--lyrics-provider",
            "fake",
            "--lyrics-fixture",
            "tests/fixtures/lyrics/simple_song_timeline.json",
            "--firered-endpoint",
            "http://127.0.0.1:8765",
            "--asr-chunk-s",
            "30",
            "--asr-overlap-s",
            "1.5",
            "--asr-strict",
        ]
    )

    overrides = {}
    apply_asr_runtime_overrides(args, overrides)

    assert args.mode == "vpbd_asr"
    assert overrides["lyrics_alignment.enabled"] is True
    assert overrides["lyrics_alignment.provider"] == "fake"
    assert overrides["lyrics_alignment.fixture_path"].endswith("simple_song_timeline.json")
    assert overrides["fire_red.endpoint"] == "http://127.0.0.1:8765"
    assert overrides["lyrics_alignment.chunk_s"] == 30.0
    assert overrides["lyrics_alignment.overlap_s"] == 1.5
    assert overrides["lyrics_alignment.strict"] is True


def test_run_splitter_endpoint_defaults_provider_to_sidecar() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["input/song.mp3", "--mode", "vpbd_asr", "--firered-endpoint", "http://127.0.0.1:8765"]
    )

    overrides = {}
    apply_asr_runtime_overrides(args, overrides)

    assert overrides["lyrics_alignment.enabled"] is True
    assert overrides["lyrics_alignment.provider"] == "sidecar"
    assert overrides["fire_red.endpoint"] == "http://127.0.0.1:8765"

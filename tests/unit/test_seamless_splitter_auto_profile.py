#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_seamless_splitter_auto_profile.py
# AI-SUMMARY: Tests SeamlessSplitter applies smart_cut auto/manual profiles from feature cache.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from vocal_smart_splitter.utils.config_manager import get_config, reset_runtime_config, set_runtime_config


def _cache() -> SimpleNamespace:
    return SimpleNamespace(
        bpm_features=SimpleNamespace(main_bpm=142.0),
        global_mdd=0.58,
        rms_series=np.asarray([0.40, 0.52, 0.47], dtype=np.float32),
        vocal_coverage_ratio=0.82,
    )


def test_seamless_splitter_applies_auto_profile_runtime_overrides() -> None:
    set_runtime_config(
        {
            "smart_cut.profile": "auto",
            "smart_cut.cut_style": "rhythmic",
            "smart_cut.target_duration_s": [4.0, 10.0],
        }
    )
    try:
        meta = SeamlessSplitter(sample_rate=44100)._apply_smart_cut_runtime(_cache())

        assert meta["style"] == "rap"
        assert get_config("global_planner.target_min_s") == 4.0
        assert get_config("global_planner.hard_max_s") == 15.0
        assert get_config("segment_layout.soft_max_s") == 10.0
        assert get_config("quality_control.segment_max_duration") == 15.0
        assert get_config("phrase_boundary.weights.beat_affinity") >= 0.18
    finally:
        reset_runtime_config()


def test_seamless_splitter_manual_profile_takes_priority_over_auto_estimate() -> None:
    set_runtime_config({"smart_cut.profile": "ballad", "smart_cut.target_duration_s": [5.0, 12.0]})
    try:
        meta = SeamlessSplitter(sample_rate=44100)._apply_smart_cut_runtime(_cache())

        assert meta is None
        assert get_config("meta.profile") == "ballad"
        assert get_config("pure_vocal_detection.min_pause_duration") == 0.6
        assert get_config("phrase_boundary.weights.acoustic_pause") == 0.40
        assert get_config("phrase_boundary.weights.beat_affinity") == 0.05
    finally:
        reset_runtime_config()


def test_seamless_splitter_derives_vocal_coverage_for_auto_profile() -> None:
    cache = SimpleNamespace(
        bpm_features=SimpleNamespace(main_bpm=142.0),
        global_mdd=0.58,
        rms_series=np.asarray([0.40, 0.52, 0.47], dtype=np.float32),
    )
    vocal = np.ones(44100, dtype=np.float32)
    set_runtime_config({"smart_cut.profile": "auto"})
    try:
        meta = SeamlessSplitter(sample_rate=44100)._apply_smart_cut_runtime(cache, vocal_track=vocal)

        assert meta["features"]["vocal_coverage_ratio"] > 0.9
        assert meta["style"] == "rap"
    finally:
        reset_runtime_config()


def test_legacy_mode_clears_previous_auto_profile_meta(monkeypatch, tmp_path) -> None:
    splitter = SeamlessSplitter(sample_rate=1000)
    audio = np.zeros(1000, dtype=np.float32)
    splitter._last_auto_profile_meta = {"style": "rap"}

    monkeypatch.setattr(splitter, "_load_and_resample_if_needed", lambda _path: audio)
    monkeypatch.setattr(splitter, "_get_gpu_pipeline_config", lambda: SimpleNamespace(enable=False))
    monkeypatch.setattr(
        splitter.separator,
        "separate_for_detection",
        lambda _audio, gpu_context=None: SimpleNamespace(
            vocal_track=audio,
            instrumental_track=None,
            feature_cache=_cache(),
            gpu_meta={},
            vad_segments=[],
            backend_used="fake",
            separation_confidence=1.0,
            quality_metrics={},
        ),
    )
    monkeypatch.setattr(splitter.pure_vocal_detector, "detect_pure_vocal_pauses", lambda *args, **kwargs: [])
    monkeypatch.setattr(splitter, "_estimate_vocal_presence", lambda _vocal: False)
    monkeypatch.setattr(
        splitter.segment_exporter,
        "export_segments",
        lambda *args, **kwargs: [str(tmp_path / "segment_001_music.wav")],
    )

    result = splitter._process_pure_vocal_split(
        "song.wav",
        str(tmp_path),
        "v2.2_mdd",
        export_plan=("music_segments",),
    )

    assert "auto_profile" not in result
    assert result["export_plan"] == ["mix_segments"]
    assert result["mix_segment_files"] == [str(tmp_path / "segment_001_music.wav")]


def test_single_segment_fallback_preserves_mix_segment_contract(monkeypatch, tmp_path) -> None:
    splitter = SeamlessSplitter(sample_rate=1000)
    calls = []

    def fake_export_segments(*args, **kwargs):
        calls.append(kwargs)
        return [str(tmp_path / "segment_001_music.wav")]

    monkeypatch.setattr(splitter.segment_exporter, "export_segments", fake_export_segments)

    result = splitter._create_single_segment_result(
        np.zeros(2000, dtype=np.float32),
        "song.wav",
        str(tmp_path),
        "no_pause_candidates",
        is_vocal=False,
        export_plan=("music_segments",),
        append_duration=False,
        include_note=True,
    )

    assert calls[0]["duration_map"] is None
    assert result["export_plan"] == ["mix_segments"]
    assert result["mix_segment_files"] == [str(tmp_path / "segment_001_music.wav")]
    assert result["full_vocal_file"] == str(tmp_path / "segment_001_music.wav")
    assert result["note"] == "no_pause_candidates"


def test_vpbd_single_segment_fallback_preserves_note_without_duration_suffix(monkeypatch, tmp_path) -> None:
    splitter = SeamlessSplitter(sample_rate=1000)
    calls = []

    def fake_export_segments(*args, **kwargs):
        calls.append(kwargs)
        return [str(tmp_path / "segment_001_music.wav")]

    monkeypatch.setattr(splitter.segment_exporter, "export_segments", fake_export_segments)

    result = splitter._create_single_segment_result(
        np.zeros(2000, dtype=np.float32),
        "song.wav",
        str(tmp_path),
        "no_vpbd_candidates",
        is_vocal=False,
        export_plan=("music_segments",),
        append_duration=False,
        include_note=True,
    )

    assert calls[0]["duration_map"] is None
    assert result["note"] == "no_vpbd_candidates"
    assert result["mix_segment_files"] == [str(tmp_path / "segment_001_music.wav")]

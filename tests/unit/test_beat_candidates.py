#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_beat_candidates.py
# AI-SUMMARY: Tests VPBD beat candidates are generated only in high-energy regions with vocal risk.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from audio_cut.analysis.chorus_regions import detect_chorus_regions
from audio_cut.cutting.beat_candidates import generate_beat_candidates
from audio_cut.cutting.cut_candidate import CandidateSource
from vocal_smart_splitter.core.vocal_phrase_boundary_detector import VocalPhraseBoundaryDetector
from vocal_smart_splitter.utils.config_manager import reset_runtime_config, set_runtime_config


class _NoPauseDetector:
    def detect_pure_vocal_pauses(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def _feature_cache() -> SimpleNamespace:
    return SimpleNamespace(
        beat_times=np.arange(0.0, 12.001, 0.5, dtype=np.float32),
        rms_series=np.array(
            [0.05] * 40
            + [0.90] * 40
            + [0.92] * 40
            + [0.94] * 40
            + [0.91] * 40
            + [0.05] * 40,
            dtype=np.float32,
        ),
        hop_s=0.05,
        duration_s=12.0,
    )


def test_detect_chorus_regions_requires_continuous_high_energy_bars() -> None:
    bars = detect_chorus_regions(
        [0.1, 0.8, 0.85, 0.9, 0.82, 0.2],
        energy_threshold=0.5,
        min_consecutive_bars=4,
    )

    assert bars == {1, 2, 3, 4}


def test_generate_beat_candidates_only_in_high_energy_regions() -> None:
    cache = _feature_cache()
    candidates = generate_beat_candidates(
        beat_times=cache.beat_times,
        rms_series=cache.rms_series,
        hop_s=cache.hop_s,
        duration_s=cache.duration_s,
        sample_rate=1000,
        vocal_track=np.zeros(12000, dtype=np.float32),
        bars_per_cut=2,
        base_score=0.3,
    )

    assert [candidate.t for candidate in candidates] == [2.0, 6.0]
    assert {candidate.source for candidate in candidates} == {CandidateSource.BEAT}
    assert all(candidate.score == 0.3 for candidate in candidates)
    assert all('vocal_cut_risk' in candidate.features for candidate in candidates)
    assert all(candidate.meta['bar_index'] in {1, 3} for candidate in candidates)


def test_beat_candidates_mark_high_vocal_cut_risk() -> None:
    cache = _feature_cache()
    vocal = np.zeros(12000, dtype=np.float32)
    vocal[5920:6080] = 0.9

    candidates = generate_beat_candidates(
        beat_times=cache.beat_times,
        rms_series=cache.rms_series,
        hop_s=cache.hop_s,
        duration_s=cache.duration_s,
        sample_rate=1000,
        vocal_track=vocal,
        bars_per_cut=2,
        base_score=0.3,
    )

    risk_by_time = {candidate.t: candidate.features['vocal_cut_risk'] for candidate in candidates}
    assert risk_by_time[2.0] == 0.0
    assert risk_by_time[6.0] > 0.8


def test_vpbd_includes_enabled_beat_candidates_in_pool(tmp_path) -> None:
    set_runtime_config(
        {
            'vpbd.beat_candidates.enable': True,
            'vpbd.beat_candidates.bars_per_cut': 2,
            'vpbd.beat_candidates.base_score': 0.3,
            'global_planner.hard_min_s': 2.0,
            'global_planner.hard_max_s': 8.0,
            'global_planner.target_min_s': 5.0,
            'global_planner.target_max_s': 7.0,
        }
    )
    try:
        sample_rate = 1000
        duration_s = 12.0
        audio = np.zeros(int(sample_rate * duration_s), dtype=np.float32)

        result = VocalPhraseBoundaryDetector(sample_rate=sample_rate).detect(
            mode='vpbd_acoustic',
            vocal_track=audio,
            original_audio=audio.copy(),
            pure_vocal_detector=_NoPauseDetector(),
            feature_cache=_feature_cache(),
            vad_segments=[],
            input_path=str(tmp_path / 'sample.wav'),
            output_dir=str(tmp_path / 'out'),
        )
    finally:
        reset_runtime_config()

    assert result.boundary_detection['candidate_counts']['beat'] == 2
    beat_selected = [candidate for candidate in result.selected_candidates if candidate.source == CandidateSource.BEAT]
    assert beat_selected
    assert 'vocal_cut_risk' in beat_selected[0].features

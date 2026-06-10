#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_breath_candidates.py
# AI-SUMMARY: Verifies breath pauses enter only the VPBD candidate pool with a rollback scale.

from typing import Any

import numpy as np

from audio_cut.cutting.candidate_adapters import adapt_legacy_acoustic_candidates
from audio_cut.cutting.cut_candidate import CandidateSource
from vocal_smart_splitter.core.pure_vocal_pause_detector import PureVocalPause, PureVocalPauseDetector
from vocal_smart_splitter.core.vocal_phrase_boundary_detector import VocalPhraseBoundaryDetector
from vocal_smart_splitter.utils.config_manager import reset_runtime_config, set_runtime_config


def _breath_pause() -> PureVocalPause:
    return PureVocalPause(
        start_time=1.0,
        end_time=1.2,
        duration=0.2,
        pause_type='uncertain',
        confidence=0.1,
        features={},
        cut_point=1.1,
    )


def test_legacy_pause_filter_still_drops_breath_by_default() -> None:
    detector = PureVocalPauseDetector(sample_rate=1000)

    assert detector._classify_and_filter([_breath_pause()]) == []


def test_vpbd_pause_filter_can_keep_breath_candidates() -> None:
    detector = PureVocalPauseDetector(sample_rate=1000)

    pauses = detector._classify_and_filter([_breath_pause()], include_breath_candidates=True)

    assert len(pauses) == 1
    assert pauses[0].pause_type == 'breath'


def test_breath_pause_maps_to_breath_source_with_score_scale() -> None:
    pause = _breath_pause()
    pause.pause_type = 'breath'

    candidates = adapt_legacy_acoustic_candidates(
        [(pause.cut_point, pause.confidence, {'pause_type': pause.pause_type})],
        source=CandidateSource.ACOUSTIC_PAUSE,
        breath_score_scale=0.6,
    )

    assert candidates[0].source == CandidateSource.BREATH
    assert candidates[0].score == 0.06
    assert 'legacy_acoustic' in candidates[0].reasons


def test_breath_score_scale_zero_drops_breath_candidates() -> None:
    pause = _breath_pause()
    pause.pause_type = 'breath'

    candidates = adapt_legacy_acoustic_candidates(
        [(pause.cut_point, pause.confidence, {'pause_type': pause.pause_type})],
        source=CandidateSource.ACOUSTIC_PAUSE,
        breath_score_scale=0.0,
    )

    assert candidates == []


class _BreathOnlyDetector:
    def detect_pure_vocal_pauses(self, *args: Any, **kwargs: Any) -> list[PureVocalPause]:
        pause = _breath_pause()
        pause.start_time = 5.9
        pause.end_time = 6.1
        pause.cut_point = 6.0
        pause.confidence = 1.0
        pause.pause_type = 'breath'
        return [pause]


def test_vpbd_planner_can_select_breath_candidate_when_no_long_pause_exists(tmp_path) -> None:
    set_runtime_config(
        {
            'vpbd.breath_score_scale': 0.6,
            'global_planner.hard_min_s': 2.0,
            'global_planner.hard_max_s': 8.0,
            'global_planner.target_min_s': 5.0,
            'global_planner.target_max_s': 7.0,
        }
    )
    try:
        sample_rate = 44100
        duration_s = 12.0
        audio = np.zeros(int(sample_rate * duration_s), dtype=np.float32)

        result = VocalPhraseBoundaryDetector(sample_rate=sample_rate).detect(
            mode='vpbd_acoustic',
            vocal_track=audio,
            original_audio=audio.copy(),
            pure_vocal_detector=_BreathOnlyDetector(),
            feature_cache=None,
            vad_segments=[],
            input_path=str(tmp_path / 'sample.wav'),
            output_dir=str(tmp_path / 'out'),
        )
    finally:
        reset_runtime_config()

    assert [candidate.source for candidate in result.selected_candidates] == [CandidateSource.BREATH]
    assert result.boundary_detection['candidate_counts']['acoustic'] == 1
    assert result.boundary_detection['candidate_counts']['merged'] == 1


def test_vpbd_breath_scale_zero_restores_no_breath_candidate_path(tmp_path) -> None:
    set_runtime_config(
        {
            'vpbd.breath_score_scale': 0.0,
            'global_planner.hard_min_s': 2.0,
            'global_planner.hard_max_s': 8.0,
            'global_planner.target_min_s': 5.0,
            'global_planner.target_max_s': 7.0,
        }
    )
    try:
        sample_rate = 44100
        duration_s = 12.0
        audio = np.zeros(int(sample_rate * duration_s), dtype=np.float32)

        result = VocalPhraseBoundaryDetector(sample_rate=sample_rate).detect(
            mode='vpbd_acoustic',
            vocal_track=audio,
            original_audio=audio.copy(),
            pure_vocal_detector=_BreathOnlyDetector(),
            feature_cache=None,
            vad_segments=[],
            input_path=str(tmp_path / 'sample.wav'),
            output_dir=str(tmp_path / 'out'),
        )
    finally:
        reset_runtime_config()

    assert result.selected_candidates == []
    assert result.boundary_detection['candidate_counts']['acoustic'] == 0
    assert result.boundary_detection['candidate_counts']['merged'] == 0

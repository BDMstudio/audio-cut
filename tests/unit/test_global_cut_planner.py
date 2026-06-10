#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_global_cut_planner.py
# AI-SUMMARY: Tests VPBD global cut planner constraints, pruning, rescue and metadata.

from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate
from audio_cut.cutting.global_cut_planner import (
    GlobalCutPlanner,
    GlobalCutPlannerConfig,
    apply_guard_shift_metadata,
    planner_result_to_cut_points,
)
from audio_cut.cutting.refine import CutAdjustment


def test_global_cut_planner_selects_dynamic_path_with_duration_constraints() -> None:
    planner = GlobalCutPlanner(
        GlobalCutPlannerConfig(hard_min_s=3.0, hard_max_s=8.0, target_min_s=4.0, target_max_s=7.0)
    )
    result = planner.plan(
        [
            CutCandidate(4.0, 0.8, CandidateSource.LYRICS_GAP),
            CutCandidate(10.0, 0.9, CandidateSource.SENTENCE_END),
            CutCandidate(16.0, 0.7, CandidateSource.ACOUSTIC_PAUSE),
        ],
        duration_s=20.0,
    )

    assert result.feasible is True
    assert result.cut_times == [0.0, 4.0, 10.0, 16.0, 20.0]
    assert [candidate.t for candidate in result.selected_candidates] == [4.0, 10.0, 16.0]
    assert result.metadata["planner"] == "dynamic_programming"


def test_global_cut_planner_prunes_candidates_per_second() -> None:
    planner = GlobalCutPlanner(
        GlobalCutPlannerConfig(
            hard_min_s=2.0,
            hard_max_s=8.0,
            max_candidates_per_second=1.0,
        )
    )
    result = planner.plan(
        [
            CutCandidate(5.10, 0.1, CandidateSource.LYRICS_GAP),
            CutCandidate(5.20, 0.9, CandidateSource.LYRICS_GAP),
            CutCandidate(5.30, 0.8, CandidateSource.LYRICS_GAP),
        ],
        duration_s=10.0,
    )

    assert [candidate.t for candidate in result.selected_candidates] == [5.2]
    assert sorted(candidate.t for candidate in result.suppressed_candidates) == [5.1, 5.3]


def test_global_cut_planner_penalizes_vocal_risk_and_beat_conflict() -> None:
    planner = GlobalCutPlanner(
        GlobalCutPlannerConfig(
            hard_min_s=4.0,
            hard_max_s=9.0,
            vocal_risk_weight=0.5,
            beat_conflict_weight=0.4,
        )
    )
    result = planner.plan(
        [
            CutCandidate(6.0, 0.9, CandidateSource.ACOUSTIC_PAUSE, features={"vocal_cut_risk": 1.0}),
            CutCandidate(7.0, 0.8, CandidateSource.LYRICS_GAP, features={"beat_conflict": 0.0}),
        ],
        duration_s=14.0,
    )

    assert result.cut_times == [0.0, 7.0, 14.0]


def test_global_cut_planner_rescues_when_no_candidate_path_exists() -> None:
    planner = GlobalCutPlanner(
        GlobalCutPlannerConfig(hard_min_s=3.0, hard_max_s=8.0, rescue_enabled=True)
    )

    result = planner.plan([], duration_s=21.0)

    assert result.feasible is True
    assert result.cut_times == [0.0, 7.0, 14.0, 21.0]
    assert result.metadata["planner"] == "rescue"


def test_planner_result_to_cut_points_and_guard_shift_metadata() -> None:
    planner = GlobalCutPlanner(GlobalCutPlannerConfig(hard_min_s=2.0, hard_max_s=8.0))
    result = planner.plan(
        [CutCandidate(4.0, 0.8, CandidateSource.LYRICS_GAP)],
        duration_s=8.0,
    )

    cut_points = planner_result_to_cut_points(result)
    updated = apply_guard_shift_metadata(
        result,
        [
            CutAdjustment(
                raw_time=4.0,
                guard_time=4.1,
                final_time=4.1,
                score=0.8,
                guard_shift_ms=100.0,
                final_shift_ms=100.0,
            )
        ],
    )

    assert [(point.t, point.score, point.kind) for point in cut_points] == [(4.0, 0.8, "lyrics_gap")]
    assert updated.metadata["guard_shift_ms_by_raw_time"][4.0] == 100.0

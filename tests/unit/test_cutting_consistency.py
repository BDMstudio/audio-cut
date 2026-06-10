#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_cutting_consistency.py
# AI-SUMMARY: Ensures planner output remains compatible with existing cut refinement.

import numpy as np

from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate
from audio_cut.cutting.global_cut_planner import (
    GlobalCutPlanner,
    GlobalCutPlannerConfig,
    apply_guard_shift_metadata,
    planner_result_to_cut_points,
)
from audio_cut.cutting.refine import CutContext, finalize_cut_points


def test_global_planner_output_can_flow_through_finalize_cut_points() -> None:
    sample_rate = 10
    mix = np.zeros(120, dtype=np.float32)
    planner = GlobalCutPlanner(GlobalCutPlannerConfig(hard_min_s=2.0, hard_max_s=6.0))
    plan = planner.plan(
        [
            CutCandidate(4.0, 0.9, CandidateSource.ACOUSTIC_PAUSE),
            CutCandidate(8.0, 0.9, CandidateSource.LYRICS_GAP),
        ],
        duration_s=12.0,
    )

    refined = finalize_cut_points(
        CutContext(sr=sample_rate, mix_wave=mix),
        planner_result_to_cut_points(plan),
        min_gap_s=1.0,
        enable_mix_guard=False,
        enable_vocal_guard=False,
        zero_cross_win_ms=0.0,
    )
    updated = apply_guard_shift_metadata(plan, refined.adjustments)

    assert refined.sample_boundaries == [0, 40, 80, 120]
    assert updated.metadata["selected_count"] == 2
    assert set(updated.metadata["guard_shift_ms_by_raw_time"]) == {4.0, 8.0}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_smart_cut_duration_derivation.py
# AI-SUMMARY: Tests smart_cut target duration derives planner, layout and quality duration settings.

from audio_cut.config.auto_profile import derive_smart_cut_overrides


def test_smart_cut_target_duration_derives_all_duration_knobs() -> None:
    overrides = derive_smart_cut_overrides(
        {
            "target_duration_s": [5.0, 12.0],
            "cut_style": "natural",
        }
    )

    assert overrides["global_planner.target_min_s"] == 5.0
    assert overrides["global_planner.target_max_s"] == 12.0
    assert overrides["global_planner.hard_min_s"] == 2.0
    assert overrides["global_planner.hard_max_s"] == 18.0
    assert overrides["segment_layout.soft_min_s"] == 5.0
    assert overrides["segment_layout.soft_max_s"] == 12.0
    assert overrides["quality_control.segment_max_duration"] == 18.0


def test_smart_cut_rejects_invalid_target_duration() -> None:
    try:
        derive_smart_cut_overrides({"target_duration_s": [12.0, 5.0]})
    except ValueError as exc:
        assert "target_duration_s" in str(exc)
    else:
        raise AssertionError("expected invalid target_duration_s to fail")

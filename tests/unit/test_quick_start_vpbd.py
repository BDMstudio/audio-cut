#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_quick_start_vpbd.py
# AI-SUMMARY: Verifies quick_start v2.8 intent-surface three-question flow.

import builtins

from quick_start import (
    build_intent_runtime_overrides,
    select_alignment_style,
    select_segment_density,
    select_processing_mode,
)


def test_quick_start_processing_menu_asks_split_or_separate_without_algorithm_terms(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "")

    mode = select_processing_mode()
    output = capsys.readouterr().out

    assert mode == "vpbd_asr"
    forbidden = ("[推荐]", "[新]", "[实验]", "VPBD", "MDD", "ASR", "Hybrid", "librosa", "FireRed")
    for token in forbidden:
        assert token not in output
    assert "要切片" in output
    assert "只分离" in output


def test_quick_start_processing_menu_can_choose_vocal_separation(monkeypatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "2")

    mode = select_processing_mode()

    assert mode == "vocal_separation"


def test_quick_start_density_defaults_to_medium(monkeypatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "")

    assert select_segment_density() == "medium"


def test_quick_start_alignment_defaults_to_balanced(monkeypatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "")

    assert select_alignment_style() == "balanced"


def test_quick_start_builds_intent_runtime_overrides() -> None:
    overrides = build_intent_runtime_overrides(segments="many", alignment="beat_lean")

    assert overrides == {
        "smart_cut.segments": "many",
        "smart_cut.alignment": "beat_lean",
        "lyrics_alignment.enabled": True,
        "lyrics_alignment.provider": "auto",
        "lyrics_alignment.strict": False,
    }

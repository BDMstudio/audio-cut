#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_quick_start_vpbd.py
# AI-SUMMARY: Verifies quick_start VPBD ASR menu selection and runtime overrides.

import builtins

from quick_start import (
    build_vpbd_asr_runtime_overrides,
    select_processing_mode,
    select_smart_profile,
    select_vpbd_asr_runtime_overrides,
)


def test_quick_start_processing_menu_uses_goal_names_without_status_labels(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "")

    mode = select_processing_mode()
    output = capsys.readouterr().out

    assert mode == "vpbd_asr"
    assert "[推荐]" not in output
    assert "[新]" not in output
    assert "[实验]" not in output
    assert "歌词辅助自然切分" in output
    assert "mvagent 默认" in output


def test_quick_start_processing_menu_keeps_legacy_modes_addressable(monkeypatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "4")

    mode = select_processing_mode()

    assert mode == "hybrid_mdd"


def test_quick_start_builds_sidecar_runtime_overrides() -> None:
    overrides = build_vpbd_asr_runtime_overrides(
        provider="sidecar",
        endpoint="http://127.0.0.1:8765",
        strict=True,
    )

    assert overrides == {
        "lyrics_alignment.enabled": True,
        "lyrics_alignment.provider": "sidecar",
        "lyrics_alignment.strict": True,
        "fire_red.endpoint": "http://127.0.0.1:8765",
    }


def test_quick_start_builds_auto_runtime_overrides_for_default_vpbd_path() -> None:
    overrides = build_vpbd_asr_runtime_overrides(provider="auto")

    assert overrides == {
        "lyrics_alignment.enabled": True,
        "lyrics_alignment.provider": "auto",
        "lyrics_alignment.strict": False,
    }


def test_quick_start_vpbd_provider_defaults_to_auto_without_extra_prompts(monkeypatch) -> None:
    answers = iter([""])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))

    overrides = select_vpbd_asr_runtime_overrides()

    assert overrides == {
        "lyrics_alignment.enabled": True,
        "lyrics_alignment.provider": "auto",
        "lyrics_alignment.strict": False,
    }


def test_quick_start_vpbd_cli_uses_default_worker_and_non_strict_by_default(monkeypatch) -> None:
    answers = iter(["2", "", ""])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))

    overrides = select_vpbd_asr_runtime_overrides()

    assert overrides == {
        "lyrics_alignment.enabled": True,
        "lyrics_alignment.provider": "cli",
        "lyrics_alignment.strict": False,
        "fire_red.cli.executable": "scripts/fireredasr2s_worker.py",
    }


def test_quick_start_smart_profile_defaults_to_auto(monkeypatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "")

    assert select_smart_profile() == "auto"

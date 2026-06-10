#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_quick_start_vpbd.py
# AI-SUMMARY: Verifies quick_start VPBD ASR menu selection and runtime overrides.

import builtins

from quick_start import build_vpbd_asr_runtime_overrides, select_processing_mode, select_smart_profile


def test_quick_start_processing_menu_includes_vpbd_asr(monkeypatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "5")

    mode = select_processing_mode()

    assert mode == "vpbd_asr"


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


def test_quick_start_smart_profile_defaults_to_auto(monkeypatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "")

    assert select_smart_profile() == "auto"

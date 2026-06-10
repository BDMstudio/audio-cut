#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/contracts/test_config_contracts.py
# AI-SUMMARY: Contract tests for unified configuration defaults and VSS environment overrides.

from vocal_smart_splitter.utils.config_manager import ConfigManager


def test_vpbd_asr_config_defaults_are_loaded() -> None:
    cfg = ConfigManager()

    assert cfg.get("vpbd.enabled") is True
    assert cfg.get("vpbd.breath_score_scale") == 0.6
    assert cfg.get("lyrics_alignment.provider") == "disabled"
    assert cfg.get("fire_red.cli.timeout_s") == 120.0
    assert cfg.get("phrase_boundary.weights.asr_gap") > 0.0
    assert cfg.get("global_planner.hard_min_s") > 0.0


def test_vpbd_asr_config_supports_vss_env_override(monkeypatch) -> None:
    monkeypatch.setenv("VSS__LYRICS_ALIGNMENT__PROVIDER", "fake")
    monkeypatch.setenv("VSS__FIRE_RED__CLI__TIMEOUT_S", "9.5")
    monkeypatch.setenv("VSS__GLOBAL_PLANNER__HARD_MIN_S", "1.25")
    monkeypatch.setenv("VSS__VPBD__BREATH_SCORE_SCALE", "0.0")

    cfg = ConfigManager()

    assert cfg.get("lyrics_alignment.provider") == "fake"
    assert cfg.get("fire_red.cli.timeout_s") == 9.5
    assert cfg.get("global_planner.hard_min_s") == 1.25
    assert cfg.get("vpbd.breath_score_scale") == 0.0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_pause_stats_adaptation_config.py
# AI-SUMMARY: Verifies VPP pause-stat multipliers use the unified relative threshold config.

from vocal_smart_splitter.core import pure_vocal_pause_detector as detector_module


def test_vpp_multiplier_uses_relative_threshold_adaptation(monkeypatch) -> None:
    config = {
        "pure_vocal_detection.relative_threshold_adaptation.pause_stats_multipliers": {
            "slow": 1.21,
            "medium": 1.0,
            "fast": 0.81,
        },
        "pure_vocal_detection.pause_stats_adaptation.multipliers": {
            "slow": 9.9,
            "medium": 9.9,
            "fast": 9.9,
        },
    }

    monkeypatch.setattr(
        detector_module,
        "get_config",
        lambda key, default=None: config.get(key, default),
    )

    assert detector_module._resolve_pause_stats_multiplier("slow") == 1.21
    assert detector_module._resolve_pause_stats_multiplier("fast") == 0.81

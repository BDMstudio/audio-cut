#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_config_migration.py
# AI-SUMMARY: Tests legacy configuration migration warnings and schema v3 key mapping.

import warnings

from audio_cut.config.migrate_v2_to_v3 import migrate_to_schema_v3


def test_migrate_v2_to_v3_preserves_core_legacy_knobs() -> None:
    legacy = {
        "meta": {"schema_name": "legacy-song", "schema_comment": "legacy profile"},
        "audio": {"sample_rate": 48000, "channels": 2},
        "pure_vocal_detection": {
            "min_pause_duration": 0.42,
            "peak_relative_threshold_ratio": 0.24,
            "relative_threshold_adaptation": {
                "bpm": {"slow_multiplier": 1.10, "fast_multiplier": 0.90},
                "mdd": {"gain": 0.30},
            },
        },
        "quality_control": {
            "min_split_gap": 0.8,
            "nms_topk_per_10s": 5,
            "enforce_quiet_cut": {
                "search_right_ms": 320.0,
                "floor_db_override": -58.0,
            },
        },
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        migrated = migrate_to_schema_v3(legacy)

    assert migrated["version"] == 3
    assert migrated["name"] == "legacy-song"
    assert migrated["comment"] == "legacy profile"
    assert migrated["audio"] == {"sample_rate": 48000, "channels": 2}
    assert migrated["min_pause_s"] == 0.42
    assert migrated["min_gap_s"] == 0.8
    assert migrated["guard"] == {"max_shift_ms": 320.0, "floor_db": -58.0}
    assert migrated["threshold"]["base_ratio"] == 0.24
    assert migrated["adapt"]["bpm_strength"] > 0.0
    assert migrated["adapt"]["mdd_strength"] == 1.5
    assert migrated["nms"]["topk"] == 5


def test_migrate_v2_to_v3_warns_about_removed_bpm_legacy_keys() -> None:
    legacy = {
        "pure_vocal_detection": {},
        "quality_control": {},
        "bpm_adaptive_core": {"tempo_categories": {}},
        "vocal_pause_splitting": {
            "bpm_adaptive_settings": {
                "pause_duration_multipliers": {
                    "slow_song_multiplier": 1.0,
                },
            },
        },
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        migrate_to_schema_v3(legacy)

    messages = [str(item.message) for item in caught]
    assert any("schema v2 is deprecated" in message for message in messages)
    assert any("bpm_adaptive_core" in message for message in messages)
    assert any("vocal_pause_splitting.bpm_adaptive_settings" in message for message in messages)

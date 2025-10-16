#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/config/migrate_v2_to_v3.py
# AI-SUMMARY: 将 legacy v2 配置映射到 schema v3 精简格式的迁移脚本。

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml

from .derive import SchemaV3Config


def _load_mapping(source: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Legacy config must be a mapping")
    return data


def _derive_strength(low: float, high: float, scale: float) -> float:
    try:
        diff = max(low - high, 0.0)
        return round(diff / scale, 3)
    except Exception:
        return 0.4


def migrate_to_schema_v3(source: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    warnings.warn(
        "config schema v2 is deprecated; call migrate_to_schema_v3 and adopt schema v3",
        DeprecationWarning,
        stacklevel=2,
    )
    legacy = _load_mapping(source)
    audio = legacy.get("audio", {})
    pure = legacy.get("pure_vocal_detection", {})
    quality = legacy.get("quality_control", {})
    guard = (quality.get("enforce_quiet_cut") or {})
    adapt = pure.get("relative_threshold_adaptation", {})
    bpm_cfg = adapt.get("bpm", {})
    mdd_cfg = adapt.get("mdd", {})

    base_ratio = float(pure.get("peak_relative_threshold_ratio", 0.26))
    slow = float(bpm_cfg.get("slow_multiplier", 1.08))
    fast = float(bpm_cfg.get("fast_multiplier", 0.92))
    bpm_strength = _derive_strength(slow, fast, 0.16)
    mdd_strength = round(float(mdd_cfg.get("gain", 0.2)) / 0.2, 3)

    schema = SchemaV3Config(
        name=str(legacy.get("meta", {}).get("schema_name", "migrated")),
        sample_rate=int(audio.get("sample_rate", 44100)),
        channels=int(audio.get("channels", 1)),
        min_pause_s=float(pure.get("min_pause_duration", 0.5)),
        min_gap_s=float(quality.get("min_split_gap", 1.0)),
        guard_max_shift_ms=float(guard.get("search_right_ms", 150.0)),
        guard_floor_db=float(guard.get("floor_db_override", -60.0)),
        threshold_base_ratio=base_ratio,
        adapt_bpm_strength=bpm_strength,
        adapt_mdd_strength=mdd_strength,
        nms_topk=int(quality.get("nms_topk_per_10s", 4)),
        comment=str(legacy.get("meta", {}).get("schema_comment", "migrated from v2")),
    )
    return schema.to_mapping()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy config to schema v3")
    parser.add_argument("legacy", type=Path, help="Path to legacy config.yaml")
    parser.add_argument("--out", type=Path, default=None, help="Optional output path (YAML)")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of YAML")
    args = parser.parse_args(argv)

    schema = migrate_to_schema_v3(args.legacy)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            yaml.dump(schema, handle, allow_unicode=True, default_flow_style=False)
    if args.json:
        print(json.dumps(schema, indent=2, ensure_ascii=False))
    else:
        print(yaml.dump(schema, allow_unicode=True, default_flow_style=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/config/derive.py
# AI-SUMMARY: 提供 Schema v3 解析、参数派生、运行时覆盖与 Profile 应用的统一工具。

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "schema_v3.yaml"
_PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
_BASE_CONFIG_PATH = _PROJECT_ROOT / "src" / "vocal_smart_splitter" / "config.yaml"


@dataclass(frozen=True)
class SchemaV3Config:
    """Normalized representation of schema v3."""

    name: str = "default"
    sample_rate: int = 44100
    channels: int = 1
    min_pause_s: float = 0.5
    min_gap_s: float = 1.0
    guard_max_shift_ms: float = 150.0
    guard_floor_db: float = -60.0
    threshold_base_ratio: float = 0.26
    adapt_bpm_strength: float = 0.4
    adapt_mdd_strength: float = 0.2
    nms_topk: int = 4
    comment: Optional[str] = None

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "version": 3,
            "name": self.name,
            "comment": self.comment,
            "audio": {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
            },
            "min_pause_s": float(self.min_pause_s),
            "min_gap_s": float(self.min_gap_s),
            "guard": {
                "max_shift_ms": float(self.guard_max_shift_ms),
                "floor_db": float(self.guard_floor_db),
            },
            "threshold": {
                "base_ratio": float(self.threshold_base_ratio),
            },
            "adapt": {
                "bpm_strength": float(self.adapt_bpm_strength),
                "mdd_strength": float(self.adapt_mdd_strength),
            },
            "nms": {
                "topk": int(self.nms_topk),
            },
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def is_v3_schema(mapping: Dict[str, Any]) -> bool:
    """Detect whether the provided mapping follows schema v3."""

    if not isinstance(mapping, dict):
        return False
    if mapping.get("version") == 3:
        return True
    core_keys = {"min_pause_s", "min_gap_s", "guard", "threshold", "adapt", "nms"}
    return core_keys.issubset(mapping.keys())


# ---------------------------------------------------------------------------
# Schema load & merge
# ---------------------------------------------------------------------------


def schema_from_mapping(mapping: Dict[str, Any]) -> SchemaV3Config:
    if not is_v3_schema(mapping):
        raise ValueError("mapping is not schema v3 compatible")
    audio = mapping.get("audio", {}) or {}
    guard = mapping.get("guard", {}) or {}
    threshold = mapping.get("threshold", {}) or {}
    adapt = mapping.get("adapt", {}) or {}
    nms = mapping.get("nms", {}) or {}

    return SchemaV3Config(
        name=str(mapping.get("name", "default")),
        comment=mapping.get("comment"),
        sample_rate=int(audio.get("sample_rate", 44100)),
        channels=int(audio.get("channels", 1)),
        min_pause_s=float(mapping.get("min_pause_s", 0.5)),
        min_gap_s=float(mapping.get("min_gap_s", 1.0)),
        guard_max_shift_ms=float(guard.get("max_shift_ms", 150.0)),
        guard_floor_db=float(guard.get("floor_db", -60.0)),
        threshold_base_ratio=float(threshold.get("base_ratio", 0.26)),
        adapt_bpm_strength=float(adapt.get("bpm_strength", 0.4)),
        adapt_mdd_strength=float(adapt.get("mdd_strength", 0.2)),
        nms_topk=int(nms.get("topk", 4)),
    )


def load_default_schema() -> SchemaV3Config:
    return schema_from_mapping(_read_yaml(_DEFAULT_SCHEMA_PATH))


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def merge_schema(schema: SchemaV3Config, overrides: Dict[str, Any]) -> SchemaV3Config:
    merged = _deep_merge(schema.to_mapping(), overrides)
    return schema_from_mapping(merged)


# ---------------------------------------------------------------------------
# Derived parameters for legacy pipeline
# ---------------------------------------------------------------------------


def _derive_rms_ratio(base_ratio: float) -> float:
    return _clamp(base_ratio + 0.06, 0.05, 0.7)


def _derive_bpm_multipliers(strength: float) -> Dict[str, float]:
    strength = _clamp(strength, 0.0, 1.5)
    spread = 0.08 * strength
    return {
        "slow_multiplier": round(1.0 + spread, 4),
        "medium_multiplier": 1.0,
        "fast_multiplier": round(1.0 - spread, 4),
    }


def _derive_bpm_clamp(strength: float) -> Tuple[float, float]:
    strength = _clamp(strength, 0.0, 1.5)
    span = 0.15 + 0.05 * strength
    return round(1.0 - span, 4), round(1.0 + span, 4)


def _derive_mdd_params(strength: float) -> Tuple[float, float]:
    strength = _clamp(strength, 0.0, 2.0)
    base = 1.0
    gain = round(0.2 * strength, 4)
    return base, gain


def _derive_topk_cap(topk: int) -> int:
    return max(60, int(topk) * 20)


def build_legacy_overrides(schema: SchemaV3Config) -> Dict[str, Any]:
    bpm_mult = _derive_bpm_multipliers(schema.adapt_bpm_strength)
    clamp_min, clamp_max = _derive_bpm_clamp(schema.adapt_bpm_strength)
    mdd_base, mdd_gain = _derive_mdd_params(schema.adapt_mdd_strength)

    overrides: Dict[str, Any] = {
        "meta": {
            "schema_version": 3,
            "schema_name": schema.name,
            "schema_comment": schema.comment,
        },
        "audio": {
            "sample_rate": schema.sample_rate,
            "channels": schema.channels,
        },
        "pure_vocal_detection": {
            "min_pause_duration": schema.min_pause_s,
            "peak_relative_threshold_ratio": schema.threshold_base_ratio,
            "rms_relative_threshold_ratio": _derive_rms_ratio(schema.threshold_base_ratio),
            "relative_threshold_adaptation": {
                "enable": True,
                "bpm": {
                    "slow_multiplier": bpm_mult["slow_multiplier"],
                    "medium_multiplier": bpm_mult["medium_multiplier"],
                    "fast_multiplier": bpm_mult["fast_multiplier"],
                },
                "mdd": {
                    "base": mdd_base,
                    "gain": mdd_gain,
                },
                "clamp_min": clamp_min,
                "clamp_max": clamp_max,
            },
            "valley_scoring": {
                "max_kept_after_nms": _derive_topk_cap(schema.nms_topk),
            },
        },
        "quality_control": {
            "min_split_gap": schema.min_gap_s,
            "nms_topk_per_10s": schema.nms_topk,
            "enforce_quiet_cut": {
                "search_right_ms": schema.guard_max_shift_ms,
                "floor_db_override": schema.guard_floor_db,
            },
        },
    }
    return overrides


def _flatten(nested: Dict[str, Any], prefix: str = "") -> Iterator[Tuple[str, Any]]:
    for key, value in nested.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten(value, path)
        else:
            yield path, value


def build_runtime_override_map(schema: SchemaV3Config) -> Dict[str, Any]:
    overrides = build_legacy_overrides(schema)
    return {key: value for key, value in _flatten(overrides) if value is not None}


# ---------------------------------------------------------------------------
# Runtime helpers (profiles + threshold derivation)
# ---------------------------------------------------------------------------


def apply_profile_overrides(
    profile_name: str,
    *,
    base_schema: Optional[SchemaV3Config] = None,
) -> Tuple[SchemaV3Config, Dict[str, Any]]:
    base_schema = base_schema or load_default_schema()
    profile_path = _PROFILE_DIR / f"{profile_name}.yaml"
    if not profile_path.exists():
        available = sorted(p.stem for p in _PROFILE_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"profile '{profile_name}' not found. available={available}")
    profile_mapping = _read_yaml(profile_path)
    overrides = profile_mapping.get("overrides", {})
    schema = merge_schema(base_schema, overrides)
    runtime_overrides = build_runtime_override_map(schema)
    runtime_overrides.setdefault("meta.profile", profile_name)
    return schema, runtime_overrides


@dataclass(frozen=True)
class AdaptStats:
    bpm: Optional[float] = None
    global_mdd: Optional[float] = None


@dataclass(frozen=True)
class DerivedThresholds:
    peak_ratio: float
    rms_ratio: float
    slow_multiplier: float
    fast_multiplier: float
    clamp_min: float
    clamp_max: float


def resolve_threshold(
    base_ratio: float,
    adapt_cfg: Dict[str, Any],
    stats: AdaptStats,
) -> DerivedThresholds:
    bpm_cfg = (adapt_cfg or {}).get("bpm", {})
    clamp_min = float((adapt_cfg or {}).get("clamp_min", 0.85))
    clamp_max = float((adapt_cfg or {}).get("clamp_max", 1.15))
    slow_mult = float(bpm_cfg.get("slow_multiplier", 1.08))
    fast_mult = float(bpm_cfg.get("fast_multiplier", 0.92))

    peak_ratio = base_ratio
    rms_ratio = _derive_rms_ratio(base_ratio)

    bpm = stats.bpm
    if bpm and bpm > 0:
        if bpm < 90.0:
            peak_ratio *= _clamp(slow_mult, clamp_min, clamp_max)
        elif bpm > 140.0:
            peak_ratio *= _clamp(fast_mult, clamp_min, clamp_max)
        peak_ratio = _clamp(peak_ratio, base_ratio * clamp_min, base_ratio * clamp_max)

    mdd_cfg = (adapt_cfg or {}).get("mdd", {})
    base = float(mdd_cfg.get("base", 1.0))
    gain = float(mdd_cfg.get("gain", 0.2))
    global_mdd = stats.global_mdd
    if global_mdd is not None:
        peak_ratio *= _clamp(base + gain * global_mdd, clamp_min, clamp_max)

    peak_ratio = _clamp(peak_ratio, 0.05, 0.6)
    rms_ratio = _clamp(rms_ratio, peak_ratio + 0.02, 0.72)

    return DerivedThresholds(
        peak_ratio=peak_ratio,
        rms_ratio=rms_ratio,
        slow_multiplier=slow_mult,
        fast_multiplier=fast_mult,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


def resolve_min_pause(base_pause: float, adapt_strength: float, stats: AdaptStats) -> float:
    bpm = stats.bpm
    if not bpm or bpm <= 0:
        return base_pause
    adapt_strength = _clamp(adapt_strength, 0.0, 1.5)
    normalized = _clamp((bpm - 110.0) / 110.0, -1.0, 1.0)
    delta = -0.18 * adapt_strength * normalized
    return max(0.3, base_pause + delta)


# ---------------------------------------------------------------------------
# Base config accessor used by migration tools
# ---------------------------------------------------------------------------


def load_base_config() -> Dict[str, Any]:
    return _read_yaml(_BASE_CONFIG_PATH)


__all__ = [
    "AdaptStats",
    "DerivedThresholds",
    "SchemaV3Config",
    "apply_profile_overrides",
    "build_legacy_overrides",
    "build_runtime_override_map",
    "is_v3_schema",
    "load_base_config",
    "load_default_schema",
    "merge_schema",
    "resolve_min_pause",
    "resolve_threshold",
    "schema_from_mapping",
]

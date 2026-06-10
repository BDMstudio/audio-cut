#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/config/auto_profile.py
# AI-SUMMARY: Estimates v2.7 smart-cut profiles and derives runtime overrides from track features.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from audio_cut.config.derive import apply_profile_overrides


@dataclass(frozen=True)
class StyleEstimate:
    """Rule-based style estimate produced from a TrackFeatureCache-like object."""

    profile: str
    confidence: float
    features: Dict[str, float]
    fallback_reason: Optional[str] = None


_PROFILE_ANCHORS = (
    (60.0, "ballad"),
    (110.0, "pop"),
    (140.0, "rap"),
    (160.0, "edm"),
)

_STYLE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "ballad": {
        "acoustic_pause": 0.40,
        "asr_gap": 0.20,
        "sentence_end": 0.20,
        "beat_affinity": 0.05,
        "mdd_affinity": 0.05,
        "breath": 0.10,
        "inside_word_penalty": 0.80,
        "singing_penalty": 0.50,
    },
    "pop": {
        "acoustic_pause": 0.35,
        "asr_gap": 0.20,
        "sentence_end": 0.15,
        "beat_affinity": 0.08,
        "mdd_affinity": 0.10,
        "breath": 0.12,
        "inside_word_penalty": 0.80,
        "singing_penalty": 0.50,
    },
    "rap": {
        "acoustic_pause": 0.28,
        "asr_gap": 0.16,
        "sentence_end": 0.12,
        "beat_affinity": 0.14,
        "mdd_affinity": 0.14,
        "breath": 0.16,
        "inside_word_penalty": 0.85,
        "singing_penalty": 0.50,
    },
    "edm": {
        "acoustic_pause": 0.25,
        "asr_gap": 0.12,
        "sentence_end": 0.10,
        "beat_affinity": 0.22,
        "mdd_affinity": 0.14,
        "breath": 0.17,
        "inside_word_penalty": 0.85,
        "singing_penalty": 0.50,
    },
}


def estimate_style(cache: Any) -> StyleEstimate:
    """Estimate the nearest smart-cut profile from cached global track features."""

    features = _extract_features(cache)
    bpm = features["bpm"]
    mdd = features["global_mdd"]
    energy_cv = features["energy_cv"]
    vocal_coverage = features["vocal_coverage_ratio"]

    if bpm <= 0.0:
        return StyleEstimate(
            profile="pop",
            confidence=0.25,
            features=features,
            fallback_reason="low_confidence",
        )

    if bpm <= 88.0 and energy_cv <= 0.25:
        profile = "ballad"
        confidence = 0.78
    elif bpm >= 122.0 and energy_cv >= 0.65 and vocal_coverage <= 0.55:
        profile = "edm"
        confidence = 0.82
    elif bpm >= 118.0 and mdd >= 0.45 and vocal_coverage >= 0.68:
        profile = "rap"
        confidence = 0.82
    else:
        profile = "pop"
        confidence = 0.70

    return StyleEstimate(profile=profile, confidence=confidence, features=features)


def build_auto_profile_overrides(
    estimate: StyleEstimate,
    *,
    cut_style: str = "natural",
) -> Dict[str, Any]:
    """Build runtime overrides for an auto-selected style estimate."""

    anchor_weights = {"pop": 1.0} if estimate.confidence < 0.6 else _anchor_weights(estimate)
    overrides = _interpolate_profile_overrides(anchor_weights)
    style_profile = "pop" if estimate.confidence < 0.6 else estimate.profile
    overrides.update(build_style_weight_overrides(style_profile, cut_style=cut_style))

    applied_keys = sorted(key for key in overrides if not key.startswith("meta."))
    overrides["meta.auto_profile"] = {
        "style": style_profile,
        "confidence": round(float(estimate.confidence), 4),
        "bpm": _round_feature(estimate.features.get("bpm")),
        "mdd": _round_feature(estimate.features.get("global_mdd")),
        "features": {key: _round_feature(value) for key, value in estimate.features.items()},
        "anchor_weights": anchor_weights,
        "fallback_reason": estimate.fallback_reason,
        "applied_overrides": applied_keys,
    }
    overrides["meta.profile"] = "auto"
    return overrides


def build_style_weight_overrides(profile: str, *, cut_style: str = "natural") -> Dict[str, float]:
    """Build flattened phrase-boundary weight overrides for one style profile."""

    return {
        f"phrase_boundary.weights.{key}": value
        for key, value in _style_phrase_weights(profile, cut_style=cut_style).items()
    }


def derive_smart_cut_overrides(smart_cut: Mapping[str, Any]) -> Dict[str, float]:
    """Derive planner/layout duration knobs from user-facing smart_cut config."""

    target = smart_cut.get("target_duration_s", [5.0, 12.0])
    if not isinstance(target, (list, tuple)) or len(target) != 2:
        raise ValueError("smart_cut.target_duration_s must be [min_s, max_s]")
    target_min = float(target[0])
    target_max = float(target[1])
    if target_min <= 0.0 or target_max <= target_min:
        raise ValueError("smart_cut.target_duration_s must be increasing positive seconds")

    hard_min = round(max(1.0, target_min * 0.4), 4)
    hard_max = round(target_max * 1.5, 4)
    return {
        "global_planner.target_min_s": round(target_min, 4),
        "global_planner.target_max_s": round(target_max, 4),
        "global_planner.hard_min_s": hard_min,
        "global_planner.hard_max_s": hard_max,
        "segment_layout.soft_min_s": round(target_min, 4),
        "segment_layout.soft_max_s": round(target_max, 4),
        "quality_control.segment_max_duration": hard_max,
    }


def _extract_features(cache: Any) -> Dict[str, float]:
    bpm_features = getattr(cache, "bpm_features", None)
    bpm = float(getattr(bpm_features, "main_bpm", 0.0) or 0.0)
    if bpm <= 0.0:
        bpm = _bpm_from_beats(getattr(cache, "beat_times", []))
    rms = np.asarray(getattr(cache, "rms_series", []), dtype=np.float32)
    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    energy_cv = float(np.std(rms) / max(rms_mean, 1e-9)) if rms_mean > 0.0 else 0.0
    return {
        "bpm": round(max(0.0, bpm), 4),
        "global_mdd": round(_clamp01(float(getattr(cache, "global_mdd", 0.0) or 0.0)), 4),
        "energy_cv": round(max(0.0, energy_cv), 4),
        "vocal_coverage_ratio": round(
            _clamp01(float(getattr(cache, "vocal_coverage_ratio", 0.0) or 0.0)),
            4,
        ),
    }


def _anchor_weights(estimate: StyleEstimate) -> Dict[str, float]:
    bpm = float(estimate.features.get("bpm", 0.0) or 0.0)
    if estimate.profile == "edm":
        return {"edm": 1.0}
    if bpm <= _PROFILE_ANCHORS[0][0]:
        return {_PROFILE_ANCHORS[0][1]: 1.0}
    for (left_bpm, left_name), (right_bpm, right_name) in zip(_PROFILE_ANCHORS, _PROFILE_ANCHORS[1:]):
        if left_bpm <= bpm <= right_bpm:
            right_weight = (bpm - left_bpm) / max(right_bpm - left_bpm, 1e-9)
            left_weight = 1.0 - right_weight
            return _clean_weights({left_name: left_weight, right_name: right_weight})
    return {_PROFILE_ANCHORS[-1][1]: 1.0}


def _interpolate_profile_overrides(anchor_weights: Mapping[str, float]) -> Dict[str, Any]:
    profile_overrides = {
        name: apply_profile_overrides(name)[1]
        for name, weight in anchor_weights.items()
        if weight > 0.0
    }
    keys = sorted({key for overrides in profile_overrides.values() for key in overrides})
    result: Dict[str, Any] = {}
    dominant = max(anchor_weights.items(), key=lambda item: item[1])[0]
    for key in keys:
        values = [profile_overrides[name].get(key) for name in anchor_weights if name in profile_overrides]
        if values and all(isinstance(value, (int, float)) for value in values):
            result[key] = round(
                sum(
                    float(profile_overrides[name][key]) * float(anchor_weights[name])
                    for name in anchor_weights
                    if name in profile_overrides
                ),
                6,
            )
        elif key in profile_overrides[dominant]:
            result[key] = profile_overrides[dominant][key]
    return result


def _style_phrase_weights(profile: str, *, cut_style: str) -> Dict[str, float]:
    weights = dict(_STYLE_WEIGHTS.get(profile, _STYLE_WEIGHTS["pop"]))
    if cut_style == "rhythmic":
        weights["beat_affinity"] = min(0.25, weights["beat_affinity"] + 0.04)
        weights["breath"] = min(0.20, weights["breath"] + 0.02)
        weights["acoustic_pause"] = max(0.20, weights["acoustic_pause"] - 0.04)
    elif cut_style == "dense":
        weights["breath"] = min(0.22, weights["breath"] + 0.04)
        weights["sentence_end"] = max(0.08, weights["sentence_end"] - 0.02)
    return {key: round(value, 4) for key, value in weights.items()}


def _bpm_from_beats(beat_times: Any) -> float:
    beats = np.asarray(list(beat_times or []), dtype=np.float32)
    if beats.size < 2:
        return 0.0
    intervals = np.diff(beats)
    intervals = intervals[intervals > 1e-6]
    if intervals.size == 0:
        return 0.0
    return 60.0 / float(np.median(intervals))


def _clean_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    cleaned = {key: round(float(value), 4) for key, value in weights.items() if float(value) > 1e-6}
    total = sum(cleaned.values())
    if total <= 0.0:
        return {"pop": 1.0}
    return {key: round(value / total, 4) for key, value in cleaned.items()}


def _round_feature(value: Any) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 4)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))

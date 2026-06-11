#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/config/auto_profile.py
# AI-SUMMARY: Estimates smart-cut profiles and derives v2.8 intent-surface runtime overrides.

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set, Tuple

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

ALIGNMENT_STOPS: Dict[str, float] = {
    "lyric": 0.0,
    "lyric_lean": 0.25,
    "balanced": 0.5,
    "beat_lean": 0.75,
    "beat": 1.0,
}

SEGMENT_DURATION_STOPS: Dict[str, Tuple[float, float]] = {
    "few": (10.0, 18.0),
    "medium": (5.0, 12.0),
    "many": (3.0, 8.0),
}

LYRIC_POLE: Dict[str, float] = {
    "acoustic_pause": 0.38,
    "asr_gap": 0.26,
    "sentence_end": 0.22,
    "beat_affinity": 0.02,
    "mdd_affinity": 0.06,
    "breath": 0.10,
    "inside_word_penalty": 0.85,
    "singing_penalty": 0.50,
}

BEAT_POLE: Dict[str, float] = {
    "acoustic_pause": 0.22,
    "asr_gap": 0.10,
    "sentence_end": 0.08,
    "beat_affinity": 0.32,
    "mdd_affinity": 0.12,
    "breath": 0.10,
    "inside_word_penalty": 0.80,
    "singing_penalty": 0.50,
}

_DEFAULT_TARGET_DURATION_S = SEGMENT_DURATION_STOPS["medium"]
_WEIGHT_KEYS = tuple(LYRIC_POLE.keys())

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


def resolve_alignment(value: Any) -> float:
    """Resolve a named or numeric alignment stop into the 0..1 intent axis."""

    if value is None or value == "":
        return 0.5
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return 0.5
        if normalized in ALIGNMENT_STOPS:
            return ALIGNMENT_STOPS[normalized]
        try:
            numeric = float(normalized)
        except ValueError as exc:
            allowed = ", ".join(sorted(ALIGNMENT_STOPS))
            raise ValueError(
                f"smart_cut.alignment must be one of {allowed} or a float between 0.0 and 1.0"
            ) from exc
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("smart_cut.alignment must be a stop name or numeric value") from exc
    if not np.isfinite(numeric):
        raise ValueError("smart_cut.alignment must be finite")
    return round(_clamp01(float(numeric)), 4)


def resolve_segment_duration(value: Any) -> Tuple[float, float]:
    """Resolve a segment-density stop or numeric pair into target duration seconds."""

    if value is None or value == "":
        return _DEFAULT_TARGET_DURATION_S
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in SEGMENT_DURATION_STOPS:
            return SEGMENT_DURATION_STOPS[normalized]
        if "-" in normalized:
            left, right = normalized.split("-", 1)
            return _validate_target_duration((float(left), float(right)))
        raise ValueError("smart_cut.segments must be few, medium, many, or MIN-MAX seconds")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return _validate_target_duration((float(value[0]), float(value[1])))
    raise ValueError("smart_cut.segments must be few, medium, many, or [min_s, max_s]")


def resolve_smart_cut_intent(
    smart_cut: Mapping[str, Any],
    *,
    explicit_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Resolve the user-facing v2.8 intent surface into machine values."""

    explicit_keys = explicit_keys or set()
    profile = str(smart_cut.get("profile", "auto") or "auto").strip().lower()
    lyrics = str(smart_cut.get("lyrics", "auto") or "auto").strip().lower()
    cut_style = str(smart_cut.get("cut_style", "") or "").strip().lower()

    alignment_raw = smart_cut.get("alignment", None)
    segments_raw = smart_cut.get("segments", None)
    target_raw = smart_cut.get("target_duration_s", None)

    alignment_is_explicit = (
        "smart_cut.alignment" in explicit_keys
        or alignment_raw not in {None, "", "balanced", 0.5}
    )
    target_is_default = _is_missing_value(target_raw) or _target_duration_equals_default(target_raw)
    segments_is_explicit = (
        "smart_cut.segments" in explicit_keys
        or segments_raw not in {None, "", "medium"}
    )

    if cut_style and cut_style != "natural":
        warnings.warn(
            "smart_cut.cut_style is deprecated; use smart_cut.alignment and smart_cut.segments instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if alignment_is_explicit and cut_style in {"natural", "rhythmic"}:
            warnings.warn(
                "smart_cut.alignment is explicit, so deprecated cut_style alignment mapping is ignored",
                DeprecationWarning,
                stacklevel=2,
            )
        elif cut_style == "rhythmic":
            alignment_raw = 0.7
        if cut_style == "dense" and not segments_is_explicit and target_is_default:
            segments_raw = "many"

    alignment = resolve_alignment(alignment_raw)
    resolved_smart_cut = dict(smart_cut)
    if alignment_raw is not None:
        resolved_smart_cut["alignment"] = alignment_raw
    if segments_raw is not None:
        resolved_smart_cut["segments"] = segments_raw
    target_duration = _resolve_target_duration(resolved_smart_cut, explicit_keys=explicit_keys)
    segments_name = _segments_name_for_value(segments_raw, target_duration)

    return {
        "target_duration_s": [round(target_duration[0], 4), round(target_duration[1], 4)],
        "segments": segments_name,
        "alignment": alignment,
        "alignment_raw": alignment_raw if alignment_raw is not None else "balanced",
        "lyrics": lyrics,
        "profile": profile,
    }


def derive_alignment_overrides(
    alignment: Any,
    style_weights: Mapping[str, Any],
    *,
    alignment_poles: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Derive phrase-boundary and beat-bias overrides from the alignment axis."""

    a = resolve_alignment(alignment)
    if abs(a - 0.5) <= 1e-9:
        return {}

    lyric_pole, beat_pole = _alignment_poles_from_config(alignment_poles)
    base = {
        key: _read_weight(style_weights, key, _STYLE_WEIGHTS["pop"].get(key, 0.0))
        for key in _WEIGHT_KEYS
    }
    if a <= 0.5:
        t = a * 2.0
        weights = {key: _lerp(lyric_pole[key], base[key], t) for key in _WEIGHT_KEYS}
    else:
        t = (a - 0.5) * 2.0
        weights = {key: _lerp(base[key], beat_pole[key], t) for key in _WEIGHT_KEYS}

    overrides = {f"phrase_boundary.weights.{key}": round(value, 4) for key, value in weights.items()}
    overrides["vpbd.beat_candidates.base_score"] = round(_beat_candidate_base_score(a), 4)
    overrides["global_planner.beat_conflict_weight"] = round(0.30 * a, 4)
    return overrides


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


def derive_smart_cut_overrides(
    smart_cut: Mapping[str, Any],
    *,
    explicit_keys: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Derive planner/layout duration knobs from user-facing smart_cut config."""

    target_min, target_max = _resolve_target_duration(smart_cut, explicit_keys=explicit_keys)
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


def _resolve_target_duration(
    smart_cut: Mapping[str, Any],
    *,
    explicit_keys: Optional[Set[str]] = None,
) -> Tuple[float, float]:
    explicit_keys = explicit_keys or set()
    segments_raw = smart_cut.get("segments", None)
    target_raw = smart_cut.get("target_duration_s", None)
    segments_target = resolve_segment_duration(segments_raw)

    if target_raw is None:
        return segments_target

    target = resolve_segment_duration(target_raw)
    target_differs_from_segments = target != segments_target
    target_explicit = "smart_cut.target_duration_s" in explicit_keys or target != _DEFAULT_TARGET_DURATION_S
    segments_explicit = "smart_cut.segments" in explicit_keys or segments_raw not in {None, "", "medium"}

    if target_differs_from_segments and target_explicit:
        if segments_explicit:
            warnings.warn(
                "smart_cut.target_duration_s is explicit and wins over smart_cut.segments",
                UserWarning,
                stacklevel=3,
            )
        return target
    if target_differs_from_segments and segments_explicit:
        return segments_target
    return target


def should_apply_duration_overrides(
    smart_cut: Mapping[str, Any],
    *,
    explicit_keys: Optional[Set[str]] = None,
) -> bool:
    """Return whether smart_cut should override planner duration knobs."""

    explicit_keys = explicit_keys or set()
    if "smart_cut.segments" in explicit_keys or "smart_cut.target_duration_s" in explicit_keys:
        return True
    segments = smart_cut.get("segments", None)
    if segments not in {None, "", "medium"}:
        return True
    target = smart_cut.get("target_duration_s", None)
    return not _is_missing_value(target) and not _target_duration_equals_default(target)


def _is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _target_duration_equals_default(value: Any) -> bool:
    try:
        target = resolve_segment_duration(value)
    except Exception:
        return False
    return tuple(round(item, 4) for item in target) == tuple(
        round(item, 4) for item in _DEFAULT_TARGET_DURATION_S
    )


def _validate_target_duration(value: Tuple[float, float]) -> Tuple[float, float]:
    target_min = float(value[0])
    target_max = float(value[1])
    if target_min <= 0.0 or target_max <= target_min:
        raise ValueError("smart_cut.target_duration_s must be increasing positive seconds")
    return (target_min, target_max)


def _segments_name_for_value(raw: Any, target: Tuple[float, float]) -> Optional[str]:
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in SEGMENT_DURATION_STOPS:
            return normalized
    for name, preset in SEGMENT_DURATION_STOPS.items():
        if tuple(round(v, 4) for v in preset) == tuple(round(v, 4) for v in target):
            return name
    return None


def _alignment_poles_from_config(
    alignment_poles: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not isinstance(alignment_poles, Mapping):
        return dict(LYRIC_POLE), dict(BEAT_POLE)
    lyric_cfg = alignment_poles.get("lyric", {})
    beat_cfg = alignment_poles.get("beat", {})
    lyric = dict(LYRIC_POLE)
    beat = dict(BEAT_POLE)
    if isinstance(lyric_cfg, Mapping):
        for key in _WEIGHT_KEYS:
            if key in lyric_cfg:
                lyric[key] = float(lyric_cfg[key])
    if isinstance(beat_cfg, Mapping):
        for key in _WEIGHT_KEYS:
            if key in beat_cfg:
                beat[key] = float(beat_cfg[key])
    return lyric, beat


def _read_weight(weights: Mapping[str, Any], key: str, default: float) -> float:
    if key in weights:
        return float(weights[key])
    flattened = f"phrase_boundary.weights.{key}"
    if flattened in weights:
        return float(weights[flattened])
    return float(default)


def _beat_candidate_base_score(a: float) -> float:
    if a <= 0.3:
        return 0.0
    if a <= 0.5:
        return 0.3 * ((a - 0.3) / 0.2)
    return 0.3 + (0.65 - 0.3) * ((a - 0.5) / 0.5)


def _lerp(left: float, right: float, t: float) -> float:
    return float(left) + (float(right) - float(left)) * float(t)


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

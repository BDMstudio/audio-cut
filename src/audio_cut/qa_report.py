#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/qa_report.py
# AI-SUMMARY: Derives lightweight QA metrics from SegmentManifest-compatible data.

from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

_EPS = 1e-9


def build_qa_report(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Build derived QA metrics without changing segmentation output."""

    duration_s = _duration_s(manifest)
    segments = [item for item in manifest.get("segments", []) or [] if isinstance(item, Mapping)]
    durations = [_float_or_none(item.get("duration")) for item in segments]
    valid_durations = [value for value in durations if value is not None]

    cuts = _internal_cuts(manifest, duration_s)
    timeline = _timeline(manifest)
    words = [item for item in timeline.get("words", []) or [] if isinstance(item, Mapping)]
    vad_regions = [item for item in timeline.get("vad_regions", []) or [] if isinstance(item, Mapping)]
    guard_shifts = [
        abs(value)
        for value in (_float_or_none(item.get("guard_shift_ms")) for item in _final_cut_items(manifest))
        if value is not None
    ]

    return {
        "segments_count": len(segments),
        "median_segment_s": _round_or_none(median(valid_durations) if valid_durations else None),
        "segment_5_15_pass_rate": _rate(
            sum(1 for value in valid_durations if 5.0 <= value <= 15.0),
            len(valid_durations),
        ),
        "cut_inside_word_rate": _inside_rate(cuts, words),
        "cut_inside_singing_rate": _inside_rate(cuts, vad_regions),
        "avg_boundary_score": _average(_cut_scores(manifest)),
        "lyrics_coverage_ratio": _coverage_ratio(words, duration_s),
        "asr_avg_confidence": _average(_confidences(words)),
        "guard_shift_p50_ms": _round_or_none(_percentile(guard_shifts, 0.50)),
        "guard_shift_p95_ms": _round_or_none(_percentile(guard_shifts, 0.95)),
        "breath_cut_ratio": _source_rate(manifest, cuts, "breath"),
        "beat_aligned_ratio": _beat_aligned_rate(manifest, cuts),
        "fallback_reason": _fallback_reason(manifest),
    }


def _source_rate(manifest: Mapping[str, Any], cuts: List[float], source: str) -> float:
    if not cuts:
        return 0.0
    matched = 0
    for item in _internal_cut_items(manifest):
        if _candidate_has_source(item, source):
            matched += 1
    return _rate(matched, len(cuts))


def _beat_aligned_rate(manifest: Mapping[str, Any], cuts: List[float]) -> float:
    if not cuts:
        return 0.0
    matched = 0
    for item in _internal_cut_items(manifest):
        features = item.get("features") if isinstance(item, Mapping) else {}
        beat_affinity = _float_or_none(features.get("beat_affinity") if isinstance(features, Mapping) else None)
        if _candidate_has_source(item, "beat") or (beat_affinity is not None and beat_affinity >= 0.8):
            matched += 1
    return _rate(matched, len(cuts))


def _internal_cut_items(manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    duration_s = _duration_s(manifest)
    items = []
    for item in _final_cut_items(manifest):
        t = _float_or_none(item.get("t"))
        if t is None or t <= _EPS:
            continue
        if duration_s is not None and t >= duration_s - _EPS:
            continue
        items.append(item)
    return items


def _candidate_has_source(item: Mapping[str, Any], source: str) -> bool:
    if str(item.get("source", "")) == source:
        return True
    meta = item.get("meta")
    if isinstance(meta, Mapping):
        sources = meta.get("sources")
        if isinstance(sources, Iterable) and not isinstance(sources, (str, bytes)):
            return source in {str(value) for value in sources}
    return False


def _duration_s(manifest: Mapping[str, Any]) -> Optional[float]:
    audio = manifest.get("audio")
    if not isinstance(audio, Mapping):
        return None
    return _float_or_none(audio.get("duration"))


def _timeline(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    lyrics = manifest.get("lyrics_alignment")
    if not isinstance(lyrics, Mapping):
        return {}
    timeline = lyrics.get("timeline")
    return timeline if isinstance(timeline, Mapping) else {}


def _fallback_reason(manifest: Mapping[str, Any]) -> Optional[str]:
    lyrics = manifest.get("lyrics_alignment")
    if not isinstance(lyrics, Mapping):
        return None
    value = lyrics.get("fallback_reason")
    return str(value) if value is not None else None


def _final_cut_items(manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    cuts = manifest.get("cuts")
    if not isinstance(cuts, Mapping):
        return []
    final = cuts.get("final", [])
    return [item for item in final if isinstance(item, Mapping)]


def _internal_cuts(manifest: Mapping[str, Any], duration_s: Optional[float]) -> List[float]:
    cuts = manifest.get("cuts")
    if not isinstance(cuts, Mapping):
        return []
    result: List[float] = []
    for item in cuts.get("final", []) or []:
        value = item.get("t") if isinstance(item, Mapping) else item
        t = _float_or_none(value)
        if t is None or t <= _EPS:
            continue
        if duration_s is not None and t >= duration_s - _EPS:
            continue
        result.append(t)
    return result


def _cut_scores(manifest: Mapping[str, Any]) -> List[float]:
    return [
        value
        for value in (_float_or_none(item.get("score")) for item in _final_cut_items(manifest))
        if value is not None
    ]


def _confidences(items: Iterable[Mapping[str, Any]]) -> List[float]:
    return [
        value
        for value in (_float_or_none(item.get("confidence")) for item in items)
        if value is not None
    ]


def _inside_rate(cuts: List[float], intervals: Iterable[Mapping[str, Any]]) -> float:
    if not cuts:
        return 0.0
    ranges = [
        (start, end)
        for start, end in (_interval(item) for item in intervals)
        if start is not None and end is not None and end > start
    ]
    inside = sum(1 for t in cuts if any(start < t < end for start, end in ranges))
    return _rate(inside, len(cuts))


def _coverage_ratio(items: Iterable[Mapping[str, Any]], duration_s: Optional[float]) -> Optional[float]:
    if duration_s is None or duration_s <= 0.0:
        return None
    intervals = [
        (max(0.0, start), min(duration_s, end))
        for start, end in (_interval(item) for item in items)
        if start is not None and end is not None and end > start
    ]
    if not intervals:
        return 0.0
    merged = _merge_intervals(intervals)
    covered = sum(end - start for start, end in merged)
    return _round_or_none(covered / duration_s)


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    ordered = sorted(intervals)
    merged: List[Tuple[float, float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _interval(item: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    return _float_or_none(item.get("start_s")), _float_or_none(item.get("end_s"))


def _average(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return _round_or_none(sum(values) / len(values))


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _round_or_none(float(numerator) / float(denominator)) or 0.0


def _round_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 12)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

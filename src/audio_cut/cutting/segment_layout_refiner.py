#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/cutting/segment_layout_refiner.py
# AI-SUMMARY: 段落布局精炼器，负责微碎片合并、软最小再合并与软最大救援切分，返回更新后的段落与守卫调整。

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from audio_cut.analysis.features_cache import TrackFeatureCache
from audio_cut.cutting.refine import CutAdjustment, CutPoint


@dataclass
class Segment:
    """音频段落表示（基于秒）"""

    start: float
    end: float
    kind: str = "human"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class LayoutConfig:
    enable: bool = False
    micro_merge_s: float = 0.0
    soft_min_s: float = 0.0
    soft_max_s: float = 0.0
    min_gap_s: float = 1.0
    beat_snap_ms: float = 0.0


@dataclass
class LayoutResult:
    segments: List[Segment]
    adjustments: List[CutAdjustment]
    suppressed_points: List[CutPoint]


def derive_layout_config(
    raw_cfg: Optional[dict],
    features: Optional[TrackFeatureCache],
    *,
    sample_rate: float,
) -> LayoutConfig:
    """根据原始配置与特征缓存派生布局配置（P0 仅直接读取静态值）。"""

    raw_cfg = raw_cfg or {}
    enable = bool(raw_cfg.get("enable", False))
    micro_merge = float(raw_cfg.get("micro_merge_s", 0.0) or 0.0)
    soft_min = float(raw_cfg.get("soft_min_s", 0.0) or 0.0)
    soft_max = float(raw_cfg.get("soft_max_s", 0.0) or 0.0)
    min_gap = float(raw_cfg.get("min_gap_s", 1.0) or 1.0)
    beat_snap_ms = float(raw_cfg.get("beat_snap_ms", 0.0) or 0.0)
    # 预留：可在后续基于 BPM/MDD 进行自适应调整
    _ = features, sample_rate  # placate linters for future use
    return LayoutConfig(
        enable=enable,
        micro_merge_s=max(0.0, micro_merge),
        soft_min_s=max(0.0, soft_min),
        soft_max_s=max(0.0, soft_max),
        min_gap_s=max(0.0, min_gap),
        beat_snap_ms=max(0.0, beat_snap_ms),
    )


def refine_layout(
    segments: Iterable[Segment],
    adjustments: Iterable[CutAdjustment],
    *,
    config: LayoutConfig,
    sample_rate: float,
    suppressed_cut_points: Optional[Iterable[CutPoint]] = None,
    features: Optional[TrackFeatureCache] = None,
    asr_boundary_times: Optional[Iterable[float]] = None,
    asr_word_intervals: Optional[Iterable[Tuple[float, float]]] = None,
    allow_midpoint_fallback: bool = False,
) -> LayoutResult:
    """执行段落布局精炼，当前实现微碎片合并、软最小合并和软最大救援切分。"""

    segments = [Segment(seg.start, seg.end, seg.kind) for seg in segments]
    adjustments = list(adjustments or [])
    suppressed = list(suppressed_cut_points or [])
    asr_boundaries = sorted(float(t) for t in (asr_boundary_times or []))
    word_intervals = sorted(
        (float(start), float(end))
        for start, end in (asr_word_intervals or [])
        if float(end) > float(start)
    )

    if not config.enable or len(segments) <= 1:
        return LayoutResult(segments, adjustments, suppressed)

    refined_segments = _apply_micro_merge(segments, config.micro_merge_s, config.soft_max_s)
    refined_segments = _apply_soft_min_merge(refined_segments, config.soft_min_s, config.soft_max_s)
    refined_segments, suppressed, split_adjustments = _apply_soft_max_splits(
        refined_segments,
        suppressed,
        config.soft_max_s,
        min_gap_s=config.min_gap_s,
        features=features,
        asr_boundary_times=asr_boundaries,
        asr_word_intervals=word_intervals,
        allow_midpoint_fallback=allow_midpoint_fallback,
    )
    refined_segments = _apply_post_split_micro_merge(
        refined_segments,
        config.micro_merge_s,
        config.soft_max_s,
    )
    refined_segments = _enforce_min_gap(refined_segments, config.min_gap_s)
    refined_segments = _apply_beat_snap(
        refined_segments,
        config.beat_snap_ms,
        features=features,
        sample_rate=sample_rate,
        min_gap_s=config.min_gap_s,
    )

    filtered_adjustments = _filter_adjustments(
        adjustments,
        refined_segments,
        sample_rate,
        extra_adjustments=split_adjustments,
    )

    return LayoutResult(refined_segments, filtered_adjustments, suppressed)


def _apply_micro_merge(
    segments: List[Segment],
    micro_threshold_s: float,
    soft_max_s: float,
) -> List[Segment]:
    if micro_threshold_s <= 0.0 or len(segments) <= 1:
        return segments

    segs = [Segment(seg.start, seg.end, seg.kind) for seg in segments]

    idx = 0
    while len(segs) > 1 and idx < len(segs):
        seg = segs[idx]
        # Skip merging _lib segments (preserve beat-aligned segments)
        if "_lib" in seg.kind:
            idx += 1
            continue
        if seg.duration >= micro_threshold_s:
            idx += 1
            continue

        left = segs[idx - 1] if idx > 0 else None
        right = segs[idx + 1] if idx + 1 < len(segs) else None

        if left is None and right is None:
            break

        direction = None
        if left is not None and right is not None:
            left_combined = seg.end - left.start
            right_combined = right.end - seg.start
            left_penalty = left_combined if soft_max_s <= 0.0 or left_combined <= soft_max_s else float("inf")
            right_penalty = right_combined if soft_max_s <= 0.0 or right_combined <= soft_max_s else float("inf")
            if left_penalty <= right_penalty:
                direction = "left"
            else:
                direction = "right"
            if direction == "left" and left_penalty == float("inf") and right_penalty != float("inf"):
                direction = "right"
            if direction == "right" and right_penalty == float("inf") and left_penalty != float("inf"):
                direction = "left"
        elif left is not None:
            direction = "left"
        else:
            direction = "right"

        if direction == "left":
            left.end = seg.end
            segs.pop(idx)
            idx = max(idx - 1, 0)
        else:
            right_seg = right or segs[idx + 1]
            new_seg = Segment(start=seg.start, end=right_seg.end, kind=right_seg.kind)
            segs[idx] = new_seg
            segs.pop(idx + 1)

    return segs


def _apply_soft_min_merge(
    segments: List[Segment],
    soft_min_s: float,
    soft_max_s: float,
) -> List[Segment]:
    if soft_min_s <= 0.0 or len(segments) <= 1:
        return segments

    segs = [Segment(seg.start, seg.end, seg.kind) for seg in segments]
    idx = 0
    while len(segs) > 1 and idx < len(segs):
        seg = segs[idx]
        # Skip merging _lib segments (preserve beat-aligned segments)
        if "_lib" in seg.kind:
            idx += 1
            continue
        if seg.duration >= soft_min_s:
            idx += 1
            continue

        left = segs[idx - 1] if idx > 0 else None
        right = segs[idx + 1] if idx + 1 < len(segs) else None
        if left is None and right is None:
            break

        def _merge_cost(candidate: Optional[Segment], target: Segment) -> Tuple[float, str]:
            if candidate is None:
                return float("inf"), ""
            combined = candidate.duration + target.duration
            over_penalty = float("inf") if (soft_max_s > 0.0 and combined > soft_max_s) else combined
            kind_penalty = 0.0 if candidate.kind == target.kind else combined + 1.0
            return over_penalty + kind_penalty, candidate.kind

        left_cost, left_kind = _merge_cost(left, seg)
        right_cost, right_kind = _merge_cost(right, seg)

        if left_cost == right_cost:
            direction = "left" if seg.kind == left_kind else "right"
        elif left_cost < right_cost:
            direction = "left"
        else:
            direction = "right"

        if direction == "left" and left is not None:
            new_kind = left.kind if left.kind == seg.kind else left.kind
            merged = Segment(start=left.start, end=seg.end, kind=new_kind)
            segs[idx - 1] = merged
            segs.pop(idx)
            idx = max(idx - 1, 0)
        elif direction == "right" and right is not None:
            new_kind = right.kind if right.kind == seg.kind else right.kind
            merged = Segment(start=seg.start, end=right.end, kind=new_kind)
            segs[idx] = merged
            segs.pop(idx + 1)
        else:
            idx += 1

    _restore_continuity(segs)
    return segs


def _apply_post_split_micro_merge(
    segments: List[Segment],
    micro_threshold_s: float,
    soft_max_s: float,
) -> List[Segment]:
    if micro_threshold_s <= 0.0 or len(segments) <= 1:
        return segments

    segs = [Segment(seg.start, seg.end, seg.kind) for seg in segments]
    idx = 0
    while len(segs) > 1 and idx < len(segs):
        seg = segs[idx]
        if "_lib" in seg.kind or seg.duration >= micro_threshold_s:
            idx += 1
            continue

        choices = []
        if idx > 0:
            left = segs[idx - 1]
            combined = seg.end - left.start
            choices.append(("left", left, combined))
        if idx + 1 < len(segs):
            right = segs[idx + 1]
            combined = right.end - seg.start
            choices.append(("right", right, combined))
        if not choices:
            idx += 1
            continue

        def _cost(item: Tuple[str, Segment, float]) -> Tuple[float, float, float]:
            _, neighbor, combined = item
            same_kind_penalty = 0.0 if neighbor.kind == seg.kind else 10.0
            if soft_max_s > 0.0 and combined > soft_max_s:
                overage = combined - soft_max_s
                if neighbor.kind != seg.kind or overage > micro_threshold_s:
                    same_kind_penalty += 100.0 + overage
            else:
                overage = 0.0
            return same_kind_penalty, overage, combined

        direction, _, _ = min(choices, key=_cost)
        if direction == "left" and idx > 0:
            left = segs[idx - 1]
            segs[idx - 1] = Segment(start=left.start, end=seg.end, kind=left.kind)
            segs.pop(idx)
            idx = max(idx - 1, 0)
        elif direction == "right" and idx + 1 < len(segs):
            right = segs[idx + 1]
            segs[idx:idx + 2] = [Segment(start=seg.start, end=right.end, kind=right.kind)]
        else:
            idx += 1

    _restore_continuity(segs)
    return segs


def _apply_soft_max_splits(
    segments: List[Segment],
    suppressed: List[CutPoint],
    soft_max_s: float,
    *,
    min_gap_s: float,
    features: Optional[TrackFeatureCache],
    asr_boundary_times: Sequence[float],
    asr_word_intervals: Sequence[Tuple[float, float]],
    allow_midpoint_fallback: bool,
) -> Tuple[List[Segment], List[CutPoint], List[CutAdjustment]]:
    if soft_max_s <= 0.0 or len(segments) <= 0:
        return segments, suppressed, []

    suppressed = list(suppressed)
    segs = [Segment(seg.start, seg.end, seg.kind) for seg in segments]
    new_adjustments: List[CutAdjustment] = []
    tolerance = 1e-3

    idx = 0
    while idx < len(segs):
        seg = segs[idx]
        if seg.duration <= soft_max_s:
            idx += 1
            continue

        candidates = [
            pt for pt in suppressed
            if (seg.start + tolerance) < float(pt.t) < (seg.end - tolerance)
        ]
        cut_time = None
        if candidates:
            best = max(candidates, key=lambda p: _layout_candidate_score(p, asr_boundary_times, asr_word_intervals))
            cut_time = float(best.t)
            suppressed.remove(best)
        else:
            cut_time = _find_acoustic_valley_split(
                seg,
                features,
                asr_boundary_times=asr_boundary_times,
                asr_word_intervals=asr_word_intervals,
                min_gap_s=min_gap_s,
            )
            if cut_time is None and allow_midpoint_fallback:
                cut_time = seg.start + (seg.duration / 2.0)
        if cut_time is None:
            idx += 1
            continue

        left_duration = cut_time - seg.start
        right_duration = seg.end - cut_time
        if left_duration <= 0.0 or right_duration <= 0.0:
            idx += 1
            continue
        if min_gap_s > 0.0 and (left_duration < min_gap_s or right_duration < min_gap_s):
            idx += 1
            continue

        left_seg = Segment(start=seg.start, end=cut_time, kind=seg.kind)
        right_seg = Segment(start=cut_time, end=seg.end, kind=seg.kind)
        segs[idx:idx + 1] = [left_seg, right_seg]
        new_adjustments.append(
            CutAdjustment(
                raw_time=cut_time,
                guard_time=cut_time,
                final_time=cut_time,
                score=1.0,
                guard_shift_ms=0.0,
                final_shift_ms=0.0,
            )
        )
        # 重新检查当前段（可能仍然超长）
        continue
    _restore_continuity(segs)
    return segs, suppressed, new_adjustments



def _layout_candidate_score(
    point: CutPoint,
    asr_boundary_times: Sequence[float],
    asr_word_intervals: Sequence[Tuple[float, float]],
) -> float:
    t = float(point.t)
    base = float(getattr(point, "score", 0.0) or 0.0)
    inside_word_penalty = 0.75 if _inside_interval(t, asr_word_intervals) else 0.0
    return base + 0.5 * _asr_boundary_affinity(t, asr_boundary_times) - inside_word_penalty


def _find_acoustic_valley_split(
    seg: Segment,
    features: Optional[TrackFeatureCache],
    *,
    asr_boundary_times: Sequence[float],
    asr_word_intervals: Sequence[Tuple[float, float]],
    min_gap_s: float,
) -> Optional[float]:
    if features is None or features.frame_count() <= 2:
        return None

    start = seg.start + max(0.0, min_gap_s)
    end = seg.end - max(0.0, min_gap_s)
    if end <= start:
        return None

    sl = features.frame_slice(start, end)
    rms = np.asarray(features.rms_series[sl], dtype=np.float64)
    if rms.size < 3 or not np.all(np.isfinite(rms)):
        return None

    median = float(np.median(rms))
    spread = float(np.percentile(rms, 75) - np.percentile(rms, 5))
    if median <= 1e-12 or spread <= max(1e-9, median * 0.02):
        return None

    threshold = min(float(np.percentile(rms, 25)), median * 0.75)
    frame_start = int(sl.start or 0)
    best_time: Optional[float] = None
    best_score = -1.0

    for local_idx in range(1, rms.size - 1):
        value = float(rms[local_idx])
        if value > threshold:
            continue
        if value > float(rms[local_idx - 1]) or value > float(rms[local_idx + 1]):
            continue
        frame_idx = frame_start + local_idx
        t = frame_idx * float(features.hop_s)
        if t <= start or t >= end:
            continue
        quiet_score = max(0.0, (median - value) / max(median, 1e-12))
        inside_word_penalty = 0.75 if _inside_interval(t, asr_word_intervals) else 0.0
        score = quiet_score + 0.5 * _asr_boundary_affinity(t, asr_boundary_times) - inside_word_penalty
        if score > best_score:
            best_score = score
            best_time = float(t)

    for boundary in asr_boundary_times:
        t = float(boundary)
        if t <= start or t >= end or _inside_interval(t, asr_word_intervals):
            continue
        local_idx = int(round((t / float(features.hop_s)) - frame_start))
        if local_idx < 0 or local_idx >= rms.size:
            continue
        lo = max(0, local_idx - 2)
        hi = min(rms.size, local_idx + 3)
        value = float(np.min(rms[lo:hi]))
        if value > median:
            continue
        quiet_score = max(0.0, (median - value) / max(median, 1e-12))
        score = quiet_score + 0.65
        if score > best_score:
            best_score = score
            best_time = t

    if best_time is None or best_score < 0.5:
        return None
    return best_time


def _inside_interval(t: float, intervals: Sequence[Tuple[float, float]]) -> bool:
    for start_s, end_s in intervals:
        if start_s < t < end_s:
            return True
        if start_s >= t:
            break
    return False

def _asr_boundary_affinity(t: float, asr_boundary_times: Sequence[float], tolerance_s: float = 0.75) -> float:
    if not asr_boundary_times:
        return 0.0
    best = 0.0
    for boundary in asr_boundary_times:
        distance = abs(float(boundary) - float(t))
        if distance > tolerance_s:
            continue
        best = max(best, 1.0 - distance / max(tolerance_s, 1e-6))
    return best

def _enforce_min_gap(segments: List[Segment], min_gap_s: float) -> List[Segment]:
    if min_gap_s <= 0.0 or len(segments) <= 1:
        return segments

    segs = [Segment(seg.start, seg.end, seg.kind) for seg in segments]
    idx = 0
    while len(segs) > 1 and idx < len(segs):
        seg = segs[idx]
        # Skip merging _lib segments (preserve beat-aligned segments)
        if "_lib" in seg.kind:
            idx += 1
            continue
        if seg.duration >= min_gap_s:
            idx += 1
            continue

        neighbors = []
        if idx > 0:
            neighbors.append(("left", segs[idx - 1]))
        if idx + 1 < len(segs):
            neighbors.append(("right", segs[idx + 1]))
        if not neighbors:
            idx += 1
            continue

        if len(neighbors) == 2:
            # 选择合并后持续时间更大的方向，以减少碎片
            combined_left = seg.end - neighbors[0][1].start
            combined_right = neighbors[1][1].end - seg.start
            direction = "left" if combined_left <= combined_right else "right"
        else:
            direction = neighbors[0][0]

        if direction == "left" and idx > 0:
            left = segs[idx - 1]
            segs[idx - 1] = Segment(start=left.start, end=seg.end, kind=left.kind)
            segs.pop(idx)
            idx = max(idx - 1, 0)
        elif direction == "right" and idx + 1 < len(segs):
            right = segs[idx + 1]
            segs[idx:idx + 2] = [Segment(start=seg.start, end=right.end, kind=right.kind)]
        else:
            idx += 1

    _restore_continuity(segs)
    return segs


def _apply_beat_snap(
    segments: List[Segment],
    beat_snap_ms: float,
    *,
    features: Optional[TrackFeatureCache],
    sample_rate: float,
    min_gap_s: float,
) -> List[Segment]:
    if beat_snap_ms <= 0.0 or features is None:
        return segments
    beat_times = getattr(features, "beat_times", None)
    if beat_times is None or len(beat_times) == 0:
        return segments

    max_offset = beat_snap_ms / 1000.0
    segs = [Segment(seg.start, seg.end, seg.kind) for seg in segments]
    # 只有内部边界可吸附
    for idx in range(1, len(segs)):
        boundary_time = segs[idx].start
        snapped_time = _snap_to_beat(boundary_time, beat_times, max_offset)
        if snapped_time is None:
            continue
        left_duration = snapped_time - segs[idx - 1].start
        right_duration = segs[idx].end - snapped_time
        if left_duration < min_gap_s or right_duration < min_gap_s:
            continue
        segs[idx - 1].end = snapped_time
        segs[idx].start = snapped_time

    _restore_continuity(segs)
    return segs


def _snap_to_beat(
    boundary_time: float,
    beat_times: Iterable[float],
    max_offset: float,
) -> Optional[float]:
    """返回吸附后的时间（秒），若无合适节拍则返回 None。"""
    best_time = None
    best_offset = None
    for beat in beat_times:
        offset = abs(float(beat) - boundary_time)
        if offset > max_offset:
            continue
        if best_offset is None or offset < best_offset:
            best_offset = offset
            best_time = float(beat)
    return best_time


def _filter_adjustments(
    adjustments: Iterable[CutAdjustment],
    segments: List[Segment],
    sample_rate: float,
    *,
    extra_adjustments: Optional[List[CutAdjustment]] = None,
) -> List[CutAdjustment]:
    if not adjustments:
        return []
    boundaries = []
    if segments:
        boundaries.append(segments[0].start)
        boundaries.extend(seg.end for seg in segments)
    internal_boundaries = set(boundaries[1:-1])
    if not internal_boundaries:
        return []

    tolerance = max(1.0 / max(sample_rate, 1.0), 1e-4)

    filtered: List[CutAdjustment] = []
    for adj in adjustments:
        for boundary in internal_boundaries:
            if abs(adj.final_time - boundary) <= tolerance:
                filtered.append(adj)
                break
    if extra_adjustments:
        for adj in extra_adjustments:
            if not any(abs(adj.final_time - existing.final_time) <= tolerance for existing in filtered):
                filtered.append(adj)
    return filtered


def _restore_continuity(segments: List[Segment]) -> None:
    if not segments:
        return
    for i in range(1, len(segments)):
        segments[i].start = segments[i - 1].end


__all__ = [
    "Segment",
    "LayoutConfig",
    "LayoutResult",
    "derive_layout_config",
    "refine_layout",
]

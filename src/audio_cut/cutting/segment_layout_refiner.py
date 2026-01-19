#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/cutting/segment_layout_refiner.py
# AI-SUMMARY: 段落布局精炼器，负责微碎片合并、软最小再合并与软最大救援切分，返回更新后的段落与守卫调整。

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

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
) -> LayoutResult:
    """执行段落布局精炼，当前实现微碎片合并、软最小合并和软最大救援切分。"""

    segments = [Segment(seg.start, seg.end, seg.kind) for seg in segments]
    adjustments = list(adjustments or [])
    suppressed = list(suppressed_cut_points or [])

    if not config.enable or len(segments) <= 1:
        return LayoutResult(segments, adjustments, suppressed)

    refined_segments = _apply_micro_merge(segments, config.micro_merge_s, config.soft_max_s)
    refined_segments = _apply_soft_min_merge(refined_segments, config.soft_min_s, config.soft_max_s)
    refined_segments, suppressed, split_adjustments = _apply_soft_max_splits(
        refined_segments,
        suppressed,
        config.soft_max_s,
        min_gap_s=config.min_gap_s,
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


def _apply_soft_max_splits(
    segments: List[Segment],
    suppressed: List[CutPoint],
    soft_max_s: float,
    *,
    min_gap_s: float,
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
            best = max(candidates, key=lambda p: float(getattr(p, "score", 0.0) or 0.0))
            cut_time = float(best.t)
            suppressed.remove(best)
        else:
            cut_time = seg.start + seg.duration / 2.0

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

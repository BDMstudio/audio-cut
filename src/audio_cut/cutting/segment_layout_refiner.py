#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/cutting/segment_layout_refiner.py
# AI-SUMMARY: 段落布局精炼器，当前仅实现微碎片合并并过滤守卫调整，为后续软最小/软最大策略奠定结构。

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

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
    # 预留：可在后续基于 BPM/MDD 进行自适应调整
    _ = features, sample_rate  # placate linters for future use
    return LayoutConfig(
        enable=enable,
        micro_merge_s=max(0.0, micro_merge),
        soft_min_s=max(0.0, soft_min),
        soft_max_s=max(0.0, soft_max),
        min_gap_s=max(0.0, min_gap),
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
    """执行段落布局精炼（P0：仅支持微碎片合并）。"""

    segments = [Segment(seg.start, seg.end, seg.kind) for seg in segments]
    adjustments = list(adjustments or [])
    suppressed = list(suppressed_cut_points or [])

    if not config.enable or len(segments) <= 1:
        return LayoutResult(segments, adjustments, suppressed)

    refined_segments = _apply_micro_merge(segments, config.micro_merge_s, config.soft_max_s)

    filtered_adjustments = _filter_adjustments(adjustments, refined_segments, sample_rate)

    return LayoutResult(refined_segments, filtered_adjustments, suppressed)


def _apply_micro_merge(
    segments: List[Segment],
    micro_threshold_s: float,
    soft_max_s: float,
) -> List[Segment]:
    if micro_threshold_s <= 0.0 or len(segments) <= 1:
        return segments

    segs = [Segment(seg.start, seg.end, seg.kind) for seg in segments]
    original_start = segs[0].start
    original_end = segs[-1].end

    idx = 0
    while len(segs) > 1 and idx < len(segs):
        seg = segs[idx]
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

    # 纠正首尾，确保连续
    if segs:
        segs[0].start = original_start
        segs[-1].end = original_end
        for i in range(1, len(segs)):
            segs[i].start = segs[i - 1].end

    return segs


def _filter_adjustments(
    adjustments: Iterable[CutAdjustment],
    segments: List[Segment],
    sample_rate: float,
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
    return filtered


__all__ = [
    "Segment",
    "LayoutConfig",
    "LayoutResult",
    "derive_layout_config",
    "refine_layout",
]

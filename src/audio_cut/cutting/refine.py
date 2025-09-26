#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/cutting/refine.py
# AI-SUMMARY: 音频切点精炼工具，提供过零吸附、守卫右推与最小间隔 NMS 等共享逻辑。

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

_EPS = 1e-12


@dataclass
class CutPoint:
    """切点候选信息"""

    t: float
    score: float
    kind: str = "pause"


@dataclass
class CutContext:
    """切点精炼上下文"""

    sr: int
    mix_wave: np.ndarray
    vocal_wave: Optional[np.ndarray] = None


@dataclass
class CutAdjustment:
    """切点校正明细"""

    raw_time: float
    guard_time: float
    final_time: float
    score: float
    guard_shift_ms: float
    final_shift_ms: float


@dataclass
class CutRefineResult:
    """精炼结果汇总"""

    final_points: List[CutPoint]
    sample_boundaries: List[int]
    adjustments: List[CutAdjustment]


def _ensure_mono(wave: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if wave is None:
        return None
    if wave.ndim == 1:
        return wave
    if wave.ndim == 2:
        return np.mean(wave, axis=0)
    return wave.reshape(-1)


def align_to_zero_cross(wave: Optional[np.ndarray], sr: int, t: float, win_ms: float = 8.0) -> float:
    """在指定窗口内吸附到最近的零交叉点。"""

    wave = _ensure_mono(wave)
    if wave is None or wave.size == 0 or sr <= 0:
        return t

    idx = int(round(t * sr))
    if idx <= 0 or idx >= wave.size:
        return t

    half_window = max(1, int(round(win_ms / 1000.0 * sr)))
    start = max(1, idx - half_window)
    end = min(wave.size - 1, idx + half_window)
    if end <= start:
        return t

    best_zero = None
    best_dist = None
    for pos in range(start, end + 1):
        left = wave[pos - 1]
        right = wave[pos]
        if left == 0.0:
            zero_pos = pos - 1
        elif right == 0.0:
            zero_pos = pos
        elif left * right < 0.0:
            denom = abs(left) + abs(right)
            frac = abs(left) / denom if denom > _EPS else 0.5
            zero_pos = (pos - 1) + frac
        else:
            continue
        dist = abs(zero_pos - idx)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_zero = zero_pos
    if best_zero is None:
        return t
    return float(best_zero) / float(sr)


def apply_quiet_guard(
    wave: Optional[np.ndarray],
    sr: int,
    t: float,
    *,
    max_shift_ms: float = 150.0,
    guard_db: float = 2.0,
    window_ms: float = 10.0,
    floor_db: float = -60.0,
) -> float:
    """向右搜索更安静的点并返回调整后的时间。"""

    wave = _ensure_mono(wave)
    if wave is None or wave.size == 0 or sr <= 0:
        return t

    idx = int(round(t * sr))
    if idx < 0:
        idx = 0
    search_samples = max(1, int(round(max_shift_ms / 1000.0 * sr)))
    end = min(wave.size, idx + search_samples)
    if end <= idx + 1:
        return t

    segment = wave[idx:end]
    win = max(1, int(round(window_ms / 1000.0 * sr)))
    if segment.size <= win:
        rms_window = segment
    else:
        padded = np.pad(segment, (0, win - 1), mode='edge')
        sq = padded * padded
        kernel = np.ones(win) / float(win)
        rms_window = np.sqrt(np.convolve(sq, kernel, mode='valid') + _EPS)
    rms_db = 20.0 * np.log10(rms_window + _EPS)

    original_db = rms_db[0]
    target_idx = int(np.argmin(rms_db))
    target_db = rms_db[target_idx]

    if (original_db - target_db) < guard_db or target_db > floor_db:
        return t

    center = idx + target_idx + win // 2
    center = min(wave.size - 1, max(0, center))
    return float(center) / float(sr)


def nms_min_gap(points: Iterable[CutPoint], min_gap_s: float, topk: Optional[int] = None) -> List[CutPoint]:
    """基于得分的最小间隔筛选。"""

    ordered = sorted(points, key=lambda p: p.score, reverse=True)
    kept: List[CutPoint] = []
    for point in ordered:
        if all(abs(point.t - other.t) >= min_gap_s for other in kept):
            kept.append(point)
        if topk is not None and len(kept) >= topk:
            break
    return sorted(kept, key=lambda p: p.t)


def _filter_cut_times(
    times: Sequence[float],
    *,
    duration_s: float,
    min_gap_s: float,
    min_boundary_s: float,
) -> List[float]:
    filtered: List[float] = []
    if duration_s <= 0.0:
        return filtered
    boundary = min(min_boundary_s, duration_s / 2.0)
    for t in sorted(times):
        if t <= boundary or t >= (duration_s - boundary):
            continue
        if filtered and (t - filtered[-1]) < min_gap_s:
            continue
        filtered.append(t)
    return filtered


def finalize_cut_points(
    ctx: CutContext,
    raw_points: Iterable[CutPoint],
    *,
    use_vocal_guard_first: bool = True,
    min_gap_s: float = 1.0,
    max_keep: Optional[int] = None,
    guard_db: float = 2.0,
    search_right_ms: float = 150.0,
    guard_win_ms: float = 10.0,
    floor_db: float = -60.0,
    enable_mix_guard: bool = True,
    enable_vocal_guard: bool = True,
    zero_cross_win_ms: float = 8.0,
    min_boundary_s: float = 0.5,
) -> CutRefineResult:
    """执行切点精炼并返回结果。"""

    sr = ctx.sr
    mix = _ensure_mono(ctx.mix_wave)
    vocal = _ensure_mono(ctx.vocal_wave) if ctx.vocal_wave is not None else None
    duration_s = len(mix) / float(sr) if sr > 0 and mix is not None else 0.0

    if mix is None or mix.size == 0 or sr <= 0:
        sample_boundaries = [0, len(mix) if mix is not None else 0]
        return CutRefineResult([], sample_boundaries, [])

    base_candidates = list(raw_points)
    if not base_candidates:
        return CutRefineResult([], [0, len(mix)], [])

    pruned = nms_min_gap(base_candidates, min_gap_s=min_gap_s, topk=max_keep)
    adjustments: List[CutAdjustment] = []
    adjusted_times: List[float] = []
    for point in pruned:
        raw_t = point.t
        guard_stage_time = raw_t

        if use_vocal_guard_first and vocal is not None and enable_vocal_guard:
            guard_stage_time = align_to_zero_cross(vocal, sr, guard_stage_time, win_ms=zero_cross_win_ms)
            guard_stage_time = apply_quiet_guard(
                vocal,
                sr,
                guard_stage_time,
                max_shift_ms=search_right_ms,
                guard_db=guard_db,
                window_ms=guard_win_ms,
                floor_db=floor_db,
            )

        mix_time = align_to_zero_cross(mix, sr, guard_stage_time, win_ms=zero_cross_win_ms)
        if enable_mix_guard:
            mix_time = apply_quiet_guard(
                mix,
                sr,
                mix_time,
                max_shift_ms=search_right_ms,
                guard_db=guard_db,
                window_ms=guard_win_ms,
                floor_db=floor_db,
            )

        mix_time = float(np.clip(mix_time, 0.0, max(duration_s, 0.0)))
        adjustments.append(
            CutAdjustment(
                raw_time=float(raw_t),
                guard_time=float(guard_stage_time),
                final_time=float(mix_time),
                score=float(point.score),
                guard_shift_ms=float((guard_stage_time - raw_t) * 1000.0),
                final_shift_ms=float((mix_time - raw_t) * 1000.0),
            )
        )
        adjusted_times.append(mix_time)

    kept_times = _filter_cut_times(
        adjusted_times,
        duration_s=duration_s,
        min_gap_s=min_gap_s,
        min_boundary_s=min_boundary_s,
    )

    kept_adjustments: List[CutAdjustment] = []
    for t in kept_times:
        match = None
        best_diff = None
        for adj in adjustments:
            diff = abs(adj.final_time - t)
            if best_diff is None or diff < best_diff:
                match = adj
                best_diff = diff
        if match is not None:
            kept_adjustments.append(match)

    final_points = [CutPoint(t=float(t), score=1.0) for t in kept_times]
    sample_boundaries = [0]
    sample_boundaries.extend(int(round(t * sr)) for t in kept_times)
    sample_boundaries.append(len(mix))
    sample_boundaries = sorted(set(sample_boundaries))

    return CutRefineResult(final_points, sample_boundaries, kept_adjustments)

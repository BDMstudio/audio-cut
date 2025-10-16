#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/cutting/metrics.py
# AI-SUMMARY: 提供切点序列对齐与误差统计工具，衡量块级与整段流程的时间一致性。
"""Cut sequence comparison helpers.

These utilities are intentionally lightweight so unit tests can assert that the
chunked GPU pipeline produces the same cut points as the full-resolution flow
within a tight tolerance.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def _to_sorted_array(values: Iterable[float], *, exclude_edges: bool) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    arr = np.unique(arr)
    if exclude_edges and arr.size >= 2:
        arr = arr[1:-1]
    return arr


def compute_cut_time_diffs_ms(
    reference_samples: Sequence[int],
    candidate_samples: Sequence[int],
    sample_rate: float,
    *,
    exclude_edges: bool = True,
) -> np.ndarray:
    """Return per-cut absolute timing error in milliseconds.

    Args:
        reference_samples: Reference cut boundaries (sample index).
        candidate_samples: Cut boundaries to compare (sample index).
        sample_rate: Audio sample rate, used to convert to milliseconds.
        exclude_edges: Whether to drop the first/last boundary before compare.

    Raises:
        ValueError: If reference/candidate shapes mismatch after processing.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    ref = _to_sorted_array(reference_samples, exclude_edges=exclude_edges)
    cand = _to_sorted_array(candidate_samples, exclude_edges=exclude_edges)
    if ref.size != cand.size:
        raise ValueError(
            f"Cut sequence length mismatch: reference={ref.size}, candidate={cand.size}"
        )
    if ref.size == 0:
        return np.asarray([], dtype=float)

    sample_to_ms = 1000.0 / float(sample_rate)
    return np.abs(ref - cand) * sample_to_ms


def summarize_diffs(diff_ms: Sequence[float]) -> List[tuple[str, float]]:
    """Summarize timing error statistics.

    Returns a list of (name, value) tuples for stable formatting downstream.
    """
    arr = np.asarray(list(diff_ms), dtype=float)
    if arr.size == 0:
        return [
            ("mean_ms", 0.0),
            ("p50_ms", 0.0),
            ("p95_ms", 0.0),
            ("max_ms", 0.0),
        ]
    return [
        ("mean_ms", float(np.mean(arr))),
        ("p50_ms", float(np.percentile(arr, 50))),
        ("p95_ms", float(np.percentile(arr, 95))),
        ("max_ms", float(np.max(arr))),
    ]

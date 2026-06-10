#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/analysis/chorus_regions.py
# AI-SUMMARY: Shared chorus/high-energy bar detection for hybrid strategies and VPBD beat candidates.

from __future__ import annotations

import logging
from typing import Iterable, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


def detect_chorus_regions(
    bar_energies: Iterable[float],
    energy_threshold: float,
    *,
    min_consecutive_bars: int = 4,
    bar_centroids: Optional[Iterable[float]] = None,
    bar_bandwidths: Optional[Iterable[float]] = None,
) -> Set[int]:
    """Detect continuous high-energy bar regions using optional spectral fusion."""

    energies = np.asarray(list(bar_energies), dtype=np.float32)
    if energies.size == 0:
        return set()

    centroids = list(bar_centroids or [])
    bandwidths = list(bar_bandwidths or [])
    if centroids and bandwidths and len(centroids) == len(energies):
        norm_energy = _normalize(energies)
        norm_centroid = _normalize(centroids)
        norm_bandwidth = _normalize(bandwidths)
        energy_cv = float(np.std(energies) / (np.mean(energies) + 1e-6))

        if energy_cv < 0.15:
            weights = {'energy': 0.3, 'centroid': 0.4, 'bandwidth': 0.3}
            logger.debug("[ChorusDetect] Low dynamics (CV=%.3f), using spectral-heavy weights", energy_cv)
        elif energy_cv > 0.4:
            weights = {'energy': 0.6, 'centroid': 0.2, 'bandwidth': 0.2}
            logger.debug("[ChorusDetect] High dynamics (CV=%.3f), using energy-heavy weights", energy_cv)
        else:
            weights = {'energy': 0.5, 'centroid': 0.25, 'bandwidth': 0.25}
            logger.debug("[ChorusDetect] Medium dynamics (CV=%.3f), using balanced weights", energy_cv)

        chorus_score = (
            norm_energy * weights['energy']
            + norm_centroid * weights['centroid']
            + norm_bandwidth * weights['bandwidth']
        )
        fused_threshold = float(np.percentile(chorus_score, 60))
        is_high_energy = chorus_score >= fused_threshold
        logger.info(
            "[ChorusDetect] Multi-feature fusion: CV=%.3f, weights=%s, threshold=%.3f",
            energy_cv,
            weights,
            fused_threshold,
        )
    else:
        is_high_energy = energies >= float(energy_threshold)
        logger.debug("[ChorusDetect] Using simple energy threshold")

    return _continuous_regions(is_high_energy, min_consecutive_bars=max(1, int(min_consecutive_bars)))


def _normalize(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return arr
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if arr_max - arr_min > 1e-6:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr)


def _continuous_regions(is_high_energy: Iterable[bool], *, min_consecutive_bars: int) -> Set[int]:
    chorus_bars: Set[int] = set()
    current_start = None
    consecutive_count = 0

    for idx, is_high in enumerate(is_high_energy):
        if bool(is_high):
            if current_start is None:
                current_start = idx
            consecutive_count += 1
            continue

        if consecutive_count >= min_consecutive_bars and current_start is not None:
            chorus_bars.update(range(current_start, idx))
        current_start = None
        consecutive_count = 0

    if consecutive_count >= min_consecutive_bars and current_start is not None:
        chorus_bars.update(range(current_start, idx + 1))

    return chorus_bars

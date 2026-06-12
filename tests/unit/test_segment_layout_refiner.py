#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_segment_layout_refiner.py
# AI-SUMMARY: Tests segment layout secondary split selection using acoustic valleys and ASR soft priors.

from typing import List

import numpy as np

from audio_cut.analysis.features_cache import TrackFeatureCache
from audio_cut.cutting.segment_layout_refiner import LayoutConfig, Segment, refine_layout


def _features_with_valley(*, duration_s: float, valley_s: float, hop_s: float = 0.5) -> TrackFeatureCache:
    frame_count = int(duration_s / hop_s) + 1
    rms = np.ones(frame_count, dtype=np.float32)
    valley_idx = int(round(valley_s / hop_s))
    rms[max(0, valley_idx - 1): min(frame_count, valley_idx + 2)] = np.array([0.08, 0.01, 0.08], dtype=np.float32)
    zeros = np.zeros(frame_count, dtype=np.float32)
    return TrackFeatureCache(
        sr=44100,
        hop_length=int(44100 * hop_s),
        hop_s=hop_s,
        duration_s=duration_s,
        rms_series=rms,
        spectral_flatness=zeros.copy(),
        onset_envelope=zeros.copy(),
        onset_strength=zeros.copy(),
        onset_frames=np.array([], dtype=np.int64),
        rms_max=float(rms.max()),
        onset_max=0.0,
        bpm_features=None,
        tempo_curve=None,
        beat_times=np.array([], dtype=np.float32),
        global_mdd=0.0,
        mdd_series=zeros.copy(),
    )


def _features_with_valleys(*, duration_s: float, valley_times: List[float], hop_s: float = 0.05) -> TrackFeatureCache:
    frame_count = int(duration_s / hop_s) + 1
    rms = np.ones(frame_count, dtype=np.float32)
    for valley_s in valley_times:
        valley_idx = int(round(valley_s / hop_s))
        width = max(2, int(round(1.0 / hop_s)))
        for idx in range(max(0, valley_idx - width), min(frame_count, valley_idx + width + 1)):
            distance = abs(idx - valley_idx) / float(width)
            rms[idx] = min(rms[idx], 0.01 + 0.4 * distance)
    zeros = np.zeros(frame_count, dtype=np.float32)
    return TrackFeatureCache(
        sr=44100,
        hop_length=int(44100 * hop_s),
        hop_s=hop_s,
        duration_s=duration_s,
        rms_series=rms,
        spectral_flatness=zeros.copy(),
        onset_envelope=zeros.copy(),
        onset_strength=zeros.copy(),
        onset_frames=np.array([], dtype=np.int64),
        rms_max=float(rms.max()),
        onset_max=0.0,
        bpm_features=None,
        tempo_curve=None,
        beat_times=np.array([], dtype=np.float32),
        global_mdd=0.0,
        mdd_series=zeros.copy(),
    )


def test_soft_max_split_uses_acoustic_valley_near_asr_boundary() -> None:
    result = refine_layout(
        [Segment(0.0, 30.0, "human"), Segment(30.0, 35.0, "music")],
        [],
        config=LayoutConfig(enable=True, soft_max_s=15.0, min_gap_s=1.0),
        sample_rate=44100,
        features=_features_with_valley(duration_s=35.0, valley_s=20.0),
        asr_boundary_times=[20.1],
    )

    boundaries = [round(seg.end, 1) for seg in result.segments[:-1]]
    assert 20.1 in boundaries
    assert 15.0 not in boundaries


def test_soft_max_split_does_not_use_midpoint_without_acoustic_evidence() -> None:
    flat_features = _features_with_valley(duration_s=35.0, valley_s=20.0)
    flat_features.rms_series[:] = 1.0

    result = refine_layout(
        [Segment(0.0, 30.0, "human"), Segment(30.0, 35.0, "music")],
        [],
        config=LayoutConfig(enable=True, soft_max_s=15.0, min_gap_s=1.0),
        sample_rate=44100,
        features=flat_features,
        asr_boundary_times=[15.0],
    )

    boundaries = [round(seg.end, 1) for seg in result.segments[:-1]]
    assert boundaries == [30.0]

def test_soft_max_split_prefers_phrase_boundary_over_inside_word_valley() -> None:
    features = _features_with_valley(duration_s=35.0, valley_s=20.0)
    features.rms_series[:] = 1.0
    inside_idx = int(round(20.0 / features.hop_s))
    phrase_idx = int(round(22.0 / features.hop_s))
    features.rms_series[inside_idx - 1: inside_idx + 2] = np.array([0.08, 0.01, 0.08], dtype=np.float32)
    features.rms_series[phrase_idx - 1: phrase_idx + 2] = np.array([0.12, 0.05, 0.12], dtype=np.float32)
    features.rms_max = float(features.rms_series.max())

    result = refine_layout(
        [Segment(0.0, 30.0, "human"), Segment(30.0, 35.0, "music")],
        [],
        config=LayoutConfig(enable=True, soft_max_s=15.0, min_gap_s=1.0),
        sample_rate=44100,
        features=features,
        asr_boundary_times=[22.0],
        asr_word_intervals=[(19.8, 20.2)],
    )

    boundaries = [round(seg.end, 1) for seg in result.segments[:-1]]
    assert 22.0 in boundaries
    assert 20.0 not in boundaries


def test_soft_max_split_remerges_new_micro_piece_with_same_kind_neighbor() -> None:
    result = refine_layout(
        [
            Segment(0.0, 21.95, "human"),
            Segment(21.95, 23.31, "human"),
            Segment(23.31, 26.47, "music"),
        ],
        [],
        config=LayoutConfig(
            enable=True,
            micro_merge_s=2.0,
            soft_min_s=5.0,
            soft_max_s=12.0,
            min_gap_s=1.0,
        ),
        sample_rate=44100,
        features=_features_with_valleys(duration_s=30.0, valley_times=[9.68, 20.70]),
    )

    segments = [(round(seg.start, 2), round(seg.end, 2), seg.kind) for seg in result.segments]
    assert segments == [
        (0.0, 9.7, "human"),
        (9.7, 21.95, "human"),
        (21.95, 26.47, "music"),
    ]
    assert all(seg.duration >= 2.0 for seg in result.segments)

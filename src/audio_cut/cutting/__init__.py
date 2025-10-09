#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/cutting/__init__.py
# AI-SUMMARY: 提供音频切点精炼与守卫相关的共享工具入口。

from .refine import (
    CutPoint,
    CutContext,
    CutAdjustment,
    CutRefineResult,
    align_to_zero_cross,
    apply_quiet_guard,
    nms_min_gap,
    finalize_cut_points,
)
from .metrics import compute_cut_time_diffs_ms, summarize_diffs
from .segment_layout_refiner import (
    Segment,
    LayoutConfig,
    LayoutResult,
    derive_layout_config,
    refine_layout,
)

__all__ = [
    'CutPoint',
    'CutContext',
    'CutAdjustment',
    'CutRefineResult',
    'align_to_zero_cross',
    'apply_quiet_guard',
    'nms_min_gap',
    'finalize_cut_points',
    'compute_cut_time_diffs_ms',
    'summarize_diffs',
    'Segment',
    'LayoutConfig',
    'LayoutResult',
    'derive_layout_config',
    'refine_layout',
]

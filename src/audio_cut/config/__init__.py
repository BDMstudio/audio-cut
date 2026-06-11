#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/config/__init__.py
# AI-SUMMARY: 配置 schema v3 工具集入口，提供检测、派生与迁移实用函数。

"""Configuration helpers for the schema v3 pipeline."""

from .derive import (
    is_v3_schema,
    load_default_schema,
    schema_from_mapping,
    build_legacy_overrides,
    apply_profile_overrides,
)
from .auto_profile import (
    ALIGNMENT_STOPS,
    BEAT_POLE,
    LYRIC_POLE,
    SEGMENT_DURATION_STOPS,
    StyleEstimate,
    build_auto_profile_overrides,
    build_style_weight_overrides,
    derive_alignment_overrides,
    derive_smart_cut_overrides,
    estimate_style,
    resolve_alignment,
    resolve_smart_cut_intent,
    should_apply_duration_overrides,
)

__all__ = [
    "is_v3_schema",
    "load_default_schema",
    "schema_from_mapping",
    "build_legacy_overrides",
    "apply_profile_overrides",
    "ALIGNMENT_STOPS",
    "BEAT_POLE",
    "LYRIC_POLE",
    "SEGMENT_DURATION_STOPS",
    "StyleEstimate",
    "build_auto_profile_overrides",
    "build_style_weight_overrides",
    "derive_alignment_overrides",
    "derive_smart_cut_overrides",
    "estimate_style",
    "resolve_alignment",
    "resolve_smart_cut_intent",
    "should_apply_duration_overrides",
]

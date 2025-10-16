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

__all__ = [
    "is_v3_schema",
    "load_default_schema",
    "schema_from_mapping",
    "build_legacy_overrides",
    "apply_profile_overrides",
]

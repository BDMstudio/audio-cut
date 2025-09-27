#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/config/__init__.py
# AI-SUMMARY: 暴露 v3 精简配置入口，提供加载、派生与兼容旧配置的工具函数。

from .settings import AudioCutAppConfig, AudioCutDetectionConfig, GuardConfig, ThresholdConfig, AdaptConfig, NMSConfig
from .settings import load_schema_config, merge_schema_override, compose_app_config
from .derive import resolve_legacy_config

__all__ = [
    'AudioCutAppConfig',
    'AudioCutDetectionConfig',
    'GuardConfig',
    'ThresholdConfig',
    'AdaptConfig',
    'NMSConfig',
    'load_schema_config',
    'merge_schema_override',
    'compose_app_config',
    'resolve_legacy_config',
]

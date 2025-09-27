#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/config/settings.py
# AI-SUMMARY: 定义 v3 精简配置的结构体与加载/合并逻辑，支持 Profile 与覆盖。 

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_SCHEMA_VERSION = 3


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并 patch 到 base，遇到 dict 继续深入，否则直接覆盖。"""
    for key, value in patch.items():
        if isinstance(value, dict):
            node = base.get(key)
            if isinstance(node, dict):
                base[key] = _deep_merge(node, value)
            else:
                base[key] = copy.deepcopy(value)
        else:
            base[key] = copy.deepcopy(value)
    return base


@dataclass
class GuardConfig:
    enable: bool = False
    max_shift_ms: float = 150.0
    win_ms: float = 80.0
    guard_db: float = 2.5
    floor_db: float = -60.0


@dataclass
class ThresholdConfig:
    base_ratio: float = 0.26
    rms_offset: float = 0.06


@dataclass
class AdaptConfig:
    bpm_strength: float = 0.6
    mdd_strength: float = 0.4


@dataclass
class NMSConfig:
    topk: int = 180
    zero_cross_win_ms: float = 8.0


@dataclass
class AudioCutDetectionConfig:
    min_pause_s: float = 0.5
    min_gap_s: float = 1.0
    pure_music_min_duration: float = 6.0
    segment_vocal_ratio: float = 0.10
    guard: GuardConfig = field(default_factory=GuardConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    nms: NMSConfig = field(default_factory=NMSConfig)


@dataclass
class AudioCutAppConfig:
    audio: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    separation: Dict[str, Any] = field(default_factory=dict)
    detection: AudioCutDetectionConfig = field(default_factory=AudioCutDetectionConfig)
    profile: str = 'default'
    version: int = _SCHEMA_VERSION


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_schema_config(schema_path: Path) -> Dict[str, Any]:
    data = load_yaml(schema_path)
    if not isinstance(data, dict):
        raise ValueError(f"v3 schema 必须是字典: {schema_path}")
    data.setdefault('version', _SCHEMA_VERSION)
    if data.get('version') != _SCHEMA_VERSION:
        raise ValueError(f"不支持的 schema 版本: {data.get('version')}")
    data.setdefault('defaults', {})
    data.setdefault('profiles', {})
    data.setdefault('profile', data.get('defaults', {}).get('profile', 'default'))
    return data


def merge_schema_override(schema: Dict[str, Any], override: Dict[str, Any]) -> None:
    if not override:
        return
    if 'version' in override and override['version'] != schema.get('version', _SCHEMA_VERSION):
        raise ValueError('覆盖配置的 version 不一致')

    if 'profiles' in override:
        schema.setdefault('profiles', {})
        _deep_merge(schema['profiles'], override['profiles'])

    if 'profile' in override:
        schema['profile'] = override['profile']

    direct_defaults: Dict[str, Any] = {}
    if 'defaults' in override:
        direct_defaults = override['defaults']
    else:
        for key, val in override.items():
            if key in {'version', 'profiles', 'profile'}:
                continue
            direct_defaults[key] = val

    if direct_defaults:
        schema.setdefault('defaults', {})
        _deep_merge(schema['defaults'], direct_defaults)


def _dict_to_guard(data: Dict[str, Any]) -> GuardConfig:
    return GuardConfig(
        enable=bool(data.get('enable', False)),
        max_shift_ms=float(data.get('max_shift_ms', 150.0)),
        win_ms=float(data.get('win_ms', 80.0)),
        guard_db=float(data.get('guard_db', 2.5)),
        floor_db=float(data.get('floor_db', -60.0)),
    )


def _dict_to_threshold(data: Dict[str, Any]) -> ThresholdConfig:
    return ThresholdConfig(
        base_ratio=float(data.get('base_ratio', 0.26)),
        rms_offset=float(data.get('rms_offset', 0.06)),
    )


def _dict_to_adapt(data: Dict[str, Any]) -> AdaptConfig:
    return AdaptConfig(
        bpm_strength=float(data.get('bpm_strength', 0.6)),
        mdd_strength=float(data.get('mdd_strength', 0.4)),
    )


def _dict_to_nms(data: Dict[str, Any]) -> NMSConfig:
    return NMSConfig(
        topk=int(data.get('topk', 180)),
        zero_cross_win_ms=float(data.get('zero_cross_win_ms', 8.0)),
    )


def _dict_to_detection(data: Dict[str, Any]) -> AudioCutDetectionConfig:
    return AudioCutDetectionConfig(
        min_pause_s=float(data.get('min_pause_s', 0.5)),
        min_gap_s=float(data.get('min_gap_s', 1.0)),
        pure_music_min_duration=float(data.get('pure_music_min_duration', 6.0)),
        segment_vocal_ratio=float(data.get('segment_vocal_ratio', 0.10)),
        guard=_dict_to_guard(data.get('guard', {})),
        threshold=_dict_to_threshold(data.get('threshold', {})),
        adapt=_dict_to_adapt(data.get('adapt', {})),
        nms=_dict_to_nms(data.get('nms', {})),
    )


def compose_app_config(schema: Dict[str, Any]) -> AudioCutAppConfig:
    defaults = copy.deepcopy(schema.get('defaults', {}))
    profile = schema.get('profile', 'default')
    profiles = schema.get('profiles', {})
    if profile and profile in profiles:
        _deep_merge(defaults, profiles[profile])

    detection_cfg = _dict_to_detection(defaults.get('detection', {}))

    return AudioCutAppConfig(
        audio=copy.deepcopy(defaults.get('audio', {})),
        output=copy.deepcopy(defaults.get('output', {})),
        logging=copy.deepcopy(defaults.get('logging', {})),
        separation=copy.deepcopy(defaults.get('separation', {})),
        detection=detection_cfg,
        profile=profile or 'default',
        version=int(schema.get('version', _SCHEMA_VERSION)),
    )

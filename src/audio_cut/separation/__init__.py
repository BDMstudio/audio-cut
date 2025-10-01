#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/separation/__init__.py
# AI-SUMMARY: 声部分离后端包，导出统一接口与默认实现。

from .backends import (
    DemucsPyTorchBackend,
    IVocalSeparatorBackend,
    MDX23OnnxBackend,
    SeparationOutputs,
)

__all__ = [
    'IVocalSeparatorBackend',
    'SeparationOutputs',
    'MDX23OnnxBackend',
    'DemucsPyTorchBackend',
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/__init__.py
# AI-SUMMARY: 顶层包入口，使用惰性导入避免循环依赖。

"""Utility package for shared audio-cutting helpers."""

__all__ = ['separate_and_segment']


def __getattr__(name: str):
    """Lazily import to avoid circular import with vocal_smart_splitter."""
    if name == 'separate_and_segment':
        from .api import separate_and_segment
        return separate_and_segment
    raise AttributeError(f"module 'audio_cut' has no attribute {name!r}")

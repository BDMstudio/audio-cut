#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/__init__.py
# AI-SUMMARY: 顶层包入口，聚合 audio_cut 通用工具模块。

"""Utility package for shared audio-cutting helpers."""

from .api import separate_and_segment

__all__ = ['separate_and_segment']

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/exceptions.py
# AI-SUMMARY: Shared exception types for optional lyrics alignment and VPBD planning.

from __future__ import annotations


class AudioCutError(Exception):
    """Base class for audio-cut controlled failures."""


class LyricsAlignmentUnavailable(AudioCutError):
    """Raised when lyrics alignment is required but no provider can supply it."""


class FireRedProviderError(AudioCutError):
    """Raised when a FireRed provider fails while strict ASR mode is enabled."""


class TimelineValidationError(AudioCutError):
    """Raised when a lyrics timeline has invalid or inconsistent timestamps."""


class GlobalCutPlanningError(AudioCutError):
    """Raised when the global cut planner cannot produce a valid path."""

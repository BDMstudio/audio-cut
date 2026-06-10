#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/lyrics/__init__.py
# AI-SUMMARY: Public lyrics alignment models and provider seams for VPBD modes.

from .cache import build_lyrics_cache_key
from .chunker import LyricsChunk, plan_asr_chunks
from .models import LyricsTimeline, Sentence, VadRegion, Word
from .timeline import merge_chunk_timelines
from .providers import (
    FakeLyricsProvider,
    LyricsProvider,
    LyricsProviderRequest,
    NullLyricsProvider,
    build_lyrics_provider,
)

__all__ = [
    "Word",
    "Sentence",
    "VadRegion",
    "LyricsTimeline",
    "LyricsChunk",
    "plan_asr_chunks",
    "merge_chunk_timelines",
    "build_lyrics_cache_key",
    "LyricsProviderRequest",
    "LyricsProvider",
    "NullLyricsProvider",
    "FakeLyricsProvider",
    "build_lyrics_provider",
]

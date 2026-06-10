#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_lyrics_timeline_merge.py
# AI-SUMMARY: Tests ASR chunk timeline merging and overlap de-duplication.

from pathlib import Path

from audio_cut.lyrics.chunker import LyricsChunk
from audio_cut.lyrics.models import LyricsTimeline, Word
from audio_cut.lyrics.timeline import merge_chunk_timelines


def test_merge_chunk_timelines_converts_local_times_to_global() -> None:
    chunks = [
        LyricsChunk(0, 0.0, 35.0, Path("chunk0.wav")),
        LyricsChunk(1, 34.0, 60.0, Path("chunk1.wav")),
    ]
    timelines = {
        0: LyricsTimeline(words=[Word("hello", 1.0, 1.4)], duration_s=35.0, source="c0"),
        1: LyricsTimeline(words=[Word("again", 2.0, 2.3)], duration_s=26.0, source="c1"),
    }

    merged = merge_chunk_timelines(chunks, timelines)

    assert [(word.text, word.start_s, word.end_s) for word in merged.words] == [
        ("hello", 1.0, 1.4),
        ("again", 36.0, 36.3),
    ]
    assert merged.duration_s == 60.0


def test_merge_chunk_timelines_deduplicates_overlap_words_by_confidence() -> None:
    chunks = [
        LyricsChunk(0, 0.0, 35.0, Path("chunk0.wav")),
        LyricsChunk(1, 34.0, 60.0, Path("chunk1.wav")),
    ]
    timelines = {
        0: LyricsTimeline(
            words=[
                Word("keep", 30.0, 30.4, confidence=0.8),
                Word("echo", 34.5, 34.8, confidence=0.4),
            ],
            duration_s=35.0,
            source="c0",
        ),
        1: LyricsTimeline(
            words=[Word("echo", 0.5, 0.8, confidence=0.9)],
            duration_s=26.0,
            source="c1",
        ),
    }

    merged = merge_chunk_timelines(chunks, timelines)

    assert [(word.text, word.confidence) for word in merged.words] == [
        ("keep", 0.8),
        ("echo", 0.9),
    ]


def test_merge_chunk_timelines_deduplicates_missing_confidence_by_chunk_center() -> None:
    chunks = [
        LyricsChunk(0, 0.0, 35.0, Path("chunk0.wav")),
        LyricsChunk(1, 34.0, 60.0, Path("chunk1.wav")),
    ]
    timelines = {
        0: LyricsTimeline(words=[Word("echo", 34.5, 34.8)], duration_s=35.0, source="c0"),
        1: LyricsTimeline(words=[Word("echo", 0.5, 0.8)], duration_s=26.0, source="c1"),
    }

    merged = merge_chunk_timelines(chunks, timelines)

    assert len(merged.words) == 1
    assert merged.words[0].start_s == 34.5
    assert merged.words[0].end_s == 34.8


def test_merge_chunk_timelines_marks_internal_chunk_boundaries_as_forbidden() -> None:
    chunks = [
        LyricsChunk(0, 0.0, 35.0, Path("chunk0.wav")),
        LyricsChunk(1, 34.0, 60.0, Path("chunk1.wav")),
        LyricsChunk(2, 59.0, 80.0, Path("chunk2.wav")),
    ]

    merged = merge_chunk_timelines(chunks, {})

    assert merged.meta["forbidden_cut_times_s"] == [35.0, 60.0]

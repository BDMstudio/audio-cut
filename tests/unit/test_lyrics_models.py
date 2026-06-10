#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_lyrics_models.py
# AI-SUMMARY: Tests lyrics timeline data models and validation behavior.

import pytest

from audio_cut.exceptions import TimelineValidationError
from audio_cut.lyrics.models import LyricsTimeline, Sentence, VadRegion, Word


def test_timeline_sorts_words_and_regions() -> None:
    timeline = LyricsTimeline(
        words=[
            Word(text="world", start_s=1.0, end_s=1.4, confidence=0.9),
            Word(text="hello", start_s=0.2, end_s=0.6, confidence=0.8),
        ],
        sentences=[Sentence(text="hello world", start_s=0.2, end_s=1.4, confidence=0.85)],
        vad_regions=[
            VadRegion(start_s=0.9, end_s=1.5, confidence=0.7, kind="singing"),
            VadRegion(start_s=0.1, end_s=0.7, confidence=0.6, kind="speech"),
        ],
        duration_s=2.0,
        source="fake",
    )

    assert [word.text for word in timeline.words] == ["hello", "world"]
    assert [region.kind for region in timeline.vad_regions] == ["speech", "singing"]
    assert timeline.to_dict()["source"] == "fake"


def test_timeline_filters_invalid_items_when_not_strict() -> None:
    timeline = LyricsTimeline.from_dict(
        {
            "duration_s": 2.0,
            "source": "fixture",
            "words": [
                {"text": "keep", "start_s": 0.1, "end_s": 0.4, "confidence": 0.8},
                {"text": "drop", "start_s": 0.8, "end_s": 0.7, "confidence": 0.8},
                {"text": "outside", "start_s": 2.1, "end_s": 2.4, "confidence": 0.8},
            ],
            "sentences": [
                {"text": "keep", "start_s": 0.1, "end_s": 0.4, "confidence": 0.8},
            ],
            "vad_regions": [
                {"start_s": 0.0, "end_s": 0.5, "confidence": 0.5, "kind": "singing"},
                {"start_s": 1.0, "end_s": 0.5, "confidence": 0.5, "kind": "singing"},
            ],
        },
        strict=False,
    )

    assert [word.text for word in timeline.words] == ["keep"]
    assert len(timeline.vad_regions) == 1
    assert timeline.warnings


def test_timeline_rejects_invalid_items_when_strict() -> None:
    with pytest.raises(TimelineValidationError):
        LyricsTimeline.from_dict(
            {
                "duration_s": 2.0,
                "words": [
                    {"text": "bad", "start_s": 1.0, "end_s": 0.5, "confidence": 0.8},
                ],
            },
            strict=True,
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_fake_lyrics_provider.py
# AI-SUMMARY: Tests fake and null lyrics providers without external ASR dependencies.

from pathlib import Path

import pytest

from audio_cut.exceptions import LyricsAlignmentUnavailable
from audio_cut.lyrics.providers import (
    FakeLyricsProvider,
    LyricsProviderRequest,
    NullLyricsProvider,
    build_lyrics_provider,
)


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lyrics" / "simple_song_timeline.json"


def test_fake_provider_loads_fixture_timeline() -> None:
    provider = FakeLyricsProvider(FIXTURE)

    timeline = provider.align(
        LyricsProviderRequest(vocal_path=Path("vocal.wav"), duration_s=8.0, strict=True)
    )

    assert provider.name == "fake"
    assert [word.text for word in timeline.words] == ["hello", "world", "again"]
    assert timeline.sentences[0].text == "hello world"
    assert timeline.source == "fake"


def test_null_provider_fails_loudly_in_strict_mode() -> None:
    provider = NullLyricsProvider(reason="disabled")

    with pytest.raises(LyricsAlignmentUnavailable):
        provider.align(LyricsProviderRequest(vocal_path=Path("vocal.wav"), strict=True))


def test_null_provider_returns_empty_timeline_when_not_strict() -> None:
    provider = NullLyricsProvider(reason="disabled")

    timeline = provider.align(LyricsProviderRequest(vocal_path=Path("vocal.wav"), duration_s=4.0))

    assert timeline.words == []
    assert timeline.duration_s == 4.0
    assert timeline.warnings == ["disabled"]


def test_build_lyrics_provider_selects_fake_and_disabled() -> None:
    fake = build_lyrics_provider(
        {"provider": "fake", "fixture_path": str(FIXTURE), "strict": True}
    )
    disabled = build_lyrics_provider({"provider": "disabled"})
    auto = build_lyrics_provider({"provider": "auto"})

    assert isinstance(fake, FakeLyricsProvider)
    assert isinstance(disabled, NullLyricsProvider)
    assert isinstance(auto, NullLyricsProvider)

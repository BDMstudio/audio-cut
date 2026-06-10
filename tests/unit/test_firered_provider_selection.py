#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_firered_provider_selection.py
# AI-SUMMARY: Verifies FireRed provider construction and auto fallback order.

from __future__ import annotations

from typing import Any

import audio_cut.lyrics.providers as providers
from audio_cut.lyrics.providers import LyricsProvider, NullLyricsProvider, build_lyrics_provider


class _ReadyProvider(LyricsProvider):
    name = "ready"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def is_available(self) -> bool:
        return True

    def align(self, request: Any) -> Any:
        raise NotImplementedError


class _DownProvider(_ReadyProvider):
    name = "down"

    def is_available(self) -> bool:
        return False


def test_auto_provider_prefers_sidecar_before_cli(monkeypatch: Any) -> None:
    class ReadySidecar(_ReadyProvider):
        name = "firered_sidecar"

    class ReadyCli(_ReadyProvider):
        name = "firered_cli"

    monkeypatch.setattr(providers, "FireRedSidecarProvider", ReadySidecar)
    monkeypatch.setattr(providers, "FireRedCliProvider", ReadyCli)

    provider = build_lyrics_provider(
        {
            "provider": "auto",
            "fire_red": {
                "provider_order": ["sidecar", "cli", "null"],
                "endpoint": "http://127.0.0.1:18888",
                "cli": {"executable": "firered-worker", "timeout_s": 3.0},
            },
        }
    )

    assert provider.name == "firered_sidecar"


def test_auto_provider_uses_cli_when_sidecar_unavailable(monkeypatch: Any) -> None:
    class ReadyCli(_ReadyProvider):
        name = "firered_cli"

    monkeypatch.setattr(providers, "FireRedSidecarProvider", _DownProvider)
    monkeypatch.setattr(providers, "FireRedCliProvider", ReadyCli)

    provider = build_lyrics_provider(
        {
            "provider": "auto",
            "fire_red": {
                "provider_order": ["sidecar", "cli", "null"],
                "endpoint": "http://127.0.0.1:18888",
                "cli": {"executable": "firered-worker", "timeout_s": 3.0},
            },
        }
    )

    assert provider.name == "firered_cli"


def test_auto_provider_returns_null_when_no_backend_is_available(monkeypatch: Any) -> None:
    monkeypatch.setattr(providers, "FireRedSidecarProvider", _DownProvider)
    monkeypatch.setattr(providers, "FireRedCliProvider", _DownProvider)

    provider = build_lyrics_provider(
        {
            "provider": "auto",
            "fire_red": {
                "provider_order": ["sidecar", "cli", "in_process", "null"],
                "endpoint": "http://127.0.0.1:18888",
                "cli": {"executable": "missing-worker"},
            },
        }
    )

    assert isinstance(provider, NullLyricsProvider)
    assert "no available FireRed backend" in provider.reason

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/lyrics/providers.py
# AI-SUMMARY: Defines optional lyrics alignment provider interface plus null, fake and FireRed providers.

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from audio_cut.exceptions import LyricsAlignmentUnavailable
from audio_cut.lyrics.models import LyricsTimeline


FireRedCliProvider: Any = None
FireRedSidecarProvider: Any = None


@dataclass
class LyricsProviderRequest:
    """Input context passed to a lyrics alignment provider."""

    vocal_path: Path
    duration_s: Optional[float] = None
    sample_rate: Optional[int] = None
    strict: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


class LyricsProvider(ABC):
    """Abstract lyrics alignment provider seam."""

    name: str = "base"

    @abstractmethod
    def align(self, request: LyricsProviderRequest) -> LyricsTimeline:
        """Return a full-track lyrics timeline for the request."""


class NullLyricsProvider(LyricsProvider):
    """Provider used when lyrics alignment is disabled or unavailable."""

    name = "null"

    def __init__(self, reason: str = "lyrics alignment disabled") -> None:
        self.reason = reason

    def align(self, request: LyricsProviderRequest) -> LyricsTimeline:
        if request.strict:
            raise LyricsAlignmentUnavailable(self.reason)
        return LyricsTimeline(
            words=[],
            sentences=[],
            vad_regions=[],
            duration_s=request.duration_s,
            source=self.name,
            warnings=[self.reason],
        )


class FakeLyricsProvider(LyricsProvider):
    """Fixture-backed provider for deterministic tests and local dry runs."""

    name = "fake"

    def __init__(self, fixture_path: Path | str) -> None:
        self.fixture_path = Path(fixture_path)

    def align(self, request: LyricsProviderRequest) -> LyricsTimeline:
        if not self.fixture_path.exists():
            message = f"lyrics fixture not found: {self.fixture_path}"
            if request.strict:
                raise LyricsAlignmentUnavailable(message)
            return LyricsTimeline(duration_s=request.duration_s, source=self.name, warnings=[message])
        with self.fixture_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("source", self.name)
        if request.duration_s is not None:
            payload.setdefault("duration_s", request.duration_s)
        return LyricsTimeline.from_dict(payload, strict=request.strict)


def build_lyrics_provider(cfg: Dict[str, Any]) -> LyricsProvider:
    """Build a lyrics provider from a lyrics_alignment-style config mapping."""

    provider = str(cfg.get("provider", "disabled")).strip().lower()
    fire_red_cfg = _mapping(cfg.get("fire_red", {}))
    if provider in {"", "disabled", "none", "null"}:
        return NullLyricsProvider(reason="lyrics alignment disabled")
    if provider == "fake":
        fixture_path = cfg.get("fixture_path")
        if not fixture_path:
            return NullLyricsProvider(reason="fake lyrics provider requires fixture_path")
        return FakeLyricsProvider(Path(str(fixture_path)))
    if provider == "sidecar":
        return _build_sidecar_provider(fire_red_cfg)
    if provider == "cli":
        return _build_cli_provider(fire_red_cfg)
    if provider == "auto":
        return _build_auto_provider(fire_red_cfg)
    return NullLyricsProvider(reason=f"unsupported lyrics provider: {provider}")


def _build_auto_provider(fire_red_cfg: Dict[str, Any]) -> LyricsProvider:
    reasons = []
    for name in _provider_order(fire_red_cfg):
        if name == "sidecar":
            provider = _build_sidecar_provider(fire_red_cfg)
            if _available(provider):
                return provider
            reasons.append(_reason(provider, "sidecar unavailable"))
        elif name == "cli":
            provider = _build_cli_provider(fire_red_cfg)
            if _available(provider):
                return provider
            reasons.append(_reason(provider, "cli unavailable"))
        elif name == "in_process":
            reasons.append("in_process provider is not configured")
        elif name == "null":
            break
    detail = "; ".join(reason for reason in reasons if reason)
    suffix = f": {detail}" if detail else ""
    return NullLyricsProvider(reason=f"no available FireRed backend{suffix}")


def _provider_order(fire_red_cfg: Dict[str, Any]) -> Iterable[str]:
    value = fire_red_cfg.get("provider_order", ["sidecar", "cli", "in_process", "null"])
    if not isinstance(value, list):
        return ["sidecar", "cli", "in_process", "null"]
    return [str(item).strip().lower() for item in value]


def _build_sidecar_provider(fire_red_cfg: Dict[str, Any]) -> LyricsProvider:
    endpoint = fire_red_cfg.get("endpoint")
    if not endpoint:
        return NullLyricsProvider(reason="FireRed sidecar endpoint is not configured")
    provider_cls = _sidecar_provider_class()
    return provider_cls(
        endpoint=str(endpoint),
        health_path=str(fire_red_cfg.get("health_path", "/health")),
        analyze_path=str(fire_red_cfg.get("analyze_path", "/analyze")),
        timeout_s=float(fire_red_cfg.get("timeout_s", 2.0)),
    )


def _build_cli_provider(fire_red_cfg: Dict[str, Any]) -> LyricsProvider:
    cli_cfg = _mapping(fire_red_cfg.get("cli", {}))
    executable = cli_cfg.get("executable")
    if not executable:
        return NullLyricsProvider(reason="FireRed CLI executable is not configured")
    provider_cls = _cli_provider_class()
    return provider_cls(
        executable=str(executable),
        model_dir=str(cli_cfg["model_dir"]) if cli_cfg.get("model_dir") else None,
        timeout_s=float(cli_cfg.get("timeout_s", 120.0)),
    )


def _cli_provider_class() -> Any:
    global FireRedCliProvider
    if FireRedCliProvider is None:
        from audio_cut.lyrics.firered_cli_provider import FireRedCliProvider as provider_cls

        FireRedCliProvider = provider_cls
    return FireRedCliProvider


def _sidecar_provider_class() -> Any:
    global FireRedSidecarProvider
    if FireRedSidecarProvider is None:
        from audio_cut.lyrics.firered_sidecar_provider import FireRedSidecarProvider as provider_cls

        FireRedSidecarProvider = provider_cls
    return FireRedSidecarProvider


def _available(provider: LyricsProvider) -> bool:
    checker = getattr(provider, "is_available", None)
    if checker is None:
        return not isinstance(provider, NullLyricsProvider)
    try:
        return bool(checker())
    except Exception:
        return False


def _reason(provider: LyricsProvider, fallback: str) -> str:
    return str(getattr(provider, "reason", fallback))


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}

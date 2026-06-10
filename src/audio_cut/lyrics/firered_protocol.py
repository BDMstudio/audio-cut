#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/lyrics/firered_protocol.py
# AI-SUMMARY: Normalizes FireRed worker JSON input and output for lyrics alignment providers.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from audio_cut.lyrics.models import LyricsTimeline
from audio_cut.lyrics.providers import LyricsProviderRequest


def build_worker_request(request: LyricsProviderRequest) -> Dict[str, Any]:
    """Return the stable JSON input protocol sent to external FireRed workers."""

    return {
        "audio_path": str(Path(request.vocal_path)),
        "duration_s": request.duration_s,
        "sample_rate": request.sample_rate,
        "strict": request.strict,
        "meta": dict(request.meta),
    }


def parse_worker_response(
    payload: Dict[str, Any],
    *,
    duration_s: Optional[float],
    global_t0_s: float,
    source: str,
    strict: bool,
) -> LyricsTimeline:
    """Normalize FireRed worker output into a global-seconds lyrics timeline."""

    resolved_duration_s = _resolve_duration(payload, duration_s, global_t0_s)
    timeline_payload: Dict[str, Any] = {
        "duration_s": resolved_duration_s,
        "source": source,
        "warnings": list(payload.get("warnings", [])),
        "meta": dict(payload.get("meta", {})),
        "words": [
            _normalize_text_item(raw, global_t0_s)
            for raw in _items(payload, "words")
        ],
        "sentences": [
            _normalize_text_item(raw, global_t0_s)
            for raw in _items(payload, "sentences", "segments")
        ],
        "vad_regions": [
            _normalize_vad_item(raw, global_t0_s)
            for raw in _items(payload, "vad_regions", "mvad", "mVAD")
        ],
    }
    return LyricsTimeline.from_dict(timeline_payload, strict=strict)


def _items(payload: Dict[str, Any], *keys: str) -> Iterable[Dict[str, Any]]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _normalize_text_item(raw: Dict[str, Any], global_t0_s: float) -> Dict[str, Any]:
    return {
        "text": str(raw.get("text") or raw.get("word") or raw.get("token") or ""),
        "start_s": _time_s(raw, "start", global_t0_s),
        "end_s": _time_s(raw, "end", global_t0_s),
        "confidence": raw.get("confidence"),
    }


def _normalize_vad_item(raw: Dict[str, Any], global_t0_s: float) -> Dict[str, Any]:
    return {
        "start_s": _time_s(raw, "start", global_t0_s),
        "end_s": _time_s(raw, "end", global_t0_s),
        "confidence": raw.get("confidence"),
        "kind": str(raw.get("kind") or "singing"),
    }


def _time_s(raw: Dict[str, Any], stem: str, global_t0_s: float) -> float:
    ms_key = f"{stem}_ms"
    seconds_key = f"{stem}_s"
    if ms_key in raw:
        return global_t0_s + float(raw[ms_key]) / 1000.0
    if seconds_key in raw:
        return global_t0_s + float(raw[seconds_key])
    if stem in raw:
        return global_t0_s + float(raw[stem])
    raise KeyError(f"{stem}_ms or {stem}_s is required")


def _resolve_duration(
    payload: Dict[str, Any],
    duration_s: Optional[float],
    global_t0_s: float,
) -> Optional[float]:
    if duration_s is not None:
        return float(duration_s)
    if payload.get("duration_s") is not None:
        return global_t0_s + float(payload["duration_s"])
    if payload.get("duration_ms") is not None:
        return global_t0_s + float(payload["duration_ms"]) / 1000.0
    return None

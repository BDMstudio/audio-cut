#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/lyrics/cache.py
# AI-SUMMARY: Builds stable cache keys for optional lyrics alignment timelines.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def build_lyrics_cache_key(
    *,
    audio_path: Path | str,
    separator: str,
    mode: str,
    provider: str,
    provider_version: Optional[str],
    chunk_s: float,
    overlap_s: float,
    scorer_config: Dict[str, Any],
    planner_config: Dict[str, Any],
) -> str:
    """Build a deterministic lyrics timeline cache key."""

    payload = {
        "audio_sha256": _sha256_file(Path(audio_path)),
        "separator": separator,
        "mode": mode,
        "provider": provider,
        "provider_version": provider_version,
        "chunk_s": float(chunk_s),
        "overlap_s": float(overlap_s),
        "scorer_config": scorer_config,
        "planner_config": planner_config,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "lyrics:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

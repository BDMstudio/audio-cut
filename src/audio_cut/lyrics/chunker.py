#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/lyrics/chunker.py
# AI-SUMMARY: Plans ASR-safe overlapping chunks for FireRed-style lyrics alignment.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

FIRERED_AED_MAX_CHUNK_S = 60.0
DEFAULT_CHUNK_S = 35.0
DEFAULT_OVERLAP_S = 1.0
DEFAULT_MAX_CHUNK_S = 55.0


@dataclass(frozen=True)
class LyricsChunk:
    """ASR chunk metadata using global track seconds."""

    chunk_id: int
    global_t0: float
    global_t1: float
    path: Path
    duration_s: float = field(init=False)

    def __post_init__(self) -> None:
        if self.global_t0 < 0.0:
            raise ValueError("global_t0 must be >= 0")
        if self.global_t1 <= self.global_t0:
            raise ValueError("global_t1 must be greater than global_t0")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "duration_s", round(self.global_t1 - self.global_t0, 9))


def plan_asr_chunks(
    audio_path: Path | str,
    *,
    duration_s: float,
    output_dir: Path | str,
    chunk_s: float = DEFAULT_CHUNK_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    max_chunk_s: float = DEFAULT_MAX_CHUNK_S,
) -> List[LyricsChunk]:
    """Plan overlapping ASR chunks without writing chunk audio."""

    duration_s = float(duration_s)
    if duration_s <= 0.0:
        return []

    effective_max = min(float(max_chunk_s), FIRERED_AED_MAX_CHUNK_S)
    effective_chunk_s = min(float(chunk_s), effective_max)
    if effective_chunk_s <= 0.0:
        raise ValueError("chunk_s must be positive")
    overlap_s = float(overlap_s)
    if overlap_s < 0.0:
        raise ValueError("overlap_s must be >= 0")
    if overlap_s >= effective_chunk_s:
        raise ValueError("overlap_s must be smaller than effective chunk length")

    output_root = Path(output_dir)
    stem = Path(audio_path).stem
    chunks: List[LyricsChunk] = []
    start = 0.0
    chunk_id = 0
    while start < duration_s:
        end = min(duration_s, start + effective_chunk_s)
        chunks.append(
            LyricsChunk(
                chunk_id=chunk_id,
                global_t0=round(start, 9),
                global_t1=round(end, 9),
                path=output_root / f"{stem}_chunk_{chunk_id:03d}.wav",
            )
        )
        if end >= duration_s:
            break
        next_start = end - overlap_s
        if next_start <= start:
            raise ValueError("chunk planning did not advance")
        start = next_start
        chunk_id += 1
    return chunks

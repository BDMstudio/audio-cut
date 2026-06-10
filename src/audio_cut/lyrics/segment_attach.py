#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/lyrics/segment_attach.py
# AI-SUMMARY: Attaches full-track lyrics timeline words to exported segment metadata.

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Sequence

from audio_cut.lyrics.models import LyricsTimeline, Word

_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def attach_lyrics_to_segments(
    segments: Sequence[Mapping[str, Any]],
    timeline: LyricsTimeline,
    *,
    min_word_overlap_ratio: float = 0.5,
) -> List[Dict[str, Any]]:
    """Return segment dicts with optional `lyrics` objects attached."""

    attached: List[Dict[str, Any]] = []
    for segment in segments:
        item = deepcopy(dict(segment))
        start = _as_float(item.get("start"))
        end = _as_float(item.get("end"))
        if start is None or end is None or end <= start:
            item["lyrics"] = None
            attached.append(item)
            continue
        words = [
            word
            for word in timeline.words
            if _word_overlap_ratio(word, start, end) >= min_word_overlap_ratio
        ]
        item["lyrics"] = _build_segment_lyrics(words) if words else None
        attached.append(item)
    return attached


def _build_segment_lyrics(words: Sequence[Word]) -> Dict[str, Any]:
    ordered = sorted(words, key=lambda word: (word.start_s, word.end_s))
    texts = [word.text for word in ordered]
    joiner = "" if texts and all(_is_cjk_text(text) for text in texts) else " "
    return {
        "text": joiner.join(texts),
        "words": [word.to_dict() for word in ordered],
        "start": ordered[0].start_s if ordered else None,
        "end": ordered[-1].end_s if ordered else None,
    }


def _word_overlap_ratio(word: Word, segment_start: float, segment_end: float) -> float:
    overlap = min(word.end_s, segment_end) - max(word.start_s, segment_start)
    if overlap <= 0.0:
        return 0.0
    duration = max(word.end_s - word.start_s, 1e-9)
    return overlap / duration


def _is_cjk_text(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

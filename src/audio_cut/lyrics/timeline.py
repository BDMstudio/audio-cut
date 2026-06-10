#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/lyrics/timeline.py
# AI-SUMMARY: Merges chunk-local lyrics timelines into a validated global timeline.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from audio_cut.lyrics.chunker import LyricsChunk
from audio_cut.lyrics.models import LyricsTimeline, Sentence, VadRegion, Word


@dataclass
class _WordWithChunk:
    word: Word
    chunk: LyricsChunk


def merge_chunk_timelines(
    chunks: Sequence[LyricsChunk],
    timelines: Mapping[int, LyricsTimeline],
) -> LyricsTimeline:
    """Merge chunk-local timelines into global track seconds."""

    ordered_chunks = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    converted_words: List[_WordWithChunk] = []
    sentences: List[Sentence] = []
    vad_regions: List[VadRegion] = []

    for chunk in ordered_chunks:
        timeline = timelines.get(chunk.chunk_id)
        if timeline is None:
            continue
        for word in timeline.words:
            converted_words.append(
                _WordWithChunk(
                    word=Word(
                        word.text,
                        chunk.global_t0 + word.start_s,
                        chunk.global_t0 + word.end_s,
                        confidence=word.confidence,
                    ),
                    chunk=chunk,
                )
            )
        for sentence in timeline.sentences:
            sentences.append(
                Sentence(
                    sentence.text,
                    chunk.global_t0 + sentence.start_s,
                    chunk.global_t0 + sentence.end_s,
                    confidence=sentence.confidence,
                )
            )
        for region in timeline.vad_regions:
            vad_regions.append(
                VadRegion(
                    chunk.global_t0 + region.start_s,
                    chunk.global_t0 + region.end_s,
                    confidence=region.confidence,
                    kind=region.kind,
                )
            )

    words = _dedupe_overlap_words(converted_words)
    duration_s = max((chunk.global_t1 for chunk in ordered_chunks), default=None)
    forbidden = [chunk.global_t1 for chunk in ordered_chunks[:-1]]
    return LyricsTimeline(
        words=words,
        sentences=sentences,
        vad_regions=vad_regions,
        duration_s=duration_s,
        source="merged_chunks",
        meta={"forbidden_cut_times_s": forbidden},
    )


def _dedupe_overlap_words(words: Sequence[_WordWithChunk]) -> List[Word]:
    kept: List[_WordWithChunk] = []
    for candidate in sorted(words, key=lambda item: (item.word.start_s, item.word.end_s, item.word.text)):
        if kept and _is_duplicate_word(kept[-1].word, candidate.word):
            kept[-1] = _select_better_word(kept[-1], candidate)
            continue
        kept.append(candidate)
    return [item.word for item in kept]


def _is_duplicate_word(left: Word, right: Word) -> bool:
    if left.text != right.text:
        return False
    overlaps = min(left.end_s, right.end_s) > max(left.start_s, right.start_s)
    center_distance = abs(_center(left.start_s, left.end_s) - _center(right.start_s, right.end_s))
    return overlaps or center_distance <= 0.25


def _select_better_word(left: _WordWithChunk, right: _WordWithChunk) -> _WordWithChunk:
    left_conf = left.word.confidence
    right_conf = right.word.confidence
    if left_conf is not None or right_conf is not None:
        if right_conf is None:
            return left
        if left_conf is None:
            return right
        return right if right_conf > left_conf else left
    left_dist = _distance_to_chunk_center(left)
    right_dist = _distance_to_chunk_center(right)
    return right if right_dist < left_dist else left


def _distance_to_chunk_center(item: _WordWithChunk) -> float:
    word_center = _center(item.word.start_s, item.word.end_s)
    chunk_center = _center(item.chunk.global_t0, item.chunk.global_t1)
    return abs(word_center - chunk_center)


def _center(start_s: float, end_s: float) -> float:
    return (start_s + end_s) / 2.0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/lyrics/candidates.py
# AI-SUMMARY: Generates lyrics-derived soft boundary candidates for VPBD scoring.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate
from audio_cut.lyrics.models import LyricsTimeline, Sentence, VadRegion, Word

_PUNCTUATION_ENDINGS = tuple(".!?。！？")


@dataclass
class LyricsBoundaryCandidateGenerator:
    """Generate lyrics and mVAD candidate boundaries from a full-track timeline."""

    min_word_gap_s: float = 0.35
    max_word_gap_s: float = 1.5
    sentence_end_score: float = 0.75
    mvad_boundary_score: float = 0.45

    def generate(self, timeline: LyricsTimeline) -> List[CutCandidate]:
        candidates: List[CutCandidate] = []
        candidates.extend(self._word_gap_candidates(timeline.words))
        candidates.extend(self._sentence_end_candidates(timeline.sentences))
        candidates.extend(self._mvad_boundary_candidates(timeline.vad_regions))
        return sorted(candidates, key=lambda candidate: (candidate.t, candidate.source.value))

    def _word_gap_candidates(self, words: List[Word]) -> List[CutCandidate]:
        candidates: List[CutCandidate] = []
        for left, right in zip(words, words[1:]):
            gap_s = right.start_s - left.end_s
            if gap_s < self.min_word_gap_s:
                continue
            score = min(1.0, gap_s / max(self.max_word_gap_s, self.min_word_gap_s))
            candidates.append(
                CutCandidate(
                    t=(left.end_s + right.start_s) / 2.0,
                    score=score,
                    source=CandidateSource.LYRICS_GAP,
                    reasons=["word_gap"],
                    meta={"gap_s": gap_s, "left_word": left.text, "right_word": right.text},
                )
            )
        return candidates

    def _sentence_end_candidates(self, sentences: List[Sentence]) -> List[CutCandidate]:
        candidates: List[CutCandidate] = []
        for sentence in sentences:
            reasons = ["sentence_end"]
            score = self.sentence_end_score
            if sentence.text.strip().endswith(_PUNCTUATION_ENDINGS):
                reasons.append("punctuation_end")
                score = min(1.0, score + 0.1)
            if sentence.confidence is not None:
                score *= sentence.confidence
            candidates.append(
                CutCandidate(
                    t=sentence.end_s,
                    score=score,
                    source=CandidateSource.SENTENCE_END,
                    reasons=reasons,
                    meta={"text": sentence.text},
                )
            )
        return candidates

    def _mvad_boundary_candidates(self, regions: List[VadRegion]) -> List[CutCandidate]:
        candidates: List[CutCandidate] = []
        for region in regions:
            score = self.mvad_boundary_score
            if region.confidence is not None:
                score *= region.confidence
            for t, reason in ((region.start_s, "mvad_start"), (region.end_s, "mvad_end")):
                candidates.append(
                    CutCandidate(
                        t=t,
                        score=score,
                        source=CandidateSource.MVAD_BOUNDARY,
                        reasons=[reason],
                        meta={"kind": region.kind},
                    )
                )
        return candidates

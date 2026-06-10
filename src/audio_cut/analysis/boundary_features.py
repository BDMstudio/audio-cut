#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/analysis/boundary_features.py
# AI-SUMMARY: Typed VPBD boundary feature extraction used by candidate scoring.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from audio_cut.lyrics.models import LyricsTimeline


@dataclass
class BoundaryFeatures:
    """Normalized feature vector for phrase-boundary scoring."""

    acoustic_pause: float = 0.0
    asr_gap: float = 0.0
    sentence_end: float = 0.0
    inside_word_penalty: float = 0.0
    singing_penalty: float = 0.0
    beat_affinity: float = 0.0
    mdd_affinity: float = 0.0

    def __post_init__(self) -> None:
        for name in self.to_dict():
            setattr(self, name, _clamp01(float(getattr(self, name))))

    def to_dict(self) -> Dict[str, float]:
        return {
            "acoustic_pause": self.acoustic_pause,
            "asr_gap": self.asr_gap,
            "sentence_end": self.sentence_end,
            "inside_word_penalty": self.inside_word_penalty,
            "singing_penalty": self.singing_penalty,
            "beat_affinity": self.beat_affinity,
            "mdd_affinity": self.mdd_affinity,
        }


@dataclass
class BoundaryFeatureExtractor:
    """Extract normalized boundary features from lyrics, beat and MDD priors."""

    timeline: LyricsTimeline
    beat_times: Iterable[float] = field(default_factory=list)
    mdd_times: Iterable[float] = field(default_factory=list)
    high_confidence: float = 0.85
    word_gap_norm_s: float = 1.5
    sentence_tolerance_s: float = 0.08
    affinity_tolerance_s: float = 0.12

    def __post_init__(self) -> None:
        self.beat_times = list(self.beat_times)
        self.mdd_times = list(self.mdd_times)

    def extract(self, t: float, *, acoustic_pause: float = 0.0) -> BoundaryFeatures:
        """Extract a normalized feature vector at global time `t`."""

        return BoundaryFeatures(
            acoustic_pause=acoustic_pause,
            asr_gap=self._asr_gap_score(t),
            sentence_end=self._sentence_end_score(t),
            inside_word_penalty=self._inside_word_penalty(t),
            singing_penalty=self._singing_penalty(t),
            beat_affinity=self._affinity(t, self.beat_times),
            mdd_affinity=self._affinity(t, self.mdd_times),
        )

    def _inside_word_penalty(self, t: float) -> float:
        for word in self.timeline.words:
            if word.start_s < t < word.end_s:
                if word.confidence is None:
                    return 0.5
                return 1.0 if word.confidence >= self.high_confidence else 0.3
        return 0.0

    def _singing_penalty(self, t: float) -> float:
        for region in self.timeline.vad_regions:
            if region.kind == "singing" and region.start_s < t < region.end_s:
                if region.confidence is None:
                    return 0.5
                return 1.0 if region.confidence >= self.high_confidence else 0.3
        return 0.0

    def _asr_gap_score(self, t: float) -> float:
        for left, right in zip(self.timeline.words, self.timeline.words[1:]):
            if left.end_s <= t <= right.start_s:
                gap_s = max(0.0, right.start_s - left.end_s)
                return _clamp01(gap_s / max(self.word_gap_norm_s, 1e-6))
        return 0.0

    def _sentence_end_score(self, t: float) -> float:
        best = 0.0
        for sentence in self.timeline.sentences:
            distance = abs(t - sentence.end_s)
            if distance > self.sentence_tolerance_s:
                continue
            confidence = sentence.confidence if sentence.confidence is not None else 1.0
            proximity = 1.0 - (distance / max(self.sentence_tolerance_s, 1e-6))
            best = max(best, confidence * proximity)
        return _clamp01(best)

    def _affinity(self, t: float, anchors: Iterable[float]) -> float:
        best = 0.0
        for anchor in anchors:
            distance = abs(t - float(anchor))
            if distance > self.affinity_tolerance_s:
                continue
            score = 1.0 - (distance / max(self.affinity_tolerance_s, 1e-6))
            best = max(best, score)
        return _clamp01(best)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value

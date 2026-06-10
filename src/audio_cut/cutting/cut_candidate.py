#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/cutting/cut_candidate.py
# AI-SUMMARY: Candidate boundary data model shared by VPBD scoring and planning.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class CandidateSource(str, Enum):
    """Known sources for candidate cut boundaries."""

    ACOUSTIC_PAUSE = "acoustic_pause"
    MDD_VALLEY = "mdd_valley"
    BREATH = "breath"
    LYRICS_GAP = "lyrics_gap"
    SENTENCE_END = "sentence_end"
    MVAD_BOUNDARY = "mvad_boundary"
    BEAT = "beat"
    RESCUE = "rescue"


@dataclass
class CutCandidate:
    """Candidate cut boundary before guard snapping and layout refinement."""

    t: float
    score: float
    source: CandidateSource
    reasons: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = float(self.t)
        self.score = min(1.0, max(0.0, float(self.score)))
        if not isinstance(self.source, CandidateSource):
            self.source = CandidateSource(str(self.source))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "score": self.score,
            "source": self.source.value,
            "reasons": list(self.reasons),
            "features": dict(self.features),
            "meta": dict(self.meta),
        }

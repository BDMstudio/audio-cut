#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/cutting/candidate_adapters.py
# AI-SUMMARY: Adapts legacy acoustic and MDD boundary outputs into VPBD cut candidates.

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate
from audio_cut.cutting.refine import CutPoint


def adapt_legacy_acoustic_candidates(
    raw_candidates: Iterable[CutPoint | Tuple[float, float] | Tuple[float, float, Dict[str, Any]]],
    *,
    source: CandidateSource = CandidateSource.ACOUSTIC_PAUSE,
    breath_score_scale: float = 0.6,
) -> List[CutCandidate]:
    """Convert legacy `(time, score, meta)` style candidates to `CutCandidate`."""

    candidates: List[CutCandidate] = []
    for raw in raw_candidates:
        if isinstance(raw, CutPoint):
            t = raw.t
            score = raw.score
            meta: Dict[str, Any] = {"legacy_kind": raw.kind}
        else:
            t = float(raw[0])
            score = float(raw[1])
            meta = dict(raw[2]) if len(raw) > 2 and isinstance(raw[2], dict) else {}
        candidate_source = source
        if str(meta.get("pause_type", "")).startswith("breath"):
            if breath_score_scale <= 0.0:
                continue
            candidate_source = CandidateSource.BREATH
            score *= float(breath_score_scale)

        candidates.append(
            CutCandidate(
                t=t,
                score=score,
                source=candidate_source,
                reasons=["legacy_acoustic"],
                meta=meta,
            )
        )
    return candidates

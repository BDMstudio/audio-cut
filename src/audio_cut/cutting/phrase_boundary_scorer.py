#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/cutting/phrase_boundary_scorer.py
# AI-SUMMARY: Scores VPBD boundary candidates from normalized feature components.

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from audio_cut.analysis.boundary_features import BoundaryFeatures
from audio_cut.cutting.cut_candidate import CutCandidate

DEFAULT_BOUNDARY_WEIGHTS: Dict[str, float] = {
    "acoustic_pause": 0.35,
    "asr_gap": 0.20,
    "sentence_end": 0.15,
    "beat_affinity": 0.08,
    "mdd_affinity": 0.10,
    "breath": 0.12,
    "inside_word_penalty": 0.80,
    "singing_penalty": 0.50,
}
_PENALTY_KEYS = {"inside_word_penalty", "singing_penalty"}


class PhraseBoundaryScorer:
    """Weighted scorer for normalized phrase-boundary features."""

    def __init__(self, weights: Optional[Mapping[str, float]] = None) -> None:
        merged = dict(DEFAULT_BOUNDARY_WEIGHTS)
        if weights:
            merged.update({key: float(value) for key, value in weights.items()})
        self.weights = merged

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, object]] = None) -> "PhraseBoundaryScorer":
        if cfg is None:
            try:
                from vocal_smart_splitter.utils.config_manager import get_config

                cfg = {"weights": get_config("phrase_boundary.weights", DEFAULT_BOUNDARY_WEIGHTS)}
            except Exception:
                cfg = {"weights": DEFAULT_BOUNDARY_WEIGHTS}
        weights = cfg.get("weights", DEFAULT_BOUNDARY_WEIGHTS) if isinstance(cfg, Mapping) else DEFAULT_BOUNDARY_WEIGHTS
        return cls(weights=weights if isinstance(weights, Mapping) else DEFAULT_BOUNDARY_WEIGHTS)

    def score(self, features: BoundaryFeatures) -> float:
        feature_values = features.to_dict()
        total = 0.0
        for name, value in feature_values.items():
            weight = float(self.weights.get(name, 0.0))
            if name in _PENALTY_KEYS:
                total -= weight * value
            else:
                total += weight * value
        return _clamp01(total)

    def score_candidate(self, candidate: CutCandidate, features: BoundaryFeatures) -> CutCandidate:
        reasons = list(candidate.reasons)
        if "vpbd_score" not in reasons:
            reasons.append("vpbd_score")
        return replace(
            candidate,
            score=self.score(features),
            features=features.to_dict(),
            reasons=reasons,
        )


def write_candidate_debug_json(candidates: Iterable[CutCandidate], path: Path | str) -> None:
    """Write candidate debug payload used by later Manifest/debug tooling."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"candidates": [candidate.to_dict() for candidate in candidates]}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value

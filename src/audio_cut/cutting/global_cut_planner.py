#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/cutting/global_cut_planner.py
# AI-SUMMARY: Dynamic-programming global cut planner for VPBD candidate boundaries.

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from audio_cut.cutting.cut_candidate import CutCandidate
from audio_cut.cutting.refine import CutAdjustment, CutPoint


@dataclass(frozen=True)
class GlobalCutPlannerConfig:
    """Configuration for global phrase-boundary planning."""

    hard_min_s: float = 2.0
    hard_max_s: float = 18.0
    target_min_s: float = 5.0
    target_max_s: float = 12.0
    duration_penalty_weight: float = 0.15
    vocal_risk_weight: float = 0.25
    beat_conflict_weight: float = 0.15
    max_candidates_per_second: float = 2.0
    rescue_enabled: bool = True


@dataclass(frozen=True)
class GlobalCutPlanResult:
    """Planner result and trace metadata."""

    cut_times: List[float]
    selected_candidates: List[CutCandidate]
    suppressed_candidates: List[CutCandidate] = field(default_factory=list)
    rescue_points: List[float] = field(default_factory=list)
    feasible: bool = True
    metadata: Dict[str, object] = field(default_factory=dict)


class GlobalCutPlanner:
    """Selects a globally feasible cut path before local guard refinement."""

    def __init__(self, config: Optional[GlobalCutPlannerConfig] = None) -> None:
        self.config = config or GlobalCutPlannerConfig()

    def plan(self, candidates: Sequence[CutCandidate], *, duration_s: float) -> GlobalCutPlanResult:
        duration_s = float(duration_s)
        if duration_s <= 0.0:
            return GlobalCutPlanResult(
                cut_times=[0.0],
                selected_candidates=[],
                feasible=True,
                metadata={"planner": "empty", "selected_count": 0, "suppressed_count": 0},
            )

        pruned, suppressed = self._prune_candidates(candidates, duration_s)
        planned = self._plan_dynamic(pruned, duration_s)
        if planned is None:
            if not self.config.rescue_enabled:
                return GlobalCutPlanResult(
                    cut_times=[0.0, duration_s],
                    selected_candidates=[],
                    suppressed_candidates=list(candidates),
                    feasible=False,
                    metadata={"planner": "dynamic_programming", "selected_count": 0, "suppressed_count": len(candidates)},
                )
            return self._rescue(duration_s, suppressed_candidates=list(candidates))

        selected, cut_times = planned
        selected_ids = {id(candidate) for candidate in selected}
        suppressed.extend(candidate for candidate in pruned if id(candidate) not in selected_ids)
        return GlobalCutPlanResult(
            cut_times=cut_times,
            selected_candidates=selected,
            suppressed_candidates=sorted(suppressed, key=lambda item: (item.t, item.score)),
            feasible=True,
            metadata={
                "planner": "dynamic_programming",
                "selected_count": len(selected),
                "suppressed_count": len(suppressed),
            },
        )

    def _prune_candidates(
        self,
        candidates: Sequence[CutCandidate],
        duration_s: float,
    ) -> Tuple[List[CutCandidate], List[CutCandidate]]:
        max_per_second = max(1, int(math.floor(self.config.max_candidates_per_second)))
        buckets: Dict[int, List[CutCandidate]] = {}
        suppressed: List[CutCandidate] = []
        for candidate in candidates:
            if candidate.t <= 0.0 or candidate.t >= duration_s:
                suppressed.append(candidate)
                continue
            bucket = int(math.floor(candidate.t))
            buckets.setdefault(bucket, []).append(candidate)

        kept: List[CutCandidate] = []
        for bucket_candidates in buckets.values():
            ordered = sorted(bucket_candidates, key=self._candidate_value, reverse=True)
            kept.extend(ordered[:max_per_second])
            suppressed.extend(ordered[max_per_second:])
        return sorted(kept, key=lambda item: item.t), suppressed

    def _plan_dynamic(
        self,
        candidates: Sequence[CutCandidate],
        duration_s: float,
    ) -> Optional[Tuple[List[CutCandidate], List[float]]]:
        nodes: List[Optional[CutCandidate]] = [None] + list(candidates) + [None]
        times = [0.0] + [candidate.t for candidate in candidates] + [duration_s]
        n = len(times)
        scores = [-math.inf] * n
        parents = [-1] * n
        scores[0] = 0.0

        for i in range(1, n):
            for j in range(0, i):
                segment_s = times[i] - times[j]
                if not self._segment_allowed(segment_s, duration_s):
                    continue
                node_score = self._candidate_value(nodes[i]) if nodes[i] is not None else 0.0
                total = scores[j] + node_score + self._duration_score(segment_s)
                if total > scores[i]:
                    scores[i] = total
                    parents[i] = j

        if parents[-1] < 0:
            return None

        selected: List[CutCandidate] = []
        path_times: List[float] = []
        index = n - 1
        while index >= 0:
            path_times.append(times[index])
            node = nodes[index]
            if node is not None:
                selected.append(node)
            index = parents[index]
            if index < 0 and path_times[-1] != 0.0:
                return None
        selected.reverse()
        path_times.reverse()
        return selected, path_times

    def _segment_allowed(self, segment_s: float, duration_s: float) -> bool:
        if duration_s <= self.config.hard_min_s:
            return True
        return self.config.hard_min_s <= segment_s <= self.config.hard_max_s

    def _duration_score(self, segment_s: float) -> float:
        if self.config.target_min_s <= segment_s <= self.config.target_max_s:
            return 0.1
        if segment_s < self.config.target_min_s:
            distance = self.config.target_min_s - segment_s
        else:
            distance = segment_s - self.config.target_max_s
        return -self.config.duration_penalty_weight * distance / max(self.config.target_max_s, 1e-6)

    def _candidate_value(self, candidate: Optional[CutCandidate]) -> float:
        if candidate is None:
            return 0.0
        vocal_risk = float(candidate.features.get("vocal_cut_risk", candidate.meta.get("vocal_cut_risk", 0.0)))
        beat_conflict = float(candidate.features.get("beat_conflict", candidate.meta.get("beat_conflict", 0.0)))
        return (
            candidate.score
            - self.config.vocal_risk_weight * _clamp01(vocal_risk)
            - self.config.beat_conflict_weight * _clamp01(beat_conflict)
        )

    def _rescue(self, duration_s: float, suppressed_candidates: List[CutCandidate]) -> GlobalCutPlanResult:
        segment_count = max(1, int(math.ceil(duration_s / max(self.config.hard_max_s, 1e-6))))
        step_s = duration_s / float(segment_count)
        if step_s < self.config.hard_min_s and segment_count > 1:
            segment_count = max(1, int(math.floor(duration_s / max(self.config.hard_min_s, 1e-6))))
            step_s = duration_s / float(segment_count)
        cut_times = [round(i * step_s, 9) for i in range(segment_count + 1)]
        cut_times[-1] = duration_s
        rescue_points = cut_times[1:-1]
        return GlobalCutPlanResult(
            cut_times=cut_times,
            selected_candidates=[],
            suppressed_candidates=suppressed_candidates,
            rescue_points=rescue_points,
            feasible=True,
            metadata={
                "planner": "rescue",
                "selected_count": 0,
                "suppressed_count": len(suppressed_candidates),
            },
        )


def planner_result_to_cut_points(result: GlobalCutPlanResult) -> List[CutPoint]:
    """Convert selected planner candidates into refine-compatible cut points."""

    return [
        CutPoint(t=candidate.t, score=candidate.score, kind=candidate.source.value)
        for candidate in result.selected_candidates
    ]


def apply_guard_shift_metadata(
    result: GlobalCutPlanResult,
    adjustments: Sequence[CutAdjustment],
) -> GlobalCutPlanResult:
    """Return a planner result whose metadata tracks guard shifts by raw time."""

    guard_shift_ms_by_raw_time = {
        adjustment.raw_time: adjustment.guard_shift_ms
        for adjustment in adjustments
    }
    final_time_by_raw_time = {
        adjustment.raw_time: adjustment.final_time
        for adjustment in adjustments
    }
    metadata = dict(result.metadata)
    metadata["guard_shift_ms_by_raw_time"] = guard_shift_ms_by_raw_time
    metadata["final_time_by_raw_time"] = final_time_by_raw_time
    return replace(result, metadata=metadata)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value

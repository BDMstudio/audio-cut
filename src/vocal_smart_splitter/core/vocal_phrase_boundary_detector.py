#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/vocal_phrase_boundary_detector.py
# AI-SUMMARY: VPBD boundary detector that fuses acoustic pause candidates with optional lyrics priors.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf

from audio_cut.analysis.boundary_features import BoundaryFeatureExtractor
from audio_cut.cutting.candidate_adapters import adapt_legacy_acoustic_candidates
from audio_cut.cutting.cut_candidate import CandidateSource, CutCandidate
from audio_cut.cutting.global_cut_planner import (
    GlobalCutPlanResult,
    GlobalCutPlanner,
    GlobalCutPlannerConfig,
)
from audio_cut.cutting.phrase_boundary_scorer import PhraseBoundaryScorer
from audio_cut.exceptions import LyricsAlignmentUnavailable
from audio_cut.lyrics.candidates import LyricsBoundaryCandidateGenerator
from audio_cut.lyrics.models import LyricsTimeline
from audio_cut.lyrics.providers import LyricsProviderRequest, build_lyrics_provider
from ..utils.config_manager import get_config


@dataclass
class VPBDDetectionResult:
    """VPBD detector output consumed by SeamlessSplitter."""

    selected_candidates: List[CutCandidate]
    planner_result: GlobalCutPlanResult
    boundary_detection: Dict[str, Any]
    lyrics_alignment: Dict[str, Any]


class VocalPhraseBoundaryDetector:
    """Fuse acoustic and optional lyrics candidates before global cut planning."""

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate

    def detect(
        self,
        *,
        mode: str,
        vocal_track: np.ndarray,
        original_audio: np.ndarray,
        pure_vocal_detector: Any,
        feature_cache: Optional[Any],
        vad_segments: Optional[List[Dict[str, float]]],
        input_path: str,
        output_dir: str,
    ) -> VPBDDetectionResult:
        duration_s = len(original_audio) / float(self.sample_rate) if self.sample_rate > 0 else 0.0
        actual_mode = mode
        fallback_reason: Optional[str] = None
        timeline = LyricsTimeline(duration_s=duration_s, source="none")
        lyrics_candidates: List[CutCandidate] = []

        lyrics_cfg = _config_section("lyrics_alignment")
        strict = bool(lyrics_cfg.get("strict", False))
        lyrics_enabled = bool(lyrics_cfg.get("enabled", False)) and mode == "vpbd_asr"
        provider_name = str(lyrics_cfg.get("provider", "disabled"))

        if mode == "vpbd_asr":
            if not lyrics_enabled:
                actual_mode = "vpbd_acoustic"
                fallback_reason = "lyrics_alignment_disabled"
            else:
                provider_cfg = dict(lyrics_cfg)
                provider_cfg["fire_red"] = _config_section("fire_red")
                provider = build_lyrics_provider(provider_cfg)
                provider_name = provider.name
                try:
                    if provider.name == "null":
                        raise LyricsAlignmentUnavailable(
                            str(getattr(provider, "reason", "lyrics alignment unavailable"))
                        )
                    asr_vocal_path = _write_asr_vocal_copy(
                        vocal_track=vocal_track,
                        output_dir=output_dir,
                        input_path=input_path,
                        source_sample_rate=self.sample_rate,
                    )
                    timeline = provider.align(
                        LyricsProviderRequest(
                            vocal_path=asr_vocal_path,
                            duration_s=duration_s,
                            sample_rate=16000,
                            strict=strict,
                        )
                    )
                    lyrics_candidates = LyricsBoundaryCandidateGenerator().generate(timeline)
                except LyricsAlignmentUnavailable:
                    if strict:
                        raise
                    actual_mode = "vpbd_acoustic"
                    fallback_reason = "lyrics_alignment_unavailable"
                except Exception as exc:
                    if strict:
                        raise
                    actual_mode = "vpbd_acoustic"
                    fallback_reason = str(exc)

        acoustic_candidates = self._build_acoustic_candidates(
            vocal_track=vocal_track,
            original_audio=original_audio,
            pure_vocal_detector=pure_vocal_detector,
            feature_cache=feature_cache,
            vad_segments=vad_segments,
            enable_mdd=True,
        )
        scored_candidates = self._score_candidates(
            candidates=acoustic_candidates,
            timeline=timeline,
            feature_cache=feature_cache,
        )
        planner = GlobalCutPlanner(_planner_config())
        planner_result = planner.plan(scored_candidates, duration_s=duration_s)

        lyrics_meta = {
            "enabled": lyrics_enabled,
            "provider": provider_name,
            "strict": strict,
            "fallback_reason": fallback_reason,
            "word_count": len(timeline.words),
            "sentence_count": len(timeline.sentences),
            "vad_region_count": len(timeline.vad_regions),
            "warnings": list(timeline.warnings),
            "timeline": timeline.to_dict(),
        }
        boundary_meta = {
            "mode": mode,
            "actual_mode": actual_mode,
            "candidate_counts": {
                "acoustic": len(acoustic_candidates),
                "lyrics": len(lyrics_candidates),
                "total": len(scored_candidates),
                "selected": len(planner_result.selected_candidates),
                "suppressed": len(planner_result.suppressed_candidates),
                "lyrics_soft_prior": len(lyrics_candidates),
            },
            "planner": dict(planner_result.metadata),
            "selected": [candidate.to_dict() for candidate in planner_result.selected_candidates],
            "suppressed": [candidate.to_dict() for candidate in planner_result.suppressed_candidates],
        }
        return VPBDDetectionResult(
            selected_candidates=list(planner_result.selected_candidates),
            planner_result=planner_result,
            boundary_detection=boundary_meta,
            lyrics_alignment=lyrics_meta,
        )

    def _build_acoustic_candidates(
        self,
        *,
        vocal_track: np.ndarray,
        original_audio: np.ndarray,
        pure_vocal_detector: Any,
        feature_cache: Optional[Any],
        vad_segments: Optional[List[Dict[str, float]]],
        enable_mdd: bool,
    ) -> List[CutCandidate]:
        pauses = pure_vocal_detector.detect_pure_vocal_pauses(
            vocal_track,
            enable_mdd_enhancement=enable_mdd,
            original_audio=original_audio,
            feature_cache=feature_cache,
            vad_segments=vad_segments,
        )
        raw = []
        for pause in pauses or []:
            t = float(getattr(pause, "cut_point", (pause.start_time + pause.end_time) / 2.0))
            score = float(getattr(pause, "confidence", 1.0))
            raw.append(
                (
                    t,
                    score,
                    {
                        "pause_start_s": float(getattr(pause, "start_time", t)),
                        "pause_end_s": float(getattr(pause, "end_time", t)),
                        "pause_duration_s": float(getattr(pause, "duration", 0.0)),
                    },
                )
            )
        return adapt_legacy_acoustic_candidates(raw, source=CandidateSource.ACOUSTIC_PAUSE)

    def _score_candidates(
        self,
        *,
        candidates: List[CutCandidate],
        timeline: LyricsTimeline,
        feature_cache: Optional[Any],
    ) -> List[CutCandidate]:
        beat_times = getattr(feature_cache, "beat_times", []) if feature_cache is not None else []
        extractor = BoundaryFeatureExtractor(timeline=timeline, beat_times=beat_times)
        scorer = PhraseBoundaryScorer.from_config(_config_section("phrase_boundary"))
        scored: List[CutCandidate] = []
        for candidate in candidates:
            acoustic_pause = candidate.score if candidate.source == CandidateSource.ACOUSTIC_PAUSE else 0.0
            features = extractor.extract(candidate.t, acoustic_pause=acoustic_pause)
            scored.append(scorer.score_candidate(candidate, features))
        return scored


def _write_asr_vocal_copy(
    *,
    vocal_track: np.ndarray,
    output_dir: str,
    input_path: str,
    source_sample_rate: int,
) -> Path:
    """Write the separated vocal track as a FireRed-safe ASR WAV copy."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{Path(input_path).stem}_vocal_for_asr.wav"

    audio = np.asarray(vocal_track, dtype=np.float32)
    if audio.ndim == 2:
        axis = 0 if audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0] else 1
        audio = np.mean(audio, axis=axis, dtype=np.float32)
    audio = np.ravel(audio).astype(np.float32, copy=False)

    if source_sample_rate != 16000 and audio.size:
        audio = librosa.resample(y=audio, orig_sr=int(source_sample_rate), target_sr=16000)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(target_path, audio, 16000, subtype="PCM_16")
    return target_path


def _config_section(name: str) -> Dict[str, Any]:
    try:
        value = get_config(name, {})
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _planner_config() -> GlobalCutPlannerConfig:
    cfg = _config_section("global_planner")
    return GlobalCutPlannerConfig(
        hard_min_s=float(cfg.get("hard_min_s", 2.0)),
        hard_max_s=float(cfg.get("hard_max_s", 18.0)),
        target_min_s=float(cfg.get("target_min_s", 5.0)),
        target_max_s=float(cfg.get("target_max_s", 12.0)),
        duration_penalty_weight=float(cfg.get("duration_penalty_weight", 0.15)),
        vocal_risk_weight=float(cfg.get("vocal_risk_weight", 0.25)),
        beat_conflict_weight=float(cfg.get("beat_conflict_weight", 0.15)),
        max_candidates_per_second=float(cfg.get("max_candidates_per_second", 2.0)),
        rescue_enabled=bool(cfg.get("rescue_enabled", True)),
    )

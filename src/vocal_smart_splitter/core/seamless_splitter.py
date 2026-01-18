#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/seamless_splitter.py
# AI-SUMMARY: Orchestrates seamless splitting, covering separation, pause detection, cut refinement, and feature cache reuse.

import os
import numpy as np
import librosa
import logging
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple, Any, Sequence, Iterable, Set
from pathlib import Path
import time
import tempfile

from audio_cut.analysis import TrackFeatureCache, build_feature_cache
from audio_cut.utils.gpu_pipeline import PipelineConfig, PipelineContext, build_pipeline_context
from audio_cut.cutting.refine import (
    CutAdjustment,
    CutContext,
    CutPoint,
    CutRefineResult,
    finalize_cut_points,
)
from audio_cut.cutting.segment_layout_refiner import (
    Segment as SegmentLayoutItem,
    derive_layout_config,
    refine_layout,
)

from ..utils.config_manager import get_config, get_librosa_onset_config, get_hybrid_mdd_config
from ..utils.audio_export import (
    build_export_options,
    ensure_supported_format,
    export_audio,
)
from ..utils.audio_processor import AudioProcessor
from .pure_vocal_pause_detector import PureVocalPauseDetector
from .quality_controller import QualityController
from .enhanced_vocal_separator import EnhancedVocalSeparator

logger = logging.getLogger(__name__)

PRECISION_GUARD_AVG_MS = 150.0
PRECISION_GUARD_P95_MS = 220.0

class SeamlessSplitter:
    """Unified seam-splitting orchestrator (v2.3). Handles every split mode through a single entry point."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate)
        self.pure_vocal_detector = PureVocalPauseDetector(sample_rate)  # default to v2.2 configuration
        self.quality_controller = QualityController(sample_rate)
        self.separator = EnhancedVocalSeparator(sample_rate)
        self._last_segment_classification_debug: List[Dict[str, Any]] = []
        self._last_guard_shift_stats: Dict[str, float] = self._blank_guard_stats()
        self._last_guard_adjustments: List[Dict[str, float]] = []
        self._last_guard_adjustments_raw: List[CutAdjustment] = []
        self._last_suppressed_cut_points: List[CutPoint] = []
        self._precision_guard_ok: bool = True
        default_export_format = ensure_supported_format(get_config('output.format', 'wav'))
        self._export_format = default_export_format
        self._export_options = build_export_options(
            default_export_format,
            get_config(f'output.{default_export_format}', {})
        )
        logger.info("SeamlessSplitter orchestrator initialized (sample_rate=%d) - PureVocalPauseDetector ready", self.sample_rate)

    @staticmethod
    def _normalize_export_plan(
        plan: Optional[Sequence[str]],
        *,
        default: Iterable[str],
    ) -> Set[str]:
        """Normalize export plan tokens to internal flags."""

        alias_map = {
            'vocal': {'full_vocal'},
            'full_vocal': {'full_vocal'},
            'vocal_full': {'full_vocal'},
            'instrumental': {'full_instrumental'},
            'full_instrumental': {'full_instrumental'},
            'instrumental_full': {'full_instrumental'},
            'human_segments': {'vocal_segments'},
            'vocal_segments': {'vocal_segments'},
            'human': {'vocal_segments'},
            'music_segments': {'mix_segments'},
            'mix_segments': {'mix_segments'},
            'music': {'mix_segments'},
        }

        if plan is None:
            return {item for item in default}

        normalized: Set[str] = set()
        for raw in plan:
            if raw is None:
                continue
            token = str(raw).strip().lower()
            if not token:
                continue
            # Special token to skip all exports
            if token == 'none':
                return set()
            if token == 'all':
                normalized.update(default)
                continue
            mapped = alias_map.get(token)
            if mapped:
                normalized.update(mapped)
                continue
            normalized.add(token)

        if not normalized:
            normalized.update(default)
        return normalized

    def split_audio_seamlessly(
        self,
        input_path: str,
        output_dir: str,
        mode: str = 'v2.2_mdd',
        export_format: Optional[str] = None,
        export_options: Optional[Dict[str, Any]] = None,
        export_plan: Optional[Sequence[str]] = None,
    ) -> Dict:
        """Run the seamless splitting pipeline. Supports pure vocal separation and v2.2 MDD mode."""
        logger.info("Start seamless split: %s (mode: %s)", input_path, mode)

        previous_format = self._export_format
        previous_options = self._export_options
        requested_format = export_format or get_config('output.format', previous_format)

        try:
            try:
                effective_format = ensure_supported_format(requested_format)
            except ValueError:
                logger.warning(
                    '[EXPORT] Unsupported export format %s, fallback to %s.',
                    requested_format,
                    previous_format,
                )
                effective_format = previous_format

            config_overrides = get_config(f'output.{effective_format}', {})
            if not isinstance(config_overrides, dict):
                config_overrides = {}

            merged_options = build_export_options(
                effective_format,
                config_overrides,
                export_options,
            )
            self._export_format = effective_format
            self._export_options = merged_options
            logger.info(f"[EXPORT] active format: {self._export_format}")

            try:
                if mode == 'vocal_separation':
                    return self._process_vocal_separation_only(
                        input_path,
                        output_dir,
                        export_plan=export_plan,
                    )
                if mode == 'librosa_onset':
                    return self._process_librosa_onset_split(
                        input_path,
                        output_dir,
                        export_plan=export_plan,
                    )
                if mode == 'hybrid_mdd':
                    return self._process_hybrid_mdd_split(
                        input_path,
                        output_dir,
                        export_plan=export_plan,
                    )
                if mode != 'v2.2_mdd':
                    logger.warning(f"Unknown mode {mode}, default to v2.2_mdd")
                return self._process_pure_vocal_split(
                    input_path,
                    output_dir,
                    'v2.2_mdd',
                    export_plan=export_plan,
                )

            except Exception as e:
                logger.error(f"split failed: {e}", exc_info=True)
                return {'success': False, 'error': str(e), 'input_file': input_path}
        finally:
            self._export_format = previous_format
            self._export_options = previous_options
    def _load_and_resample_if_needed(self, input_path: str):
        """Load audio and resample when necessary."""
        original_audio, sr = self.audio_processor.load_audio(input_path, normalize=False)
        if sr != self.sample_rate:
            logger.info("Resample from %d Hz to target %d Hz", sr, self.sample_rate)
            original_audio = librosa.resample(original_audio, orig_sr=sr, target_sr=self.sample_rate)
        return original_audio

    def _process_pure_vocal_split(
        self,
        input_path: str,
        output_dir: str,
        mode: str,
        *,
        export_plan: Optional[Sequence[str]] = None,
    ) -> Dict:
        """
        Core flow for v2.2 MDD mode (v2.9 adjustments)
        关键修复: 调用新的双音频输入接口，确保音乐分析在 original_audio 上，而VAD检测在 vocal_track 上。
        """
        logger.info("[%s] Running pure vocal split pipeline...", mode.upper())
        overall_start_time = time.time()

        # 1. Load audio
        original_audio = self._load_and_resample_if_needed(input_path)

        gpu_cfg = self._get_gpu_pipeline_config()
        gpu_context: Optional[PipelineContext] = None
        if gpu_cfg.enable:
            try:
                gpu_context = self._build_gpu_pipeline_context(original_audio, gpu_cfg)
            except Exception as exc:
                gpu_context = None
                logger.warning("[GPU Pipeline] initialization failed, falling back to CPU: %s", exc, exc_info=True)

        if gpu_context and gpu_context.enabled:
            logger.info(
                "[GPU Pipeline] enabled: device=%s, chunk_count=%d",
                gpu_context.device,
                len(gpu_context.plans),
            )
        else:
            logger.info("[GPU Pipeline] disabled; continue with CPU path.")

        # 2. Perform vocal separation
        logger.info("[%s-STEP1] Performing vocal separation...", mode.upper())
        separation_start = time.time()
        separation_result = self.separator.separate_for_detection(
            original_audio,
            gpu_context=gpu_context,
        )
        separation_time = time.time() - separation_start

        if separation_result.vocal_track is None:
            gpu_meta = dict(separation_result.gpu_meta or {})
            failure = {
                'success': False,
                'error': 'vocal_track_missing',
                'input_file': input_path,
            }
            failure.update(gpu_meta)
            return failure

        vocal_track = separation_result.vocal_track
        logger.info("[%s-STEP1] Separation completed - backend=%s, confidence=%.3f, time=%.1fs", mode.upper(), separation_result.backend_used, separation_result.separation_confidence, separation_time)

        gpu_meta = dict(separation_result.gpu_meta or {})
        if gpu_meta.get('gpu_pipeline_used'):
            logger.info(
                "[GPU Pipeline] used device=%s chunks=%s",
                gpu_meta.get('gpu_pipeline_device'),
                gpu_meta.get('gpu_pipeline_chunks'),
            )
        elif gpu_meta.get('gpu_pipeline_enabled'):
            logger.info(
                "[GPU Pipeline] not used, reason=%s",
                gpu_meta.get('fallback_reason', gpu_meta.get('gpu_pipeline_failures', 'n/a')),
            )

        # 3. Detect pauses with PureVocalPauseDetector
        logger.info("[%s-STEP2] Running PureVocalPauseDetector on the vocal track...", mode.upper())

        # Determine whether we are in v2.2 MDD mode
        enable_mdd = (mode == 'v2.2_mdd')
        feature_cache: Optional[TrackFeatureCache] = separation_result.feature_cache
        if feature_cache is None:
            try:
                feature_cache = build_feature_cache(original_audio, vocal_track, self.sample_rate)
                logger.debug(
                    "Feature cache built (fallback): frames=%d, global_mdd=%.4f",
                    feature_cache.frame_count(),
                    feature_cache.global_mdd,
                )
            except Exception as exc:
                logger.warning("Failed to build feature cache: %s", exc)
                feature_cache = None

        vocal_pauses = self.pure_vocal_detector.detect_pure_vocal_pauses(
            vocal_track,
            enable_mdd_enhancement=enable_mdd,
            original_audio=original_audio,
            feature_cache=feature_cache,
            vad_segments=separation_result.vad_segments,
        )

        if not vocal_pauses:
            has_vocal = self._estimate_vocal_presence(vocal_track)
            return self._create_single_segment_result(
                original_audio,
                input_path,
                output_dir,
                "no_pause_candidates",
                is_vocal=has_vocal,
                gpu_meta=gpu_meta,
            )

        cut_points_samples: List[int] = []
        for pause in vocal_pauses:
            if hasattr(pause, 'cut_point'):
                cut_points_samples.append(int(pause.cut_point * self.sample_rate))
            else:
                center_time = (pause.start_time + pause.end_time) / 2
                cut_points_samples.append(int(center_time * self.sample_rate))

        logger.info("[%s-STEP3] Produced %d candidate split points", mode.upper(), len(cut_points_samples))
        # 将样本点转换为(time, score)形式，暂以score=confidence占位（若有）
        cut_candidates = []
        markers = getattr(separation_result, 'quality_metrics', {}) or {}
        marker_times = []
        try:
            marker_times = [float(t) for t in markers.get('vocal_presence_cut_points_sec', []) if t is not None]
        except Exception:
            marker_times = []
        protected_marker_samples = set()
        for p in vocal_pauses:
            t = float(getattr(p, 'cut_point', (p.start_time + p.end_time)/2))
            s = float(getattr(p, 'confidence', 1.0))
            # Filter long pauses that resemble interludes.
            dur = float(getattr(p, 'duration', (p.end_time - p.start_time)))
            interlude_min_s = get_config('pure_vocal_detection.pause_stats_adaptation.interlude_min_s', 4.0)
            if False and dur >= interlude_min_s:
                continue
            cut_candidates.append((t, s))


        # Pure music (no vocal) intervals -> inject split boundaries when exceeding the threshold.
        try:
            min_pure_music = float(get_config('quality_control.pure_music_min_duration', 0.0))
        except Exception:
            min_pure_music = 0.0
        if min_pure_music > 0.0:
            spans = self._find_no_vocal_runs(vocal_track, min_pure_music)
            if spans:
                logger.info("[%s-STEP3] Found %d pure music spans >= %.2fs", mode.upper(), len(spans), min_pure_music)
            else:
                logger.info("[%s-STEP3] No pure music spans >= %.2fs detected; skipping", mode.upper(), min_pure_music)
            for a, b in spans:
                cut_candidates.append((float(a), 1.0))
                cut_candidates.append((float(b), 1.0))
        else:
            logger.info("[%s-STEP3] Pure music detection disabled (quality_control.pure_music_min_duration <= 0)", mode.upper())
        audio_duration = len(original_audio) / self.sample_rate
        for t in marker_times:
            if t <= 0.0 or t >= audio_duration:
                continue
            cut_candidates.append((t, 1.0))
            protected_marker_samples.add(int(round(t * self.sample_rate)))

        refine_result = self._finalize_and_filter_cuts_v2(
            cut_candidates,
            original_audio,
            pure_vocal_audio=vocal_track
        )
        self._last_suppressed_cut_points = list(refine_result.suppressed_points or [])
        final_cut_points = sorted(set(refine_result.sample_boundaries))

        if protected_marker_samples:
            total_samples = len(original_audio)
            augmented = set(int(point) for point in final_cut_points)
            for sample_idx in protected_marker_samples:
                sample_idx = int(min(max(sample_idx, 0), total_samples))
                if sample_idx not in (0, total_samples):
                    augmented.add(sample_idx)
            final_cut_points = sorted(augmented)

        instrumental_audio = separation_result.instrumental_track
        if isinstance(instrumental_audio, np.ndarray):
            if len(instrumental_audio) != len(vocal_track):
                min_len = min(len(instrumental_audio), len(vocal_track))
                if min_len > 0:
                    instrumental_audio = instrumental_audio[:min_len]
                    if len(vocal_track) > min_len:
                        instrumental_audio = np.pad(instrumental_audio, (0, len(vocal_track) - min_len))
                else:
                    instrumental_audio = None
        else:
            instrumental_audio = None
        segment_vocal_flags = self._classify_segments_vocal_presence(
            vocal_track,
            final_cut_points,
            marker_segments=markers.get('vocal_presence_segments'),
            pure_music_segments=markers.get('pure_music_segments'),
            instrumental_audio=instrumental_audio,
            original_audio=original_audio
        )
        classification_debug = list(getattr(self, '_last_segment_classification_debug', []))

        layout_cfg_raw = get_config('segment_layout', {}) or {}
        if isinstance(layout_cfg_raw, dict):
            layout_cfg_raw = dict(layout_cfg_raw)
        else:
            layout_cfg_raw = {}
        fallback_micro = get_config('quality_control.segment_min_mix_piece', None)
        if fallback_micro is not None:
            layout_cfg_raw.setdefault('micro_merge_s', float(fallback_micro))
            layout_cfg_raw.setdefault('enable', bool(float(fallback_micro) > 0.0))
        fallback_max = get_config('quality_control.segment_max_duration', None)
        if fallback_max is not None:
            layout_cfg_raw.setdefault('soft_max_s', float(fallback_max))
        layout_cfg_raw.setdefault('min_gap_s', float(get_config('quality_control.min_split_gap', 1.0)))
        layout_cfg_raw.setdefault('beat_snap_ms', float(get_config('segment_layout.beat_snap_ms', 0.0) or 0.0))

        layout_config = derive_layout_config(
            layout_cfg_raw,
            feature_cache,
            sample_rate=self.sample_rate,
        )
        layout_applied = False
        if layout_config.enable and len(final_cut_points) >= 2:
            boundaries_sec = [sample / float(self.sample_rate) for sample in final_cut_points]
            layout_segments = [
                SegmentLayoutItem(
                    start=boundaries_sec[i],
                    end=boundaries_sec[i + 1],
                    kind='human' if segment_vocal_flags[i] else 'music',
                )
                for i in range(len(boundaries_sec) - 1)
            ]
            layout_result = refine_layout(
                layout_segments,
                self._last_guard_adjustments_raw,
                config=layout_config,
                sample_rate=self.sample_rate,
                suppressed_cut_points=self._last_suppressed_cut_points,
                features=feature_cache,
            )
            if layout_result.segments:
                layout_boundaries_sec = [layout_result.segments[0].start]
                layout_boundaries_sec.extend(seg.end for seg in layout_result.segments)
                sample_count = len(original_audio)
                updated_cut_points: List[int] = []
                for t in layout_boundaries_sec:
                    sample_idx = int(round(t * self.sample_rate))
                    sample_idx = max(0, min(sample_idx, sample_count))
                    updated_cut_points.append(sample_idx)
                if updated_cut_points:
                    updated_cut_points[0] = 0
                    updated_cut_points[-1] = sample_count
                updated_cut_points = sorted(set(updated_cut_points))
                if updated_cut_points != final_cut_points:
                    layout_applied = True
                final_cut_points = updated_cut_points if updated_cut_points else final_cut_points
                self._set_guard_adjustments(layout_result.adjustments)
                self._last_suppressed_cut_points = list(layout_result.suppressed_points or [])
                segment_vocal_flags = self._classify_segments_vocal_presence(
                    vocal_track,
                    final_cut_points,
                    marker_segments=markers.get('vocal_presence_segments'),
                    pure_music_segments=markers.get('pure_music_segments'),
                    instrumental_audio=instrumental_audio,
                    original_audio=original_audio
                )
                classification_debug = list(getattr(self, '_last_segment_classification_debug', []))
        else:
            layout_applied = False

        # Optional local valley refinement (used to fine-tune boundaries when guards are disabled).
        local_refine_cfg = get_config('quality_control.local_boundary_refine', {}) or {}
        if local_refine_cfg.get('enable') and len(final_cut_points) >= 2:
            refined_points = self._refine_boundaries_local_valley(
                final_cut_points,
                vocal_track,
                local_refine_cfg,
                min_gap_s=float(get_config('quality_control.min_split_gap', 1.0)),
            )
            if refined_points != final_cut_points:
                final_cut_points = refined_points
                layout_applied = True
                segment_vocal_flags = self._classify_segments_vocal_presence(
                    vocal_track,
                    final_cut_points,
                    marker_segments=markers.get('vocal_presence_segments'),
                    pure_music_segments=markers.get('pure_music_segments'),
                    instrumental_audio=instrumental_audio,
                    original_audio=original_audio
                )
                classification_debug = list(getattr(self, '_last_segment_classification_debug', []))

        segments, segment_vocal_flags, merged_debug = self._split_at_sample_level(
            original_audio,
            final_cut_points,
            segment_flags=segment_vocal_flags,
            debug_entries=classification_debug,
        )
        segment_durations = [len(seg) / float(self.sample_rate) for seg in segments]
        segment_durations_map = {i: duration for i, duration in enumerate(segment_durations)}
        if merged_debug is not None:
            classification_debug = merged_debug
            self._last_segment_classification_debug = merged_debug
        else:
            self._last_segment_classification_debug = classification_debug

        export_flags = self._normalize_export_plan(
            export_plan,
            default=('mix_segments', 'vocal_segments', 'full_vocal', 'full_instrumental'),
        )

        saved_files: List[str] = []

        mix_segment_files: List[str] = []
        if 'mix_segments' in export_flags:
            mix_segment_files = self._save_segments(
                segments,
                output_dir,
                segment_is_vocal=segment_vocal_flags,
                duration_map=segment_durations_map,
            )
            saved_files.extend(mix_segment_files)

        vocal_segment_files: List[str] = []
        if 'vocal_segments' in export_flags:
            vocal_segments, _, _ = self._split_at_sample_level(vocal_track, final_cut_points)
            vocal_segment_files = self._save_segments(
                vocal_segments,
                output_dir,
                segment_is_vocal=segment_vocal_flags,
                subdir='segments_vocal',
                file_suffix='_vocal',
                duration_map=segment_durations_map,
            )
            saved_files.extend(vocal_segment_files)

        # 5. Save full-length separated files
        input_name = Path(input_path).stem
        full_vocal_file: Optional[str] = None
        if 'full_vocal' in export_flags:
            full_vocal_duration = len(vocal_track) / float(self.sample_rate)
            full_vocal_base = Path(output_dir) / f"{input_name}_{mode}_vocal_full_{full_vocal_duration:.1f}"
            full_vocal_path = export_audio(
                vocal_track,
                self.sample_rate,
                full_vocal_base,
                self._export_format,
                options=self._export_options,
            )
            full_vocal_file = str(full_vocal_path)
            saved_files.append(full_vocal_file)

        full_instrumental_file: Optional[str] = None
        if 'full_instrumental' in export_flags:
            if separation_result.instrumental_track is not None:
                instrumental_duration = len(separation_result.instrumental_track) / float(self.sample_rate)
                full_instrumental_base = (
                    Path(output_dir) / f"{input_name}_{mode}_instrumental_{instrumental_duration:.1f}"
                )
                full_instrumental_path = export_audio(
                    separation_result.instrumental_track,
                    self.sample_rate,
                    full_instrumental_base,
                    self._export_format,
                    options=self._export_options,
                )
                full_instrumental_file = str(full_instrumental_path)
                saved_files.append(full_instrumental_file)
            else:
                logger.warning("[EXPORT] Instrumental track unavailable; skip full_instrumental export.")

        total_time = time.time() - overall_start_time

        cuts_samples = list(final_cut_points)
        cuts_seconds = [sample / float(self.sample_rate) for sample in cuts_samples]
        guard_adjustments = self._get_guard_adjustments()

        result = {
            'success': True, 'method': f'pure_vocal_split_{mode}', 'num_segments': len(segments),
            'saved_files': saved_files,
            'mix_segment_files': mix_segment_files,
            'vocal_segment_files': vocal_segment_files,
            'full_vocal_file': full_vocal_file,
            'full_instrumental_file': full_instrumental_file,
            'export_plan': sorted(export_flags),
            'backend_used': separation_result.backend_used,
            'separation_confidence': separation_result.separation_confidence, 'processing_time': total_time,
            'segment_durations': segment_durations,
            'segment_vocal_flags': segment_vocal_flags,
            'segment_labels': ['human' if flag else 'music' for flag in segment_vocal_flags],
            'segment_classification_debug': getattr(self, '_last_segment_classification_debug', []),
            'guard_shift_stats': self._get_guard_shift_stats(),
            'guard_adjustments': guard_adjustments,
            'segment_layout_applied': bool(layout_applied),
            'suppressed_cut_points_sec': [float(cut.t) for cut in self._last_suppressed_cut_points],
            'cut_points_samples': cuts_samples,
            'cut_points_sec': cuts_seconds,
            'precision_guard_ok': bool(self._precision_guard_ok),
            'precision_guard_threshold_ms': {
                'avg': PRECISION_GUARD_AVG_MS,
                'p95': PRECISION_GUARD_P95_MS,
            },
            'input_file': input_path, 'output_dir': output_dir
        }
        result.update(gpu_meta)
        return result

    def _get_gpu_pipeline_config(self) -> PipelineConfig:
        try:
            mapping = get_config('gpu_pipeline', {})
        except Exception:
            mapping = {}
        if not isinstance(mapping, dict):
            mapping = {}
        return PipelineConfig.from_mapping(mapping)

    def _build_gpu_pipeline_context(self, audio: np.ndarray, cfg: PipelineConfig) -> Optional[PipelineContext]:
        if not cfg.enable or self.sample_rate <= 0:
            return None
        duration_s = float(audio.shape[-1]) / float(self.sample_rate)
        ctx = build_pipeline_context(duration_s, cfg)
        if not ctx.enabled:
            return None
        return ctx

    def _process_vocal_separation_only(
        self,
        input_path: str,
        output_dir: str,
        *,
        export_plan: Optional[Sequence[str]] = None,
    ) -> Dict:
        """Process vocal separation only mode."""
        logger.info("[VOCAL_SEPARATION] Running vocal-only separation...")
        start_time = time.time()
        self._precision_guard_ok = True
        original_audio = self._load_and_resample_if_needed(input_path)
        separation_result = self.separator.separate_for_detection(original_audio)

        if separation_result.vocal_track is None:
            return {'success': False, 'error': 'vocal_separation_failed', 'input_file': input_path}

        export_flags = self._normalize_export_plan(
            export_plan,
            default=('full_vocal', 'full_instrumental'),
        )

        input_name = Path(input_path).stem
        saved_files: List[str] = []

        full_vocal_file: Optional[str] = None
        if 'full_vocal' in export_flags:
            vocal_duration = len(separation_result.vocal_track) / float(self.sample_rate)
            vocal_base = Path(output_dir) / f"{input_name}_vocal_{vocal_duration:.1f}"
            vocal_path = export_audio(
                separation_result.vocal_track,
                self.sample_rate,
                vocal_base,
                self._export_format,
                options=self._export_options,
            )
            full_vocal_file = str(vocal_path)
            saved_files.append(full_vocal_file)

        full_instrumental_file: Optional[str] = None
        if 'full_instrumental' in export_flags:
            if separation_result.instrumental_track is not None:
                instrumental_duration = len(separation_result.instrumental_track) / float(self.sample_rate)
                instrumental_base = Path(output_dir) / f"{input_name}_instrumental_{instrumental_duration:.1f}"
                instrumental_path = export_audio(
                    separation_result.instrumental_track,
                    self.sample_rate,
                    instrumental_base,
                    self._export_format,
                    options=self._export_options,
                )
                full_instrumental_file = str(instrumental_path)
                saved_files.append(full_instrumental_file)
            else:
                logger.warning("[EXPORT] Instrumental track unavailable; skip full_instrumental export.")

        processing_time = time.time() - start_time

        base_result = {
            'success': True,
            'method': 'vocal_separation_only',
            'num_segments': 0,
            'saved_files': saved_files,
            'mix_segment_files': [],
            'vocal_segment_files': [],
            'full_vocal_file': full_vocal_file,
            'full_instrumental_file': full_instrumental_file,
            'export_plan': sorted(export_flags),
            'backend_used': separation_result.backend_used,
            'separation_confidence': separation_result.separation_confidence,
            'processing_time': processing_time,
            'segment_durations': [],
            'guard_shift_stats': self._get_guard_shift_stats(),
            'precision_guard_ok': bool(self._precision_guard_ok),
            'precision_guard_threshold_ms': {
                'avg': PRECISION_GUARD_AVG_MS,
                'p95': PRECISION_GUARD_P95_MS,
            },
            'input_file': input_path,
            'output_dir': output_dir,
        }
        base_result.update(dict(separation_result.gpu_meta or {}))
        return base_result

    def _process_librosa_onset_split(
        self,
        input_path: str,
        output_dir: str,
        *,
        export_plan: Optional[Sequence[str]] = None,
    ) -> Dict:
        """
        Smart music segmentation v2: 情感感知的智能分割

        Features:
        - BPM 检测 + 小节边界对齐
        - 能量曲线分析 → 主歌/副歌自动识别
        - 静音段落强制分割
        - 密度控制 (low/medium/high)
        - 保持 human/music 标签

        配置来源 (优先级从高到低):
        - 环境变量 AUDIOCUT_*
        - config/unified.yaml 中的 librosa_onset 节
        - 内置默认值
        """
        logger.info("[SMART_SEGMENT] Running emotion-aware segmentation v2...")
        start_time = time.time()
        self._precision_guard_ok = True

        # ========== 0. Load configuration ==========
        lo_config = get_librosa_onset_config()
        logger.info("[SMART_SEGMENT] Config loaded: density=%s, silence_db=%.1f, use_vocal=%s",
                    lo_config['density'], lo_config['silence']['threshold_db'], lo_config['use_vocal_separation'])

        # ========== 1. Load audio ==========
        audio = self._load_and_resample_if_needed(input_path)
        audio_duration = len(audio) / float(self.sample_rate)
        logger.info("[SMART_SEGMENT] Audio duration: %.1fs", audio_duration)

        # ========== 2. MDX23 vocal separation ==========
        use_vocal = lo_config['use_vocal_separation']
        vocal_track = None
        instrumental_track = None
        separation_result = None
        
        if use_vocal:
            logger.info("[SMART_SEGMENT] MDX23 vocal separation enabled")
            separation_result = self.separator.separate_for_detection(audio)
            if separation_result.vocal_track is not None:
                vocal_track = separation_result.vocal_track
                instrumental_track = separation_result.instrumental_track
            else:
                logger.warning("[SMART_SEGMENT] Vocal separation failed")
        
        # ========== 3. Feature extraction ==========
        hop_length = lo_config['energy_analysis'].get('hop_length', 512)
        time_signature = lo_config['beat'].get('time_signature', 4)

        # 3.1 BPM detection
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate, hop_length=hop_length)
        tempo = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate, hop_length=hop_length)
        bar_duration = 60.0 / tempo * time_signature  # Use configured time signature
        logger.info("[SMART_SEGMENT] BPM: %.1f, bar duration: %.2fs, %d beats detected", tempo, bar_duration, len(beat_times))
        
        # 3.2 RMS energy curve (for chorus/verse detection)
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sample_rate, hop_length=hop_length)
        
        # Calculate bar boundaries for energy analysis
        bar_times_all = np.arange(0, audio_duration + bar_duration, bar_duration)
        
        # Compute average RMS energy for each bar
        bar_energies: List[float] = []
        for i in range(len(bar_times_all) - 1):
            bar_start = bar_times_all[i]
            bar_end = bar_times_all[i + 1]
            # Find RMS frames within this bar
            mask = (rms_times >= bar_start) & (rms_times < bar_end)
            if np.any(mask):
                bar_energies.append(float(np.mean(rms[mask])))
            else:
                bar_energies.append(0.0)
        
        # Use configured percentiles for chorus/verse detection
        chorus_percentile = lo_config['energy_analysis'].get('chorus_percentile', 60)
        chorus_peak_percentile = lo_config['energy_analysis'].get('chorus_peak_percentile', 80)

        if bar_energies:
            rms_threshold_chorus = float(np.percentile(bar_energies, chorus_percentile))
            rms_threshold_peak = float(np.percentile(bar_energies, chorus_peak_percentile))
        else:
            rms_threshold_chorus = 0.0
            rms_threshold_peak = 0.0

        # Classify each bar as chorus (high energy) or verse (low energy)
        bar_types: List[str] = []
        for energy in bar_energies:
            if energy >= rms_threshold_peak:
                bar_types.append('chorus_peak')  # Very high energy
            elif energy >= rms_threshold_chorus:
                bar_types.append('chorus')  # High energy
            else:
                bar_types.append('verse')  # Low energy

        chorus_count = sum(1 for t in bar_types if 'chorus' in t)
        logger.info("[SMART_SEGMENT] Bar classification: %d chorus, %d verse (threshold P%d=%.4f, P%d=%.4f)",
                    chorus_count, len(bar_types) - chorus_count,
                    chorus_percentile, rms_threshold_chorus, chorus_peak_percentile, rms_threshold_peak)
        
        # 3.3 Silence detection (mandatory cut points)
        silence_threshold_db = lo_config['silence']['threshold_db']
        silence_min_duration = lo_config['silence']['min_duration']
        rms_db = 20.0 * np.log10(rms + 1e-10)
        silence_mask = rms_db < silence_threshold_db
        
        # Find silence boundaries
        silence_boundaries: List[float] = []
        in_silence = False
        silence_start = 0.0
        for i, is_silent in enumerate(silence_mask):
            t = float(rms_times[i]) if i < len(rms_times) else audio_duration
            if is_silent and not in_silence:
                in_silence = True
                silence_start = t
            elif not is_silent and in_silence:
                in_silence = False
                silence_duration = t - silence_start
                if silence_duration >= silence_min_duration:
                    silence_boundaries.append(silence_start + silence_duration / 2)  # Cut at midpoint
        logger.info("[SMART_SEGMENT] Found %d silence boundaries", len(silence_boundaries))
        
        # ========== 4. Density control ==========
        density = lo_config['density']
        density_custom = lo_config.get('density_custom', {})

        # Check if custom density is enabled
        if density_custom.get('enable', False):
            density_cfg = {
                'verse_bars': density_custom.get('verse_bars', 4),
                'chorus_bars': density_custom.get('chorus_bars', 2),
            }
            logger.info("[SMART_SEGMENT] Using custom density: verse=%d bars, chorus=%d bars",
                        density_cfg['verse_bars'], density_cfg['chorus_bars'])
        else:
            # Use preset density
            density_map = {
                'low': {'verse_bars': 8, 'chorus_bars': 4},
                'medium': {'verse_bars': 4, 'chorus_bars': 2},
                'high': {'verse_bars': 2, 'chorus_bars': 1},
            }
            density_cfg = density_map.get(density, density_map['medium'])
            logger.info("[SMART_SEGMENT] Density: %s, verse=%d bars, chorus=%d bars",
                        density, density_cfg['verse_bars'], density_cfg['chorus_bars'])
        
        # ========== 5. Generate smart cut points ==========
        cut_times: List[float] = [0.0]  # Start
        
        # Use pre-computed bar_times_all from energy analysis
        bar_times = bar_times_all
        
        # Generate cuts based on bar boundaries and segment type
        last_cut_time = 0.0
        bars_since_last_cut = 0
        
        for bar_idx, bar_time in enumerate(bar_times[1:]):  # Skip first (0.0)
            bars_since_last_cut += 1
            
            # Get bar type from pre-computed classification
            bar_type = bar_types[bar_idx] if bar_idx < len(bar_types) else 'verse'
            is_chorus = 'chorus' in bar_type
            required_bars = density_cfg['chorus_bars'] if is_chorus else density_cfg['verse_bars']
            
            # Check if we should cut here
            should_cut = bars_since_last_cut >= required_bars
            
            # Force cut at silence boundaries
            for silence_t in silence_boundaries:
                if last_cut_time < silence_t <= bar_time:
                    should_cut = True
                    break
            
            if should_cut:
                cut_times.append(float(bar_time))
                last_cut_time = bar_time
                bars_since_last_cut = 0
        
        # Add all silence boundaries that weren't added
        for silence_t in silence_boundaries:
            if silence_t not in cut_times and 0 < silence_t < audio_duration:
                cut_times.append(silence_t)
        
        cut_times.append(audio_duration)  # End
        cut_times = sorted(set(cut_times))
        
        # Merge segments that are too short (< 2 seconds)
        min_segment_duration = float(get_config('segment_layout.soft_min_s', 2.0))
        merged_cuts: List[float] = [cut_times[0]]
        for t in cut_times[1:]:
            if t - merged_cuts[-1] >= min_segment_duration:
                merged_cuts.append(t)
            elif t == cut_times[-1]:  # Always keep the last point
                merged_cuts[-1] = t
        cut_times = merged_cuts
        
        logger.info("[SMART_SEGMENT] Final cut points: %d", len(cut_times))
        
        # ========== 6. Convert to sample points ==========
        audio_len = len(audio)
        cut_points = [0]
        for t in cut_times[1:-1]:
            sample_idx = int(t * self.sample_rate)
            if 0 < sample_idx < audio_len:
                cut_points.append(sample_idx)
        cut_points.append(audio_len)
        cut_points = sorted(set(cut_points))
        
        # ========== 7. Classify segments as human/music ==========
        segment_vocal_flags: List[bool] = []
        for i in range(len(cut_points) - 1):
            start_sample = cut_points[i]
            end_sample = cut_points[i + 1]
            
            if vocal_track is not None:
                # Check vocal energy in this segment
                vocal_segment = vocal_track[start_sample:end_sample]
                vocal_rms = float(np.sqrt(np.mean(vocal_segment ** 2)))
                # Compare with instrumental
                if instrumental_track is not None:
                    inst_segment = instrumental_track[start_sample:end_sample]
                    inst_rms = float(np.sqrt(np.mean(inst_segment ** 2)))
                    is_vocal = vocal_rms > inst_rms * 0.3  # Vocal is significant if > 30% of instrumental
                else:
                    is_vocal = vocal_rms > 0.01  # Simple threshold
            else:
                # No separation available, assume all human
                is_vocal = True
            
            segment_vocal_flags.append(is_vocal)
        
        logger.info("[SMART_SEGMENT] Segment classification: %d human, %d music", 
                    sum(segment_vocal_flags), len(segment_vocal_flags) - sum(segment_vocal_flags))
        
        # ========== 8. Split and export ==========
        num_segments = len(cut_points) - 1
        default_flags = segment_vocal_flags if segment_vocal_flags else [True] * num_segments
        
        segments, returned_flags, _ = self._split_at_sample_level(
            audio, cut_points, segment_flags=default_flags
        )
        if returned_flags is None:
            returned_flags = segment_vocal_flags if segment_vocal_flags else [True] * len(segments)
        
        segment_durations = [len(seg) / float(self.sample_rate) for seg in segments]
        segment_durations_map = {i: duration for i, duration in enumerate(segment_durations)}
        
        export_flags = self._normalize_export_plan(
            export_plan,
            default=('mix_segments',),
        )
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved_files: List[str] = []
        mix_segment_files: List[str] = []
        
        if 'mix_segments' in export_flags:
            mix_segment_files = self._save_segments(
                segments,
                output_dir,
                segment_is_vocal=returned_flags,
                duration_map=segment_durations_map,
            )
            saved_files.extend(mix_segment_files)
        
        vocal_segment_files: List[str] = []
        if 'vocal_segments' in export_flags and vocal_track is not None:
            vocal_segments, _, _ = self._split_at_sample_level(vocal_track, cut_points)
            vocal_segment_files = self._save_segments(
                vocal_segments,
                output_dir,
                segment_is_vocal=returned_flags,
                subdir='segments_vocal',
                file_suffix='_vocal',
                duration_map=segment_durations_map,
            )
            saved_files.extend(vocal_segment_files)
        
        processing_time = time.time() - start_time
        cuts_seconds = [sample / float(self.sample_rate) for sample in cut_points]
        
        # ========== 9. Build result ==========
        result = {
            'success': True,
            'method': 'smart_segment_v2',
            'num_segments': len(segments),
            'saved_files': saved_files,
            'mix_segment_files': mix_segment_files,
            'vocal_segment_files': vocal_segment_files,
            'export_plan': sorted(export_flags),
            'use_vocal_preprocessing': use_vocal,
            'processing_time': processing_time,
            'segment_durations': segment_durations,
            'segment_vocal_flags': returned_flags,
            'segment_labels': ['human' if flag else 'music' for flag in returned_flags],
            'cut_points_samples': list(cut_points),
            'cut_points_sec': cuts_seconds,
            # Smart segmentation metadata
            'bpm': tempo,
            'bar_duration_s': bar_duration,
            'density': density,
            'silence_boundaries': silence_boundaries,
            'precision_guard_ok': True,
            'input_file': input_path,
            'output_dir': output_dir,
        }
        if separation_result is not None:
            result['backend_used'] = separation_result.backend_used
            result['separation_confidence'] = separation_result.separation_confidence
            result.update(dict(separation_result.gpu_meta or {}))
        return result

    def _process_hybrid_mdd_split(
        self,
        input_path: str,
        output_dir: str,
        *,
        export_plan: Optional[Sequence[str]] = None,
        density_override: Optional[str] = None,
    ) -> Dict:
        """
        Hybrid MDD 模式：以 MDD 人声分割为基础 + librosa_onset 节拍卡点增强

        Features:
        - MDD 人声停顿检测作为基础切点
        - 在高能量段（副歌）添加 librosa_onset 节拍边界切点
        - 节拍卡点片段添加 _lib 后缀标签
        - 支持"多/中/少"三档卡点数量控制

        配置来源：
        - config/unified.yaml 中的 hybrid_mdd 节
        - density_override 参数覆盖
        """
        logger.info("[HYBRID_MDD] Running hybrid MDD + beat-cut segmentation...")
        start_time = time.time()
        self._precision_guard_ok = True

        # ========== 0. Load configuration ==========
        hybrid_config = get_hybrid_mdd_config(density_override)
        enable_beat_cuts = hybrid_config['enable_beat_cuts']
        energy_percentile = hybrid_config['energy_percentile']
        bars_per_cut = hybrid_config['bars_per_cut']
        hop_length = hybrid_config['beat_detection']['hop_length']
        time_signature = hybrid_config['beat_detection']['time_signature']
        snap_to_pause_ms = hybrid_config['beat_detection']['snap_to_pause_ms']
        lib_suffix = hybrid_config['labeling']['lib_suffix']

        logger.info("[HYBRID_MDD] Config: density=%s, enable_beat_cuts=%s, energy_percentile=%d, bars_per_cut=%d",
                    hybrid_config['density'], enable_beat_cuts, energy_percentile, bars_per_cut)

        # ========== 1. Run standard MDD pipeline to get base cut points ==========
        # Call MDD split to get base result (we reuse its cut points, not its exports)
        mdd_result = self._process_pure_vocal_split(
            input_path,
            output_dir,
            'v2.2_mdd',
            export_plan=('none',),  # Skip all exports - we'll do our own with _lib suffix
        )

        if not mdd_result.get('success'):
            return mdd_result

        # Get MDD cut points
        mdd_cut_points_samples: List[int] = list(mdd_result.get('cut_points_samples', []))
        mdd_cut_points_sec = [s / float(self.sample_rate) for s in mdd_cut_points_samples]

        logger.info("[HYBRID_MDD] MDD provided %d cut points", len(mdd_cut_points_samples))

        # ========== 2. Load original audio for beat detection ==========
        original_audio = self._load_and_resample_if_needed(input_path)
        audio_duration = len(original_audio) / float(self.sample_rate)

        # Track which cut points are from librosa beat detection
        lib_cut_flags: List[bool] = [False] * (len(mdd_cut_points_samples) - 1)  # Per segment

        if not enable_beat_cuts:
            logger.info("[HYBRID_MDD] Beat cuts disabled (density=low), using MDD cuts only")
            final_cut_points = mdd_cut_points_samples
        else:
            # ========== 3. Beat detection on ORIGINAL audio ==========
            tempo, beat_frames = librosa.beat.beat_track(y=original_audio, sr=self.sample_rate, hop_length=hop_length)
            tempo = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate, hop_length=hop_length)
            bar_duration = 60.0 / tempo * time_signature
            logger.info("[HYBRID_MDD] BPM: %.1f, bar duration: %.2fs, detected %d beats", tempo, bar_duration, len(beat_times))

            # ========== 4. Energy analysis to identify high-energy segments ==========
            rms = librosa.feature.rms(y=original_audio, hop_length=hop_length)[0]
            rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sample_rate, hop_length=hop_length)

            # Calculate bar boundaries from ACTUAL detected beats (not from 0)
            # Each bar = time_signature beats (e.g., 4 beats for 4/4 time)
            # Group beats into bars
            bar_times_list: List[float] = []
            if len(beat_times) >= time_signature:
                for i in range(0, len(beat_times), time_signature):
                    bar_times_list.append(float(beat_times[i]))
                # Add the end of the last bar
                if len(beat_times) > 0:
                    bar_times_list.append(float(audio_duration))
            else:
                # Fallback if not enough beats detected
                bar_times_list = list(np.arange(0, audio_duration + bar_duration, bar_duration))

            bar_times = np.array(bar_times_list)
            logger.info("[HYBRID_MDD] Generated %d bar boundaries from detected beats", len(bar_times))

            # Compute average RMS energy for each bar
            bar_energies: List[float] = []
            for i in range(len(bar_times) - 1):
                bar_start = bar_times[i]
                bar_end = bar_times[i + 1]
                mask = (rms_times >= bar_start) & (rms_times < bar_end)
                if np.any(mask):
                    bar_energies.append(float(np.mean(rms[mask])))
                else:
                    bar_energies.append(0.0)

            # Determine energy threshold for beat cuts
            if bar_energies:
                energy_threshold = float(np.percentile(bar_energies, energy_percentile))
            else:
                energy_threshold = 0.0

            # Identify high-energy bars
            high_energy_bars: Set[int] = set()
            for i, energy in enumerate(bar_energies):
                if energy >= energy_threshold:
                    high_energy_bars.add(i)

            logger.info("[HYBRID_MDD] High-energy bars: %d / %d (threshold P%d=%.4f)",
                        len(high_energy_bars), len(bar_energies), energy_percentile, energy_threshold)

            # ========== 5. Generate beat cut points for high-energy segments ==========
            # Use ACTUAL beat times for cut points, not bar boundaries
            beat_cut_times: List[float] = []
            bars_since_last_cut = 0

            for bar_idx in range(len(bar_times) - 1):
                bars_since_last_cut += 1
                # Get the beat time at the END of this bar (start of next bar)
                # This ensures the cut is at an actual detected beat
                if bar_idx + 1 < len(bar_times):
                    bar_end_time = bar_times[bar_idx + 1]
                else:
                    continue

                if bar_idx in high_energy_bars and bars_since_last_cut >= bars_per_cut:
                    beat_cut_times.append(float(bar_end_time))
                    bars_since_last_cut = 0

            logger.info("[HYBRID_MDD] Generated %d beat cut points", len(beat_cut_times))

            # ========== 6. Merge MDD and beat cut points ==========
            # Convert beat cuts to samples
            beat_cut_samples = [int(t * self.sample_rate) for t in beat_cut_times]

            # Check for overlap with existing MDD cuts (within snap_to_pause_ms)
            snap_samples = int(snap_to_pause_ms / 1000.0 * self.sample_rate)
            new_cuts: List[int] = []
            new_cut_is_lib: List[bool] = []

            for beat_sample in beat_cut_samples:
                # Check if there's an MDD cut point nearby
                is_duplicate = False
                for mdd_sample in mdd_cut_points_samples:
                    if abs(beat_sample - mdd_sample) <= snap_samples:
                        is_duplicate = True
                        break

                if not is_duplicate and 0 < beat_sample < len(original_audio):
                    new_cuts.append(beat_sample)
                    new_cut_is_lib.append(True)

            # Merge and sort all cut points
            all_cuts = list(mdd_cut_points_samples)
            cut_sources: Dict[int, bool] = {c: False for c in all_cuts}  # False = MDD, True = librosa

            for cut, is_lib in zip(new_cuts, new_cut_is_lib):
                all_cuts.append(cut)
                cut_sources[cut] = is_lib

            final_cut_points = sorted(set(all_cuts))

            # Ensure first and last cut points are correct
            if final_cut_points[0] != 0:
                final_cut_points.insert(0, 0)
            if final_cut_points[-1] != len(original_audio):
                final_cut_points.append(len(original_audio))
            final_cut_points = sorted(set(final_cut_points))

            logger.info("[HYBRID_MDD] Merged cut points: %d total (%d from MDD, %d new beat cuts)",
                        len(final_cut_points), len(mdd_cut_points_samples), len(new_cuts))

            # Build lib flags for each segment (segment i is between cut i and cut i+1)
            # A segment is marked as lib if its END cut point is from librosa
            lib_cut_flags = []
            for i in range(len(final_cut_points) - 1):
                end_cut = final_cut_points[i + 1]
                is_lib = cut_sources.get(end_cut, False)
                lib_cut_flags.append(is_lib)

        # ========== 7. Classify segments as human/music ==========
        # Reuse vocal track from MDD result if available
        vocal_track = None
        separation_result = None
        if hasattr(self, '_last_separation_result'):
            separation_result = self._last_separation_result
            if separation_result and separation_result.vocal_track is not None:
                vocal_track = separation_result.vocal_track

        # If no cached result, perform separation
        if vocal_track is None:
            separation_result = self.separator.separate_for_detection(original_audio)
            if separation_result.vocal_track is not None:
                vocal_track = separation_result.vocal_track

        segment_vocal_flags = self._classify_segments_vocal_presence(
            vocal_track if vocal_track is not None else original_audio,
            final_cut_points,
            marker_segments=None,
            pure_music_segments=None,
            instrumental_audio=separation_result.instrumental_track if separation_result else None,
            original_audio=original_audio,
        )

        # ========== 8. Apply segment layout refinement (merge short segments) ==========
        # Load segment_layout config
        layout_cfg_raw = get_config('segment_layout', {}) or {}
        if isinstance(layout_cfg_raw, dict):
            layout_cfg_raw = dict(layout_cfg_raw)
        else:
            layout_cfg_raw = {}

        micro_merge_s = float(layout_cfg_raw.get('micro_merge_s', 2.0))
        soft_min_s = float(layout_cfg_raw.get('soft_min_s', 2.0))

        if micro_merge_s > 0 and len(final_cut_points) > 2:
            logger.info("[HYBRID_MDD] Applying segment layout refinement (micro_merge=%.1fs, soft_min=%.1fs)",
                        micro_merge_s, soft_min_s)

            # Merge segments shorter than micro_merge_s
            merged_cut_points: List[int] = [final_cut_points[0]]
            merged_lib_flags: List[bool] = []

            for i in range(len(final_cut_points) - 1):
                start_sample = final_cut_points[i]
                end_sample = final_cut_points[i + 1]
                duration_s = (end_sample - start_sample) / float(self.sample_rate)

                if duration_s < micro_merge_s and i < len(final_cut_points) - 2:
                    # Merge with next segment: skip this cut point
                    # Preserve lib flag if either segment was lib
                    if i < len(lib_cut_flags) and lib_cut_flags[i]:
                        # Carry lib flag forward to the next segment
                        if i + 1 < len(lib_cut_flags):
                            lib_cut_flags[i + 1] = True
                    logger.debug("[HYBRID_MDD] Merging short segment %d (%.1fs) with next", i, duration_s)
                else:
                    merged_cut_points.append(end_sample)
                    if i < len(lib_cut_flags):
                        merged_lib_flags.append(lib_cut_flags[i])
                    else:
                        merged_lib_flags.append(False)

            # Update the data
            if len(merged_cut_points) != len(final_cut_points):
                logger.info("[HYBRID_MDD] Merged %d short segments: %d → %d cut points",
                            len(final_cut_points) - len(merged_cut_points),
                            len(final_cut_points), len(merged_cut_points))
                final_cut_points = merged_cut_points
                lib_cut_flags = merged_lib_flags

                # Re-classify after merging
                segment_vocal_flags = self._classify_segments_vocal_presence(
                    vocal_track if vocal_track is not None else original_audio,
                    final_cut_points,
                    marker_segments=None,
                    pure_music_segments=None,
                    instrumental_audio=separation_result.instrumental_track if separation_result else None,
                    original_audio=original_audio,
                )

        # ========== 9. Split and export with _lib suffix for beat cuts ==========
        segments, returned_flags, _ = self._split_at_sample_level(
            original_audio,
            final_cut_points,
            segment_flags=segment_vocal_flags,
        )

        segment_durations = [len(seg) / float(self.sample_rate) for seg in segments]
        segment_durations_map = {i: duration for i, duration in enumerate(segment_durations)}

        export_flags = self._normalize_export_plan(
            export_plan,
            default=('mix_segments', 'vocal_segments', 'full_vocal'),
        )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved_files: List[str] = []
        mix_segment_files: List[str] = []

        if 'mix_segments' in export_flags:
            mix_segment_files = self._save_segments_with_lib_suffix(
                segments,
                output_dir,
                segment_is_vocal=returned_flags if returned_flags else segment_vocal_flags,
                lib_flags=lib_cut_flags,
                lib_suffix=lib_suffix,
                duration_map=segment_durations_map,
            )
            saved_files.extend(mix_segment_files)

        vocal_segment_files: List[str] = []
        if 'vocal_segments' in export_flags and vocal_track is not None:
            vocal_segments, _, _ = self._split_at_sample_level(vocal_track, final_cut_points)
            vocal_segment_files = self._save_segments_with_lib_suffix(
                vocal_segments,
                output_dir,
                segment_is_vocal=returned_flags if returned_flags else segment_vocal_flags,
                lib_flags=lib_cut_flags,
                lib_suffix=lib_suffix,
                subdir='segments_vocal',
                file_suffix='_vocal',
                duration_map=segment_durations_map,
            )
            saved_files.extend(vocal_segment_files)

        # Export full vocal/instrumental
        input_name = Path(input_path).stem
        full_vocal_file: Optional[str] = None
        if 'full_vocal' in export_flags and vocal_track is not None:
            full_vocal_duration = len(vocal_track) / float(self.sample_rate)
            full_vocal_base = Path(output_dir) / f"{input_name}_hybrid_mdd_vocal_full_{full_vocal_duration:.1f}"
            full_vocal_path = export_audio(
                vocal_track,
                self.sample_rate,
                full_vocal_base,
                self._export_format,
                options=self._export_options,
            )
            full_vocal_file = str(full_vocal_path)
            saved_files.append(full_vocal_file)

        processing_time = time.time() - start_time
        cuts_seconds = [sample / float(self.sample_rate) for sample in final_cut_points]

        # Count lib segments
        lib_segment_count = sum(1 for f in lib_cut_flags if f)

        result = {
            'success': True,
            'method': 'hybrid_mdd',
            'num_segments': len(segments),
            'saved_files': saved_files,
            'mix_segment_files': mix_segment_files,
            'vocal_segment_files': vocal_segment_files,
            'full_vocal_file': full_vocal_file,
            'export_plan': sorted(export_flags),
            'processing_time': processing_time,
            'segment_durations': segment_durations,
            'segment_vocal_flags': returned_flags if returned_flags else segment_vocal_flags,
            'segment_labels': ['human' if flag else 'music' for flag in (returned_flags or segment_vocal_flags)],
            'segment_lib_flags': lib_cut_flags,
            'lib_segment_count': lib_segment_count,
            'cut_points_samples': list(final_cut_points),
            'cut_points_sec': cuts_seconds,
            'hybrid_config': {
                'density': hybrid_config['density'],
                'enable_beat_cuts': enable_beat_cuts,
                'energy_percentile': energy_percentile,
                'bars_per_cut': bars_per_cut,
            },
            'precision_guard_ok': True,
            'input_file': input_path,
            'output_dir': output_dir,
        }

        if separation_result is not None:
            result['backend_used'] = separation_result.backend_used
            result['separation_confidence'] = separation_result.separation_confidence
            result.update(dict(separation_result.gpu_meta or {}))

        logger.info("[HYBRID_MDD] Completed: %d segments (%d with _lib suffix), processing_time=%.1fs",
                    len(segments), lib_segment_count, processing_time)
        return result

    def _save_segments_with_lib_suffix(
        self,
        segments: List[np.ndarray],
        output_dir: str,
        segment_is_vocal: List[bool],
        lib_flags: List[bool],
        lib_suffix: str = "_lib",
        subdir: Optional[str] = None,
        file_suffix: str = "",
        duration_map: Optional[Dict[int, float]] = None,
    ) -> List[str]:
        """Save segments with optional _lib suffix for beat-cut segments.

        Args:
            segments: Audio segment data
            segment_is_vocal: Whether each segment is vocal (human) or music
            lib_flags: Whether each segment ends at a librosa beat cut point
            lib_suffix: Suffix to add for beat-cut segments (e.g., "_lib")
            subdir: Optional subdirectory for output
            file_suffix: Additional suffix for filenames
            duration_map: Optional mapping of segment index to duration

        Returns:
            List of saved file paths
        """
        final_output_dir = Path(output_dir)
        if subdir:
            final_output_dir = final_output_dir / subdir
        final_output_dir.mkdir(parents=True, exist_ok=True)

        saved_files: List[str] = []

        for i, segment in enumerate(segments):
            is_vocal = segment_is_vocal[i] if i < len(segment_is_vocal) else True
            is_lib = lib_flags[i] if i < len(lib_flags) else False
            label = "human" if is_vocal else "music"

            # Build filename with optional _lib suffix
            lib_part = lib_suffix if is_lib else ""
            duration = duration_map.get(i, len(segment) / float(self.sample_rate)) if duration_map else len(segment) / float(self.sample_rate)

            base_name = f"segment_{i:03d}_{label}{lib_part}{file_suffix}_{duration:.1f}"
            output_base = final_output_dir / base_name

            output_path = export_audio(
                segment,
                self.sample_rate,
                output_base,
                self._export_format,
                options=self._export_options,
            )
            saved_files.append(str(output_path))

        return saved_files

    def _find_no_vocal_runs(self, vocal_audio: np.ndarray, min_duration: float):
        """
        Construct a voice-active mask from the vocal track energy envelope and gather continuous "no-vocal" intervals (seconds).
        修复：改用鲁棒阈值（噪底与人声分布中值之间的中点）+ 轻度形态学，避免阈值过低导致全程“活跃”。
        """
        sr = self.sample_rate
        hop = max(1, int(0.01 * sr))  # 10ms
        # RMS envelope -> dB
        rms = librosa.feature.rms(y=vocal_audio, hop_length=hop)[0]
        db = 20.0 * np.log10(rms + 1e-12)
        # Robust threshold: midpoint between noise floor and vocal upper percentile
        try:
            noise_pct = float(get_config('quality_control.enforce_quiet_cut.floor_percentile', 10))
        except Exception:
            noise_pct = 10.0
        try:
            voice_pct = float(get_config('pure_vocal_detection.pause_stats_adaptation.voice_percentile_hint', 90))
        except Exception:
            voice_pct = 90.0
        noise_db = float(np.percentile(db, np.clip(noise_pct, 0, 50)))
        voice_db = float(np.percentile(db, np.clip(voice_pct, 50, 100)))
        delta_db = float(get_config('pure_vocal_detection.pause_stats_adaptation.delta_db', 3.0))
        thr_mid = 0.5 * (noise_db + voice_db)
        thr_db = max(noise_db + delta_db, thr_mid)
        active = db > thr_db
        # Light morphology: close then open (milliseconds -> frames)
        close_ms = int(get_config('pure_vocal_detection.pause_stats_adaptation.morph_close_ms', 150))
        open_ms = int(get_config('pure_vocal_detection.pause_stats_adaptation.morph_open_ms', 50))
        frame_sec = hop / float(sr)
        close_k = max(1, int(close_ms / 1000.0 / frame_sec))
        open_k = max(1, int(open_ms / 1000.0 / frame_sec))
        def fill_false_runs(m: np.ndarray, max_len: int) -> np.ndarray:
            m = m.astype(bool).copy(); n = len(m); i = 0
            while i < n:
                if not m[i]:
                    j = i
                    while j < n and not m[j]:
                        j += 1
                    if (j - i) <= max_len:
                        m[i:j] = True
                    i = j
                else:
                    i += 1
            return m
        def remove_true_runs(m: np.ndarray, max_len: int) -> np.ndarray:
            m = m.astype(bool).copy(); n = len(m); i = 0
            while i < n:
                if m[i]:
                    j = i
                    while j < n and m[j]:
                        j += 1
                    if (j - i) <= max_len:
                        m[i:j] = False
                    i = j
                else:
                    i += 1
            return m
        active = fill_false_runs(active, close_k)
        active = remove_true_runs(active, open_k)
        inactive = ~active
        # Frame timeline
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
        spans = []
        in_run = False
        start_t = 0.0
        for i, flag in enumerate(inactive):
            if flag and not in_run:
                in_run = True
                start_t = float(times[i])
            elif (not flag) and in_run:
                end_t = float(times[i])
                if end_t - start_t >= float(min_duration):
                    spans.append((start_t, end_t))
                in_run = False
        # Finalize
        if in_run:
            end_t = float(len(vocal_audio) / float(sr))
            if end_t - start_t >= float(min_duration):
                spans.append((start_t, end_t))
        try:
            n_inactive = int(np.sum(inactive.astype(int)))
            logger.info(f"[NoVocalRuns] thr_db={thr_db:.2f}, noise_db={noise_db:.2f}, voice_db={voice_db:.2f}, inactive_frames={n_inactive}/{len(inactive)}")
        except Exception:
            pass
        return spans

    def _finalize_and_filter_cuts_v2(
        self,
        cut_candidates,
        audio_for_split: np.ndarray,
        pure_vocal_audio: Optional[np.ndarray] = None
    ) -> CutRefineResult:
        """Refined candidate filtering: leverage shared refiner for guards/NMS."""

        sr = self.sample_rate
        if sr <= 0 or audio_for_split.size == 0:
            self._set_guard_adjustments([])
            return CutRefineResult([], [0, len(audio_for_split)], [])

        points: List[CutPoint] = []
        if isinstance(cut_candidates, list) and cut_candidates:
            first = cut_candidates[0]
            if isinstance(first, tuple) and len(first) >= 2:
                for candidate in cut_candidates:
                    t, s = candidate[:2]
                    points.append(CutPoint(t=float(t), score=float(s)))
            elif isinstance(first, int):
                for sample_idx in cut_candidates:
                    points.append(CutPoint(t=float(sample_idx) / float(sr), score=1.0))
            else:
                for t in cut_candidates:
                    points.append(CutPoint(t=float(t), score=1.0))

        if not points:
            self._set_guard_adjustments([])
            return CutRefineResult([], [0, len(audio_for_split)], [])

        min_gap_s = float(get_config('quality_control.min_split_gap', 1.0))
        try:
            max_keep = int(get_config('pure_vocal_detection.valley_scoring.max_kept_after_nms', 150))
        except Exception:
            max_keep = None

        guard_enabled = bool(get_config('quality_control.enforce_quiet_cut.enable', False))
        guard_db = float(get_config('quality_control.enforce_quiet_cut.guard_db', 2.5))
        search_right_ms = float(get_config('quality_control.enforce_quiet_cut.search_right_ms', 150))
        guard_win_ms = float(get_config('quality_control.enforce_quiet_cut.win_ms', 80))

        floor_db = -60.0
        if guard_enabled:
            override_floor = get_config('quality_control.enforce_quiet_cut.floor_db_override', None)
            if override_floor is not None:
                floor_db = float(override_floor)
            else:
                try:
                    floor_cfg = get_config('quality_control.enforce_quiet_cut.floor_percentile', 5)
                    floor_pct = float(floor_cfg) / 100.0 if float(floor_cfg) > 1 else float(floor_cfg)
                except Exception:
                    floor_pct = 0.05
                mono = audio_for_split if audio_for_split.ndim == 1 else np.mean(audio_for_split, axis=0)
                if mono.size > 0:
                    hop_length = max(1, int(sr * 0.01))
                    rms = librosa.feature.rms(y=mono, hop_length=hop_length)[0]
                    rms_db = 20.0 * np.log10(rms + 1e-12)
                    floor_db = float(np.percentile(rms_db, max(0.0, min(100.0, floor_pct * 100.0))))

        ctx = CutContext(sr=sr, mix_wave=audio_for_split, vocal_wave=pure_vocal_audio)
        use_vocal_guard = pure_vocal_audio is not None
        nms_topk_cfg = get_config('quality_control.nms_topk_per_10s', None)
        topk_per_10s = int(nms_topk_cfg) if nms_topk_cfg is not None else None
        nms_window_s = float(get_config('quality_control.nms_window_s', 10.0))

        result = finalize_cut_points(
            ctx,
            points,
            use_vocal_guard_first=use_vocal_guard,
            min_gap_s=min_gap_s,
            max_keep=max_keep,
            topk_per_10s=topk_per_10s,
            nms_window_s=nms_window_s,
            guard_db=guard_db,
            search_right_ms=search_right_ms,
            guard_win_ms=guard_win_ms,
            floor_db=floor_db,
            enable_mix_guard=guard_enabled,
            enable_vocal_guard=(guard_enabled and use_vocal_guard),
        )

        kept_adjustments = list(result.adjustments or [])
        self._set_guard_adjustments(kept_adjustments)

        boundaries = result.sample_boundaries or [0, len(audio_for_split)]
        unique_boundaries = sorted(set(boundaries))
        return CutRefineResult(result.final_points, unique_boundaries, kept_adjustments, result.suppressed_points)
    def _split_at_sample_level(
        self,
        audio: np.ndarray,
        final_cut_points: List[int],
        *,
        segment_flags: Optional[List[bool]] = None,
        debug_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[np.ndarray], Optional[List[bool]], Optional[List[Dict[str, Any]]]]:
        """按照样本级切点拆分音频，并可选合并标记与调试信息."""
        segments: List[np.ndarray] = []
        merged_flags: Optional[List[bool]] = [] if segment_flags is not None else None
        merged_debug: Optional[List[Dict[str, Any]]] = [] if debug_entries is not None else None

        min_keep_samples = max(1, int(0.01 * self.sample_rate))
        carry_audio: Optional[np.ndarray] = None
        carry_flag: Optional[bool] = None
        carry_debug: Optional[Dict[str, Any]] = None

        def _flag_at(index: int) -> bool:
            if segment_flags is not None and index < len(segment_flags):
                return bool(segment_flags[index])
            return True

        def _debug_at(index: int) -> Optional[Dict[str, Any]]:
            if debug_entries is not None and index < len(debug_entries):
                entry = dict(debug_entries[index])
                entry.setdefault('merged_from_segments', [index])
                entry.setdefault('decision_reason', entry.get('decision_reason', entry.get('reason')))
                return entry
            return None

        def _merge_debug(base: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if base is None:
                return extra
            if extra is None:
                return base
            merged = dict(base)
            merged['start_s'] = min(base.get('start_s', merged.get('start_s')), extra.get('start_s', merged.get('start_s')))
            merged['end_s'] = max(base.get('end_s', merged.get('end_s')), extra.get('end_s', merged.get('end_s')))
            merged_duration = float(base.get('duration_s', 0.0)) + float(extra.get('duration_s', 0.0))
            merged_seconds = float(base.get('vocal_activity_seconds', 0.0)) + float(extra.get('vocal_activity_seconds', 0.0))
            merged['duration_s'] = merged_duration
            merged['vocal_activity_seconds'] = merged_seconds
            merged['vocal_activity_ratio'] = merged_seconds / merged_duration if merged_duration > 0 else 0.0
            merged['decision'] = bool(base.get('decision')) or bool(extra.get('decision'))
            reason = extra.get('decision_reason') or base.get('decision_reason') or extra.get('reason') or base.get('reason')
            if reason:
                merged['reason'] = reason
                merged['decision_reason'] = reason
            merged_from = list(base.get('merged_from_segments', []))
            merged_from += list(extra.get('merged_from_segments', []))
            merged['merged_from_segments'] = sorted({seg for seg in merged_from if seg is not None})
            merged['activity_ratio_threshold'] = extra.get('activity_ratio_threshold', base.get('activity_ratio_threshold'))
            merged['activity_threshold_db'] = extra.get('activity_threshold_db', base.get('activity_threshold_db'))
            return merged

        for idx in range(len(final_cut_points) - 1):
            start = final_cut_points[idx]
            end = final_cut_points[idx + 1]
            chunk = audio[start:end]
            flag = _flag_at(idx)
            entry = _debug_at(idx)

            if carry_audio is not None:
                if chunk.size:
                    chunk = np.concatenate((carry_audio, chunk))
                else:
                    chunk = carry_audio
                flag = bool(carry_flag) or bool(flag)
                entry = _merge_debug(carry_debug, entry)
                carry_audio = None
                carry_flag = None
                carry_debug = None

            if end - start >= min_keep_samples and chunk.size:
                segments.append(chunk)
                if merged_flags is not None:
                    merged_flags.append(flag)
                if merged_debug is not None:
                    if entry is None:
                        entry = {
                            'start_s': start / self.sample_rate,
                            'end_s': end / self.sample_rate,
                            'duration_s': (end - start) / self.sample_rate,
                            'vocal_activity_ratio': 1.0 if flag else 0.0,
                            'vocal_activity_seconds': ((end - start) / self.sample_rate) if flag else 0.0,
                            'activity_ratio_threshold': None,
                            'activity_threshold_db': None,
                            'decision': flag,
                            'decision_reason': 'inferred_flag',
                            'reason': 'inferred_flag',
                            'merged_from_segments': [idx],
                        }
                    merged_debug.append(entry)
            else:
                if chunk.size:
                    carry_audio = chunk if carry_audio is None else np.concatenate((carry_audio, chunk))
                    carry_flag = flag if carry_flag is None else (bool(carry_flag) or bool(flag))
                    if merged_debug is not None:
                        carry_debug = _merge_debug(carry_debug, entry)
                else:
                    if merged_debug is not None:
                        carry_debug = _merge_debug(carry_debug, entry)

        if carry_audio is not None:
            if segments:
                segments[-1] = np.concatenate((segments[-1], carry_audio))
                if merged_flags is not None:
                    merged_flags[-1] = bool(merged_flags[-1]) or bool(carry_flag)
                if merged_debug is not None:
                    merged_debug[-1] = _merge_debug(merged_debug[-1], carry_debug)
            else:
                segments.append(carry_audio)
                if merged_flags is not None:
                    merged_flags.append(bool(carry_flag))
                if merged_debug is not None:
                    entry = carry_debug or {
                        'start_s': 0.0,
                        'end_s': len(carry_audio) / self.sample_rate,
                        'duration_s': len(carry_audio) / self.sample_rate,
                        'vocal_activity_ratio': 1.0 if carry_flag else 0.0,
                        'vocal_activity_seconds': (len(carry_audio) / self.sample_rate) if carry_flag else 0.0,
                        'activity_ratio_threshold': None,
                        'activity_threshold_db': None,
                        'decision': bool(carry_flag),
                        'decision_reason': 'merged_trailing_segment',
                        'reason': 'merged_trailing_segment',
                        'merged_from_segments': [],
                    }
                    merged_debug.append(entry)

        if merged_debug is not None:
            for new_idx, entry in enumerate(merged_debug):
                entry['index'] = new_idx

        if not isinstance(self, SeamlessSplitter):
            return segments
        return segments, merged_flags, merged_debug
    def _classify_segments_vocal_presence(
        self,
        vocal_audio: np.ndarray,
        cut_points: List[int],
        marker_segments: Optional[List[Dict]] = None,
        pure_music_segments: Optional[List[Dict]] = None,
        instrumental_audio: Optional[np.ndarray] = None,
        original_audio: Optional[np.ndarray] = None,
    ) -> List[bool]:
        """Classify segments via pure vocal occupancy for _human/_music suffixes."""
        num_segments = max(len(cut_points) - 1, 0)
        self._last_segment_classification_debug = []
        if num_segments == 0:
            return []

        sr = self.sample_rate
        if sr <= 0 or vocal_audio is None or getattr(vocal_audio, 'size', 0) == 0:
            flags = [True] * num_segments
            for idx in range(num_segments):
                self._last_segment_classification_debug.append({
                    'index': idx,
                    'reason': 'fallback_invalid_input',
                    'decision': True,
                })
            return flags

        activity_ratio_threshold = float(get_config('quality_control.segment_vocal_activity_ratio', 0.10))
        threshold_db = float(get_config('quality_control.segment_vocal_threshold_db', -50.0))
        hop = max(1, int(0.02 * sr))
        frame_length = max(hop * 2, int(0.05 * sr))
        min_samples = max(1, int(0.02 * sr))

        def _slice_audio(array: Optional[np.ndarray], start_idx: int, end_idx: int) -> Optional[np.ndarray]:
            if array is None or not isinstance(array, np.ndarray):
                return None
            n = len(array)
            if n == 0:
                return None
            start_i = max(0, min(start_idx, n))
            end_i = max(start_i, min(end_idx, n))
            if end_i <= start_i:
                return None
            return array[start_i:end_i]

        flags: List[bool] = []
        debug_entries: List[Dict[str, Any]] = []

        for i in range(num_segments):
            start_idx = max(0, min(int(cut_points[i]), len(vocal_audio)))
            end_idx = max(start_idx, min(int(cut_points[i + 1]), len(vocal_audio)))
            seg_start_s = start_idx / sr
            seg_end_s = end_idx / sr
            seg_duration = max(seg_end_s - seg_start_s, 1e-6)

            vocal_segment = _slice_audio(vocal_audio, start_idx, end_idx)

            vocal_activity_ratio = 0.0
            vocal_activity_seconds = 0.0
            segment_rms_db = None

            if vocal_segment is not None and len(vocal_segment) >= frame_length:
                try:
                    rms_frames = librosa.feature.rms(y=vocal_segment, frame_length=frame_length, hop_length=hop)[0]
                except Exception:
                    rms_scalar = float(np.sqrt(np.mean(np.square(vocal_segment)) + 1e-12))
                    rms_frames = np.full(max(1, len(vocal_segment) // hop + 1), rms_scalar, dtype=np.float32)
                rms_db_frames = 20.0 * np.log10(rms_frames + 1e-12)
                active_mask = rms_db_frames > threshold_db
                if active_mask.size > 0:
                    vocal_activity_ratio = float(np.mean(active_mask))
                    active_seconds = float(active_mask.sum()) * (hop / sr)
                    vocal_activity_seconds = float(min(seg_duration, active_seconds))
            elif vocal_segment is not None and len(vocal_segment) > 0:
                rms_scalar = float(np.sqrt(np.mean(np.square(vocal_segment)) + 1e-12))
                segment_rms_db = 20.0 * np.log10(rms_scalar)
                if segment_rms_db > threshold_db:
                    vocal_activity_ratio = 1.0
                    vocal_activity_seconds = seg_duration

            if vocal_segment is not None and segment_rms_db is None and len(vocal_segment) > 0:
                rms_scalar = float(np.sqrt(np.mean(np.square(vocal_segment)) + 1e-12))
                segment_rms_db = 20.0 * np.log10(rms_scalar)

            decision = vocal_activity_ratio >= activity_ratio_threshold
            decision_reason = 'vocal_activity_ratio_gte_threshold' if decision else 'vocal_activity_ratio_lt_threshold'

            # Energy ratio vote temporarily disabled; keep placeholders for future reinstatement.
            energy_ratio = None
            energy_vote = None
            energy_reason = None

            debug_entries.append({
                'index': i,
                'start_s': seg_start_s,
                'end_s': seg_end_s,
                'duration_s': seg_duration,
                'vocal_activity_ratio': vocal_activity_ratio,
                'vocal_activity_seconds': vocal_activity_seconds,
                'activity_ratio_threshold': activity_ratio_threshold,
                'activity_threshold_db': threshold_db,
                'marker_vote': None,
                'marker_reason': None,
                'vocal_overlap_s': None,
                'music_overlap_s': None,
                'vocal_ratio': None,
                'music_ratio': None,
                'energy_ratio': energy_ratio,
                'energy_vote': energy_vote,
                'presence_ratio': None,
                'presence_vote': None,
                'presence_baseline_db': None,
                'vocal_energy': None,
                'instrumental_energy': None,
                'total_energy': None,
                'rms_db': segment_rms_db,
                'rms_vote': None,
                'decision': decision,
                'decision_reason': decision_reason,
                'reason': decision_reason,
                'decision_threshold_db': threshold_db,
                'threshold_source': 'vocal_activity_ratio',
                'noise_floor_db': None,
                'energy_reason': energy_reason,
            })
            flags.append(bool(decision))

        self._last_segment_classification_debug = debug_entries
        return flags
    def _estimate_vocal_presence(self, vocal_audio: np.ndarray) -> bool:
        """Estimate whether the provided vocal track contains human vocal presence."""
        if vocal_audio is None or getattr(vocal_audio, 'size', 0) == 0:
            return False
        flags = self._classify_segments_vocal_presence(vocal_audio, [0, len(vocal_audio)])
        return bool(flags[0]) if flags else False

    def _save_segments(
        self,
        segments: List[np.ndarray],
        output_dir: str,
        segment_is_vocal: Optional[List[bool]] = None,
        *,
        subdir: Optional[str] = None,
        file_suffix: str = '',
        duration_map: Optional[Dict[int, float]] = None,
    ) -> List[str]:
        """Persist split segments using the configured export format."""
        base_dir = Path(output_dir)
        if subdir:
            base_dir = base_dir / subdir
            base_dir.mkdir(parents=True, exist_ok=True)

        saved_files: List[str] = []
        for i, segment_audio in enumerate(segments):
            is_vocal = True
            if segment_is_vocal is not None and i < len(segment_is_vocal):
                is_vocal = bool(segment_is_vocal[i])

            duration_s = None
            if duration_map and i in duration_map:
                duration_s = max(0.0, float(duration_map[i]))

            label = 'human' if is_vocal else 'music'
            suffix = file_suffix
            if duration_s is not None:
                formatted = f"_{duration_s:.1f}"
                suffix = f"{file_suffix}{formatted}" if file_suffix else formatted

            output_base = base_dir / f"segment_{i + 1:03d}_{label}{suffix}"
            exported_path = export_audio(
                segment_audio,
                self.sample_rate,
                output_base,
                self._export_format,
                options=self._export_options,
            )
            saved_files.append(str(exported_path))

        return saved_files

    @staticmethod
    def _blank_guard_stats() -> Dict[str, float]:
        return {
            'avg_shift_ms': 0.0,
            'max_shift_ms': 0.0,
            'avg_guard_only_shift_ms': 0.0,
            'avg_vocal_guard_shift_ms': 0.0,
            'avg_mix_guard_shift_ms': 0.0,
            'p95_shift_ms': 0.0,
            'count': 0,
        }

    def _set_guard_adjustments(self, adjustments: List[CutAdjustment]) -> None:
        adjustments_list = list(adjustments or [])
        self._last_guard_adjustments_raw = adjustments_list
        self._last_guard_adjustments = [asdict(adj) for adj in adjustments_list]

        if not adjustments_list:
            self._last_guard_shift_stats = self._blank_guard_stats()
            self._precision_guard_ok = True
            return

        total_shifts = [adj.final_shift_ms for adj in adjustments_list]
        vocal_shifts = [adj.guard_shift_ms for adj in adjustments_list]
        mix_shifts = [adj.final_shift_ms - adj.guard_shift_ms for adj in adjustments_list]

        def _avg_abs(values: List[float]) -> float:
            return float(sum(abs(v) for v in values) / len(values)) if values else 0.0

        def _avg_positive(values: List[float]) -> float:
            positives = [v for v in values if v > 0]
            return float(sum(positives) / len(positives)) if positives else 0.0

        def _percentile_abs(values: List[float], percentile: float) -> float:
            if not values:
                return 0.0
            arr = np.abs(np.asarray(values, dtype=float))
            return float(np.percentile(arr, percentile))

        self._last_guard_shift_stats = {
            'avg_shift_ms': _avg_abs(total_shifts),
            'max_shift_ms': float(max((abs(v) for v in total_shifts), default=0.0)),
            'avg_guard_only_shift_ms': _avg_positive(total_shifts),
            'avg_vocal_guard_shift_ms': _avg_positive(vocal_shifts),
            'avg_mix_guard_shift_ms': _avg_positive(mix_shifts),
            'p95_shift_ms': _percentile_abs(total_shifts, 95.0),
            'count': len(adjustments_list),
        }
        self._precision_guard_ok = (
            self._last_guard_shift_stats['avg_shift_ms'] <= PRECISION_GUARD_AVG_MS
            and self._last_guard_shift_stats['p95_shift_ms'] <= PRECISION_GUARD_P95_MS
        )
        if not self._precision_guard_ok:
            logger.warning(
                "[PrecisionGuard] shift exceeds threshold: avg=%.2fms (<=%.2f), p95=%.2fms (<=%.2f)",
                self._last_guard_shift_stats['avg_shift_ms'],
                PRECISION_GUARD_AVG_MS,
                self._last_guard_shift_stats['p95_shift_ms'],
                PRECISION_GUARD_P95_MS,
            )

    def _get_guard_adjustments(self) -> List[Dict[str, float]]:
        adjustments = getattr(self, '_last_guard_adjustments', None)
        if not adjustments:
            return []
        return [dict(item) for item in adjustments]

    def _get_guard_shift_stats(self) -> Dict[str, float]:
        stats = getattr(self, '_last_guard_shift_stats', None)
        if not stats:
            return self._blank_guard_stats()
        return dict(stats)

    def _refine_boundaries_local_valley(
        self,
        sample_boundaries: List[int],
        vocal_audio: np.ndarray,
        cfg: Dict[str, Any],
        *,
        min_gap_s: float,
    ) -> List[int]:
        if vocal_audio is None or vocal_audio.size == 0 or len(sample_boundaries) <= 2:
            return sample_boundaries

        sr = float(self.sample_rate)
        radius = max(1, int(float(cfg.get('search_radius_ms', 200)) / 1000.0 * sr))
        win = max(1, int(float(cfg.get('window_ms', 20)) / 1000.0 * sr))
        drop_db = float(cfg.get('min_drop_db', 3.0))
        min_gap_samples = max(1, int(min_gap_s * sr))

        refined = list(sample_boundaries)
        for idx in range(1, len(refined) - 1):
            center = refined[idx]
            start = max(0, center - radius)
            end = min(len(vocal_audio), center + radius)
            segment = vocal_audio[start:end]
            if segment.size <= win:
                continue

            sq = np.square(segment.astype(np.float64))
            kernel = np.ones(win, dtype=np.float64) / float(win)
            rms = np.sqrt(np.convolve(sq, kernel, mode='valid') + 1e-12)
            if rms.size == 0:
                continue
            rms_db = 20.0 * np.log10(rms + 1e-12)

            original_idx = int(np.clip(center - start - win // 2, 0, rms_db.size - 1))
            original_db = rms_db[original_idx]
            valley_idx = int(np.argmin(rms_db))
            valley_db = rms_db[valley_idx]

            if (original_db - valley_db) < drop_db:
                continue

            candidate = start + valley_idx + win // 2
            if candidate <= refined[idx - 1] + min_gap_samples:
                continue
            if candidate >= refined[idx + 1] - min_gap_samples:
                continue

            refined[idx] = candidate

        return refined

    def _create_single_segment_result(self, audio: np.ndarray, input_path: str, output_dir: str, reason: str, is_vocal: bool = True, gpu_meta: Optional[Dict[str, Any]] = None) -> Dict:
        """Create a single-segment result when splitting fails."""
        logger.warning("%s; exporting as a single file.", reason)
        self._set_guard_adjustments([])
        self._last_suppressed_cut_points = []
        sample_count = audio.shape[-1] if hasattr(audio, 'shape') else len(audio)
        duration_s = float(sample_count) / float(self.sample_rate) if self.sample_rate > 0 else 0.0
        saved_files = self._save_segments([audio], output_dir, segment_is_vocal=[is_vocal])
        result = {
            'success': True, 'num_segments': 1, 'saved_files': saved_files,
            'mix_segment_files': saved_files,
            'vocal_segment_files': [],
            'full_vocal_file': saved_files[0] if saved_files else None,
            'full_instrumental_file': None,
            'export_plan': [],
            'segment_durations': [duration_s],
            'segment_vocal_flags': [is_vocal],
            'segment_labels': ['human' if is_vocal else 'music'],
            'segment_classification_debug': getattr(self, '_last_segment_classification_debug', []),
            'guard_shift_stats': self._get_guard_shift_stats(),
            'guard_adjustments': self._get_guard_adjustments(),
            'cut_points_samples': [0, sample_count],
            'cut_points_sec': [0.0, duration_s],
            'precision_guard_ok': bool(self._precision_guard_ok),
            'precision_guard_threshold_ms': {
                'avg': PRECISION_GUARD_AVG_MS,
                'p95': PRECISION_GUARD_P95_MS,
            },
            'note': reason, 'input_file': input_path, 'output_dir': output_dir
        }
        if gpu_meta:
            result.update(gpu_meta)
        return result





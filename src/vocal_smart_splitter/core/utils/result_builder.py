#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/utils/result_builder.py
# AI-SUMMARY: Helpers to build consistent result dictionaries for split/export flows.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .segment_exporter import ExportResult


class ResultBuilder:
    """Builds result dictionaries with shared fields across split modes."""

    def __init__(self, *, precision_guard_avg_ms: float, precision_guard_p95_ms: float) -> None:
        self._precision_guard_threshold_ms = {
            'avg': float(precision_guard_avg_ms),
            'p95': float(precision_guard_p95_ms),
        }

    def build_base(
        self,
        *,
        method: str,
        export_result: ExportResult,
        export_plan: Sequence[str],
        processing_time: float,
        input_path: str,
        output_dir: str,
        cut_points_samples: Sequence[int],
        cut_points_sec: Sequence[float],
        segment_durations: Sequence[float],
        segment_vocal_flags: Optional[Sequence[bool]],
        precision_guard_ok: bool,
        success: bool = True,
        include_precision_guard_threshold: bool = False,
        guard_shift_stats: Optional[Dict[str, float]] = None,
        guard_adjustments: Optional[List[Dict[str, float]]] = None,
        segment_classification_debug: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        flags = list(segment_vocal_flags or [])
        labels = ['human' if flag else 'music' for flag in flags]

        result: Dict[str, Any] = {
            'success': bool(success),
            'method': method,
            'num_segments': len(segment_durations),
            'saved_files': list(export_result.saved_files),
            'mix_segment_files': list(export_result.mix_segment_files),
            'vocal_segment_files': list(export_result.vocal_segment_files),
            'full_vocal_file': export_result.full_vocal_file,
            'full_instrumental_file': export_result.full_instrumental_file,
            'export_plan': list(export_plan),
            'processing_time': float(processing_time),
            'segment_durations': list(segment_durations),
            'segment_vocal_flags': flags,
            'segment_labels': labels,
            'cut_points_samples': list(cut_points_samples),
            'cut_points_sec': list(cut_points_sec),
            'precision_guard_ok': bool(precision_guard_ok),
            'input_file': input_path,
            'output_dir': output_dir,
        }

        if include_precision_guard_threshold:
            result['precision_guard_threshold_ms'] = dict(self._precision_guard_threshold_ms)

        if guard_shift_stats is not None:
            result['guard_shift_stats'] = guard_shift_stats

        if guard_adjustments is not None:
            result['guard_adjustments'] = guard_adjustments

        if segment_classification_debug is not None:
            result['segment_classification_debug'] = segment_classification_debug

        return result

    def add_separation_metadata(self, result: Dict[str, Any], separation_result: Any) -> Dict[str, Any]:
        if separation_result is None:
            return result
        result['backend_used'] = separation_result.backend_used
        result['separation_confidence'] = separation_result.separation_confidence
        result.update(dict(getattr(separation_result, 'gpu_meta', {}) or {}))
        return result

    def add_hybrid_metadata(
        self,
        result: Dict[str, Any],
        *,
        lib_flags: Sequence[bool],
        hybrid_config: Dict[str, Any],
        beat_analysis: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        result['segment_lib_flags'] = list(lib_flags)
        result['lib_segment_count'] = sum(1 for flag in lib_flags if flag)
        result['hybrid_config'] = dict(hybrid_config)
        if beat_analysis is not None:
            result['beat_analysis'] = beat_analysis
        if strategy is not None:
            result['strategy'] = strategy
        return result

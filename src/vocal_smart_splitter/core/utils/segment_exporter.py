#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/utils/segment_exporter.py
# AI-SUMMARY: Shared export helper for split segments and full-length tracks.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...utils.audio_export import export_audio


@dataclass
class ExportResult:
    saved_files: List[str] = field(default_factory=list)
    mix_segment_files: List[str] = field(default_factory=list)
    vocal_segment_files: List[str] = field(default_factory=list)
    full_vocal_file: Optional[str] = None
    full_instrumental_file: Optional[str] = None


class SegmentExporter:
    """Export split segments with consistent naming and optional lib suffix."""

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate

    def export_segments(
        self,
        segments: Sequence[np.ndarray],
        output_dir: str,
        *,
        segment_is_vocal: Sequence[bool],
        export_format: str,
        export_options: Dict[str, Any],
        lib_flags: Optional[Sequence[bool]] = None,
        lib_suffix: str = "_lib",
        subdir: Optional[str] = None,
        file_suffix: str = "",
        duration_map: Optional[Dict[int, float]] = None,
        index_offset: int = 1,
        always_append_duration: bool = False,
    ) -> List[str]:
        base_dir = Path(output_dir)
        if subdir:
            base_dir = base_dir / subdir
        base_dir.mkdir(parents=True, exist_ok=True)

        saved_files: List[str] = []
        for i, segment_audio in enumerate(segments):
            is_vocal = True
            if i < len(segment_is_vocal):
                is_vocal = bool(segment_is_vocal[i])

            is_lib = False
            if lib_flags is not None and i < len(lib_flags):
                is_lib = bool(lib_flags[i])

            duration_s: Optional[float] = None
            if duration_map is not None and i in duration_map:
                duration_s = max(0.0, float(duration_map[i]))
            elif always_append_duration:
                duration_s = len(segment_audio) / float(self.sample_rate)

            label = "human" if is_vocal else "music"
            suffix = file_suffix
            if duration_s is not None:
                formatted = f"_{duration_s:.1f}"
                suffix = f"{file_suffix}{formatted}" if file_suffix else formatted

            lib_part = lib_suffix if is_lib else ""
            index = i + index_offset
            output_base = base_dir / f"segment_{index:03d}_{label}{lib_part}{suffix}"
            output_path = export_audio(
                segment_audio,
                self.sample_rate,
                output_base,
                export_format,
                options=export_options,
            )
            saved_files.append(str(output_path))

        return saved_files

    def export_full_track(
        self,
        audio: np.ndarray,
        output_base: Path,
        *,
        export_format: str,
        export_options: Dict[str, Any],
    ) -> str:
        output_base.parent.mkdir(parents=True, exist_ok=True)
        output_path = export_audio(
            audio,
            self.sample_rate,
            output_base,
            export_format,
            options=export_options,
        )
        return str(output_path)

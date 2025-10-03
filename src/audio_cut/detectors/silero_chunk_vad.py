#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/detectors/silero_chunk_vad.py
# AI-SUMMARY: 分块 Silero VAD 适配 GPU 流水线，负责局部推理、halo 裁剪与跨块合并。

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from audio_cut.utils.gpu_pipeline import ChunkPlan

logger = logging.getLogger(__name__)

VadFn = Callable[[np.ndarray], Sequence[Dict[str, int]]]


@dataclass
class SileroChunkVAD:
    """增量式 Silero VAD：按 ChunkPlan 推理并输出全局时间轴。"""

    sample_rate: int
    merge_gap_ms: float = 120.0
    focus_pad_s: float = 0.2
    inference_fn: Optional[VadFn] = None

    _segments: List[Tuple[float, float]] = field(default_factory=list, init=False)
    _track_duration_s: float = field(default=0.0, init=False)
    _finalized: Optional[List[Dict[str, float]]] = field(default=None, init=False)

    def _ensure_inference_fn(self) -> Optional[VadFn]:
        if self.inference_fn is not None:
            return self.inference_fn
        try:
            from src.vocal_smart_splitter.core.vocal_pause_detector import (  # noqa: WPS433
                VocalPauseDetectorV2,
            )

            detector = VocalPauseDetectorV2(self.sample_rate)
            self.inference_fn = detector._detect_speech_timestamps  # type: ignore[attr-defined]
            return self.inference_fn
        except Exception as exc:  # pragma: no cover - Silero 依赖可能缺失
            logger.warning("SileroChunkVAD 无法初始化 Silero VAD：%s", exc)
            self.inference_fn = lambda audio: []  # type: ignore[assignment]
            return self.inference_fn

    def process_chunk(self, plan: ChunkPlan, vocal_chunk: np.ndarray, sr: int) -> None:
        """执行当前 chunk 的 VAD，保留有效区间，并缓存至全局时间轴。"""

        if vocal_chunk.size == 0:
            return
        if sr != self.sample_rate:
            raise ValueError(f"SileroChunkVAD sr mismatch: expected {self.sample_rate}, got {sr}")

        infer_fn = self._ensure_inference_fn()
        if infer_fn is None:
            return

        try:
            timestamps = infer_fn(vocal_chunk)
        except Exception as exc:  # pragma: no cover - 运行时异常降级
            logger.error("SileroChunkVAD chunk 推理失败: %s", exc, exc_info=True)
            return

        effective_start = plan.effective_start_s
        effective_end = plan.effective_end_s
        base_time = plan.start_s
        self._track_duration_s = max(self._track_duration_s, float(plan.end_s))

        for ts in timestamps:
            start_sample = int(ts.get("start", 0))
            end_sample = int(ts.get("end", 0))
            if end_sample <= start_sample:
                continue

            start_s = base_time + (start_sample / float(self.sample_rate))
            end_s = base_time + (end_sample / float(self.sample_rate))

            start_s = max(start_s, effective_start)
            end_s = min(end_s, effective_end)
            if end_s - start_s <= 1e-6:
                continue

            self._segments.append((start_s, end_s))

        self._segments.sort(key=lambda item: item[0])
        self._finalized = None

    def _merge_segments(self) -> List[Tuple[float, float]]:
        """Merge raw per-chunk speech spans into a global timeline."""
        if not self._segments:
            return []
        merged: List[Tuple[float, float]] = []
        gap_s = float(self.merge_gap_ms) / 1000.0
        for start_s, end_s in self._segments:
            if end_s <= start_s:
                continue
            if not merged:
                merged.append((start_s, end_s))
                continue
            last_start, last_end = merged[-1]
            if start_s - last_end <= gap_s:
                merged[-1] = (last_start, max(last_end, end_s))
            else:
                merged.append((start_s, end_s))
        return merged

    def finalize(self) -> List[Dict[str, float]]:
        """Finalize speech segments after all chunks have been processed."""
        if self._finalized is None:
            merged = self._merge_segments()
            self._finalized = [
                {
                    'start': float(start),
                    'end': float(end),
                    'duration': float(max(0.0, end - start)),
                }
                for start, end in merged
            ]
        return list(self._finalized or [])

    def to_focus_windows(self, *, pad_s: Optional[float] = None, min_width_s: float = 0.0) -> List[Tuple[float, float]]:
        """Project merged speech spans into padded windows for downstream detectors."""
        segments = self._merge_segments() if self._finalized is None else [
            (float(entry.get('start', 0.0)), float(entry.get('end', 0.0)))
            for entry in self._finalized
        ]
        if not segments:
            return []
        pad = self.focus_pad_s if pad_s is None else float(pad_s)
        pad = max(0.0, float(pad))
        min_width = max(0.0, float(min_width_s))
        track_end = max(self._track_duration_s, max(end for _, end in segments))
        windows: List[Tuple[float, float]] = []
        for start, end in segments:
            left = max(0.0, start - pad)
            right = min(track_end, end + pad)
            if right - left <= 0.0:
                continue
            windows.append((left, right))
        windows.sort(key=lambda item: item[0])
        merged_windows: List[Tuple[float, float]] = []
        for start, end in windows:
            if not merged_windows or start > merged_windows[-1][1]:
                merged_windows.append((start, end))
            else:
                merged_windows[-1] = (merged_windows[-1][0], max(merged_windows[-1][1], end))
        if min_width > 0.0:
            merged_windows = [
                (start, end) for start, end in merged_windows if (end - start) >= min_width
            ]
        return merged_windows

    def build_focus_windows(self) -> List[Tuple[float, float]]:
        """Convert merged speech spans into focus windows with padding."""
        return self.to_focus_windows(pad_s=self.focus_pad_s)



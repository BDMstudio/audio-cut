#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/detectors/energy_gate.py
# AI-SUMMARY: 诊断用能量门控 VAD，提供纯 CPU 兜底检测与简单片段报告，默认不在主流程启用。

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class EnergyGateConfig:
    """Configuration for the diagnostic energy-gate fallback detector."""

    sample_rate: int = 44100
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    threshold_db: float = -38.0
    min_silence_s: float = 0.25
    min_voice_s: float = 0.45


class EnergyGateDetector:
    """Simple energy gate VAD for diagnostic / CPU fallback scenarios.

    The detector intentionally favours deterministic behaviour over accuracy.
    It is not part of the main GPU pipeline and should be used only when
    Silero is unavailable.
    """

    def __init__(self, config: EnergyGateConfig | None = None) -> None:
        self.config = config or EnergyGateConfig()
        self._sr = max(1, int(self.config.sample_rate))
        self._frame = max(1, int(self._sr * (self.config.frame_ms / 1000.0)))
        self._hop = max(1, int(self._sr * (self.config.hop_ms / 1000.0)))
        self._threshold = float(self.config.threshold_db)
        self._min_voice = max(0.0, float(self.config.min_voice_s))
        self._min_silence = max(0.0, float(self.config.min_silence_s))

    def detect_segments(self, waveform: Sequence[float]) -> List[Dict[str, float]]:
        """Return coarse speech segments using RMS energy gating."""

        if waveform is None:
            return []
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.size == 0:
            return []
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        envelope = self._rms_envelope(audio)
        if envelope.size == 0:
            return []

        threshold = self._db_to_linear(self._threshold)
        active = envelope >= threshold

        windows: List[Dict[str, float]] = []
        start_idx = None
        for idx, flag in enumerate(active):
            if flag and start_idx is None:
                start_idx = idx
            elif not flag and start_idx is not None:
                windows.append(self._make_window(start_idx, idx))
                start_idx = None
        if start_idx is not None:
            windows.append(self._make_window(start_idx, len(active)))

        merged: List[Dict[str, float]] = []
        last_end = None
        for window in windows:
            if window["duration"] < self._min_voice:
                continue
            if merged and last_end is not None:
                gap = window["start"] - last_end
                if gap < self._min_silence:
                    previous = merged[-1]
                    previous["end"] = window["end"]
                    previous["duration"] = previous["end"] - previous["start"]
                    last_end = previous["end"]
                    continue
            merged.append(window)
            last_end = window["end"]

        for window in merged:
            window.setdefault("confidence", 0.5)
        return merged

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _rms_envelope(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return np.empty(0, dtype=np.float32)
        if self._frame <= 1:
            return np.abs(audio)
        pad = self._frame - 1
        padded = np.pad(audio, (0, pad), mode="constant")
        squared = padded.astype(np.float32) ** 2
        kernel = np.ones(self._frame, dtype=np.float32)
        conv = np.convolve(squared, kernel, mode="valid")
        rms = np.sqrt(conv / float(self._frame))
        if self._hop == 1:
            return rms
        return rms[:: self._hop]

    def _make_window(self, start_frame: int, end_frame: int) -> Dict[str, float]:
        start_s = (start_frame * self._hop) / float(self._sr)
        end_s = (end_frame * self._hop) / float(self._sr)
        return {
            "start": start_s,
            "end": end_s,
            "duration": max(0.0, end_s - start_s),
        }

    @staticmethod
    def _db_to_linear(db_value: float) -> float:
        return math.pow(10.0, db_value / 20.0)


__all__ = ["EnergyGateDetector", "EnergyGateConfig"]

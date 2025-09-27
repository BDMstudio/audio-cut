# File: tests/unit/test_energy_gate.py
# AI-SUMMARY: Unit tests validating the energy-gate VAD behaviour and threshold handling.

import numpy as np

from src.audio_cut.detectors.energy_gate import detect_activity_segments


def _make_wave(sr: int) -> np.ndarray:
    silence = np.zeros(int(0.4 * sr), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)).astype(np.float32)
    tail = np.zeros(int(0.3 * sr), dtype=np.float32)
    return np.concatenate([silence, tone, tail])


def test_energy_gate_detects_single_segment():
    sr = 16000
    wave = _make_wave(sr)
    segments = detect_activity_segments(wave, sr)
    assert len(segments) == 1
    seg = segments[0]
    start_s = seg['start'] / sr
    end_s = seg['end'] / sr
    assert 0.35 < start_s < 0.5
    assert 0.8 < end_s < 1.0


def test_energy_gate_respects_threshold():
    sr = 16000
    noisy = np.random.normal(scale=0.01, size=sr).astype(np.float32)
    segments = detect_activity_segments(noisy, sr, threshold_db=-20.0)
    assert segments == []

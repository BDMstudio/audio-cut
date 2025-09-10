#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from vocal_smart_splitter.core.vocal_pause_detector import VocalPauseDetectorV2, VocalPause
from vocal_smart_splitter.utils.config_manager import set_runtime_config, reset_runtime_config

def synth(sr=44100):
    t1 = np.arange(int(0.50 * sr)) / sr
    voice1 = 0.5 * np.sin(2 * np.pi * 220.0 * t1)
    breath = 0.03 * np.random.randn(int(0.15 * sr))
    t2 = np.arange(int(0.20 * sr)) / sr
    voice2 = 0.5 * np.sin(2 * np.pi * 220.0 * t2)
    return np.concatenate([voice1, breath, voice2]).astype(np.float32)

sr=44100
y = synth(sr)
set_runtime_config({
    'vocal_pause_splitting.cut_at_speech_end': True,
    'vocal_pause_splitting.max_shift_from_silence_center': 0.06,
    'vocal_pause_splitting.enable_zero_crossing_align': True,
    'vocal_pause_splitting.enable_valley_mode': True,
    'vocal_pause_splitting.auto_valley_fallback': True,
    'vocal_pause_splitting.local_rms_window_ms': 25,
    'vocal_pause_splitting.silence_floor_percentile': 5,
    'vocal_pause_splitting.lookahead_guard_ms': 120,
})
try:
    det = VocalPauseDetectorV2(sample_rate=sr)
    pause = VocalPause(start_time=0.50, end_time=0.65, duration=0.15, position_type='middle', confidence=0.8, cut_point=0.0)
    out = det._calculate_cut_points([pause], bpm_features=None, waveform=y)
    cp = out[0].cut_point
    print(f"cut_point={cp:.6f}")
    ok = (0.50 <= cp <= 0.65) and ((cp - 0.50) >= 0.02) and ((0.65 - cp) >= 0.02)
    if not ok:
        print("FAIL: cut point not in expected breath valley")
        sys.exit(2)
finally:
    reset_runtime_config()


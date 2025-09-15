#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick VPP regression: run PureVocalPauseDetector on a given audio file (raw mix ok),
summarize pause count and segment duration distribution after applying VPP capping.
"""
import os
import sys
import json
import numpy as np
import librosa
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'src'))

from vocal_smart_splitter.core.pure_vocal_pause_detector import PureVocalPauseDetector

def summarize(audio_path: str):
    audio, sr = librosa.load(audio_path, sr=44100, mono=True)
    det = PureVocalPauseDetector(sample_rate=44100)
    pauses = det.detect_pure_vocal_pauses(audio, enable_mdd_enhancement=False, original_audio=audio)
    cuts = [float(getattr(p, 'cut_point', 0.0)) for p in pauses if getattr(p, 'cut_point', None) is not None]
    dur = len(audio)/sr
    cuts = [max(0.0, min(dur - 1e-3, c)) for c in cuts]
    cuts = sorted(set(cuts))
    bounds = [0.0] + cuts + [dur]
    segs = [bounds[i+1]-bounds[i] for i in range(len(bounds)-1)]
    if segs:
        arr = np.array(segs, dtype=float)
        summary = {
            'segments': int(arr.size),
            'min': float(np.min(arr)),
            'p10': float(np.percentile(arr, 10)),
            'median': float(np.median(arr)),
            'mean': float(np.mean(arr)),
            'p90': float(np.percentile(arr, 90)),
            'max': float(np.max(arr)),
        }
    else:
        summary = {'segments': 0, 'min': 0.0, 'p10': 0.0, 'median': 0.0, 'mean': 0.0, 'p90': 0.0, 'max': 0.0}
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else str(project_root / 'input' / 'yesterday once more.MP3')
    summarize(path)


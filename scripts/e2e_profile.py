#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2E profiling (seamless v2.2_mdd) with separation stubbed.
Measures phase durations and reports segment count and duration stats.
"""
import os
import sys
import time
import json
import numpy as np
import librosa
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'src'))

from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from vocal_smart_splitter.core.enhanced_vocal_separator import SeparationResult

def main(audio_path: str):
    sr = 44100
    t0 = time.time()
    audio, file_sr = librosa.load(audio_path, sr=sr, mono=True)
    t_load = time.time() - t0

    # Build splitter and stub separation to return the loaded mix as "vocal"
    splitter = SeamlessSplitter(sample_rate=sr)

    class StubSep:
        def __init__(self, sr):
            self.sample_rate = sr
        def separate_for_detection(self, x):
            return SeparationResult(vocal_track=audio.astype(np.float32),
                                    instrumental_track=None,
                                    separation_confidence=0.5,
                                    backend_used='stub',
                                    processing_time=0.0,
                                    quality_metrics={})

    splitter.separator = StubSep(sr)

    # 1) Detect pauses on vocal (pure vocal detector)
    t1 = time.time()
    sep = splitter.separator.separate_for_detection(audio)
    vocal = sep.vocal_track
    t_sep = time.time() - t1

    t2 = time.time()
    pauses = splitter.pure_vocal_detector.detect_pure_vocal_pauses(vocal, enable_mdd_enhancement=True, original_audio=audio)
    t_detect = time.time() - t2

    # 2) Build cut candidates (time, score)
    cut_candidates = []
    for p in pauses:
        t = float(getattr(p, 'cut_point', (p.start_time + p.end_time) / 2.0))
        s = float(getattr(p, 'confidence', 1.0))
        cut_candidates.append((t, s))

    # 3) Finalize (NMS + guard + governance)
    t3 = time.time()
    boundaries = splitter._finalize_and_filter_cuts_v2(cut_candidates, audio, pure_vocal_audio=vocal)
    t_finalize = time.time() - t3

    # 4) Compute segment durations only (no disk IO)
    segs = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        segs.append((end - start) / sr)

    summary = {
        'segments': len(segs),
        'load_s': round(t_load, 3),
        'separation_s': round(t_sep, 3),
        'detect_s': round(t_detect, 3),
        'finalize_s': round(t_finalize, 3),
        'min': float(np.min(segs)) if segs else 0.0,
        'p10': float(np.percentile(segs, 10)) if segs else 0.0,
        'median': float(np.median(segs)) if segs else 0.0,
        'mean': float(np.mean(segs)) if segs else 0.0,
        'p90': float(np.percentile(segs, 90)) if segs else 0.0,
        'max': float(np.max(segs)) if segs else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    apath = sys.argv[1] if len(sys.argv) > 1 else str(project_root / 'input' / 'yesterday once more.MP3')
    main(apath)


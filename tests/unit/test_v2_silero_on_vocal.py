# File: tests/unit/test_v2_silero_on_vocal.py
# AI-SUMMARY: Unit tests for VocalPauseDetectorV2 covering Silero-monkeypatched behaviour and energy-gate fallback.

import numpy as np

from src.vocal_smart_splitter.core.vocal_pause_detector import VocalPauseDetectorV2


def make_sine(sr: int, dur_s: float, freq: float = 220.0):
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    return 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_v2_detector_on_pure_vocal_with_edge_pause(monkeypatch):
    sr = 16000
    # 构造 4s 音频：
    # 0.0-0.8s 静音（头部静音平台/“edge pause”）
    # 0.8-2.0s 人声
    # 2.0-2.4s 静音（中间停顿）
    # 2.4-3.5s 人声
    # 3.5-4.0s 静音（尾部静音平台）
    silent = np.zeros(int(sr * 0.8), dtype=np.float32)
    voice1 = make_sine(sr, 1.2)
    mid_silent = np.zeros(int(sr * 0.4), dtype=np.float32)
    voice2 = make_sine(sr, 1.1, freq=330.0)
    tail_silent = np.zeros(int(sr * 0.5), dtype=np.float32)
    vocal = np.concatenate([silent, voice1, mid_silent, voice2, tail_silent])

    # 1) 绕过真实 Silero 加载
    monkeypatch.setattr(VocalPauseDetectorV2, "_init_silero_vad", lambda self: None, raising=True)

    # 2) 伪造 _detect_speech_timestamps：直接返回语音区间
    def fake_detect_stamps(self, audio):
        # 0.8-2.0s 与 2.4-3.5s 为语音
        return [
            {"start": int(sr * 0.8), "end": int(sr * 2.0)},
            {"start": int(sr * 2.4), "end": int(sr * 3.5)},
        ]

    monkeypatch.setattr(VocalPauseDetectorV2, "_detect_speech_timestamps", fake_detect_stamps, raising=True)

    det = VocalPauseDetectorV2(sample_rate=sr)
    det.enable_bpm_adaptation = False

    pauses = det.detect_vocal_pauses(vocal)

    # 断言：应至少有一个 cut_point 且位于 (0, duration) 内
    assert len(pauses) >= 1
    dur = len(vocal) / sr
    cut_points = [p.cut_point for p in pauses if getattr(p, "cut_point", 0) > 0]
    assert any(0.0 < cp < dur for cp in cut_points)


def test_vocal_pause_detector_energy_gate_fallback():
    from src.vocal_smart_splitter.utils.config_manager import reset_runtime_config, set_runtime_config

    sr = 16000
    silent = np.zeros(int(sr * 0.5), dtype=np.float32)
    voice = make_sine(sr, 0.8)
    tail = np.zeros(int(sr * 0.4), dtype=np.float32)
    audio = np.concatenate([silent, voice, tail])

    reset_runtime_config()
    set_runtime_config({
        'advanced_vad.use_silero': False,
        'vocal_pause_splitting.enable_bpm_adaptation': False,
        'vocal_pause_splitting.min_pause_duration': 0.2,
    })
    try:
        detector = VocalPauseDetectorV2(sample_rate=sr)
        assert detector.use_silero is False

        pauses = detector.detect_vocal_pauses(audio, context_audio=audio)
        assert pauses, 'energy gate fallback should produce pause segments'
        durations = [p.duration for p in pauses if getattr(p, 'duration', 0) > 0]
        assert any(d > 0.2 for d in durations)
    finally:
        reset_runtime_config()

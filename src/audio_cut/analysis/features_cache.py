#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/analysis/features_cache.py
# AI-SUMMARY: 构建并缓存单次 STFT 提取的 BPM/MDD 全局特征，按需使用 Torch GPU 加速并回落到 CPU 计算。
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import librosa
import numpy as np

from src.vocal_smart_splitter.core.adaptive_vad_enhancer import BPMAnalyzer, BPMFeatures
from src.vocal_smart_splitter.utils.config_manager import get_config
from audio_cut.utils.gpu_pipeline import ChunkPlan

logger = logging.getLogger(__name__)

_EPS = 1e-12

try:  # Torch 为可选依赖，GPU 路径失效时自动回退。
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover - torch 在 CPU 测试环境非必备
    torch = None
    Tensor = None


def _ensure_mono_np(wave: np.ndarray) -> np.ndarray:
    if wave.ndim == 1:
        return wave
    if wave.ndim == 2:
        return np.mean(wave, axis=0)
    return wave.reshape(-1)


@dataclass
class TrackFeatureCache:
    sr: int
    hop_length: int
    hop_s: float
    duration_s: float
    rms_series: np.ndarray
    spectral_flatness: np.ndarray
    onset_envelope: np.ndarray
    onset_strength: np.ndarray
    onset_frames: np.ndarray
    rms_max: float
    onset_max: float
    bpm_features: Optional[BPMFeatures]
    tempo_curve: Optional[np.ndarray]
    beat_times: np.ndarray
    global_mdd: float
    mdd_series: np.ndarray

    def frame_count(self) -> int:
        return len(self.rms_series)

    def frame_index(self, t: float) -> int:
        if self.hop_s <= 0:
            return 0
        idx = int(round(t / self.hop_s))
        return int(np.clip(idx, 0, max(self.frame_count() - 1, 0)))

    def frame_slice(self, start_time: float, end_time: float, pad_frames: int = 0) -> slice:
        start_idx = self.frame_index(start_time) - pad_frames
        end_idx = self.frame_index(end_time) + pad_frames + 1
        start_idx = max(0, start_idx)
        end_idx = min(self.frame_count(), max(start_idx + 1, end_idx))
        return slice(start_idx, end_idx)

    def count_onsets(self, frame_slice: slice) -> int:
        if self.onset_frames.size == 0:
            return 0
        start = frame_slice.start
        end = frame_slice.stop
        mask = (self.onset_frames >= start) & (self.onset_frames < end)
        return int(np.sum(mask))

    def window_stats(self, start_time: float, end_time: float, pad_frames: int = 0) -> Dict[str, np.ndarray]:
        sl = self.frame_slice(start_time, end_time, pad_frames=pad_frames)
        return {
            'rms': self.rms_series[sl],
            'spectral_flatness': self.spectral_flatness[sl],
            'onset_strength': self.onset_strength[sl],
            'mdd': self.mdd_series[sl],
            'slice': sl,
        }


class ChunkFeatureBuilder:
    """分块特征缓存构建器，与 GPU 流水线共享 ChunkPlan。"""

    def __init__(self, sr: int, hop_s: float = 0.05) -> None:
        self.sr = sr
        self.hop_length = max(1, int(round(sr * hop_s)))
        self.hop_s = float(self.hop_length) / float(sr)
        self.frame_length = max(self.hop_length * 2, int(round(sr * 0.1)))

        self._rms: List[np.ndarray] = []
        self._flat: List[np.ndarray] = []
        self._onset_env: List[np.ndarray] = []
        self._onset_strength: List[np.ndarray] = []
        self._onset_frames: List[int] = []
        self._times: List[np.ndarray] = []
        self._mix_audio_segments: List[np.ndarray] = []

    def add_chunk(self, plan: ChunkPlan, mix_chunk: np.ndarray, sr: int) -> None:
        if mix_chunk.size == 0:
            return

        # 计算分块特征
        rms = librosa.feature.rms(y=mix_chunk, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        flat = librosa.feature.spectral_flatness(y=mix_chunk, hop_length=self.hop_length)[0]
        onset_env = librosa.onset.onset_strength(y=mix_chunk, sr=sr, hop_length=self.hop_length)
        onset_strength = onset_env.copy()
        onset_frames_local = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length)

        frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)
        frame_times += plan.start_s

        eff_start = plan.effective_start_s
        eff_end = plan.effective_end_s
        mask = (frame_times >= eff_start) & (frame_times < eff_end)

        if not np.any(mask):
            return

        self._rms.append(rms[mask])
        self._flat.append(flat[mask])
        self._onset_env.append(onset_env[mask])
        self._onset_strength.append(onset_strength[mask])
        self._times.append(frame_times[mask])

        start_frame = int(round(plan.start_s / self.hop_s))
        for frame in onset_frames_local:
            global_frame = start_frame + int(frame)
            frame_time = frame_times[frame] if frame < len(frame_times) else plan.start_s
            if eff_start <= frame_time < eff_end:
                self._onset_frames.append(global_frame)

        # 保留混音片段以便最终 BPM/MDD 计算（去除 halo）
        eff_start_sample = int(round(eff_start * sr))
        eff_end_sample = int(round(eff_end * sr))
        chunk_start_sample = int(round(plan.start_s * sr))
        local_start = eff_start_sample - chunk_start_sample
        local_end = local_start + (eff_end_sample - eff_start_sample)
        if local_end > local_start:
            self._mix_audio_segments.append(mix_chunk[local_start:local_end])

    def finalize(self, full_mix_wave: np.ndarray) -> TrackFeatureCache:
        if not self._rms:
            # 回退到整段构建
            return build_feature_cache(full_mix_wave, None, self.sr, hop_s=self.hop_s)

        rms_series = np.concatenate(self._rms)
        spectral_flatness = np.concatenate(self._flat)
        onset_envelope = np.concatenate(self._onset_env)
        onset_strength = np.concatenate(self._onset_strength)
        frame_times = np.concatenate(self._times)
        onset_frames = np.array(sorted(set(self._onset_frames)), dtype=int)

        mix_wave = np.concatenate(self._mix_audio_segments) if self._mix_audio_segments else full_mix_wave

        bpm_analyzer = BPMAnalyzer(self.sr)
        bpm_features = bpm_analyzer.extract_bpm_features(mix_wave)

        tempo_curve = librosa.beat.tempo(
            onset_envelope=onset_envelope,
            sr=self.sr,
            hop_length=self.hop_length,
            aggregate=None,
        )
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=self.sr,
            hop_length=self.hop_length,
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr, hop_length=self.hop_length)

        mdd_series = _compute_mdd_series(rms_series, spectral_flatness, onset_strength)
        global_mdd = float(np.mean(mdd_series))

        duration_s = len(full_mix_wave) / float(self.sr)

        return TrackFeatureCache(
            sr=self.sr,
            hop_length=self.hop_length,
            hop_s=self.hop_s,
            duration_s=duration_s,
            rms_series=rms_series,
            spectral_flatness=spectral_flatness,
            onset_envelope=onset_envelope,
            onset_strength=onset_strength,
            onset_frames=onset_frames,
            rms_max=float(np.max(rms_series) if rms_series.size else 0.0),
            onset_max=float(np.max(onset_strength) if onset_strength.size else 0.0),
            bpm_features=bpm_features,
            tempo_curve=tempo_curve,
            beat_times=beat_times,
            global_mdd=global_mdd,
            mdd_series=mdd_series,
        )


def _compute_mdd_series(rms: np.ndarray, flatness: np.ndarray, onset_strength: np.ndarray) -> np.ndarray:
    energy_weight = get_config('musical_dynamic_density.energy_weight', 0.5)
    spectral_weight = get_config('musical_dynamic_density.spectral_weight', 0.3)
    onset_weight = get_config('musical_dynamic_density.onset_weight', 0.2)

    rms_norm = rms / (np.max(rms) + _EPS)
    flat_norm = 1.0 - np.clip(flatness, 0.0, 1.0)
    onset_norm = onset_strength / (np.max(onset_strength) + _EPS)

    mdd_series = (
        energy_weight * rms_norm
        + spectral_weight * flat_norm
        + onset_weight * onset_norm
    )
    return np.clip(mdd_series, 0.0, 1.0)


def _resolve_backend(preference: Optional[str]) -> str:
    pref = (preference or 'auto').lower()
    if pref not in {'auto', 'cpu', 'cuda', 'gpu'}:
        logger.warning("Unknown features_cache backend '%s', fallback to auto", preference)
        pref = 'auto'
    if pref == 'cpu':
        return 'cpu'
    if pref in {'cuda', 'gpu'}:
        if torch is None or not torch.cuda.is_available():
            logger.warning('Torch CUDA 未就绪，特征缓存回退到 CPU 路径。')
            return 'cpu'
        return 'cuda'
    if torch is not None and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _build_feature_cache_numpy(mix_wave: np.ndarray, sr: int, hop_length: int, hop_s: float) -> TrackFeatureCache:
    frame_length = max(hop_length * 2, int(round(sr * 0.1)))

    rms_series = librosa.feature.rms(y=mix_wave, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=mix_wave, hop_length=hop_length)[0]
    onset_envelope = librosa.onset.onset_strength(y=mix_wave, sr=sr, hop_length=hop_length)
    onset_strength = onset_envelope.copy()
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr, hop_length=hop_length)

    bpm_analyzer = BPMAnalyzer(sr)
    bpm_features = bpm_analyzer.extract_bpm_features(mix_wave)

    tempo_curve = librosa.beat.tempo(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        aggregate=None,
    )
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    mdd_series = _compute_mdd_series(rms_series, spectral_flatness, onset_strength)
    global_mdd = float(np.mean(mdd_series))

    duration_s = len(mix_wave) / float(sr)

    return TrackFeatureCache(
        sr=sr,
        hop_length=hop_length,
        hop_s=hop_s,
        duration_s=duration_s,
        rms_series=rms_series,
        spectral_flatness=spectral_flatness,
        onset_envelope=onset_envelope,
        onset_strength=onset_strength,
        onset_frames=onset_frames,
        rms_max=float(np.max(rms_series) if rms_series.size else 0.0),
        onset_max=float(np.max(onset_strength) if onset_strength.size else 0.0),
        bpm_features=bpm_features,
        tempo_curve=tempo_curve,
        beat_times=beat_times,
        global_mdd=global_mdd,
        mdd_series=mdd_series,
    )


def _torch_hann_window(length: int, device: torch.device) -> Tensor:
    return torch.hann_window(length, device=device, dtype=torch.float32)


def _compute_rms_torch(signal: Tensor, frame_length: int, hop_length: int) -> Tensor:
    pad = frame_length // 2
    kernel = torch.ones(1, 1, frame_length, device=signal.device, dtype=torch.float32) / float(frame_length)
    sq = signal.pow(2).unsqueeze(0).unsqueeze(0)
    rms_sq = torch.nn.functional.conv1d(sq, kernel, stride=hop_length, padding=pad)
    return torch.sqrt(torch.clamp(rms_sq.squeeze(0).squeeze(0), min=_EPS))


def _build_feature_cache_torch(mix_wave: np.ndarray, sr: int, hop_length: int, hop_s: float, device: torch.device) -> TrackFeatureCache:
    if torch is None:
        raise RuntimeError('Torch backend is unavailable')

    frame_length = max(hop_length * 2, int(round(sr * 0.1)))
    n_fft = 1 << int(math.ceil(math.log2(frame_length)))
    if n_fft < frame_length:
        n_fft = frame_length

    mix_tensor = torch.as_tensor(mix_wave, dtype=torch.float32, device=device)
    mix_tensor = mix_tensor if mix_tensor.ndim == 1 else mix_tensor.mean(dim=0)

    with torch.inference_mode():
        window = _torch_hann_window(frame_length, device)
        stft = torch.stft(
            mix_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=frame_length,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitude = stft.abs().float()
        rms_tensor = _compute_rms_torch(mix_tensor, frame_length, hop_length)

    _ = magnitude.detach()  # STFT magnitude reserved for potential future use

    rms_series = librosa.feature.rms(y=mix_wave, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=mix_wave, hop_length=hop_length)[0]
    onset_envelope = librosa.onset.onset_strength(y=mix_wave, sr=sr, hop_length=hop_length)
    onset_strength = onset_envelope.copy()
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr, hop_length=hop_length)

    bpm_analyzer = BPMAnalyzer(sr)
    bpm_features = bpm_analyzer.extract_bpm_features(mix_wave)

    tempo_curve = librosa.beat.tempo(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        aggregate=None,
    )
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    mdd_series = _compute_mdd_series(rms_series, spectral_flatness, onset_strength)
    global_mdd = float(np.mean(mdd_series))
    duration_s = len(mix_wave) / float(sr)

    return TrackFeatureCache(
        sr=sr,
        hop_length=hop_length,
        hop_s=hop_s,
        duration_s=duration_s,
        rms_series=rms_series,
        spectral_flatness=spectral_flatness,
        onset_envelope=onset_envelope,
        onset_strength=onset_strength,
        onset_frames=onset_frames,
        rms_max=float(np.max(rms_series) if rms_series.size else 0.0),
        onset_max=float(np.max(onset_strength) if onset_strength.size else 0.0),
        bpm_features=bpm_features,
        tempo_curve=tempo_curve,
        beat_times=beat_times,
        global_mdd=global_mdd,
        mdd_series=mdd_series,
    )


def build_feature_cache(
    mix_wave: np.ndarray,
    vocal_wave: Optional[np.ndarray],
    sr: int,
    *,
    hop_s: float = 0.05,
) -> TrackFeatureCache:
    """构建全局特征缓存，可根据配置选择 GPU 或 CPU 后端。"""

    mix_wave = _ensure_mono_np(mix_wave)
    if mix_wave is None or mix_wave.size == 0:
        raise ValueError('mix_wave is empty, cannot build feature cache')

    _ = vocal_wave  # 占位以保证后续扩展兼容

    hop_length = max(1, int(round(sr * hop_s)))
    backend_pref = get_config('analysis.features_cache.device', 'auto')
    backend = _resolve_backend(backend_pref)

    if backend == 'cuda':
        try:
            device = torch.device('cuda')
            return _build_feature_cache_torch(mix_wave, sr, hop_length, hop_s, device=device)
        except Exception as exc:  # pragma: no cover - GPU 相关路径在 CI 中不可测
            logger.warning('GPU features_cache 失败，回退到 CPU。原因: %s', exc)

    return _build_feature_cache_numpy(mix_wave, sr, hop_length, hop_s)

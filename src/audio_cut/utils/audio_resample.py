#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/utils/audio_resample.py
# AI-SUMMARY: Creates ASR-safe 16 kHz mono PCM WAV detection copies without touching export audio.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import soundfile as sf


@dataclass(frozen=True)
class AsrAudioCopy:
    """Metadata for an ASR detection-copy WAV."""

    path: Path
    source_path: Path
    sample_rate: int
    channels: int
    subtype: str
    duration_s: float


def ensure_16k_mono_pcm_wav(input_path: Path | str, output_dir: Path | str) -> AsrAudioCopy:
    """Write a 16 kHz mono PCM_16 WAV copy for ASR workers."""

    source_path = Path(input_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{source_path.stem}_asr_16k_mono.wav"

    audio, sample_rate = librosa.load(str(source_path), sr=16000, mono=True)
    sf.write(target_path, audio, sample_rate, subtype="PCM_16")

    info = sf.info(target_path)
    return AsrAudioCopy(
        path=target_path,
        source_path=source_path,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        subtype=str(info.subtype),
        duration_s=float(info.duration),
    )

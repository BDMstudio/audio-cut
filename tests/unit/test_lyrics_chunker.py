#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_lyrics_chunker.py
# AI-SUMMARY: Tests ASR-safe audio copies, lyrics chunk planning and cache key stability.

from pathlib import Path

import numpy as np
import soundfile as sf

from audio_cut.lyrics.cache import build_lyrics_cache_key
from audio_cut.lyrics.chunker import plan_asr_chunks
from audio_cut.utils.audio_resample import ensure_16k_mono_pcm_wav


def test_ensure_16k_mono_pcm_wav_writes_detection_copy(tmp_path: Path) -> None:
    input_path = tmp_path / "stereo.wav"
    intermediate_dir = tmp_path / "intermediate"
    sr = 44100
    t = np.linspace(0.0, 0.25, int(sr * 0.25), endpoint=False)
    stereo = np.stack(
        [
            np.sin(2.0 * np.pi * 220.0 * t),
            np.sin(2.0 * np.pi * 330.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    sf.write(input_path, stereo, sr, subtype="PCM_24")

    prepared = ensure_16k_mono_pcm_wav(input_path, intermediate_dir)

    assert prepared.path.parent == intermediate_dir
    assert prepared.path != input_path
    assert input_path.exists()

    info = sf.info(prepared.path)
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert prepared.sample_rate == 16000
    assert prepared.channels == 1


def test_plan_asr_chunks_uses_default_overlap_and_metadata(tmp_path: Path) -> None:
    chunks = plan_asr_chunks(
        tmp_path / "vocal_asr.wav",
        duration_s=95.0,
        output_dir=tmp_path / "chunks",
    )

    assert [(chunk.global_t0, chunk.global_t1) for chunk in chunks] == [
        (0.0, 35.0),
        (34.0, 69.0),
        (68.0, 95.0),
    ]
    assert [chunk.chunk_id for chunk in chunks] == [0, 1, 2]
    assert chunks[0].path.name == "vocal_asr_chunk_000.wav"
    assert chunks[-1].duration_s == 27.0


def test_plan_asr_chunks_enforces_max_chunk_and_firered_limit(tmp_path: Path) -> None:
    chunks = plan_asr_chunks(
        tmp_path / "vocal_asr.wav",
        duration_s=130.0,
        output_dir=tmp_path / "chunks",
        chunk_s=90.0,
        overlap_s=2.0,
        max_chunk_s=55.0,
    )

    assert all(chunk.duration_s <= 55.0 for chunk in chunks)
    assert chunks[1].global_t0 == 53.0


def test_lyrics_cache_key_changes_with_audio_and_config(tmp_path: Path) -> None:
    audio_path = tmp_path / "vocal.wav"
    audio_path.write_bytes(b"audio-a")

    key_a = build_lyrics_cache_key(
        audio_path=audio_path,
        separator="mdx23:Kim_Vocal_1.onnx",
        mode="vpbd_asr",
        provider="fake",
        provider_version="fixture-v1",
        chunk_s=35.0,
        overlap_s=1.0,
        scorer_config={"asr_gap": 0.25},
        planner_config={"hard_min_s": 2.0},
    )
    key_b = build_lyrics_cache_key(
        audio_path=audio_path,
        separator="mdx23:Kim_Vocal_1.onnx",
        mode="vpbd_asr",
        provider="fake",
        provider_version="fixture-v1",
        chunk_s=30.0,
        overlap_s=1.0,
        scorer_config={"asr_gap": 0.25},
        planner_config={"hard_min_s": 2.0},
    )

    assert key_a.startswith("lyrics:")
    assert key_a == build_lyrics_cache_key(
        audio_path=audio_path,
        separator="mdx23:Kim_Vocal_1.onnx",
        mode="vpbd_asr",
        provider="fake",
        provider_version="fixture-v1",
        chunk_s=35.0,
        overlap_s=1.0,
        scorer_config={"asr_gap": 0.25},
        planner_config={"hard_min_s": 2.0},
    )
    assert key_a != key_b

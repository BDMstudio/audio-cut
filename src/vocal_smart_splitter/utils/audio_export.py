#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/utils/audio_export.py
# AI-SUMMARY: 提供音频导出工具，支持多种格式写入并统一扩展点。

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import soundfile as sf

try:
    from pydub import AudioSegment  # type: ignore
except ImportError:  # pragma: no cover - 运行时检查
    AudioSegment = None  # type: ignore

logger = logging.getLogger(__name__)


AudioWriter = Callable[[Path, np.ndarray, int, Dict[str, object]], None]


@dataclass(frozen=True)
class AudioExportFormat:
    """音频导出格式描述"""

    name: str
    extension: str
    description: str
    writer: AudioWriter
    defaults: Dict[str, object] = field(default_factory=dict)


_FORMATS: Dict[str, AudioExportFormat] = {}


def register_format(format_info: AudioExportFormat) -> None:
    key = format_info.name.lower()
    if key in _FORMATS:
        raise ValueError(f"音频导出格式已注册: {format_info.name}")
    _FORMATS[key] = format_info


def ensure_supported_format(name: Optional[str]) -> str:
    if not name:
        raise ValueError("未指定导出格式")
    key = name.lower()
    if key not in _FORMATS:
        raise ValueError(f"不支持的导出格式: {name}")
    return key


def get_supported_formats() -> List[AudioExportFormat]:
    return sorted(_FORMATS.values(), key=lambda fmt: fmt.name)


def build_export_options(format_name: str, *overrides: Optional[Dict[str, object]]) -> Dict[str, object]:
    fmt = _FORMATS[ensure_supported_format(format_name)]
    options: Dict[str, object] = dict(fmt.defaults)
    for override in overrides:
        if override:
            options.update(override)
    return options


def export_audio(
    audio: np.ndarray,
    sample_rate: int,
    base_path: Path,
    format_name: str,
    *,
    options: Optional[Dict[str, object]] = None,
) -> Path:
    """将音频写入指定格式，并返回最终输出路径。"""

    key = ensure_supported_format(format_name)
    fmt = _FORMATS[key]
    if base_path.suffix:
        export_path = base_path.parent / f"{base_path.name}.{fmt.extension}"
    else:
        export_path = base_path.with_suffix(f".{fmt.extension}")
    export_options = build_export_options(key, options)
    fmt.writer(export_path, audio, sample_rate, export_options)
    return export_path


# --- 写入实现 -----------------------------------------------------------------

def _prepare_audio_array(audio: np.ndarray) -> np.ndarray:
    """归一化并确保为 (num_samples, num_channels) 的连续数组。"""
    array = np.asarray(audio)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim == 2:
        # 如果检测到通道在第一维（如 (channels, samples)），进行转置
        if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            array = array.T
        if array.shape[1] == 1 and array.shape[0] > 1:
            array = array.reshape(-1, 1)
    else:
        raise ValueError(f"暂不支持 {array.ndim} 维音频数组")
    return np.ascontiguousarray(array)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int, options: Dict[str, object]) -> None:
    subtype = str(options.get("subtype", "PCM_24"))
    sf.write(path, audio, sample_rate, subtype=subtype)


def _write_mp3(path: Path, audio: np.ndarray, sample_rate: int, options: Dict[str, object]) -> None:
    if AudioSegment is None:
        raise RuntimeError("未安装 pydub，无法导出 MP3，请先安装并配置 FFmpeg。")

    if not getattr(AudioSegment, "converter", None):
        raise RuntimeError("pydub 未找到 FFmpeg/avlib，可通过设置 AudioSegment.converter 指向 ffmpeg 可执行文件解决。")

    array = _prepare_audio_array(audio).astype(np.float32)
    array = np.clip(array, -1.0, 1.0)
    pcm16 = np.round(array * 32767.0).astype(np.int16)
    channels = pcm16.shape[1]
    frame_data = pcm16.reshape(-1).tobytes()

    segment = AudioSegment(
        data=frame_data,
        sample_width=2,
        frame_rate=sample_rate,
        channels=channels,
    )
    bitrate = str(options.get("bitrate", "320k"))
    segment.export(path, format="mp3", bitrate=bitrate)


# 注册默认格式
register_format(
    AudioExportFormat(
        name="wav",
        extension="wav",
        description="无损 PCM 24-bit WAV",
        writer=_write_wav,
        defaults={"subtype": "PCM_24"},
    )
)

register_format(
    AudioExportFormat(
        name="mp3",
        extension="mp3",
        description="有损 MP3（默认 320kbps）",
        writer=_write_mp3,
        defaults={"bitrate": "320k"},
    )
)


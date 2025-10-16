#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/api.py
# AI-SUMMARY: 提供 audio-cut 对外统一 API，封装 SeamlessSplitter，输出标准化 Manifest 供上层模块调用。

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

try:  # soundfile 对部分格式（如 mp3）可能不可用，运行时回退
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sf = None  # type: ignore

from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from vocal_smart_splitter.utils.config_manager import get_config_manager

logger = logging.getLogger(__name__)


def separate_and_segment(
    *,
    input_uri: str,
    export_dir: str,
    mode: str = 'v2.2_mdd',
    device: Optional[str] = None,
    export_types: Optional[Sequence[str]] = None,
    layout: Optional[Mapping[str, Any]] = None,
    strict_gpu: Optional[bool] = None,
    export_manifest: bool = False,
    manifest_filename: str = 'SegmentManifest.json',
    runtime_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    对外统一入口：执行声部分离 + 片段切分 + 布局精炼，并产出标准 Manifest。

    Args:
        input_uri: 待处理音频路径。
        export_dir: 输出目录（函数内部会创建）。
        mode: 处理模式，默认 `v2.2_mdd`。
        device: GPU 设备标识（如 `cuda:0` / `cpu`），为空沿用配置文件。
        export_types: 导出类别控制，例如 `("vocal","human_segments","music_segments")`。
        layout: 布局参数覆盖，如 `{"micro_merge_s":2.0,"soft_min_s":6.0}`。
        strict_gpu: 是否启用 strict GPU 模式。
        export_manifest: 是否落盘 Manifest JSON。
        manifest_filename: Manifest 文件名（位于 `export_dir` 下）。
        runtime_overrides: 其他临时配置覆盖（key 使用 `.` 分隔，例如 `{"audio.sample_rate":48000}`）。

    Returns:
        结构化 Manifest 字典，可直接写入 JSON。
    """

    input_path = Path(input_uri).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input audio not found: {input_path}")

    export_path = Path(export_dir).expanduser().resolve()
    export_path.mkdir(parents=True, exist_ok=True)

    cfg_manager = get_config_manager()
    config_snapshot = copy.deepcopy(cfg_manager.config)

    try:
        _apply_runtime_overrides(
            cfg_manager.config,
            device=device,
            strict_gpu=strict_gpu,
            layout=layout,
            runtime_overrides=runtime_overrides,
        )

        layout_cfg_snapshot = copy.deepcopy(_get_nested(cfg_manager.config, ('segment_layout',), default={})) or {}
        sample_rate = int(_get_nested(cfg_manager.config, ('audio', 'sample_rate'), default=44100))
        audio_channels = int(_get_nested(cfg_manager.config, ('audio', 'channels'), default=1))

        splitter = SeamlessSplitter(sample_rate=sample_rate)
        export_plan = list(export_types) if export_types is not None else None
        result = splitter.split_audio_seamlessly(
            str(input_path),
            str(export_path),
            mode=mode,
            export_plan=export_plan,
        )
    finally:
        cfg_manager.config = config_snapshot

    manifest = _build_manifest(
        result=result,
        input_path=input_path,
        export_dir=export_path,
        mode=mode,
        sample_rate=sample_rate,
        channels=audio_channels,
        layout_cfg=layout_cfg_snapshot,
    )

    if export_manifest:
        manifest_path = export_path / manifest_filename
        with manifest_path.open('w', encoding='utf-8') as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        manifest['manifest_path'] = manifest_path.as_posix()

    return manifest


def _apply_runtime_overrides(
    config: MutableMapping[str, Any],
    *,
    device: Optional[str],
    strict_gpu: Optional[bool],
    layout: Optional[Mapping[str, Any]],
    runtime_overrides: Optional[Mapping[str, Any]],
) -> None:
    if device:
        _set_nested(config, ('gpu_pipeline', 'prefer_device'), device)

    if strict_gpu is not None:
        _set_nested(config, ('gpu_pipeline', 'strict_gpu'), bool(strict_gpu))

    if layout:
        layout_dict = dict(layout)
        enable = layout_dict.pop('enable', True)
        _set_nested(config, ('segment_layout', 'enable'), bool(enable))
        for key, value in layout_dict.items():
            _set_nested(config, ('segment_layout', str(key)), value)

    if runtime_overrides:
        for dotted_path, value in runtime_overrides.items():
            if not dotted_path:
                continue
            path_tuple = tuple(part for part in str(dotted_path).split('.') if part)
            if not path_tuple:
                continue
            _set_nested(config, path_tuple, value)


def _build_manifest(
    *,
    result: Dict[str, Any],
    input_path: Path,
    export_dir: Path,
    mode: str,
    sample_rate: int,
    channels: int,
    layout_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    success = bool(result.get('success', False))
    export_plan = result.get('export_plan') or []

    duration = _estimate_duration(result, input_path, sample_rate)
    audio_hash = f"sha256:{_compute_sha256(input_path)}"

    manifest: Dict[str, Any] = {
        'version': str(mode),
        'success': success,
        'job': {'source': input_path.as_posix()},
        'export_plan': export_plan,
        'audio': {
            'sr': sample_rate,
            'channels': channels,
            'duration': duration,
            'hash': audio_hash,
        },
        'layout_cfg': dict(layout_cfg) | {'applied': bool(result.get('segment_layout_applied', False))},
        'cuts': {
            'final': result.get('cut_points_sec', []),
            'samples': result.get('cut_points_samples', []),
            'suppressed': result.get('suppressed_cut_points_sec', []),
        },
        'segments': _build_segments(result, export_dir),
        'artifacts': _collect_artifacts(result, export_dir),
        'guard': {
            'shift_stats': result.get('guard_shift_stats', {}),
            'adjustments': result.get('guard_adjustments', []),
            'precision_ok': bool(result.get('precision_guard_ok', True)),
            'threshold_ms': result.get('precision_guard_threshold_ms', {}),
        },
        'separation': {
            'backend': result.get('backend_used'),
            'confidence': result.get('separation_confidence'),
        },
        'timings_ms': {
            'total': _to_milliseconds(result.get('processing_time')),
        },
        'stats': {
            'num_segments': int(result.get('num_segments', 0)),
        },
    }

    note = result.get('note')
    if note:
        manifest['note'] = note

    gpu_meta = {k: result[k] for k in result.keys() if k.startswith('gpu_pipeline_')}
    if gpu_meta:
        manifest['gpu'] = gpu_meta

    return manifest


def _build_segments(result: Mapping[str, Any], export_dir: Path) -> list[Dict[str, Any]]:
    cut_points = list(result.get('cut_points_sec', []))
    labels = list(result.get('segment_labels', []))
    durations = list(result.get('segment_durations', []))
    mix_files = list(result.get('mix_segment_files', []))
    vocal_files = list(result.get('vocal_segment_files', []))
    debug_info = list(result.get('segment_classification_debug', []))

    segments: list[Dict[str, Any]] = []
    for idx, label in enumerate(labels):
        start = cut_points[idx] if idx < len(cut_points) else None
        end = cut_points[idx + 1] if idx + 1 < len(cut_points) else None

        if start is None:
            start = sum(durations[:idx])
        if end is None and start is not None:
            end = start + (durations[idx] if idx < len(durations) else 0.0)

        segment_entry: Dict[str, Any] = {
            'id': f"{idx + 1:04d}",
            'start': start,
            'end': end,
            'duration': durations[idx] if idx < len(durations) else (end - start if end is not None and start is not None else None),
            'label': label,
        }

        if idx < len(mix_files):
            segment_entry['mix_path'] = _to_relative_path(mix_files[idx], export_dir)
        if idx < len(vocal_files):
            segment_entry['vocal_path'] = _to_relative_path(vocal_files[idx], export_dir)
        if idx < len(debug_info) and debug_info[idx]:
            segment_entry['debug'] = debug_info[idx]

        segments.append(segment_entry)

    return segments


def _collect_artifacts(result: Mapping[str, Any], export_dir: Path) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}

    mix_segment_files = result.get('mix_segment_files') or []
    if mix_segment_files:
        artifacts['music_segments'] = [_to_relative_path(path, export_dir) for path in mix_segment_files]

    vocal_segment_files = result.get('vocal_segment_files') or []
    if vocal_segment_files:
        artifacts['human_segments'] = [_to_relative_path(path, export_dir) for path in vocal_segment_files]

    if result.get('full_vocal_file'):
        artifacts['vocal_full'] = _to_relative_path(result['full_vocal_file'], export_dir)
    if result.get('full_instrumental_file'):
        artifacts['instrumental_full'] = _to_relative_path(result['full_instrumental_file'], export_dir)

    saved_files = result.get('saved_files') or []
    if saved_files:
        artifacts['all'] = [_to_relative_path(path, export_dir) for path in saved_files]

    artifacts['output_dir'] = export_dir.as_posix()
    return artifacts


def _estimate_duration(result: Mapping[str, Any], input_path: Path, sample_rate: int) -> Optional[float]:
    cut_points = result.get('cut_points_sec')
    if cut_points:
        try:
            return float(cut_points[-1])
        except Exception:
            pass

    if sf is not None:
        try:
            info = sf.info(str(input_path))
            if info.frames and info.samplerate:
                return float(info.frames) / float(info.samplerate)
        except Exception:  # pragma: no cover - 格式不受支持时回退
            logger.debug("soundfile info failed for %s", input_path, exc_info=True)

    processing_time = result.get('segment_durations')
    if processing_time:
        try:
            return float(sum(processing_time))
        except Exception:
            pass

    return None


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _to_milliseconds(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(round(float(value) * 1000.0))
    except Exception:
        return None


def _to_relative_path(path_value: Any, base_dir: Path) -> Optional[str]:
    if not path_value:
        return None
    path = Path(str(path_value))
    try:
        rel = path.resolve().relative_to(base_dir)
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def _set_nested(mapping: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    if not path:
        return
    cursor: MutableMapping[str, Any] = mapping
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], MutableMapping):
            cursor[key] = {}
        cursor = cursor[key]  # type: ignore[assignment]
    cursor[path[-1]] = value


def _get_nested(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cursor: Any = mapping
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


__all__ = ['separate_and_segment']

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/bench/run_multi_gpu_probe.py
# AI-SUMMARY: 逐卡运行分割流程，采集 GPU 指标与运行结果并生成 JSON 报告，辅助验证多 GPU 部署。

"""Multi-GPU probe script for the audio-cut pipeline.

The script enumerates selected CUDA devices, runs the SeamlessSplitter once per
card (or on CPU when GPUs are unavailable) and collects the GPU telemetry
captured by the pipeline. The resulting JSON report can be attached to PRs or
used to compare utilisation/peak memory across devices.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch 未安装
    torch = None  # type: ignore

from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from src.vocal_smart_splitter.utils.config_manager import (
    get_config,
    reset_runtime_config,
    set_runtime_config,
)

logger = logging.getLogger("multi_gpu_probe")


def _available_cuda_devices() -> List[str]:
    if torch is None or not torch.cuda.is_available():
        return []
    try:
        count = torch.cuda.device_count()
    except Exception:  # pragma: no cover - CUDA 查询失败
        return []
    return [f"cuda:{idx}" for idx in range(count)]


def _parse_device_list(token: str) -> List[str]:
    devices: List[str] = []
    for part in token.split(","):
        name = part.strip()
        if not name:
            continue
        lowered = name.lower()
        if lowered in {"cpu", "none"}:
            devices.append("cpu")
        else:
            if lowered.startswith("cuda:") or lowered.startswith("gpu:"):
                _, _, suffix = lowered.partition(":")
            else:
                suffix = lowered
            try:
                index = int(suffix)
            except ValueError:
                raise ValueError(f"无法解析设备编号: {name!r}") from None
            devices.append(f"cuda:{index}")
    if not devices:
        raise ValueError("设备列表为空")
    return devices


def _resolve_devices(args: argparse.Namespace) -> List[str]:
    if args.devices:
        return _parse_device_list(args.devices)
    cuda_devices = _available_cuda_devices()
    if cuda_devices:
        return cuda_devices
    return ["cpu"]


def _collect_gpu_meta(result: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in result.items() if str(k).startswith("gpu_pipeline")}


def run_probe(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = (args.output_root / f"multi_gpu_probe_{timestamp}").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    devices = _resolve_devices(args)
    sample_rate = int(get_config('audio.sample_rate', 44100))

    report: Dict[str, object] = {
        "generated_at": timestamp,
        "input_file": str(input_path),
        "mode": args.mode,
        "strict_gpu": bool(args.strict_gpu),
        "devices": [],
    }

    for device in devices:
        logger.info("[multi-gpu] 处理设备 %s", device)
        reset_runtime_config()
        overrides = {'gpu_pipeline.prefer_device': device}
        if args.strict_gpu:
            overrides['gpu_pipeline.strict_gpu'] = True
        set_runtime_config(overrides)

        splitter = SeamlessSplitter(sample_rate=sample_rate)
        device_dir = base_dir / device.replace(':', '_')
        device_dir.mkdir(parents=True, exist_ok=True)

        entry: Dict[str, object] = {
            "device": device,
            "output_dir": str(device_dir),
            "success": False,
            "gpu_meta": {},
        }

        start = time.perf_counter()
        try:
            result = splitter.split_audio_seamlessly(str(input_path), str(device_dir), mode=args.mode)
        except Exception as exc:  # pragma: no cover - 捕获运行时异常
            logger.error("[multi-gpu] 设备 %s 运行失败: %s", device, exc, exc_info=True)
            entry["error"] = str(exc)
        else:
            entry["success"] = bool(result.get("success", True))
            entry["num_segments"] = result.get("num_segments")
            entry["saved_files"] = result.get("saved_files", [])
            entry["gpu_meta"] = _collect_gpu_meta(result)
        finally:
            entry["processing_time_s"] = time.perf_counter() - start
            report["devices"].append(entry)

    reset_runtime_config()

    json_path = base_dir / "multi_gpu_probe.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info("[multi-gpu] 结果已写入 %s", json_path)
    return json_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="逐卡执行 SeamlessSplitter 并收集 GPU 指标",
    )
    parser.add_argument('input', type=Path, help='输入音频文件路径')
    parser.add_argument('--mode', choices=['vocal_separation', 'v2.2_mdd'], default='v2.2_mdd', help='运行模式，默认 v2.2_mdd')
    parser.add_argument('--devices', help='指定设备列表，例如 "0,1" 或 "cpu"，默认枚举全部 CUDA 设备')
    parser.add_argument('--strict-gpu', action='store_true', help='开启 strict GPU 模式，GPU 失败将直接抛出')
    parser.add_argument('--output-root', type=Path, default=Path('output/bench'), help='结果输出根目录 (默认: output/bench)')
    parser.add_argument('--log-level', default='INFO', help='日志级别 (默认: INFO)')

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        json_path = run_probe(args)
    except Exception as exc:
        logger.error("[multi-gpu] 运行失败: %s", exc, exc_info=True)
        return 1

    print(json_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())

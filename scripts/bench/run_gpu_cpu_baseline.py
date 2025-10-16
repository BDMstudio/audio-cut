#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU vs CPU baseline runner.

Runs the SeamlessSplitter pipeline twice (GPU + CPU) on the provided inputs and
records throughput, memory and transfer timings. Results are emitted as JSON and
optional Markdown tables under output/bench.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import soundfile as sf

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from src.vocal_smart_splitter.utils.config_manager import (
    get_config,
    reset_runtime_config,
    set_runtime_config,
)


@dataclass
class RunMetrics:
    mode: str
    processing_time_s: float
    audio_duration_s: float
    throughput_ratio: float
    segment_count: int
    gpu_meta: Dict[str, float]

    @property
    def has_gpu_metrics(self) -> bool:
        return bool(self.gpu_meta.get("gpu_pipeline_used"))

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        payload["gpu_meta"] = self.gpu_meta
        return payload


def _timed_split(input_path: Path, *, enable_gpu: bool, output_root: Path) -> RunMetrics:
    reset_runtime_config()

    set_runtime_config({'gpu_pipeline.enable': bool(enable_gpu)})
    if not enable_gpu:
        # ensure deterministic CPU path
        set_runtime_config({'gpu_pipeline.use_cuda_streams': False})

    sample_rate = get_config('audio.sample_rate', 44100)
    splitter = SeamlessSplitter(sample_rate=sample_rate)

    run_dir = output_root / ('gpu' if enable_gpu else 'cpu')
    run_dir.mkdir(parents=True, exist_ok=True)

    start_ts = time.perf_counter()
    result = splitter.split_audio_seamlessly(
        str(input_path),
        str(run_dir),
        mode='v2.2_mdd',
    )
    elapsed = time.perf_counter() - start_ts

    if not result.get('success'):
        raise RuntimeError(f"Split failed for {input_path}: {result.get('error')}")

    durations = result.get('segment_durations') or []
    if durations:
        audio_dur = float(sum(durations))
    else:
        info = sf.info(str(input_path))
        audio_dur = float(info.duration)

    throughput = audio_dur / elapsed if elapsed > 0 else 0.0
    gpu_meta = {k: result.get(k) for k in result.keys() if k.startswith('gpu_pipeline')}
    gpu_meta.update(result.get('gpu_meta', {}))

    return RunMetrics(
        mode='gpu' if enable_gpu else 'cpu',
        processing_time_s=float(elapsed),
        audio_duration_s=float(audio_dur),
        throughput_ratio=float(throughput),
        segment_count=int(result.get('num_segments', 0)),
        gpu_meta={k: float(v) if isinstance(v, (int, float)) else v for k, v in gpu_meta.items()},
    )


def _resolve_inputs(paths: Iterable[str]) -> List[Path]:
    resolved: List[Path] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            for file in sorted(p.iterdir()):
                if file.suffix.lower() in {'.mp3', '.wav', '.flac', '.m4a'}:
                    resolved.append(file)
        elif p.is_file():
            resolved.append(p)
    if not resolved:
        raise FileNotFoundError('No audio files found for benchmarking')
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description='GPU vs CPU baseline runner')
    parser.add_argument('inputs', nargs='+', help='Audio files or directories')
    parser.add_argument('--output', default='output/bench', help='Output directory for reports')
    parser.add_argument('--write-markdown', action='store_true', help='Emit Markdown summary alongside JSON')
    args = parser.parse_args()

    inputs = _resolve_inputs(args.inputs)
    output_root = Path(args.output).resolve()
    run_id = time.strftime('baseline_%Y%m%d_%H%M%S')
    benchmark_dir = output_root / run_id

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for audio_path in inputs:
        item_dir = benchmark_dir / audio_path.stem
        gpu_metrics = _timed_split(audio_path, enable_gpu=True, output_root=item_dir)
        cpu_metrics = _timed_split(audio_path, enable_gpu=False, output_root=item_dir)

        improvement = 0.0
        if gpu_metrics.processing_time_s and cpu_metrics.processing_time_s:
            improvement = 1.0 - (gpu_metrics.processing_time_s / cpu_metrics.processing_time_s)

        entry = {
            'gpu': gpu_metrics.to_dict(),
            'cpu': cpu_metrics.to_dict(),
            'summary': {
                'speedup_ratio': float(improvement),
                'meets_target': bool(improvement >= 0.30),
            },
        }
        summary[audio_path.name] = entry

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    json_path = benchmark_dir / 'gpu_cpu_baseline.json'
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    if args.write_markdown:
        md_lines = ['| file | cpu_time_s | gpu_time_s | throughput_cpu | throughput_gpu | speedup | meets_target |',
                    '| --- | --- | --- | --- | --- | --- | --- |']
        for name, entry in summary.items():
            cpu = entry['cpu']
            gpu = entry['gpu']
            speed = entry['summary']['speedup_ratio']
            meets = '✅' if entry['summary']['meets_target'] else '⚠️'
            md_lines.append(
                f"| {name} | {cpu['processing_time_s']:.2f} | {gpu['processing_time_s']:.2f} | "
                f"{cpu['throughput_ratio']:.2f} | {gpu['throughput_ratio']:.2f} | {speed:.2%} | {meets} |"
            )
        (benchmark_dir / 'gpu_cpu_baseline.md').write_text('\n'.join(md_lines), encoding='utf-8')

    print(f'Report written to {json_path}')


if __name__ == '__main__':
    main()

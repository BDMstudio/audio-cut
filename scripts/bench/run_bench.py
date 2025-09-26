#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/bench/run_bench.py
# AI-SUMMARY: 批量运行 SeamlessSplitter 并统计耗时、片段分布与守卫位移的基准脚本，可生成/校验质量护栏。
import argparse
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - 可选依赖
    psutil = None  # type: ignore

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from src.vocal_smart_splitter.utils.config_manager import get_config


@dataclass
class FileBenchmark:
    path: str
    duration_s: float
    processing_time_s: float
    segment_count: int
    long_segment_ratio: float
    short_segment_ratio: float
    guard_avg_shift_ms: float
    guard_max_shift_ms: float
    guard_avg_guard_only_ms: float
    guard_avg_vocal_shift_ms: float
    guard_avg_mix_shift_ms: float
    success: bool
    notes: Optional[str] = None

    def to_summary(self) -> Dict[str, float]:
        payload = asdict(self)
        return payload


def iter_audio_files(input_dir: Path, patterns: Optional[List[str]] = None) -> List[Path]:
    if patterns:
        patterns = [p.lower() for p in patterns]
    supported_suffix = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'}
    files: List[Path] = []
    for path in sorted(input_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if patterns and suffix not in patterns:
            continue
        if suffix in supported_suffix:
            files.append(path)
    return files


def load_manifest(manifest_path: Path) -> List[Path]:
    files: List[Path] = []
    for line in manifest_path.read_text(encoding='utf-8').splitlines():
        candidate = Path(line.strip())
        if not candidate:
            continue
        if not candidate.is_file():
            logging.warning("跳过不存在的文件: %s", candidate)
            continue
        files.append(candidate)
    return files


def get_audio_duration(path: Path) -> float:
    with sf.SoundFile(str(path)) as handle:
        frames = len(handle)
        sr = handle.samplerate
    return frames / float(sr) if sr else 0.0


def summarize_guard_stats(stats: Optional[Dict[str, float]]) -> Dict[str, float]:
    defaults = {
        'guard_avg_shift_ms': 0.0,
        'guard_max_shift_ms': 0.0,
        'guard_avg_guard_only_ms': 0.0,
        'guard_avg_vocal_shift_ms': 0.0,
        'guard_avg_mix_shift_ms': 0.0,
    }
    if not stats:
        return defaults
    return {
        'guard_avg_shift_ms': float(stats.get('avg_shift_ms', 0.0)),
        'guard_max_shift_ms': float(stats.get('max_shift_ms', 0.0)),
        'guard_avg_guard_only_ms': float(stats.get('avg_guard_only_shift_ms', 0.0)),
        'guard_avg_vocal_shift_ms': float(stats.get('avg_vocal_guard_shift_ms', 0.0)),
        'guard_avg_mix_shift_ms': float(stats.get('avg_mix_guard_shift_ms', 0.0)),
    }


def weighted_average(values: List[float], weights: List[int]) -> float:
    if not values:
        return 0.0
    total_weight = sum(weights)
    if total_weight <= 0:
        return float(sum(values) / len(values))
    return float(sum(v * w for v, w in zip(values, weights)) / total_weight)


def compute_guard_metrics(
    rows: List[FileBenchmark],
    total_processing: float,
    total_audio: float,
    peak_mem_bytes: Optional[float],
) -> Dict[str, float]:
    total_files = len(rows)
    success_rows = [r for r in rows if r.success]
    success_count = len(success_rows)

    total_segments = sum(r.segment_count for r in success_rows)
    total_long_segments = sum(r.segment_count * r.long_segment_ratio for r in success_rows)
    total_short_segments = sum(r.segment_count * r.short_segment_ratio for r in success_rows)

    seg_weights = [max(r.segment_count, 1) for r in success_rows]

    avg_guard_shift = weighted_average([r.guard_avg_shift_ms for r in success_rows], seg_weights)
    avg_guard_only = weighted_average([r.guard_avg_guard_only_ms for r in success_rows], seg_weights)
    avg_guard_vocal = weighted_average([r.guard_avg_vocal_shift_ms for r in success_rows], seg_weights)
    avg_guard_mix = weighted_average([r.guard_avg_mix_shift_ms for r in success_rows], seg_weights)
    max_guard_shift = max((r.guard_max_shift_ms for r in success_rows), default=0.0)

    realtime_factor = (total_audio / total_processing) if total_processing else 0.0
    avg_processing = (total_processing / success_count) if success_count else 0.0
    avg_audio = (total_audio / success_count) if success_count else 0.0
    avg_segment_count = (total_segments / success_count) if success_count else 0.0

    avg_long_ratio = (total_long_segments / total_segments) if total_segments else 0.0
    avg_short_ratio = (total_short_segments / total_segments) if total_segments else 0.0

    metrics: Dict[str, float] = {
        'total_files': float(total_files),
        'success_count': float(success_count),
        'failure_count': float(total_files - success_count),
        'success_rate': (success_count / total_files) if total_files else 0.0,
        'total_audio_s': float(total_audio),
        'total_processing_s': float(total_processing),
        'avg_processing_s': float(avg_processing),
        'avg_audio_s': float(avg_audio),
        'realtime_factor': float(realtime_factor),
        'avg_segment_count': float(avg_segment_count),
        'avg_long_segment_ratio': float(avg_long_ratio),
        'avg_short_segment_ratio': float(avg_short_ratio),
        'guard_avg_shift_ms': float(avg_guard_shift),
        'guard_max_shift_ms': float(max_guard_shift),
        'guard_avg_guard_only_ms': float(avg_guard_only),
        'guard_avg_vocal_guard_shift_ms': float(avg_guard_vocal),
        'guard_avg_mix_guard_shift_ms': float(avg_guard_mix),
        'peak_rss_bytes': float(peak_mem_bytes or 0.0),
        'dataset_segment_count': float(total_segments),
        'avg_long_segments_per_track': float((total_long_segments / success_count) if success_count else 0.0),
        'avg_short_segments_per_track': float((total_short_segments / success_count) if success_count else 0.0),
    }
    return metrics


def build_default_guardrails(metrics: Dict[str, float], args) -> Dict[str, Dict[str, float]]:
    def relative_min(value: float, tolerance: float, floor: float = 0.0) -> float:
        if value <= 0.0:
            return floor
        return max(floor, value * (1.0 - tolerance))

    def relative_max(value: float, tolerance: float, floor: float = 0.0) -> float:
        if value <= 0.0:
            return floor
        return max(floor, value * (1.0 + tolerance))

    quality_tol = args.quality_tolerance
    shift_tol = args.shift_tolerance
    speed_tol = args.speed_tolerance
    memory_tol = args.memory_tolerance

    guardrails: Dict[str, Dict[str, float]] = {
        'realtime_factor': {'min': relative_min(metrics['realtime_factor'], speed_tol)},
        'avg_long_segment_ratio': {'max': relative_max(metrics['avg_long_segment_ratio'], quality_tol, quality_tol)},
        'avg_short_segment_ratio': {'max': relative_max(metrics['avg_short_segment_ratio'], quality_tol, quality_tol)},
        'avg_segment_count': {'min': relative_min(metrics['avg_segment_count'], quality_tol)},
        'success_rate': {'min': relative_min(metrics['success_rate'], quality_tol)},
        'guard_avg_shift_ms': {'max': relative_max(metrics['guard_avg_shift_ms'], shift_tol, shift_tol)},
        'guard_avg_guard_only_ms': {'max': relative_max(metrics['guard_avg_guard_only_ms'], shift_tol, shift_tol)},
        'guard_avg_vocal_guard_shift_ms': {'max': relative_max(metrics['guard_avg_vocal_guard_shift_ms'], shift_tol, shift_tol)},
        'guard_avg_mix_guard_shift_ms': {'max': relative_max(metrics['guard_avg_mix_guard_shift_ms'], shift_tol, shift_tol)},
        'guard_max_shift_ms': {'max': relative_max(metrics['guard_max_shift_ms'], shift_tol, shift_tol)},
    }

    if metrics.get('peak_rss_bytes', 0.0) > 0.0:
        guardrails['peak_rss_bytes'] = {'max': relative_max(metrics['peak_rss_bytes'], memory_tol)}

    return guardrails


def write_guardrail_file(path: Path, metrics: Dict[str, float], args) -> None:
    guardrails = build_default_guardrails(metrics, args)
    payload = {
        'version': args.mode,
        'created_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'dataset_label': args.guardrail_dataset_label or args.input_dir.name,
        'metrics': metrics,
        'guardrails': guardrails,
        'tolerances': {
            'quality_relative': args.quality_tolerance,
            'shift_relative': args.shift_tolerance,
            'speed_relative': args.speed_tolerance,
            'memory_relative': args.memory_tolerance,
        },
        'notes': '生成于 scripts/bench/run_bench.py --save-guardrails',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')
    logging.info("已写入质量护栏文件: %s", path)


def check_guardrails(path: Path, metrics: Dict[str, float], allow_violation: bool) -> None:
    data = json.loads(path.read_text(encoding='utf-8'))
    guardrails = data.get('guardrails') or {}
    failures = []
    for metric, limits in guardrails.items():
        value = metrics.get(metric)
        if value is None:
            continue
        min_limit = limits.get('min')
        max_limit = limits.get('max')
        if min_limit is not None and value < min_limit:
            failures.append((metric, value, '>=', min_limit))
        if max_limit is not None and value > max_limit:
            failures.append((metric, value, '<=', max_limit))
    if failures:
        logging.error("检测到质量护栏告警 (基于 %s):", path)
        for metric, actual, op, expected in failures:
            logging.error("  %s: %.6f 不满足 %s %.6f", metric, actual, op, expected)
        if not allow_violation:
            sys.exit(2)
    else:
        logging.info("质量护栏校验通过 (基于 %s)", path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SeamlessSplitter 性能与质量基准脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input-dir', type=Path, default=project_root / 'input', help='待处理音频所在目录')
    parser.add_argument('--manifest', type=Path, help='音频列表文件，每行一个路径（优先于 --input-dir）')
    parser.add_argument('--mode', default='v2.2_mdd', choices=['v2.2_mdd'], help='运行模式')
    parser.add_argument('--output-dir', type=Path, help='基准输出目录，默认自动生成')
    parser.add_argument('--keep-output', action='store_true', help='保留切分结果，默认跑完即清理')
    parser.add_argument('--limit', type=int, help='限制处理的文件数量（可用于快速抽样）')
    parser.add_argument('--long-threshold', type=float, default=15.0, help='长片段阈值（秒）')
    parser.add_argument('--short-threshold', type=float, default=2.0, help='短片段阈值（秒）')
    parser.add_argument('--json', type=Path, help='将结果写入 JSON 文件')
    parser.add_argument('--save-guardrails', type=Path, help='将本次统计写入质量护栏文件 (baseline)')
    parser.add_argument('--guardrails', type=Path, help='读取质量护栏文件并进行校验')
    parser.add_argument('--guardrail-dataset-label', type=str, help='写护栏时记录的数据集标签')
    parser.add_argument('--quality-tolerance', type=float, default=0.05, help='质量项相对容差 (±)')
    parser.add_argument('--shift-tolerance', type=float, default=0.10, help='守卫位移相对容差 (±)')
    parser.add_argument('--speed-tolerance', type=float, default=0.10, help='速度项允许退化比例')
    parser.add_argument('--memory-tolerance', type=float, default=0.10, help='内存峰值允许增加比例')
    parser.add_argument('--allow-guardrail-violations', action='store_true', help='仅记录护栏告警而不退出非零状态')
    parser.add_argument('--verbose', action='store_true', help='启用 DEBUG 日志')
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    if args.manifest:
        audio_files = load_manifest(args.manifest)
    else:
        audio_files = iter_audio_files(args.input_dir)

    if args.limit is not None:
        audio_files = audio_files[:max(args.limit, 0)]

    if not audio_files:
        logging.error("未找到待处理音频，请检查输入参数")
        sys.exit(1)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_root = args.output_dir
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root = project_root / 'output' / f'bench_{timestamp}'
        output_root.mkdir(parents=True, exist_ok=True)

    sample_rate = get_config('audio.sample_rate', 44100)
    splitter = SeamlessSplitter(sample_rate=sample_rate)

    process = psutil.Process() if psutil else None
    peak_mem_bytes: float = 0.0

    rows: List[FileBenchmark] = []
    total_processing = 0.0
    total_audio = 0.0

    logging.info("开始运行基准，共 %d 个文件，输出目录: %s", len(audio_files), output_root)
    for idx, audio_path in enumerate(audio_files, start=1):
        job_dir = output_root / f"job_{idx:03d}_{audio_path.stem}"
        job_dir.mkdir(parents=True, exist_ok=True)

        duration_s = get_audio_duration(audio_path)
        total_audio += duration_s

        rss_before = process.memory_info().rss if process else None
        start_time = time.perf_counter()
        try:
            result = splitter.split_audio_seamlessly(str(audio_path), str(job_dir), mode=args.mode)
        except Exception as exc:  # pragma: no cover - 运行期保障
            logging.exception("处理 %s 失败: %s", audio_path, exc)
            rows.append(
                FileBenchmark(
                    path=str(audio_path),
                    duration_s=duration_s,
                    processing_time_s=0.0,
                    segment_count=0,
                    long_segment_ratio=0.0,
                    short_segment_ratio=0.0,
                    guard_avg_shift_ms=0.0,
                    guard_max_shift_ms=0.0,
                    guard_avg_guard_only_ms=0.0,
                    guard_avg_vocal_shift_ms=0.0,
                    guard_avg_mix_shift_ms=0.0,
                    success=False,
                    notes=str(exc),
                )
            )
            if not args.keep_output:
                shutil.rmtree(job_dir, ignore_errors=True)
            continue

        elapsed = time.perf_counter() - start_time
        processing_time = float(result.get('processing_time', elapsed))
        total_processing += processing_time

        if process and rss_before is not None:
            try:
                current_rss = process.memory_info().rss
            except psutil.Error:  # pragma: no cover
                current_rss = rss_before
            peak_mem_bytes = max(peak_mem_bytes, float(current_rss))

        segment_durations = [float(d) for d in result.get('segment_durations', [])]
        segment_count = len(segment_durations)
        long_ratio = (
            sum(1 for d in segment_durations if d > args.long_threshold) / segment_count
            if segment_count else 0.0
        )
        short_ratio = (
            sum(1 for d in segment_durations if d < args.short_threshold) / segment_count
            if segment_count else 0.0
        )

        guard_summary = summarize_guard_stats(result.get('guard_shift_stats'))

        rows.append(
            FileBenchmark(
                path=str(audio_path),
                duration_s=duration_s,
                processing_time_s=processing_time,
                segment_count=segment_count,
                long_segment_ratio=long_ratio,
                short_segment_ratio=short_ratio,
                guard_avg_shift_ms=guard_summary['guard_avg_shift_ms'],
                guard_max_shift_ms=guard_summary['guard_max_shift_ms'],
                guard_avg_guard_only_ms=guard_summary['guard_avg_guard_only_ms'],
                guard_avg_vocal_shift_ms=guard_summary['guard_avg_vocal_shift_ms'],
                guard_avg_mix_shift_ms=guard_summary['guard_avg_mix_shift_ms'],
                success=bool(result.get('success', False)),
                notes=result.get('error'),
            )
        )

        if not args.keep_output:
            shutil.rmtree(job_dir, ignore_errors=True)

    if not args.keep_output and not args.output_dir:
        try:
            shutil.rmtree(output_root, ignore_errors=True)
        except Exception as exc:  # pragma: no cover
            logging.warning("清理输出目录失败: %s", exc)

    guard_metrics = compute_guard_metrics(rows, total_processing, total_audio, peak_mem_bytes)

    logging.info("=== 汇总统计 ===")
    for key in (
        'total_files', 'success_count', 'failure_count', 'success_rate', 'total_audio_s',
        'total_processing_s', 'realtime_factor', 'avg_segment_count',
        'avg_long_segment_ratio', 'avg_short_segment_ratio',
        'guard_avg_shift_ms', 'guard_avg_vocal_guard_shift_ms', 'guard_avg_mix_guard_shift_ms',
        'guard_max_shift_ms', 'peak_rss_bytes'
    ):
        value = guard_metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            logging.info("%s: %.6f", key, value)
        else:
            logging.info("%s: %s", key, value)

    if args.save_guardrails:
        write_guardrail_file(args.save_guardrails, guard_metrics, args)

    if args.guardrails:
        check_guardrails(args.guardrails, guard_metrics, args.allow_guardrail_violations)

    payload = {
        'summary': guard_metrics,
        'files': [item.to_summary() for item in rows],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        logging.info("已写入 JSON 结果: %s", args.json)


if __name__ == '__main__':
    main()

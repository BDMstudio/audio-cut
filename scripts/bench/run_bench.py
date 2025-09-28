
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/bench/run_bench.py
# AI-SUMMARY: GPU 感知的基准脚本，批量运行 SeamlessSplitter 并输出端到端耗时、质量护栏与 GPU/H2D 指标。

import argparse
import json
import logging
import shutil
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import soundfile as sf

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - 可选依赖
    psutil = None  # type: ignore

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - 可选依赖
    torch = None  # type: ignore

try:  # pragma: no cover - 可选依赖
    import torchaudio  # type: ignore
except ImportError:  # pragma: no cover
    torchaudio = None  # type: ignore

try:  # pragma: no cover - 可选依赖
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from src.vocal_smart_splitter.utils.config_manager import get_config

_NVML_INITIALIZED = False


def ensure_nvml() -> bool:
    """尝试初始化 NVML，失败时返回 False。"""
    global _NVML_INITIALIZED
    if pynvml is None:
        return False
    if not _NVML_INITIALIZED:
        try:
            pynvml.nvmlInit()
            _NVML_INITIALIZED = True
        except Exception as exc:  # pragma: no cover - NVML 初始化失败
            logging.warning("NVML 初始化失败: %s", exc)
            return False
    return True


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
    gpu_util_avg: Optional[float] = None
    gpu_util_peak: Optional[float] = None
    gpu_mem_avg_bytes: Optional[float] = None
    gpu_mem_peak_bytes: Optional[float] = None
    h2d_time_s: Optional[float] = None
    dtoh_time_s: Optional[float] = None
    h2d_bytes: Optional[float] = None
    dtoh_bytes: Optional[float] = None

    def to_summary(self) -> Dict[str, float]:
        return asdict(self)


class GPUSampler:
    """基于 NVML 的 GPU 采样器，周期性抓取利用率/显存/PCIe 吞吐。"""

    def __init__(self, device_index: int, interval: float = 0.2) -> None:
        self.device_index = device_index
        self.interval = max(0.05, interval)
        self.samples: List[Tuple[float, float, float, float, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self.available = ensure_nvml()
        if self.available:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception as exc:  # pragma: no cover - 非法 GPU index
                logging.warning("无法获取 NVML 句柄 (index=%s): %s", device_index, exc)
                self.available = False

    def _collect_sample(self) -> None:
        if not self.available or self._handle is None:
            return
        try:
            ts = time.perf_counter()
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            mem_used = pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
            tx_kb_s = pynvml.nvmlDeviceGetPcieThroughput(self._handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            rx_kb_s = pynvml.nvmlDeviceGetPcieThroughput(self._handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
        except Exception:  # pragma: no cover - NVML 读数失败
            return
        self.samples.append((ts, float(util), float(mem_used), float(tx_kb_s), float(rx_kb_s)))

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._collect_sample()

    def start(self) -> None:
        if not self.available:
            return
        self.samples.clear()
        self._stop.clear()
        self._collect_sample()
        self._thread = threading.Thread(target=self._run, name='gpu-sampler', daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        if not self.available:
            return {}
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._collect_sample()
        if len(self.samples) < 2:
            return {}
        return self._summarize()

    def _summarize(self) -> Dict[str, float]:
        tx_bytes = 0.0
        rx_bytes = 0.0
        tx_active = 0.0
        rx_active = 0.0
        util_integral = 0.0
        mem_integral = 0.0
        peak_util = 0.0
        peak_mem = 0.0
        total_time = 0.0
        for i in range(len(self.samples) - 1):
            t0, util, mem_used, tx_kb_s, rx_kb_s = self.samples[i]
            t1 = self.samples[i + 1][0]
            dt = max(0.0, t1 - t0)
            total_time += dt
            util_integral += util * dt
            mem_integral += mem_used * dt
            tx_bytes += tx_kb_s * 1024.0 * dt
            rx_bytes += rx_kb_s * 1024.0 * dt
            if tx_kb_s > 0.0:
                tx_active += dt
            if rx_kb_s > 0.0:
                rx_active += dt
            peak_util = max(peak_util, util)
            peak_mem = max(peak_mem, mem_used)
        if self.samples:
            peak_util = max(peak_util, max(sample[1] for sample in self.samples))
            peak_mem = max(peak_mem, max(sample[2] for sample in self.samples))
        avg_util = util_integral / total_time if total_time > 0 else peak_util
        avg_mem = mem_integral / total_time if total_time > 0 else peak_mem
        return {
            'avg_gpu_util_percent': float(avg_util),
            'peak_gpu_util_percent': float(peak_util),
            'avg_gpu_mem_bytes': float(avg_mem),
            'peak_gpu_mem_bytes': float(peak_mem),
            'pcie_tx_time_s': float(tx_active),
            'pcie_rx_time_s': float(rx_active),
            'pcie_tx_bytes': float(tx_bytes),
            'pcie_rx_bytes': float(rx_bytes),
        }


class GPUProfiler:
    """封装 GPU/NVML 指标采样与 PyTorch 峰值显存统计。"""

    def __init__(self, device: Optional['torch.device'], interval: float = 0.2) -> None:
        self.device = device
        self.enabled = bool(torch) and device is not None and device.type == 'cuda'
        self.monitor: Optional[GPUSampler] = None
        if self.enabled:
            index = device.index if device.index is not None else torch.cuda.current_device()
            self.monitor = GPUSampler(index, interval=interval)

    def start(self) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.reset_accumulated_memory_stats(self.device)
        except Exception:  # pragma: no cover - 旧版 PyTorch
            pass
        if self.monitor:
            self.monitor.start()

    def stop(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        torch.cuda.synchronize(self.device)
        monitor_stats = self.monitor.stop() if self.monitor else {}
        peak_mem = 0.0
        try:
            peak_mem = float(torch.cuda.max_memory_allocated(self.device))
        except Exception:  # pragma: no cover - 旧版 PyTorch
            pass
        stats = dict(monitor_stats)
        stats.setdefault('avg_gpu_util_percent', None)
        stats.setdefault('peak_gpu_util_percent', None)
        stats.setdefault('avg_gpu_mem_bytes', 0.0)
        stats.setdefault('peak_gpu_mem_bytes', peak_mem)
        stats['peak_memory_bytes'] = peak_mem
        stats['h2d_time_s'] = float(stats.get('pcie_rx_time_s', 0.0))
        stats['dtoh_time_s'] = float(stats.get('pcie_tx_time_s', 0.0))
        stats['h2d_bytes'] = float(stats.get('pcie_rx_bytes', 0.0))
        stats['dtoh_bytes'] = float(stats.get('pcie_tx_bytes', 0.0))
        return stats


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


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values or not weights:
        return 0.0
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        return float(sum(values) / len(values))
    return float(sum(v * w for v, w in zip(values, weights)) / total_weight)


def compute_guard_metrics(
    rows: List[FileBenchmark],
    total_processing: float,
    total_audio: float,
    peak_rss_bytes: Optional[float],
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
        'guard_avg_vocal_shift_ms': float(avg_guard_vocal),
        'guard_avg_mix_shift_ms': float(avg_guard_mix),
        'peak_rss_bytes': float(peak_rss_bytes or 0.0),
        'dataset_segment_count': float(total_segments),
        'avg_long_segments_per_track': float((total_long_segments / success_count) if success_count else 0.0),
        'avg_short_segments_per_track': float((total_short_segments / success_count) if success_count else 0.0),
    }

    processing_weights = [max(r.processing_time_s, 1e-3) for r in success_rows]

    gpu_util_pairs = [
        (float(r.gpu_util_avg), w)
        for r, w in zip(success_rows, processing_weights)
        if r.gpu_util_avg is not None
    ]
    if gpu_util_pairs:
        util_values, util_weights = zip(*gpu_util_pairs)
        metrics['avg_gpu_util_percent'] = weighted_average(util_values, util_weights)
        metrics['peak_gpu_util_percent'] = max(
            (r.gpu_util_peak or 0.0) for r in success_rows if r.gpu_util_peak is not None
        )

    gpu_mem_pairs = [
        (float(r.gpu_mem_avg_bytes), w)
        for r, w in zip(success_rows, processing_weights)
        if r.gpu_mem_avg_bytes is not None
    ]
    if gpu_mem_pairs:
        mem_values, mem_weights = zip(*gpu_mem_pairs)
        metrics['avg_gpu_mem_bytes'] = weighted_average(mem_values, mem_weights)
        metrics['peak_gpu_mem_bytes'] = max(
            (r.gpu_mem_peak_bytes or 0.0) for r in success_rows if r.gpu_mem_peak_bytes is not None
        )

    h2d_pairs = [
        (float(r.h2d_time_s), w)
        for r, w in zip(success_rows, processing_weights)
        if r.h2d_time_s is not None
    ]
    if h2d_pairs:
        h2d_values, h2d_weights = zip(*h2d_pairs)
        metrics['avg_h2d_time_s'] = weighted_average(h2d_values, h2d_weights)
        metrics['total_h2d_time_s'] = float(sum(h2d_values))
        metrics['total_h2d_bytes'] = float(sum((r.h2d_bytes or 0.0) for r in success_rows))

    dtoh_pairs = [
        (float(r.dtoh_time_s), w)
        for r, w in zip(success_rows, processing_weights)
        if r.dtoh_time_s is not None
    ]
    if dtoh_pairs:
        dtoh_values, dtoh_weights = zip(*dtoh_pairs)
        metrics['avg_dtoh_time_s'] = weighted_average(dtoh_values, dtoh_weights)
        metrics['total_dtoh_time_s'] = float(sum(dtoh_values))
        metrics['total_dtoh_bytes'] = float(sum((r.dtoh_bytes or 0.0) for r in success_rows))

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
        'guard_avg_vocal_shift_ms': {'max': relative_max(metrics['guard_avg_vocal_shift_ms'], shift_tol, shift_tol)},
        'guard_avg_mix_shift_ms': {'max': relative_max(metrics['guard_avg_mix_shift_ms'], shift_tol, shift_tol)},
        'guard_max_shift_ms': {'max': relative_max(metrics['guard_max_shift_ms'], shift_tol, shift_tol)},
    }

    if metrics.get('peak_rss_bytes', 0.0) > 0.0:
        guardrails['peak_rss_bytes'] = {'max': relative_max(metrics['peak_rss_bytes'], memory_tol)}

    if 'avg_gpu_util_percent' in metrics:
        guardrails['avg_gpu_util_percent'] = {'min': relative_min(metrics['avg_gpu_util_percent'], speed_tol)}
    if metrics.get('avg_gpu_mem_bytes', 0.0) > 0.0:
        guardrails['avg_gpu_mem_bytes'] = {'max': relative_max(metrics['avg_gpu_mem_bytes'], memory_tol)}
    if metrics.get('peak_gpu_mem_bytes', 0.0) > 0.0:
        guardrails['peak_gpu_mem_bytes'] = {'max': relative_max(metrics['peak_gpu_mem_bytes'], memory_tol)}
    if metrics.get('avg_h2d_time_s', 0.0) > 0.0:
        guardrails['avg_h2d_time_s'] = {'max': relative_max(metrics['avg_h2d_time_s'], memory_tol, memory_tol)}
    if metrics.get('avg_dtoh_time_s', 0.0) > 0.0:
        guardrails['avg_dtoh_time_s'] = {'max': relative_max(metrics['avg_dtoh_time_s'], memory_tol, memory_tol)}

    return guardrails




def _load_metrics_payload(path: Path) -> Dict[str, float]:
    """Load metrics dict from guardrail or bench JSON."""
    text = path.read_text(encoding='utf-8-sig')
    data = json.loads(text)
    if isinstance(data, dict):
        if isinstance(data.get('metrics'), dict):
            return data['metrics']
        if isinstance(data.get('summary'), dict):
            return data['summary']
    raise ValueError(f'无法在 {path} 中找到 metrics/summary 字段供基线比较')


def compare_against_baseline(current: Dict[str, float], baseline: Dict[str, float], args) -> List[str]:
    failures: List[str] = []

    def fmt_pct(value: float) -> str:
        return f'{value * 100.0:.2f}%'

    base_rt = baseline.get('realtime_factor')
    curr_rt = current.get('realtime_factor')
    if base_rt and curr_rt:
        improvement = (curr_rt / base_rt) - 1.0
        if improvement < args.min_speed_improvement:
            failures.append(
                f'速度护栏未达标: realtime_factor 提升 {fmt_pct(improvement)} < 要求 {fmt_pct(args.min_speed_improvement)}'
            )
        else:
            logging.info(
                '速度护栏通过: realtime_factor 提升 %s (baseline %.6f -> current %.6f)',
                fmt_pct(improvement),
                base_rt,
                curr_rt,
            )

    for key in ('avg_long_segment_ratio', 'avg_short_segment_ratio'):
        base_val = baseline.get(key)
        curr_val = current.get(key)
        if base_val is None or curr_val is None:
            continue
        drift = abs(curr_val - base_val)
        if drift > args.quality_drift_tolerance:
            failures.append(
                f'质量护栏未达标: {key} 漂移 {drift:.4f} > 允许 {args.quality_drift_tolerance:.4f}'
            )
        else:
            logging.info(
                '质量护栏通过: %s 漂移 %.4f (baseline %.4f -> current %.4f)',
                key,
                drift,
                base_val,
                curr_val,
            )

    if current.get('success_rate', 1.0) < 1.0 or current.get('failure_count', 0.0) > 0:
        failures.append('可逆性护栏未达标: 存在失败任务或 success_rate < 1.0')

    baseline_peak_mem = baseline.get('peak_gpu_mem_bytes') or baseline.get('avg_gpu_mem_bytes')
    current_peak_mem = current.get('peak_gpu_mem_bytes') or current.get('avg_gpu_mem_bytes')
    if baseline_peak_mem and current_peak_mem:
        delta = (current_peak_mem / baseline_peak_mem) - 1.0
        if delta > args.max_mem_increase:
            failures.append(
                f'显存护栏未达标: 峰值显存增加 {fmt_pct(delta)} > 允许 {fmt_pct(args.max_mem_increase)}'
            )
        else:
            logging.info(
                '显存护栏通过: 峰值显存变化 %s (baseline %.0f -> current %.0f)',
                fmt_pct(delta),
                baseline_peak_mem,
                current_peak_mem,
            )

    for key in ('avg_h2d_time_s', 'avg_dtoh_time_s'):
        base_val = baseline.get(key)
        curr_val = current.get(key)
        if not base_val or curr_val is None:
            continue
        reduction = 1.0 - (curr_val / base_val)
        if reduction < args.min_transfer_reduction:
            failures.append(
                f'传输护栏未达标: {key} 降幅 {fmt_pct(reduction)} < 要求 {fmt_pct(args.min_transfer_reduction)}'
            )
        else:
            logging.info(
                '传输护栏通过: %s 降幅 %s (baseline %.6f -> current %.6f)',
                key,
                fmt_pct(reduction),
                base_val,
                curr_val,
            )

    return failures


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


def resolve_device(device_arg: str) -> Tuple[str, Optional['torch.device']]:
    if torch is None:
        if device_arg not in ('cpu', 'auto'):
            logging.warning("PyTorch 未安装，忽略设备设置 %s，回退 CPU", device_arg)
        return 'cpu', None
    if device_arg in (None, '', 'auto'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(device_arg)
        except Exception:
            logging.warning("无法解析设备 %s，回退 CPU", device_arg)
            device = torch.device('cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA 不可用，回退 CPU")
        device = torch.device('cpu')
    return device.type, device


def log_runtime_environment(device_kind: str, device: Optional['torch.device']) -> None:
    logging.info("PyTorch 版本: %s", getattr(torch, '__version__', '未安装'))
    logging.info("torchaudio 版本: %s", getattr(torchaudio, '__version__', '未安装'))
    if device_kind == 'cuda' and torch is not None:
        index = device.index if device and device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        logging.info(
            "CUDA 设备 %d: %s (CC %d.%d, %.2f GB)",
            index,
            props.name,
            props.major,
            props.minor,
            props.total_memory / (1024 ** 3),
        )
        try:
            driver_version = torch.cuda.driver_version()
        except Exception:
            driver_version = None
        logging.info("CUDA Driver: %s", driver_version if driver_version else '未知')
        logging.info("CUDA Runtime: %s", getattr(torch.version, 'cuda', '未知'))
        try:
            cudnn_version = torch.backends.cudnn.version()
        except Exception:
            cudnn_version = None
        logging.info("cuDNN: %s", cudnn_version if cudnn_version else '未知')


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SeamlessSplitter GPU 基准脚本",
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
    parser.add_argument('--baseline', type=Path, help='用于对比的 GPU 基线 JSON (guardrails 或 bench summary)')
    parser.add_argument('--min-speed-improvement', type=float, default=0.30, help='相对基线 realtime_factor 至少提升比例 (例如 0.30=提升30%)')
    parser.add_argument('--quality-drift-tolerance', type=float, default=0.05, help='长段/短段比例允许的绝对漂移上限')
    parser.add_argument('--max-mem-increase', type=float, default=0.10, help='显存峰值允许增加比例 (默认 +10%)')
    parser.add_argument('--min-transfer-reduction', type=float, default=0.15, help='H2D/DtoH 平均耗时所需下降幅度 (默认 >=15%)')
    parser.add_argument('--device', type=str, default='auto', help='运行设备 (auto/cpu/cuda/cuda:N)')
    parser.add_argument('--gpu-metrics', action='store_true', help='采集 GPU/NVML 指标 (需 torch + pynvml)')
    parser.add_argument('--gpu-poll-interval', type=float, default=0.2, help='NVML 采样间隔 (秒)')
    parser.add_argument('--verbose', action='store_true', help='启用 DEBUG 日志')
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    device_kind, torch_device = resolve_device(args.device)
    if torch_device is not None and torch_device.type == 'cuda':
        torch.cuda.set_device(torch_device)
    log_runtime_environment(device_kind, torch_device)

    use_gpu_metrics = bool(args.gpu_metrics and torch_device is not None and torch_device.type == 'cuda')
    if args.gpu_metrics and not use_gpu_metrics:
        logging.warning("GPU 指标采集被禁用（缺少 CUDA 或依赖），仅记录 CPU 统计")

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
    peak_rss_bytes: float = 0.0

    rows: List[FileBenchmark] = []
    total_processing = 0.0
    total_audio = 0.0

    logging.info(
        "开始运行基准，共 %d 个文件，输出目录: %s，采集 GPU 指标: %s",
        len(audio_files),
        output_root,
        'YES' if use_gpu_metrics else 'NO',
    )

    for idx, audio_path in enumerate(audio_files, start=1):
        job_dir = output_root / f"job_{idx:03d}_{audio_path.stem}"
        job_dir.mkdir(parents=True, exist_ok=True)

        duration_s = get_audio_duration(audio_path)
        total_audio += duration_s

        rss_before = process.memory_info().rss if process else None
        gpu_profiler = GPUProfiler(torch_device, interval=args.gpu_poll_interval) if use_gpu_metrics else None

        start_time = time.perf_counter()
        gpu_stats: Dict[str, float] = {}
        try:
            if gpu_profiler:
                gpu_profiler.start()
            result = splitter.split_audio_seamlessly(str(audio_path), str(job_dir), mode=args.mode)
        except Exception as exc:  # pragma: no cover - 运行期保障
            logging.exception("处理 %s 失败: %s", audio_path, exc)
            if gpu_profiler:
                gpu_stats = gpu_profiler.stop()
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
                    gpu_util_avg=gpu_stats.get('avg_gpu_util_percent') if gpu_stats else None,
                    gpu_util_peak=gpu_stats.get('peak_gpu_util_percent') if gpu_stats else None,
                    gpu_mem_avg_bytes=gpu_stats.get('avg_gpu_mem_bytes') if gpu_stats else None,
                    gpu_mem_peak_bytes=gpu_stats.get('peak_gpu_mem_bytes') if gpu_stats else None,
                    h2d_time_s=gpu_stats.get('h2d_time_s') if gpu_stats else None,
                    dtoh_time_s=gpu_stats.get('dtoh_time_s') if gpu_stats else None,
                    h2d_bytes=gpu_stats.get('h2d_bytes') if gpu_stats else None,
                    dtoh_bytes=gpu_stats.get('dtoh_bytes') if gpu_stats else None,
                )
            )
            if not args.keep_output:
                shutil.rmtree(job_dir, ignore_errors=True)
            continue

        elapsed = time.perf_counter() - start_time

        processing_time = float(result.get('processing_time', elapsed))
        total_processing += processing_time

        if gpu_profiler and not gpu_stats:
            gpu_stats = gpu_profiler.stop()

        if process and rss_before is not None:
            try:
                current_rss = process.memory_info().rss
            except psutil.Error:  # pragma: no cover
                current_rss = rss_before
            peak_rss_bytes = max(peak_rss_bytes, float(current_rss))

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
                gpu_util_avg=gpu_stats.get('avg_gpu_util_percent'),
                gpu_util_peak=gpu_stats.get('peak_gpu_util_percent'),
                gpu_mem_avg_bytes=gpu_stats.get('avg_gpu_mem_bytes'),
                gpu_mem_peak_bytes=gpu_stats.get('peak_gpu_mem_bytes') or gpu_stats.get('peak_memory_bytes'),
                h2d_time_s=gpu_stats.get('h2d_time_s'),
                dtoh_time_s=gpu_stats.get('dtoh_time_s'),
                h2d_bytes=gpu_stats.get('h2d_bytes'),
                dtoh_bytes=gpu_stats.get('dtoh_bytes'),
            )
        )

        if not args.keep_output:
            shutil.rmtree(job_dir, ignore_errors=True)

    if not args.keep_output and not args.output_dir:
        try:
            shutil.rmtree(output_root, ignore_errors=True)
        except Exception as exc:  # pragma: no cover
            logging.warning("清理输出目录失败: %s", exc)

    guard_metrics = compute_guard_metrics(rows, total_processing, total_audio, peak_rss_bytes)

    baseline_failures: List[str] = []
    if args.baseline:
        try:
            baseline_metrics = _load_metrics_payload(args.baseline)
            baseline_failures = compare_against_baseline(guard_metrics, baseline_metrics, args)
        except Exception as exc:
            msg = f'基线文件 {args.baseline} 解析失败: {exc}'
            logging.error(msg)
            baseline_failures = [msg]

    if baseline_failures:
        for item in baseline_failures:
            logging.error(item)
        if not args.allow_guardrail_violations:
            sys.exit(3)

    logging.info("=== 汇总统计 ===")
    important_keys = [
        'total_files', 'success_count', 'failure_count', 'success_rate', 'total_audio_s',
        'total_processing_s', 'realtime_factor', 'avg_segment_count',
        'avg_long_segment_ratio', 'avg_short_segment_ratio',
        'guard_avg_shift_ms', 'guard_avg_vocal_shift_ms', 'guard_avg_mix_shift_ms',
        'guard_max_shift_ms', 'peak_rss_bytes', 'avg_gpu_util_percent', 'peak_gpu_mem_bytes',
        'avg_h2d_time_s', 'avg_dtoh_time_s', 'total_h2d_bytes', 'total_dtoh_bytes'
    ]
    for key in important_keys:
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
        'device': device_kind,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        logging.info("已写入 JSON 结果: %s", args.json)


if __name__ == '__main__':
    main()

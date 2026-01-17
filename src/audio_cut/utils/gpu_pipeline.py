#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/utils/gpu_pipeline.py
# AI-SUMMARY: GPU 流水线工具，负责设备选择、流/事件管理、切片调度与固定内存缓冲池。

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover - torch 在 CPU-only 环境可能不可用
    torch = None  # type: ignore

try:  # 延迟导入 onnxruntime 以便在纯 CPU 环境降噪
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - ORT 未安装
    ort = None  # type: ignore


_ORT_DEPS_INJECTED = False
_NVML_INITIALIZED = False
_NVML_UNAVAILABLE = False
_NVML_LOCK = threading.Lock()

try:  # 可选 NVML 依赖，用于采集 GPU 运行指标
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - NVML 未安装
    pynvml = None  # type: ignore


@dataclass
class Streams:
    """GPU 流集合：分离 / VAD / 特征"""

    s_sep: Optional["torch.cuda.Stream"] = None
    s_vad: Optional["torch.cuda.Stream"] = None
    s_feat: Optional["torch.cuda.Stream"] = None

    def as_tuple(self) -> Sequence[Optional["torch.cuda.Stream"]]:
        return (self.s_sep, self.s_vad, self.s_feat)


@dataclass
class ChunkPlan:
    """单个分块调度计划"""

    index: int
    start_s: float
    end_s: float
    halo_left_s: float
    halo_right_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def effective_start_s(self) -> float:
        return self.start_s + self.halo_left_s

    @property
    def effective_end_s(self) -> float:
        return self.end_s - self.halo_right_s

    def as_slice(self, sample_rate: int) -> slice:
        start = max(0, int(round(self.start_s * sample_rate)))
        stop = max(start, int(round(self.end_s * sample_rate)))
        return slice(start, stop)

    def halo_slices(self, sample_rate: int) -> tuple[slice, slice]:
        left = max(0, int(round(self.halo_left_s * sample_rate)))
        right = max(0, int(round(self.halo_right_s * sample_rate)))
        return (slice(None, left if left > 0 else None), slice(-right if right > 0 else None, None))


def select_device(preferred: Optional[str] = None) -> str:
    """选择 GPU 管线使用的设备标识。"""

    if torch is None or not torch.cuda.is_available():
        return "cpu"

    normalized = (preferred or "cuda").strip().lower()
    if not normalized:
        normalized = "cuda"

    if normalized in {"cpu", "none"}:
        return "cpu"

    if normalized in {"cuda", "gpu"}:
        try:
            current = torch.cuda.current_device()
        except Exception:  # pragma: no cover - CUDA context 未初始化
            current = 0
        return f"cuda:{current}"

    if normalized.startswith("cuda:") or normalized.startswith("gpu:"):
        _, _, idx_str = normalized.partition(":")
    elif normalized.isdigit():
        idx_str = normalized
    else:
        idx_str = "0"

    try:
        index = int(idx_str)
    except ValueError:  # pragma: no cover - 字符串非数字
        index = 0

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_count == 0:
        return "cpu"
    if index < 0 or index >= device_count:
        logger.warning(
            "[GPU Pipeline] 请求的设备 cuda:%s 不存在，已回退到 cuda:0 (device_count=%s)",
            index,
            device_count,
        )
        index = 0

    return f"cuda:{index}"


def create_streams(device: str, enable: bool = True) -> Streams:
    """创建分离/VAD/特征三个 CUDA stream。"""

    if torch is None or not enable or not device.startswith("cuda"):
        return Streams()

    torch_device = torch.device(device)
    return Streams(
        s_sep=torch.cuda.Stream(device=torch_device),
        s_vad=torch.cuda.Stream(device=torch_device),
        s_feat=torch.cuda.Stream(device=torch_device),
    )


def record_event(stream: Optional["torch.cuda.Stream"], *, enable_timing: bool = False) -> Optional["torch.cuda.Event"]:
    if torch is None or stream is None:
        return None
    event = torch.cuda.Event(blocking=False, enable_timing=enable_timing)
    event.record(stream)
    return event


def wait_event(stream: Optional["torch.cuda.Stream"], event: Optional["torch.cuda.Event"]) -> None:
    if torch is None or stream is None or event is None:
        return
    stream.wait_event(event)


def _parse_cuda_device_index(device: str) -> Optional[int]:
    if not device:
        return None
    device = device.lower()
    if device.startswith("cuda:"):
        _, _, idx_str = device.partition(":")
    elif device == "cuda":
        try:
            return torch.cuda.current_device() if torch and torch.cuda.is_available() else None
        except Exception:  # pragma: no cover - CUDA context 未初始化
            return 0
    else:
        return None

    try:
        return int(idx_str)
    except (TypeError, ValueError):  # pragma: no cover - 字符串解析失败
        return None


def _resolve_device_name(device: str) -> Optional[str]:
    index = _parse_cuda_device_index(device)
    if index is None or torch is None or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(index)
    except Exception:  # pragma: no cover - 设备不可访问
        return None


def _ensure_nvml_initialized() -> bool:
    global _NVML_INITIALIZED, _NVML_UNAVAILABLE
    if pynvml is None or _NVML_UNAVAILABLE:
        return False
    with _NVML_LOCK:
        if _NVML_INITIALIZED:
            return True
        try:
            pynvml.nvmlInit()  # type: ignore[attr-defined]
            _NVML_INITIALIZED = True
        except Exception:  # pragma: no cover - NVML 初始化失败
            _NVML_UNAVAILABLE = True
            logger.debug("[GPU Pipeline] NVML 初始化失败", exc_info=True)
            return False
    return True


def _collect_nvml_metrics(device_index: int) -> Optional[Dict[str, float]]:
    if not _ensure_nvml_initialized():
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)  # type: ignore[attr-defined]
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[attr-defined]
        return {
            "gpu_pipeline_nvml_gpu_util_percent": float(util.gpu),
            "gpu_pipeline_nvml_mem_util_percent": float(util.memory),
            "gpu_pipeline_nvml_mem_used_bytes": float(mem.used),
            "gpu_pipeline_nvml_mem_total_bytes": float(mem.total),
        }
    except Exception:  # pragma: no cover - 采集失败
        logger.debug("[GPU Pipeline] NVML 指标采集失败", exc_info=True)
        return None


def _collect_nvidia_smi_metrics(device_index: int) -> Optional[Dict[str, float]]:
    command = shutil.which("nvidia-smi")
    if not command:
        return None
    query = "utilization.gpu,memory.used,memory.total"
    try:
        completed = subprocess.run(
            [command, "-i", str(device_index), "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # pragma: no cover - 命令执行失败
        logger.debug("[GPU Pipeline] 调用 nvidia-smi 失败", exc_info=True)
        return None

    line = completed.stdout.strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 3:
        return None
    try:
        util = float(parts[0])
        mem_used_mib = float(parts[1])
        mem_total_mib = float(parts[2])
    except ValueError:  # pragma: no cover - 文本解析失败
        return None
    mib_to_bytes = 1024.0 * 1024.0
    return {
        "gpu_pipeline_nvidia_smi_gpu_util_percent": util,
        "gpu_pipeline_nvidia_smi_mem_used_bytes": mem_used_mib * mib_to_bytes,
        "gpu_pipeline_nvidia_smi_mem_total_bytes": mem_total_mib * mib_to_bytes,
    }


def _collect_device_metrics(device: str) -> Optional[Dict[str, float]]:
    index = _parse_cuda_device_index(device)
    if index is None or index < 0:
        return None
    metrics = _collect_nvml_metrics(index)
    if metrics is not None:
        return metrics
    return _collect_nvidia_smi_metrics(index)

@dataclass
class OrtExecutionConfig:
    """ONNX Runtime 执行配置。"""

    graph_optimization_level: str = "basic"
    cudnn_conv_algo_search: str = "HEURISTIC"
    disable_trt: bool = True

    def _graph_level(self) -> "ort.GraphOptimizationLevel":  # type: ignore[override]
        if ort is None:  # pragma: no cover - 未安装 ORT
            raise RuntimeError("onnxruntime 未安装，无法配置 graph optimization level")
        mapping = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        key = str(self.graph_optimization_level).lower()
        return mapping.get(key, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)

    def apply(self, so: "ort.SessionOptions") -> None:  # type: ignore[override]
        so.graph_optimization_level = self._graph_level()
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True

    def providers(self, *, prefer: Optional[str] = None) -> List[object]:
        prefer_key = prefer or "CUDAExecutionProvider"
        cuda_options = {
            "cudnn_conv_algo_search": str(self.cudnn_conv_algo_search).upper() or "HEURISTIC"
        }
        providers: List[object] = []
        if not self.disable_trt:
            providers.append("TensorrtExecutionProvider")
        if prefer_key == "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        else:
            providers.append(("CUDAExecutionProvider", cuda_options))
            providers.append("CPUExecutionProvider")
        return providers


def ensure_ort_dependencies() -> None:
    """在 Windows 环境下注入 ORT DLL 依赖。"""

    global _ORT_DEPS_INJECTED
    if _ORT_DEPS_INJECTED:
        return
    if ort is None or os.name != "nt":
        _ORT_DEPS_INJECTED = True
        return
    try:
        capi_dir = Path(ort.__file__).parent / "capi"
        deps_dir = capi_dir / "deps"
        os.add_dll_directory(str(capi_dir))
        if deps_dir.exists():
            os.add_dll_directory(str(deps_dir))
    except Exception as exc:  # pragma: no cover - 仅在 Windows 执行
        logger.warning("[ORT] DLL 注入失败: %s", exc)
    finally:
        _ORT_DEPS_INJECTED = True


def chunk_schedule(
    total_s: float,
    *,
    chunk_s: float = 10.0,
    overlap_s: float = 2.5,
    halo_s: float = 0.5,
) -> List[ChunkPlan]:
    """按给定窗口/重叠/halo 生成分块计划。"""

    total_s = max(0.0, float(total_s))
    chunk_s = max(0.1, float(chunk_s))
    overlap_s = max(0.0, min(float(overlap_s), chunk_s * 0.9))
    halo_s = max(0.0, min(float(halo_s), chunk_s * 0.5))

    if total_s <= chunk_s:
        return [ChunkPlan(index=0, start_s=0.0, end_s=total_s, halo_left_s=0.0, halo_right_s=0.0)]

    stride = chunk_s - overlap_s
    if stride <= 0:
        stride = chunk_s

    plans: List[ChunkPlan] = []
    index = 0
    start = 0.0
    while start < total_s - 1e-6:
        end = min(total_s, start + chunk_s)
        halo_left = halo_s if index > 0 else 0.0
        has_next = end < total_s - 1e-6
        halo_right = halo_s if has_next else 0.0
        plans.append(
            ChunkPlan(
                index=index,
                start_s=start,
                end_s=end,
                halo_left_s=halo_left,
                halo_right_s=halo_right,
            )
        )
        index += 1
        if not has_next:
            break
        start += stride
    return plans


@dataclass
class PinnedBufferPool:
    """Pinned host tensor cache to accelerate repeated H2D/DtoH transfers."""

    dtype: "torch.dtype"
    capacity: int = 2
    _buffers: List["torch.Tensor"] = field(default_factory=list)

    def __post_init__(self) -> None:  # pragma: no cover - torch availability guard
        if torch is None:
            self.capacity = 0

    def _make_tensor(self, num_elements: int) -> "torch.Tensor":
        return torch.empty(int(num_elements), dtype=self.dtype, pin_memory=True)

    def acquire(self, num_elements: int) -> Optional["torch.Tensor"]:
        if torch is None or num_elements <= 0:
            return None
        while self._buffers:
            buf = self._buffers.pop()
            if buf.numel() >= num_elements:
                return buf[:num_elements]
        return self._make_tensor(num_elements)

    def acquire_view(self, shape: Sequence[int]) -> Optional["torch.Tensor"]:
        if torch is None:
            return None
        numel = 1
        for dim in shape:
            numel *= int(dim)
        tensor = self.acquire(numel)
        if tensor is None:
            return None
        return tensor.view(*shape)

    def release(self, tensor: Optional["torch.Tensor"]) -> None:
        if torch is None or tensor is None:
            return
        base = tensor.reshape(-1)
        if len(self._buffers) < self.capacity:
            self._buffers.append(base)

    def clear(self) -> None:
        self._buffers.clear()






@dataclass
class InflightLimiter:
    """Thread-safe in-flight counter used to apply backpressure on the GPU pipeline."""

    limit: int
    _condition: threading.Condition = field(default_factory=threading.Condition, init=False)
    _inflight: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.limit = int(self.limit)
        if self.limit < 0:
            self.limit = 0

    @contextmanager
    def acquire(self, timeout: Optional[float] = None) -> Iterator[None]:
        """Acquire an in-flight slot. Block until available or raise on timeout."""

        if self.limit == 0:
            yield
            return

        with self._condition:
            if timeout is None:
                while self._inflight >= self.limit:
                    self._condition.wait()
            else:
                ready = self._condition.wait_for(lambda: self._inflight < self.limit, timeout=timeout)
                if not ready:
                    raise RuntimeError("inflight limit exceeded")
            self._inflight += 1

        try:
            yield
        finally:
            with self._condition:
                if self._inflight > 0:
                    self._inflight -= 1
                self._condition.notify()


@dataclass
class PipelineConfig:
    enable: bool = False
    prefer_device: str = "cuda"
    chunk_s: float = 10.0
    overlap_s: float = 2.5
    halo_s: float = 0.5
    align_hop: int = 4096
    use_cuda_streams: bool = True
    prefetch_pinned_buffers: int = 2
    inflight_chunks_limit: int = 2
    strict_gpu: bool = False
    ort_config: OrtExecutionConfig = field(default_factory=OrtExecutionConfig)

    @classmethod
    def from_mapping(cls, mapping: Optional[dict]) -> "PipelineConfig":
        if not mapping:
            return cls()
        ort_cfg = mapping.get("ort", {}) if isinstance(mapping, dict) else {}
        ort_config = OrtExecutionConfig(
            graph_optimization_level=str(ort_cfg.get("graph_optimization_level", "basic")),
            cudnn_conv_algo_search=str(ort_cfg.get("cudnn_conv_algo_search", "HEURISTIC")),
            disable_trt=bool(ort_cfg.get("disable_trt", True)),
        )
        return cls(
            enable=bool(mapping.get("enable", False)),
            prefer_device=str(mapping.get("prefer_device", "cuda")),
            chunk_s=float(mapping.get("chunk_seconds", mapping.get("chunk_s", 10.0))),
            overlap_s=float(mapping.get("overlap_seconds", mapping.get("overlap_s", 2.5))),
            halo_s=float(mapping.get("halo_seconds", mapping.get("halo_s", 0.5))),
            align_hop=int(mapping.get("align_hop", mapping.get("align_hop_samples", 4096))),
            use_cuda_streams=bool(mapping.get("use_cuda_streams", True)),
            prefetch_pinned_buffers=int(mapping.get("prefetch_pinned_buffers", 2)),
            inflight_chunks_limit=int(mapping.get("inflight_chunks_limit", 2)),
            strict_gpu=bool(mapping.get("strict_mode", mapping.get("strict_gpu", False))),
            ort_config=ort_config,
        )


@dataclass
class PipelineContext:
    """GPU 流水线运行期上下文"""

    device: str
    streams: Streams
    plans: List[ChunkPlan]
    pinned_pool: Optional[PinnedBufferPool]
    limiter: Optional[InflightLimiter]
    config: PipelineConfig = field(repr=False)
    use_streams: bool = False
    strict_gpu: bool = False
    mdx23_input: Optional[Dict[str, List[int]]] = None
    gpu_meta: Dict[str, object] = field(default_factory=dict)
    failures: List[Dict[str, str]] = field(default_factory=list)
    device_index: Optional[int] = None
    device_name: Optional[str] = None

    @property
    def enabled(self) -> bool:
        is_cuda = isinstance(self.device, str) and self.device.startswith("cuda")
        return bool(self.config.enable and is_cuda and self.use_streams and self.streams.s_sep)

    @contextmanager
    def acquire_inflight(self, timeout: Optional[float] = None) -> Iterator[None]:
        """Context-managed access to in-flight slots; no-op without limiter."""

        if self.limiter is None:
            yield
        else:
            with self.limiter.acquire(timeout=timeout):
                yield

    def register_mdx23_input(self, info: Dict[str, List[int]]) -> None:
        self.mdx23_input = info

    def mark_failure(self, stage: str, reason: str) -> None:
        self.failures.append({"stage": stage, "reason": reason})

    def to_meta(self) -> Dict[str, object]:
        meta = dict(self.gpu_meta)
        meta.setdefault("gpu_pipeline_enabled", bool(self.config.enable))
        meta.setdefault("gpu_pipeline_used", bool(self.enabled))
        meta.setdefault("gpu_pipeline_device", self.device)
        if self.device_index is not None:
            meta.setdefault("gpu_pipeline_device_index", int(self.device_index))
        if self.device_name:
            meta.setdefault("gpu_pipeline_device_name", self.device_name)
        meta.setdefault("gpu_pipeline_chunks", len(self.plans))
        meta.setdefault("gpu_pipeline_streams", bool(self.use_streams))
        meta.setdefault("gpu_pipeline_inflight_limit", int(self.limiter.limit) if self.limiter else 0)
        meta.setdefault("gpu_pipeline_prefetch", int(self.pinned_pool.capacity) if self.pinned_pool else 0)
        meta.setdefault("gpu_pipeline_align_hop", int(self.config.align_hop))
        meta.setdefault(
            "gpu_pipeline_config",
            {
                "chunk_seconds": float(self.config.chunk_s),
                "overlap_seconds": float(self.config.overlap_s),
                "halo_seconds": float(self.config.halo_s),
            },
        )
        if self.mdx23_input:
            meta.setdefault("gpu_pipeline_mdx23_input", self.mdx23_input)
        if self.failures:
            meta.setdefault("gpu_pipeline_failures", list(self.failures))
        return meta

    def capture_device_metrics(self) -> None:
        snapshot = _collect_device_metrics(self.device)
        if snapshot:
            self.gpu_meta.update(snapshot)


def build_pipeline_context(duration_s: float, cfg: PipelineConfig) -> PipelineContext:
    """根据配置构建流水线上下文。"""

    device = select_device(cfg.prefer_device)
    torch_device_ctx = nullcontext()
    if torch is not None and device.startswith("cuda"):
        try:
            torch_device = torch.device(device)
            torch.cuda.set_device(torch_device)
            torch_device_ctx = torch.cuda.device(torch_device)
        except Exception:  # pragma: no cover - 设备切换失败
            logger.warning("[GPU Pipeline] 无法切换到设备 %s，已回退默认 CUDA 上下文", device, exc_info=True)
            torch_device_ctx = nullcontext()

    with torch_device_ctx:
        streams = create_streams(device, cfg.use_cuda_streams)

    plans = chunk_schedule(
        duration_s,
        chunk_s=cfg.chunk_s,
        overlap_s=cfg.overlap_s,
        halo_s=cfg.halo_s,
    )
    pinned_pool: Optional[PinnedBufferPool] = None
    if torch is not None and device.startswith("cuda") and cfg.prefetch_pinned_buffers > 0:
        pinned_pool = PinnedBufferPool(
            dtype=torch.float32,
            capacity=max(1, cfg.prefetch_pinned_buffers),
        )
    limiter: Optional[InflightLimiter] = None
    if device.startswith("cuda") and cfg.inflight_chunks_limit > 0:
        limiter = InflightLimiter(limit=cfg.inflight_chunks_limit)
    logger.debug(
        "[GPU Pipeline] device=%s, plans=%d, pinned=%s, inflight_limit=%s",
        device,
        len(plans),
        "yes" if pinned_pool else "no",
        cfg.inflight_chunks_limit if limiter else 0,
    )
    device_index = _parse_cuda_device_index(device)
    device_name = _resolve_device_name(device)
    ctx = PipelineContext(
        device=device,
        streams=streams,
        plans=plans,
        pinned_pool=pinned_pool,
        limiter=limiter,
        config=cfg,
        use_streams=bool(device.startswith("cuda") and cfg.use_cuda_streams),
        strict_gpu=bool(cfg.strict_gpu),
        device_index=device_index,
        device_name=device_name,
    )
    ctx.gpu_meta = {
        "gpu_pipeline_enabled": bool(cfg.enable),
        "gpu_pipeline_device": device,
        "gpu_pipeline_chunks": len(plans),
    }
    if device_index is not None:
        ctx.gpu_meta["gpu_pipeline_device_index"] = device_index
    if device_name:
        ctx.gpu_meta["gpu_pipeline_device_name"] = device_name
    return ctx


__all__ = [
    "Streams",
    "ChunkPlan",
    "PipelineConfig",
    "PipelineContext",
    "PinnedBufferPool",
    "InflightLimiter",
    "OrtExecutionConfig",
    "ensure_ort_dependencies",
    "build_pipeline_context",
    "chunk_schedule",
    "create_streams",
    "record_event",
    "select_device",
    "wait_event",
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/separation/backends.py
# AI-SUMMARY: 声部分离后端接口及实现骨架，提供 ONNX 与 PyTorch 后端以支持分块推理与 OLA。

from __future__ import annotations

import abc
import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import time

from contextlib import nullcontext

import numpy as np

from audio_cut.utils.gpu_pipeline import OrtExecutionConfig, ensure_ort_dependencies
from src.vocal_smart_splitter.utils.config_manager import get_config

logger = logging.getLogger(__name__)

_CUDA_PATHS_INJECTED = False


def _ensure_cuda_paths() -> None:
    global _CUDA_PATHS_INJECTED
    if _CUDA_PATHS_INJECTED:
        return

    candidates = []
    for module_name in ("nvidia.cudnn", "nvidia.cublas", "nvidia.cuda_nvrtc"):
        try:
            module = importlib.import_module(module_name)
            candidates.append(Path(module.__file__).resolve().parent / "bin")
        except Exception:
            continue

    path_env = os.environ.get("PATH", "")
    injections = []
    for candidate in candidates:
        if candidate and candidate.is_dir():
            str_path = str(candidate)
            if str_path not in path_env:
                injections.append(str_path)

    if injections:
        os.environ["PATH"] = os.pathsep.join(injections + [path_env]) if path_env else os.pathsep.join(injections)

    _CUDA_PATHS_INJECTED = True

try:  # Optional dependency, lazily imported when available
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - CI may miss onnxruntime
    ort = None  # type: ignore


@dataclass
class SeparationOutputs:
    """单个分块推理的输出。"""

    vocal: np.ndarray
    instrumental: np.ndarray


class IVocalSeparatorBackend(abc.ABC):
    """声部分离后端的统一接口。"""

    @abc.abstractmethod
    def load_model(self) -> None:
        """加载权重/初始化运行期对象。"""

    @abc.abstractmethod
    def sample_rate(self) -> int:
        """返回模型预期采样率。"""

    @abc.abstractmethod
    def infer_chunk(self, mix_chunk: np.ndarray, **kwargs) -> SeparationOutputs:
        """对输入混音分块执行分离，返回人声与伴奏。"""

    def flush(self) -> Optional[SeparationOutputs]:  # pragma: no cover - 默认无状态
        """流式推理完成后触发的尾部分离，可按需覆盖。"""

        return None


class MDX23OnnxBackend(IVocalSeparatorBackend):
    """基于 ONNXRuntime 的 MDX23 分离后端（单声道→双声道）。"""

    def __init__(
        self,
        model_dir: Path,
        *,
        provider: str = "CUDAExecutionProvider",
        execution_device: str = "cuda",
        ort_config: Optional[OrtExecutionConfig] = None,
        align_hop: Optional[int] = None,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._provider = provider
        self._execution_device = execution_device
        self._session: Optional["ort.InferenceSession"] = None
        self._chunk_model = None
        self._onnx_input: Optional[str] = None
        self._onnx_output: Optional[str] = None
        self._sr = 44100
        self._model_path: Optional[Path] = None
        self._mdx_module = None
        if align_hop is None:
            align_hop = int(os.getenv("MDX23_ALIGN_HOP", 4096))
        self._align_hop = int(align_hop)
        self._ort_config = ort_config or OrtExecutionConfig()
        self._cuda_failed = False
        self._input_signature: Optional[dict] = None
        self._perf_metrics: Dict[str, float] = {}
        self.reset_performance_metrics()

        try:
            output_type_pref = get_config('enhanced_separation.mdx23.output_type', 'auto')
        except Exception:
            output_type_pref = 'auto'
        self._output_type_pref = str(output_type_pref).strip().lower() or 'auto'
        if self._output_type_pref not in {'auto', 'vocal', 'instrumental'}:
            logger.warning('[MDX23Onnx] 未知 output_type 配置 %s，回退 auto', self._output_type_pref)
            self._output_type_pref = 'auto'
        self._resolved_output_type: Optional[str] = None

    def sample_rate(self) -> int:  # noqa: D401
        return self._sr

    def describe_input(self) -> Optional[dict]:
        return dict(self._input_signature) if self._input_signature else None

    def load_model(self) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime 未安装，无法使用 MDX23OnnxBackend")

        ensure_ort_dependencies()
        _ensure_cuda_paths()

        model_pref = os.getenv("MDX23_MODEL_FILENAME")
        if not model_pref:
            try:
                model_pref = get_config("enhanced_separation.mdx23.model_filename")
            except Exception:
                model_pref = None

        if model_pref:
            candidate = (self._model_dir / model_pref).resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"指定的 MDX23 模型不存在: {candidate}")
            onnx_files = [candidate]
        else:
            onnx_files = sorted(self._model_dir.glob("*.onnx"))

        if not onnx_files:
            raise FileNotFoundError(f"未在 {self._model_dir} 找到 ONNX 模型")
        model_path = onnx_files[0]
        self._model_path = model_path
        self._resolved_output_type = self._resolve_output_type(model_path)
        logger.info("[MDX23Onnx] 使用模型: %s (output=%s)", model_path.name, self._resolved_output_type)

        self._create_session(self._provider)

        import importlib.util

        inference_py = self._model_dir.parent / "inference.py"
        if not inference_py.exists():
            raise FileNotFoundError("缺少 inference.py 以执行 STFT/ISTFT 流程")

        spec = importlib.util.spec_from_file_location("mdx23_inference_module", inference_py.as_posix())
        if spec is None or spec.loader is None:  # pragma: no cover - exceptional
            raise RuntimeError("无法加载 mdx23 inference 模块")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._mdx_module = module
        self._init_chunk_model(self._execution_device)
        self.reset_performance_metrics()

    def reset_performance_metrics(self) -> None:
        self._perf_metrics = {
            'h2d_ms': 0.0,
            'dtoh_ms': 0.0,
            'compute_ms': 0.0,
            'chunks': 0.0,
            'max_alloc_bytes': 0.0,
        }

    def get_performance_metrics(self, *, reset: bool = False) -> Dict[str, float]:
        metrics = dict(self._perf_metrics)
        if reset:
            self.reset_performance_metrics()
        return metrics

    def _resolve_output_type(self, model_path: Path) -> str:
        if self._output_type_pref == 'auto':
            name = model_path.name.lower()
            if any(tag in name for tag in ('vocal', 'vocals')) and not any(tag in name for tag in ('inst', 'instrumental', 'accomp')):
                return 'vocal'
            return 'instrumental'
        return self._output_type_pref

    def get_output_type(self) -> str:
        resolved = self._resolved_output_type or self._output_type_pref
        return resolved if resolved in {'vocal', 'instrumental'} else 'instrumental'

    def _record_perf(self, key: str, value: float) -> None:
        if key == 'max_alloc_bytes':
            self._perf_metrics[key] = max(self._perf_metrics.get(key, 0.0), float(value))
        else:
            self._perf_metrics[key] = self._perf_metrics.get(key, 0.0) + float(value)

    def _create_session(self, provider: str) -> None:
        if self._model_path is None:
            raise RuntimeError("MDX23 模型路径未知，无法创建 Session")
        if ort is None:
            raise RuntimeError("onnxruntime 未安装，无法创建 Session")

        sess_options = ort.SessionOptions()
        self._ort_config.apply(sess_options)

        providers = self._ort_config.providers(prefer=provider)

        try:
            self._session = ort.InferenceSession(
                self._model_path.as_posix(),
                sess_options=sess_options,
                providers=providers,
            )
            self._provider = self._session.get_providers()[0]
        except Exception as exc:  # pragma: no cover - depends on runtime
            if provider == "CPUExecutionProvider":
                raise
            logger.warning("[MDX23Onnx] CUDA Session 创建失败，改用 CPU: %s", exc)
            providers = self._ort_config.providers(prefer="CPUExecutionProvider")
            self._session = ort.InferenceSession(
                self._model_path.as_posix(),
                sess_options=sess_options,
                providers=providers,
            )
            self._provider = "CPUExecutionProvider"

        self._execution_device = "cuda" if self._provider == "CUDAExecutionProvider" else "cpu"
        input_info = self._session.get_inputs()[0]
        self._onnx_input = input_info.name
        self._input_signature = {
            "name": input_info.name,
            "shape": [int(dim) if isinstance(dim, int) else 1 for dim in input_info.shape],
        }
        self._onnx_output = self._session.get_outputs()[0].name
        if self._mdx_module is not None:
            self._init_chunk_model(self._execution_device)

    def _init_chunk_model(self, device: str) -> None:
        if self._mdx_module is None:
            raise RuntimeError("MDX23 模块未初始化")
        self._chunk_model = self._mdx_module.Conv_TDF_net_trim_model(
            device=device,
            target_name="vocals",
            L=11,
            n_fft=6144,
        )
        self._chunk_model.eval()

    def _prepare_input(self, mix_chunk: np.ndarray) -> tuple[np.ndarray, int]:
        if mix_chunk.ndim == 1:
            mix_stereo = np.stack([mix_chunk, mix_chunk], axis=0)
        elif mix_chunk.ndim == 2:
            mix_stereo = mix_chunk
        else:
            raise ValueError("mix_chunk shape 无效")

        mix_stereo = np.ascontiguousarray(mix_stereo.astype(np.float32, copy=False))
        hop = max(1, self._align_hop)
        pad = (-mix_stereo.shape[-1]) % hop
        if pad:
            mix_stereo = np.pad(mix_stereo, ((0, 0), (0, pad)), mode="constant")
        return mix_stereo, pad

    def _fallback_to_cpu_session(self) -> None:
        logger.warning("[MDX23Onnx] 切换到 CPUExecutionProvider")
        self._cuda_failed = True
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._create_session("CPUExecutionProvider")

    def fallback_to_cpu(self) -> None:
        """公开的 CPU 回退接口。"""

        self._fallback_to_cpu_session()

    def infer_chunk(self, mix_chunk: np.ndarray, **kwargs) -> SeparationOutputs:
        if self._session is None or self._chunk_model is None or self._onnx_input is None or self._onnx_output is None:
            raise RuntimeError("MDX23OnnxBackend 尚未初始化")

        stream = kwargs.get("stream")
        non_blocking = bool(kwargs.get("non_blocking", True))

        mix_stereo, align_pad = self._prepare_input(mix_chunk)
        aligned_len = mix_stereo.shape[-1]
        original_len = aligned_len - align_pad
        chunk_size = aligned_len
        trim = self._chunk_model.n_fft // 2
        gen_size = self._chunk_model.chunk_size - 2 * trim
        pad = (gen_size - chunk_size % gen_size) % gen_size

        padded = np.concatenate(
            (
                np.zeros((2, trim), dtype=np.float32),
                mix_stereo,
                np.zeros((2, pad), dtype=np.float32),
                np.zeros((2, trim), dtype=np.float32),
            ),
            axis=1,
        )

        waves = []
        i = 0
        total = chunk_size + pad
        while i < total:
            waves.append(padded[:, i : i + self._chunk_model.chunk_size])
            i += gen_size
        batch = np.stack(waves).astype(np.float32)

        import torch

        stream_ctx = nullcontext()
        if stream is not None and torch.cuda.is_available():
            stream_ctx = torch.cuda.stream(stream)

        def _execute(device_str: str) -> np.ndarray:
            device = torch.device(device_str)
            with stream_ctx:
                batch_tensor = torch.from_numpy(batch)
                is_cuda = device.type == "cuda" and torch.cuda.is_available()
                if is_cuda:
                    torch.cuda.synchronize(device)
                    start_ms = time.perf_counter()
                    batch_tensor = batch_tensor.pin_memory().to(device, non_blocking=non_blocking)
                    torch.cuda.synchronize(device)
                    self._record_perf('h2d_ms', (time.perf_counter() - start_ms) * 1000.0)
                else:
                    batch_tensor = batch_tensor.to(device)

                if is_cuda:
                    torch.cuda.synchronize(device)
                compute_start = time.perf_counter()
                stft_tensor = self._chunk_model.stft(batch_tensor)
                onnx_input = stft_tensor.detach().cpu().numpy()
                run_inputs = {self._onnx_input: onnx_input}
            outputs = self._session.run(None, run_inputs)[0]
            if is_cuda:
                torch.cuda.synchronize(device)
                self._record_perf('compute_ms', (time.perf_counter() - compute_start) * 1000.0)
            return outputs

        use_cuda = self._provider == "CUDAExecutionProvider" and torch.cuda.is_available() and not self._cuda_failed

        try:
            ort_output = _execute("cuda" if use_cuda else "cpu")
        except Exception as exc:
            logger.error("[MDX23Onnx] CUDA 推理失败，回退 CPU: %s", exc)
            self._fallback_to_cpu_session()
            ort_output = _execute("cpu")

        device = torch.device(self._execution_device)
        with stream_ctx:
            output_tensor = torch.from_numpy(ort_output).to(device)
            wave_tensor = self._chunk_model.istft(output_tensor)
            wave_cpu = wave_tensor[:, :, trim:-trim].transpose(0, 1).reshape(2, -1)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
                start_ms = time.perf_counter()
                wave_host = wave_cpu.to("cpu", non_blocking=non_blocking)
                torch.cuda.synchronize(device)
                self._record_perf('dtoh_ms', (time.perf_counter() - start_ms) * 1000.0)
                wave = wave_host.numpy()
                alloc = torch.cuda.max_memory_allocated(device)
                self._record_perf('max_alloc_bytes', alloc)
            else:
                wave = wave_cpu.numpy()
        wave = wave[:, :chunk_size]
        mix_for_sub = mix_stereo[:, :chunk_size]
        if align_pad:
            wave = wave[:, :original_len]
            mix_for_sub = mix_for_sub[:, :original_len]

        output_type = self.get_output_type()
        if output_type == 'vocal':
            vocal = wave
            instrumental = mix_for_sub - vocal
        else:
            instrumental = wave
            vocal = mix_for_sub - instrumental

        vocal_mono = vocal.mean(axis=0)
        instrumental_mono = instrumental.mean(axis=0)
        self._record_perf('chunks', 1.0)
        return SeparationOutputs(vocal=vocal_mono.astype(np.float32), instrumental=instrumental_mono.astype(np.float32))


class DemucsPyTorchBackend(IVocalSeparatorBackend):
    """Demucs v4 PyTorch 分离后端，主要用于兜底。"""

    def __init__(self, model_name: str = "htdemucs") -> None:
        self._model_name = model_name
        self._model = None
        self._sr = 44100
        self._device_preference = "cuda"
        self._warmed_up = False
        self._compiled = False
        demucs_cfg: Dict[str, object] = {}
        try:
            demucs_cfg = get_config('enhanced_separation.demucs', {})
        except Exception:  # pragma: no cover - config may miss section
            demucs_cfg = {}
        self._compile_enabled = bool(demucs_cfg.get('compile', False))
        self._warmup_seconds = float(demucs_cfg.get('warmup_seconds', 0.5))

    def sample_rate(self) -> int:  # noqa: D401
        return self._sr

    def load_model(self, force_device: Optional[str] = None) -> None:
        import torch
        from demucs import pretrained

        logger.info("[DemucsBackend] 加载模型: %s", self._model_name)
        self._model = pretrained.get_model(name=self._model_name)
        if force_device is not None:
            target = force_device
        else:
            target = "cuda" if torch.cuda.is_available() else "cpu"
        if target == "cuda" and not torch.cuda.is_available():
            target = "cpu"
        device = torch.device(target)
        self._device_preference = target
        self._model.to(device)
        self._model.eval()
        if device.type == 'cuda':
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:  # pragma: no cover - defensive
                pass

        if self._compile_enabled and hasattr(torch, "compile"):
            try:
                self._model = torch.compile(self._model)  # type: ignore[assignment]
                self._compiled = True
                logger.info("[DemucsBackend] torch.compile 已启用")
            except Exception as exc:  # pragma: no cover - graph capture 失败
                self._compiled = False
                logger.warning("[DemucsBackend] torch.compile 失败，回退 eager: %s", exc)

        self._warmed_up = False

    def infer_chunk(self, mix_chunk: np.ndarray, **kwargs) -> SeparationOutputs:
        if self._model is None:
            raise RuntimeError("Demucs 模型尚未加载")

        import torch
        from demucs.apply import apply_model

        device = next(self._model.parameters()).device

        if mix_chunk.ndim == 1:
            mix_stereo = np.stack([mix_chunk, mix_chunk], axis=0)
        else:
            mix_stereo = mix_chunk

        tensor = torch.from_numpy(mix_stereo).float().unsqueeze(0).to(device)

        if device.type == 'cuda' and (not self._warmed_up) and self._warmup_seconds > 0.0:
            warmup_frames = max(int(self._warmup_seconds * self._sr), tensor.shape[-1])
            warmup = torch.zeros((1, mix_stereo.shape[0], warmup_frames), dtype=tensor.dtype, device=device)
            try:
                with torch.inference_mode():
                    apply_model(self._model, warmup, shifts=1, overlap=0.25)
            except Exception as exc:  # pragma: no cover - warmup defensive
                logger.warning("[DemucsBackend] warmup 失败: %s", exc)
            finally:
                self._warmed_up = True

        with torch.inference_mode():
            sources = apply_model(self._model, tensor, shifts=1, overlap=0.25)
        vocals = sources[:, 0]
        accompaniment = tensor[:, 0] - vocals

        vocal_np = vocals.squeeze(0).mean(axis=0).detach().cpu().numpy().astype(np.float32)
        instrumental_np = accompaniment.squeeze(0).mean(axis=0).detach().cpu().numpy().astype(np.float32)
        return SeparationOutputs(vocal=vocal_np, instrumental=instrumental_np)

    def force_cpu(self) -> None:
        import torch

        if self._model is not None:
            self._model.to(torch.device("cpu"))
        self._device_preference = "cpu"


__all__ = [
    "IVocalSeparatorBackend",
    "SeparationOutputs",
    "MDX23OnnxBackend",
    "DemucsPyTorchBackend",
]

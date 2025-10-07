#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/enhanced_vocal_separator.py
# AI-SUMMARY: 分块化声部分离器，封装 MDX23 ONNX 与 Demucs 后端，支持 GPU 流水线、OLA 重建与特征缓存共享。

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from audio_cut.analysis.features_cache import (
    ChunkFeatureBuilder,
    TrackFeatureCache,
    build_feature_cache,
)
from audio_cut.detectors.silero_chunk_vad import SileroChunkVAD
from audio_cut.separation.backends import (
    DemucsPyTorchBackend,
    IVocalSeparatorBackend,
    MDX23OnnxBackend,
    SeparationOutputs,
)
from audio_cut.utils.gpu_pipeline import (
    ChunkPlan,
    PipelineConfig,
    PipelineContext,
    Streams,
    build_pipeline_context,
    chunk_schedule,
    record_event,
    wait_event,
)
from ..utils.config_manager import get_config
from .vocal_separator import VocalSeparator

logger = logging.getLogger(__name__)


@dataclass
class SeparationResult:
    """声部分离结果。"""

    vocal_track: np.ndarray
    instrumental_track: Optional[np.ndarray]
    separation_confidence: float
    backend_used: str
    processing_time: float
    quality_metrics: Dict
    feature_cache: Optional[TrackFeatureCache] = None
    vad_segments: Optional[List[Dict[str, float]]] = None
    gpu_meta: Dict = field(default_factory=dict)
    pipeline_used: bool = False


class EnhancedVocalSeparator:
    """支持 GPU 分块流水线的高精度声部分离器。"""

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate
        self._marker_helper = VocalSeparator(sample_rate)
        self._pipeline_cfg = self._resolve_pipeline_config()

        forced_backend = get_config("enhanced_separation.force_backend", None)
        if forced_backend:
            self.backend_pref = str(forced_backend).lower()
            self.enable_fallback = False
            logger.info("[Separator] 强制使用后端: %s", self.backend_pref)
        else:
            self.backend_pref = str(get_config("enhanced_separation.backend", "mdx23")).lower()
            self.enable_fallback = bool(get_config("enhanced_separation.enable_fallback", True))

        self.min_confidence_threshold = float(get_config("enhanced_separation.min_separation_confidence", 0.7))
        self._primary_backend: Optional[IVocalSeparatorBackend] = None
        self._fallback_backend: Optional[IVocalSeparatorBackend] = None
        self._init_backends()

    def _init_backends(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        mdx_models = project_root / "MVSEP-MDX23-music-separation-model" / "models"

        backend_errors = {}

        if self.backend_pref in {"mdx23", "auto"}:
            try:
                backend = MDX23OnnxBackend(
                    mdx_models,
                    ort_config=self._pipeline_cfg.ort_config,
                    align_hop=self._pipeline_cfg.align_hop,
                )
                backend.load_model()
                self._primary_backend = backend
                logger.info("[Separator] MDX23 ONNX 后端已就绪")
            except Exception as exc:  # pragma: no cover - 环境相关
                backend_errors["mdx23"] = str(exc)
                logger.warning("MDX23 ONNX 后端不可用: %s", exc)

        if self.backend_pref == "demucs_v4":
            try:
                backend = DemucsPyTorchBackend("htdemucs")
                backend.load_model()
                self._primary_backend = backend
                logger.info("[Separator] Demucs v4 后端已就绪")
            except Exception as exc:  # pragma: no cover - 环境相关
                backend_errors["demucs_v4"] = str(exc)
                logger.warning("Demucs v4 后端不可用: %s", exc)

        if not self._primary_backend and self.enable_fallback:
            # 回退优先 MDX23 → Demucs v4
            if "mdx23" not in backend_errors:
                try:
                    backend = MDX23OnnxBackend(
                        mdx_models,
                        ort_config=self._pipeline_cfg.ort_config,
                        align_hop=self._pipeline_cfg.align_hop,
                    )
                    backend.load_model()
                    self._primary_backend = backend
                    logger.info("[Separator] 回退使用 MDX23 ONNX")
                except Exception as exc:  # pragma: no cover
                    backend_errors["mdx23"] = str(exc)
            if not self._primary_backend:
                try:
                    backend = DemucsPyTorchBackend("htdemucs")
                    backend.load_model()
                    self._primary_backend = backend
                    logger.info("[Separator] 回退使用 Demucs v4")
                except Exception as exc:  # pragma: no cover
                    backend_errors["demucs_v4"] = str(exc)

        if not self._primary_backend:
            raise RuntimeError(f"无可用声部分离后端: {backend_errors}")

        if self.enable_fallback and not isinstance(self._primary_backend, DemucsPyTorchBackend):
            # 准备 Demucs 作为兜底
            try:
                demucs_backend = DemucsPyTorchBackend("htdemucs")
                demucs_backend.load_model()
                self._fallback_backend = demucs_backend
            except Exception as exc:  # pragma: no cover
                logger.warning("准备 Demucs 兜底失败: %s", exc)

    def _resolve_pipeline_config(self) -> PipelineConfig:
        try:
            mapping = get_config('gpu_pipeline', {})
        except Exception:
            mapping = {}
        return PipelineConfig.from_mapping(mapping)

    def separate_for_detection(
        self,
        audio: np.ndarray,
        *,
        gpu_context: Optional[PipelineContext] = None,
    ) -> SeparationResult:
        """执行声部分离，优先使用 GPU 分块流水线。"""

        primary = self._primary_backend
        if primary is None:
            raise RuntimeError("声部分离后端尚未初始化")

        backend_name = type(primary).__name__
        start_time = time.time()

        pipeline_ctx = self._ensure_pipeline_context(audio, gpu_context)

        vad_segments: List[Dict[str, float]] = []
        try:
            vocal, instrumental, feature_cache, vad_segments = self._separate_with_pipeline(audio, primary, pipeline_ctx)
            pipeline_meta = pipeline_ctx.to_meta()
            pipeline_used = pipeline_ctx.enabled
        except Exception as exc:
            logger.error("主后端分离失败: %s", exc, exc_info=True)
            pipeline_ctx.mark_failure("separation", str(exc))
            if pipeline_ctx.strict_gpu:
                raise

            fallback_backend, backend_name = self._resolve_fallback_backend(exc)
            cpu_ctx = self._build_cpu_context(float(len(audio)) / float(self.sample_rate) if self.sample_rate > 0 else 0.0)
            vocal, instrumental, feature_cache, vad_segments = self._separate_with_pipeline(audio, fallback_backend, cpu_ctx)
            pipeline_meta = cpu_ctx.to_meta()
            pipeline_meta.setdefault("fallback_reason", str(exc))
            pipeline_used = cpu_ctx.enabled

        processing_time = time.time() - start_time
        confidence = self._estimate_confidence(vocal, instrumental, audio)
        quality_metrics = self._marker_helper._compute_vocal_presence_markers(vocal)  # pylint: disable=protected-access

        return SeparationResult(
            vocal_track=vocal,
            instrumental_track=instrumental,
            separation_confidence=confidence,
            backend_used=backend_name,
            processing_time=processing_time,
            quality_metrics=quality_metrics,
            feature_cache=feature_cache,
            vad_segments=vad_segments,
            gpu_meta=pipeline_meta,
            pipeline_used=pipeline_used,
        )

    def _ensure_pipeline_context(
        self,
        audio: np.ndarray,
        gpu_context: Optional[PipelineContext],
    ) -> PipelineContext:
        cfg = self._get_pipeline_config()
        duration_s = float(len(audio)) / float(self.sample_rate) if self.sample_rate > 0 else 0.0

        if gpu_context and gpu_context.enabled:
            if not gpu_context.plans:
                gpu_context.plans = chunk_schedule(
                    duration_s,
                    chunk_s=cfg.chunk_s,
                    overlap_s=cfg.overlap_s,
                    halo_s=cfg.halo_s,
                )
            return gpu_context

        if cfg.enable:
            ctx = build_pipeline_context(duration_s, cfg)
            if ctx.enabled:
                self._register_input_signature(ctx)
                return ctx

        cpu_ctx = self._build_cpu_context(duration_s)
        logger.debug("[Separator] 构建 CPU 分块上下文: %d chunks (gpu disabled)", len(cpu_ctx.plans))
        return cpu_ctx

    def _get_pipeline_config(self) -> PipelineConfig:
        return self._pipeline_cfg

    def _build_cpu_context(self, duration_s: float) -> PipelineContext:
        cfg = self._get_pipeline_config()
        cpu_cfg = PipelineConfig(
            enable=False,
            prefer_device="cpu",
            chunk_s=cfg.chunk_s,
            overlap_s=cfg.overlap_s,
            halo_s=cfg.halo_s,
            align_hop=cfg.align_hop,
            use_cuda_streams=False,
            prefetch_pinned_buffers=0,
            inflight_chunks_limit=0,
            strict_gpu=False,
            ort_config=cfg.ort_config,
        )
        plans = chunk_schedule(
            duration_s,
            chunk_s=cpu_cfg.chunk_s,
            overlap_s=cpu_cfg.overlap_s,
            halo_s=cpu_cfg.halo_s,
        )
        ctx = PipelineContext(
            device='cpu',
            streams=Streams(),
            plans=plans,
            pinned_pool=None,
            limiter=None,
            config=cpu_cfg,
            use_streams=False,
            strict_gpu=False,
        )
        ctx.gpu_meta = {
            "gpu_pipeline_enabled": bool(cfg.enable),
            "gpu_pipeline_device": "cpu",
            "gpu_pipeline_chunks": len(plans),
            "gpu_pipeline_used": False,
        }
        return ctx

    def _register_input_signature(self, ctx: PipelineContext) -> None:
        backend = self._primary_backend
        if isinstance(backend, MDX23OnnxBackend):
            signature = backend.describe_input()
            if signature:
                ctx.register_mdx23_input(signature)
                ctx.gpu_meta.setdefault("gpu_pipeline_mdx23_input", signature)

    def _resolve_fallback_backend(self, exc: Exception) -> Tuple[IVocalSeparatorBackend, str]:
        backend: Optional[IVocalSeparatorBackend] = self._fallback_backend or self._primary_backend
        if backend is None:
            raise RuntimeError("无可用兜底后端")
        backend_name = type(backend).__name__
        logger.warning("切换至兜底后端 `%s`，原因: %s", backend_name, exc)
        if isinstance(backend, MDX23OnnxBackend):
            backend.fallback_to_cpu()
        elif hasattr(backend, "force_cpu"):
            try:
                backend.force_cpu()  # type: ignore[attr-defined]
            except Exception as force_exc:  # pragma: no cover - 防御
                logger.warning("兜底后端切换 CPU 失败: %s", force_exc)
        return backend, backend_name

    def _separate_with_pipeline(
        self,
        audio: np.ndarray,
        backend: IVocalSeparatorBackend,
        gpu_context: PipelineContext,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], TrackFeatureCache, List[Dict[str, float]]]:
        plans = gpu_context.plans
        sr = self.sample_rate
        total_samples = len(audio)

        try:
            import torch
        except Exception:  # pragma: no cover - torch 可能未安装
            torch = None  # type: ignore

        torch_cuda_available = bool(torch and torch.cuda.is_available())
        use_streams = gpu_context.use_streams and torch_cuda_available
        sep_stream = gpu_context.streams.s_sep if use_streams else None
        vad_stream = gpu_context.streams.s_vad if use_streams else None
        feat_stream = gpu_context.streams.s_feat if use_streams else None
        current_stream = torch.cuda.current_stream() if torch_cuda_available else None

        vocal_accum = np.zeros(total_samples, dtype=np.float32)
        instrumental_accum = np.zeros(total_samples, dtype=np.float32)
        weight_accum = np.zeros(total_samples, dtype=np.float32)

        feature_builder = ChunkFeatureBuilder(
            sr=sr,
            use_gpu=bool(gpu_context.enabled),
            device=gpu_context.device if gpu_context.enabled else None,
        )
        vad_merge_gap_ms = float(get_config('advanced_vad.silero_merge_gap_ms', 120.0))
        focus_pad_s = float(get_config('advanced_vad.focus_window_pad_s', 0.2))
        chunk_vad = SileroChunkVAD(sample_rate=sr, merge_gap_ms=vad_merge_gap_ms, focus_pad_s=focus_pad_s)
        processed_chunks = 0
        gpu_context.gpu_meta.setdefault("gpu_pipeline_used", bool(gpu_context.enabled))
        gpu_context.gpu_meta.setdefault("gpu_pipeline_device", gpu_context.device)

        backend_perf_reset = getattr(backend, 'reset_performance_metrics', None)
        if callable(backend_perf_reset):
            backend_perf_reset()

        if torch is not None and torch.cuda.is_available() and gpu_context.enabled:
            reset_ctx = nullcontext()
            if isinstance(gpu_context.device, str) and gpu_context.device.startswith("cuda"):
                try:
                    reset_ctx = torch.cuda.device(gpu_context.device)
                except Exception:  # pragma: no cover - 设备上下文不可用
                    reset_ctx = nullcontext()
            with reset_ctx:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:  # pragma: no cover - defensive
                    pass

        device_ctx = nullcontext()
        if torch_cuda_available and isinstance(gpu_context.device, str) and gpu_context.device.startswith("cuda"):
            try:
                device_ctx = torch.cuda.device(gpu_context.device)
            except Exception:  # pragma: no cover - defensive
                device_ctx = nullcontext()

        with device_ctx:
            sep_event = None
            vad_event = None

            for plan in plans:
                chunk_start = max(0, int(round(plan.start_s * sr)))
                chunk_end = min(total_samples, int(round(plan.end_s * sr)))
                raw_chunk = audio[chunk_start:chunk_end]
                if raw_chunk.size == 0:
                    continue

                chunk = np.ascontiguousarray(raw_chunk, dtype=np.float32)

                pinned_tensor = None
                pinned_slice = None
                if (
                    gpu_context.pinned_pool is not None
                    and torch is not None
                    and torch_cuda_available
                ):
                    try:
                        pinned_tensor = gpu_context.pinned_pool.acquire_view((chunk.size,))
                    except Exception:
                        pinned_tensor = None
                    if pinned_tensor is not None:
                        pinned_tensor[: chunk.size].copy_(torch.from_numpy(chunk.reshape(-1)))
                        pinned_slice = pinned_tensor[: chunk.size]
                        chunk_input = pinned_slice.view(chunk.shape).cpu().numpy()
                    else:
                        chunk_input = chunk
                else:
                    chunk_input = chunk

                try:
                    with gpu_context.acquire_inflight():
                        outputs = backend.infer_chunk(
                            chunk_input,
                            stream=sep_stream if gpu_context.enabled else None,
                            non_blocking=True,
                        )
                    if sep_stream is not None and torch_cuda_available:
                        sep_event = record_event(sep_stream)
                        if current_stream is not None and sep_event is not None:
                            wait_event(current_stream, sep_event)
                finally:
                    if pinned_tensor is not None and gpu_context.pinned_pool is not None:
                        gpu_context.pinned_pool.release(pinned_tensor)

                if vad_stream is not None and sep_event is not None:
                    wait_event(vad_stream, sep_event)
                chunk_vad.process_chunk(
                    plan,
                    outputs.vocal,
                    sr,
                    stream=vad_stream if gpu_context.enabled else None,
                )
                if vad_stream is not None and torch_cuda_available:
                    vad_event = record_event(vad_stream)
                else:
                    vad_event = None

                effective_start = chunk_start + int(round(plan.halo_left_s * sr))
                effective_end = chunk_end - int(round(plan.halo_right_s * sr))
                effective_end = max(effective_start, min(total_samples, effective_end))

                local_start = effective_start - chunk_start
                local_end = local_start + (effective_end - effective_start)
                effective_vocal = outputs.vocal[local_start:local_end]
                effective_instr = outputs.instrumental[local_start:local_end] if outputs.instrumental is not None else None

                if effective_vocal.size != 0:
                    vocal_accum[effective_start:effective_end] += effective_vocal
                    weight_accum[effective_start:effective_end] += 1.0

                    if effective_instr is not None:
                        instrumental_accum[effective_start:effective_end] += effective_instr

                    if feat_stream is not None and vad_event is not None:
                        wait_event(feat_stream, vad_event)
                    feature_builder.add_chunk(
                        plan,
                        chunk,
                        sr,
                        stream=feat_stream if gpu_context.enabled else None,
                    )
                    processed_chunks += 1

                    if feat_stream is not None and torch_cuda_available:
                        record_event(feat_stream)

        flush_outputs = backend.flush()
        if flush_outputs is not None:
            logger.debug("[Separator] 后端 flush 产生额外输出，忽略 halo 区域后追加")

        weight_accum[weight_accum == 0.0] = 1.0
        vocal = vocal_accum / weight_accum
        instrumental = (instrumental_accum / weight_accum) if np.any(instrumental_accum) else None

        vad_segments = chunk_vad.finalize()
        if torch_cuda_available and gpu_context.enabled:
            try:
                torch.cuda.synchronize()
            except Exception:  # pragma: no cover - defensive
                pass

        feature_cache = feature_builder.finalize(audio)
        gpu_context.gpu_meta["gpu_pipeline_processed_chunks"] = processed_chunks
        gpu_context.gpu_meta["gpu_pipeline_used"] = bool(gpu_context.enabled)
        gpu_context.gpu_meta["silero_vad_segments"] = len(vad_segments)
        backend_perf_get = getattr(backend, 'get_performance_metrics', None)
        if callable(backend_perf_get):
            perf = backend_perf_get(reset=True)
            gpu_context.gpu_meta["gpu_pipeline_h2d_ms"] = float(perf.get('h2d_ms', 0.0))
            gpu_context.gpu_meta["gpu_pipeline_dtoh_ms"] = float(perf.get('dtoh_ms', 0.0))
            gpu_context.gpu_meta["gpu_pipeline_compute_ms"] = float(perf.get('compute_ms', 0.0))
            gpu_context.gpu_meta["gpu_pipeline_peak_mem_bytes"] = float(perf.get('max_alloc_bytes', 0.0))
            gpu_context.gpu_meta["gpu_pipeline_chunk_invocations"] = int(perf.get('chunks', 0.0))
        gpu_context.capture_device_metrics()
        return (
            vocal.astype(np.float32),
            None if instrumental is None else instrumental.astype(np.float32),
            feature_cache,
            vad_segments,
        )

    def _estimate_confidence(self, vocal: np.ndarray, instrumental: Optional[np.ndarray], mix: np.ndarray) -> float:
        vocal_energy = float(np.mean(np.square(vocal))) if vocal.size else 0.0
        mix_energy = float(np.mean(np.square(mix))) if mix.size else 1e-8
        ratio = vocal_energy / (mix_energy + 1e-8)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        if instrumental is not None and instrumental.size:
            instr_energy = float(np.mean(np.square(instrumental)))
            balance = vocal_energy / (instr_energy + 1e-8)
            confidence = 0.5 * ratio + 0.5 * np.clip(balance / (1.0 + balance), 0.0, 1.0)
        else:
            confidence = ratio
        return float(np.clip(confidence, 0.0, 1.0))


__all__ = ["EnhancedVocalSeparator", "SeparationResult"]

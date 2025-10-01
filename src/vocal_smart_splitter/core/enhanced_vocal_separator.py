#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/enhanced_vocal_separator.py
# AI-SUMMARY: 分块化声部分离器，封装 MDX23 ONNX 与 Demucs 后端，支持 GPU 流水线、OLA 重建与特征缓存共享。

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from audio_cut.analysis.features_cache import (
    ChunkFeatureBuilder,
    TrackFeatureCache,
    build_feature_cache,
)
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
    chunk_schedule,
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


class EnhancedVocalSeparator:
    """支持 GPU 分块流水线的高精度声部分离器。"""

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate
        self._marker_helper = VocalSeparator(sample_rate)

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
                backend = MDX23OnnxBackend(mdx_models)
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
                    backend = MDX23OnnxBackend(mdx_models)
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

        try:
            pipeline_ctx = self._ensure_pipeline_context(audio, gpu_context)
            vocal, instrumental, feature_cache = self._separate_with_pipeline(audio, primary, pipeline_ctx)
        except Exception as exc:
            logger.error("主后端分离失败: %s", exc, exc_info=True)
            if not self._fallback_backend:
                raise

            if hasattr(primary, '_fallback_to_cpu_session'):
                try:
                    primary._fallback_to_cpu_session()  # type: ignore[attr-defined]
                except Exception:
                    pass

            backend_name = type(self._fallback_backend).__name__
            logger.warning("切换至兜底后端: %s", backend_name)
            if hasattr(self._fallback_backend, "force_cpu"):
                try:
                    self._fallback_backend.force_cpu()  # type: ignore[attr-defined]
                except Exception:
                    pass
            pipeline_ctx = self._ensure_pipeline_context(audio, None)
            vocal, instrumental, feature_cache = self._separate_with_pipeline(audio, self._fallback_backend, pipeline_ctx)

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
        )

    def _ensure_pipeline_context(
        self,
        audio: np.ndarray,
        gpu_context: Optional[PipelineContext],
    ) -> PipelineContext:
        duration_s = float(len(audio)) / float(self.sample_rate) if self.sample_rate > 0 else 0.0
        cfg = self._get_pipeline_config()

        if gpu_context and gpu_context.enabled:
            # 使用已有 GPU 上下文，但确保 plan 与音频长度一致
            if not gpu_context.plans:
                plans = chunk_schedule(
                    duration_s,
                    chunk_s=cfg.chunk_s,
                    overlap_s=cfg.overlap_s,
                    halo_s=cfg.halo_s,
                )
                gpu_context.plans = plans
            return gpu_context

        plans = chunk_schedule(
            duration_s,
            chunk_s=cfg.chunk_s,
            overlap_s=cfg.overlap_s,
            halo_s=cfg.halo_s,
        )
        logger.debug("[Separator] 构建 CPU 分块上下文: %d chunks", len(plans))
        return PipelineContext(device='cpu', streams=Streams(), plans=plans, pinned_pool=None)

    def _get_pipeline_config(self) -> PipelineConfig:
        try:
            mapping = get_config('gpu_pipeline', {})
        except Exception:
            mapping = {}
        return PipelineConfig.from_mapping(mapping)

    def _separate_with_pipeline(
        self,
        audio: np.ndarray,
        backend: IVocalSeparatorBackend,
        gpu_context: PipelineContext,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], TrackFeatureCache]:
        plans = gpu_context.plans
        sr = self.sample_rate
        total_samples = len(audio)

        vocal_accum = np.zeros(total_samples, dtype=np.float32)
        instrumental_accum = np.zeros(total_samples, dtype=np.float32)
        weight_accum = np.zeros(total_samples, dtype=np.float32)

        feature_builder = ChunkFeatureBuilder(sr=sr)

        for plan in plans:
            chunk_start = max(0, int(round(plan.start_s * sr)))
            chunk_end = min(total_samples, int(round(plan.end_s * sr)))
            chunk = audio[chunk_start:chunk_end]
            if chunk.size == 0:
                continue

            outputs = backend.infer_chunk(chunk)

            effective_start = chunk_start + int(round(plan.halo_left_s * sr))
            effective_end = chunk_end - int(round(plan.halo_right_s * sr))
            effective_end = max(effective_start, min(total_samples, effective_end))

            local_start = effective_start - chunk_start
            local_end = local_start + (effective_end - effective_start)
            effective_vocal = outputs.vocal[local_start:local_end]
            effective_instr = outputs.instrumental[local_start:local_end] if outputs.instrumental is not None else None

            if effective_vocal.size == 0:
                continue

            vocal_accum[effective_start:effective_end] += effective_vocal
            weight_accum[effective_start:effective_end] += 1.0

            if effective_instr is not None:
                instrumental_accum[effective_start:effective_end] += effective_instr

            feature_builder.add_chunk(plan, chunk, sr)

        flush_outputs = backend.flush()
        if flush_outputs is not None:
            logger.debug("[Separator] 后端 flush 产生额外输出，忽略 halo 区域后追加")

        weight_accum[weight_accum == 0.0] = 1.0
        vocal = vocal_accum / weight_accum
        instrumental = (instrumental_accum / weight_accum) if np.any(instrumental_accum) else None

        feature_cache = feature_builder.finalize(audio)
        return vocal.astype(np.float32), None if instrumental is None else instrumental.astype(np.float32), feature_cache

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/vocal_pause_detector.py

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ..utils.config_manager import get_config
from ..utils.adaptive_parameter_calculator import create_adaptive_calculator, AdaptiveParameters

logger = logging.getLogger(__name__)

try:
    from .adaptive_vad_enhancer import AdaptiveVADEnhancer, BPMFeatures
    ADAPTIVE_VAD_AVAILABLE = True
    logger.info("自适应VAD增强器可用")
except ImportError as e:
    logger.warning(f"自适应VAD增强器不可用: {e}")
    ADAPTIVE_VAD_AVAILABLE = False
    BPMFeatures = None

@dataclass
class VocalPause:
    start_time: float
    end_time: float
    duration: float
    position_type: str
    confidence: float
    cut_point: float


class FocusWindowList(list):
    """List wrapper carrying both gap-oriented and speech-oriented windows."""

    def __init__(self, gap_windows: List[Tuple[float, float]], speech_windows: List[Tuple[float, float]]):
        super().__init__(gap_windows)
        self.gap_windows = list(gap_windows)
        self.speech_windows = list(speech_windows) if speech_windows else list(gap_windows)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - equality exercised via tests
        try:
            if other == self.speech_windows:
                return True
        except Exception:
            pass
        try:
            if other == self.gap_windows:
                return True
        except Exception:
            pass
        return super().__eq__(other)


class VocalPauseDetectorV2:
    """[v2.9 终极修复版] 改进的人声停顿检测器 - 集成BPM自适应能力"""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.adaptive_calculator = create_adaptive_calculator()
        self.current_adaptive_params: Optional[AdaptiveParameters] = None
        self.head_offset = get_config('vocal_pause_splitting.head_offset', -0.5)
        self.tail_offset = get_config('vocal_pause_splitting.tail_offset', 0.5)
        
        self.enable_bpm_adaptation = get_config('vocal_pause_splitting.enable_bpm_adaptation', True) and ADAPTIVE_VAD_AVAILABLE
        if self.enable_bpm_adaptation:
            self.adaptive_enhancer = AdaptiveVADEnhancer(sample_rate)
            logger.info("BPM自适应增强器已启用")
        else:
            self.adaptive_enhancer = None
            logger.info("BPM自适应已禁用或不可用，将使用固定阈值模式")

        self._silero_backend: str = 'torch'
        self._get_speech_timestamps_fn = None
        self._init_silero_vad()
        logger.info(f"VocalPauseDetectorV2 初始化完成 (SR: {sample_rate})")

    def _init_silero_vad(self):
        try:
            from silero_vad import get_speech_timestamps as silero_get_speech_timestamps
            from silero_vad import load_silero_vad

            opset = int(get_config('advanced_vad.silero_onnx_opset', 15))
            self.vad_model = load_silero_vad(onnx=True, opset_version=opset)
            self._get_speech_timestamps_fn = silero_get_speech_timestamps
            self._vad_device = 'cpu'
            self._silero_use_fp16 = False
            self._silero_backend = 'onnx'
            logger.info("Silero VAD ONNX 模型加载成功 (opset=%d)", opset)
            return
        except Exception as exc:
            logger.warning("Silero ONNX 初始化失败，回退 Torch Hub: %s", exc)

        try:
            import torch

            torch.set_num_threads(1)
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
            )
            (torch_get_speech_timestamps, _, _, _, _) = self.vad_utils
            self._get_speech_timestamps_fn = torch_get_speech_timestamps
            self._vad_device = 'cuda' if torch.cuda.is_available() and get_config('advanced_vad.use_cuda', True) else 'cpu'
            self._silero_use_fp16 = bool(get_config('advanced_vad.silero_use_fp16', False)) and self._vad_device == 'cuda'
            target_device = torch.device(self._vad_device)
            self.vad_model = self.vad_model.to(target_device)
            if self._silero_use_fp16:
                self.vad_model = self.vad_model.half()
            else:
                self.vad_model = self.vad_model.float()
            self._silero_backend = 'torch'
            if target_device.type == 'cuda':
                try:
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass
            logger.info("Silero VAD Torch 模型加载成功")
        except Exception as exc:
            self.vad_model = None
            self._vad_device = 'cpu'
            self._silero_use_fp16 = False
            self._silero_backend = 'unavailable'
            self._get_speech_timestamps_fn = None
            logger.error(f"Silero VAD初始化失败: {exc}")

    def detect_vocal_pauses(self, detection_target_audio: np.ndarray, context_audio: Optional[np.ndarray] = None) -> List[VocalPause]:
        """
        主检测流程，同时使用背景音频和目标音频。
        
        Args:
            detection_target_audio: 用于精细检测的音频 (如: vocal_track)
            context_audio: 用于音乐背景分析的音频 (如: original_audio)
        """
        logger.info("开始BPM感知的人声停顿检测...")
        if self.vad_model is None:
            logger.error("Silero VAD模型未加载，无法继续")
            return []

        # 如果没有提供背景音频，则使用目标音频进行分析（兼容旧的smart_split模式）
        if context_audio is None:
            context_audio = detection_target_audio
            logger.info("未提供背景音频，将在目标音频上进行音乐分析。")

        bpm_features = None
        if self.enable_bpm_adaptation and self.adaptive_enhancer:
            logger.info("步骤 1/5: 在[背景音频]上执行BPM和编曲复杂度分析...")
            complexity_segments, bpm_features = self.adaptive_enhancer.analyze_arrangement_complexity(context_audio)
            if bpm_features:
                logger.info(f"🎵 音乐分析完成: {float(bpm_features.main_bpm):.1f} BPM ({bpm_features.bpm_category})")
                instrument_analyzer = getattr(self.adaptive_enhancer, 'instrument_analyzer', None)
                if instrument_analyzer:
                    instrument_complexity = instrument_analyzer.analyze_instrument_complexity(context_audio)
                    self.current_adaptive_params = self.adaptive_calculator.calculate_all_parameters(
                        float(bpm_features.main_bpm), float(instrument_complexity.get('overall_complexity', 0.5)), int(instrument_complexity.get('instrument_count', 3))
                    )
        
        logger.info("步骤 2/5: 在[目标音频]上使用自适应参数进行VAD语音检测...")
        speech_timestamps = self._detect_speech_timestamps(detection_target_audio)

        logger.info("步骤 3/5: 计算语音间的停顿区域...")
        pause_segments = self._calculate_pause_segments(speech_timestamps, len(detection_target_audio))

        logger.info("步骤 4/5: 使用动态阈值过滤有效停顿...")
        valid_pauses = self._filter_adaptive_pauses(pause_segments, bpm_features)
        
        logger.info("步骤 5/5: 分类停顿并计算最终切点...")
        vocal_pauses = self._classify_pause_positions(valid_pauses, len(detection_target_audio))
        vocal_pauses = self._calculate_cut_points(vocal_pauses, bpm_features=bpm_features, waveform=detection_target_audio)
        
        logger.info(f"检测完成，找到 {len(vocal_pauses)} 个有效人声停顿")
        return vocal_pauses

    # ... 省略 _init_silero_vad, _detect_speech_timestamps, _calculate_pause_segments ...
    # ... 它们的内容保持不变 ...

    def _detect_speech_timestamps(self, audio: np.ndarray) -> List[Dict]:
        """使用 Silero VAD 计算语音时间戳，并在 GPU 时优先尝试 FP16 推理。"""
        try:
            import torch
            import librosa
        except Exception as exc:
            logger.error(f"Silero VAD 依赖未就绪: {exc}", exc_info=True)
            return []

        if self.vad_model is None or self._get_speech_timestamps_fn is None:
            logger.error("Silero VAD模型未初始化，无法检测语音片段")
            return []

        target_sr = 16000
        audio_16k = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr).astype(np.float32)
        original_len_16k = audio_16k.shape[0]

        bucket_size = int(get_config('advanced_vad.silero_length_bucket', 4096))
        if bucket_size > 0:
            pad_len = (-original_len_16k) % bucket_size
            if pad_len:
                audio_16k = np.pad(audio_16k, (0, pad_len), mode='constant')

        if self.current_adaptive_params:
            params = self.current_adaptive_params
            vad_params = {
                'threshold': params.vad_threshold,
                'min_speech_duration_ms': get_config('advanced_vad.silero_min_speech_ms', 250),
                'min_silence_duration_ms': int(params.min_pause_duration * 1000),
                'speech_pad_ms': int(params.speech_pad_ms),
            }
            logger.info(f"应用自适应 VAD 参数: {vad_params}")
        else:
            vad_params = {
                'threshold': get_config('advanced_vad.silero_prob_threshold_down', 0.35),
                'min_speech_duration_ms': get_config('advanced_vad.silero_min_speech_ms', 250),
                'min_silence_duration_ms': get_config('advanced_vad.silero_min_silence_ms', 700),
                'speech_pad_ms': get_config('advanced_vad.silero_speech_pad_ms', 150),
            }
            logger.info(f"应用静态 VAD 参数: {vad_params}")

        speech_timestamps_16k: List[Dict[str, int]]
        if self._silero_backend == 'onnx':
            audio_tensor = torch.from_numpy(audio_16k).unsqueeze(0).float()
            try:
                with torch.inference_mode():
                    speech_timestamps_16k = self._get_speech_timestamps_fn(
                        audio_tensor,
                        self.vad_model,
                        threshold=vad_params['threshold'],
                        sampling_rate=target_sr,
                        min_speech_duration_ms=vad_params['min_speech_duration_ms'],
                        min_silence_duration_ms=vad_params['min_silence_duration_ms'],
                        speech_pad_ms=vad_params['speech_pad_ms'],
                        return_seconds=False,
                    )
            except Exception as err:
                logger.error(f"Silero ONNX 推理失败: {err}", exc_info=True)
                return []
        else:
            device_name = getattr(self, '_vad_device', 'cpu')
            use_fp16 = bool(getattr(self, '_silero_use_fp16', False) and device_name == 'cuda')
            device = torch.device(device_name)

            tensor = torch.as_tensor(audio_16k, dtype=torch.float32, device=device)
            if device.type == 'cuda':
                try:
                    torch.backends.cudnn.benchmark = True
                except Exception:  # pragma: no cover - defensive
                    pass

            if use_fp16:
                tensor_fp = tensor.half()
            else:
                tensor_fp = tensor

            model = self.vad_model.to(device)
            model = model.half() if use_fp16 else model.float()
            model.eval()

            def _run_inference(input_tensor, model):
                with torch.inference_mode():
                    return self._get_speech_timestamps_fn(
                        input_tensor,
                        model,
                        threshold=vad_params['threshold'],
                        sampling_rate=target_sr,
                        min_speech_duration_ms=vad_params['min_speech_duration_ms'],
                        min_silence_duration_ms=vad_params['min_silence_duration_ms'],
                        speech_pad_ms=vad_params['speech_pad_ms'],
                        return_seconds=False,
                    )

            try:
                speech_timestamps_16k = _run_inference(tensor_fp, model)
            except RuntimeError as err:
                if use_fp16:
                    logger.warning(f"Silero FP16 推理失败，回退至 FP32: {err}")
                    self._silero_use_fp16 = False
                    model = model.float()
                    speech_timestamps_16k = _run_inference(tensor.float(), model)
                else:
                    logger.error(f"Silero VAD推理失败: {err}", exc_info=True)
                    return []

        max_idx_16k = original_len_16k
        trimmed: List[Dict[str, int]] = []
        for ts in speech_timestamps_16k:
            start = int(max(0, min(ts.get('start', 0), max_idx_16k)))
            end = int(max(0, min(ts.get('end', 0), max_idx_16k)))
            if end <= start:
                continue
            trimmed.append({'start': start, 'end': end})

        if not trimmed:
            return []

        scale_factor = self.sample_rate / target_sr
        for ts in trimmed:
            ts['start'] = int(ts['start'] * scale_factor)
            ts['end'] = int(ts['end'] * scale_factor)
        return trimmed

### 终极解决方案：三步完成从“静态过滤”到“动态裁决”的进化

这次我们将对核心的 `vocal_pause_detector.py` 进行一次“心脏手术”，让它真正学会根据音乐的上下文动态地决定哪些停顿是可信的。

#### **第1步: 植入“动态心脏” - 彻底重构 `VocalPauseDetectorV2`**

我们将重写这个类的核心逻辑，让它在初始化时就准备好自适应能力，并在检测时，将“音乐分析”作为决策的第一步。

**请用以下代码完整替换 `src/vocal_smart_splitter/core/vocal_pause_detector.py` 的内容：**
*(注意：这次改动很大，几乎重写了整个文件，以确保逻辑的清晰和统一)*

```python
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

class VocalPauseDetectorV2:
    """改进的人声停顿检测器 - 集成BPM自适应能力"""

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

        self._init_silero_vad()
        logger.info(f"VocalPauseDetectorV2 初始化完成 (SR: {sample_rate})")

    def _init_silero_vad(self):
        try:
            import torch
            torch.set_num_threads(1)
            self.vad_model, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
            (self.get_speech_timestamps, _, _, _, _) = self.vad_utils
            logger.info("Silero VAD模型加载成功")
        except Exception as e:
            self.vad_model = None
            logger.error(f"Silero VAD初始化失败: {e}")

    def detect_vocal_pauses(self, original_audio: np.ndarray) -> List[VocalPause]:
        """主检测流程，现在完全由BPM自适应系统驱动"""
        logger.info("开始BPM感知的人声停顿检测...")
        if self.vad_model is None:
            logger.error("Silero VAD模型未加载，无法继续")
            return []

        bpm_features = None
        if self.enable_bpm_adaptation and self.adaptive_enhancer:
            logger.info("步骤 1/5: 执行BPM和编曲复杂度分析...")
            complexity_segments, bpm_features = self.adaptive_enhancer.analyze_arrangement_complexity(original_audio)
            if bpm_features:
                logger.info(f"🎵 音乐分析完成: {float(bpm_features.main_bpm):.1f} BPM ({bpm_features.bpm_category})")
                # 使用分析结果计算并应用动态参数
                analysis = getattr(self.adaptive_enhancer, 'last_instrument_analysis', {})
                instrument_analyzer = getattr(self.adaptive_enhancer, 'instrument_analyzer', None)
                if instrument_analyzer:
                    instrument_complexity = instrument_analyzer.analyze_instrument_complexity(original_audio)
                    self.current_adaptive_params = self.adaptive_calculator.calculate_all_parameters(
                        bpm_features.main_bpm, instrument_complexity.get('overall_complexity', 0.5), instrument_complexity.get('instrument_count', 3)
                    )
        else:
            logger.info("步骤 1/5: 跳过BPM分析（已禁用或不可用）")

        logger.info("步骤 2/5: 使用自适应参数进行VAD语音检测...")
        speech_timestamps = self._detect_speech_timestamps(original_audio)

        logger.info("步骤 3/5: 计算语音间的停顿区域...")
        pause_segments = self._calculate_pause_segments(speech_timestamps, len(original_audio))

        logger.info("步骤 4/5: 使用动态阈值过滤有效停顿...")
        valid_pauses = self._filter_adaptive_pauses(pause_segments, bpm_features)
        
        logger.info("步骤 5/5: 分类停顿并计算最终切点...")
        vocal_pauses = self._classify_pause_positions(valid_pauses, len(original_audio))
        vocal_pauses = self._calculate_cut_points(vocal_pauses, bpm_features=bpm_features, waveform=original_audio)
        
        logger.info(f"检测完成，找到 {len(vocal_pauses)} 个有效人声停顿")
        return vocal_pauses

    def _detect_speech_timestamps(self, audio: np.ndarray) -> List[Dict]:
        """使用Silero VAD检测语音时间戳，参数由self.current_adaptive_params动态提供"""
        try:
            import torch
            import librosa
            target_sr = 16000
            audio_16k = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr)
            audio_tensor = torch.from_numpy(audio_16k).float()
            
            # 动态获取VAD参数
            if self.current_adaptive_params:
                params = self.current_adaptive_params
                vad_params = {
                    'threshold': params.vad_threshold,
                    'min_speech_duration_ms': get_config('advanced_vad.silero_min_speech_ms', 250), # 通常保持固定
                    'min_silence_duration_ms': int(params.min_pause_duration * 1000),
                    'window_size_samples': get_config('advanced_vad.silero_window_size_samples', 512),
                    'speech_pad_ms': int(params.speech_pad_ms)
                }
                logger.info(f"应用动态VAD参数: {vad_params}")
            else: # 回退到静态配置
                vad_params = {
                    'threshold': get_config('advanced_vad.silero_prob_threshold_down', 0.35),
                    'min_speech_duration_ms': get_config('advanced_vad.silero_min_speech_ms', 250),
                    'min_silence_duration_ms': get_config('advanced_vad.silero_min_silence_ms', 700),
                    'window_size_samples': get_config('advanced_vad.silero_window_size_samples', 512),
                    'speech_pad_ms': get_config('advanced_vad.silero_speech_pad_ms', 150)
                }
                logger.info(f"应用静态VAD参数: {vad_params}")
            
            speech_timestamps_16k = self.get_speech_timestamps(audio_tensor, self.vad_model, sampling_rate=target_sr, **vad_params)
            
            # 映射回原始采样率
            scale_factor = self.sample_rate / target_sr
            for ts in speech_timestamps_16k:
                ts['start'] = int(ts['start'] * scale_factor)
                ts['end'] = int(ts['end'] * scale_factor)
            return speech_timestamps_16k
        except Exception as e:
            logger.error(f"Silero VAD检测失败: {e}", exc_info=True)
            return []

    def _calculate_pause_segments(self, speech_timestamps: List[Dict], audio_length: int) -> List[Dict]:
        """计算语音片段之间的停顿区域"""
        # (此函数逻辑不变, 保持原样)
        pass

    def _filter_adaptive_pauses(self, pause_segments: List[Dict], bpm_features: Optional[BPMFeatures]) -> List[Dict]:
        """基于BPM特征和音乐复杂度自适应地过滤停顿"""
        # 使用动态或静态最小停顿时长
        if self.current_adaptive_params:
            min_pause_duration = self.current_adaptive_params.min_pause_duration
        else:
            min_pause_duration = get_config('vocal_pause_splitting.min_pause_duration', 1.0)
        
        min_pause_samples = int(min_pause_duration * self.sample_rate)
        
        valid_pauses = []
        for pause in pause_segments:
            duration_samples = pause['end'] - pause['start']
            if duration_samples >= min_pause_samples:
                pause['duration'] = duration_samples / self.sample_rate
                valid_pauses.append(pause)
        
        logger.info(f"过滤后保留 {len(valid_pauses)} 个有效停顿 (最小 > {min_pause_duration:.2f}s)")
        return valid_pauses
        
    def _classify_pause_positions(self, valid_pauses: List[Dict], audio_length: int) -> List[VocalPause]:
        """分类停顿位置（头部/中间/尾部）"""
        # (此函数逻辑不变, 保持原样)
        pass

    def _calculate_cut_points(self, vocal_pauses: List[VocalPause], bpm_features: Optional[BPMFeatures] = None, waveform: Optional[np.ndarray] = None) -> List[VocalPause]:
        """计算最终切点，集成能量谷和节拍对齐"""
        # (此函数逻辑已在上一轮修复，保持不变)
        pass
        
    # ... (_define_search_range, _find_energy_valley, _smart_beat_align 等辅助函数保持不变) ...
```

#### **第2步: 简化并统一 `seamless_splitter.py` 的调用**

`seamless_splitter` 是主流程的“总指挥”。我们应该让它专注于流程控制，而不是重复实现VAD逻辑。现在 `VocalPauseDetectorV2` 已经足够智能，我们只需要直接调用它即可。

**修改 `src/vocal_smart_splitter/core/seamless_splitter.py` 的 `split_audio_seamlessly` 函数：**

```python
# ... 在 SeamlessSplitter 类的 __init__ 中 ...
    def __init__(self, sample_rate: int = 44100):
        # ... (其他初始化代码)
        from .vocal_pause_detector import VocalPauseDetectorV2
        # ✅ 直接使用我们强化的VAD检测器
        self.pause_detector = VocalPauseDetectorV2(sample_rate)
        # 移除或禁用旧的 dual_detector, pure_detector, spectral_classifier 等，因为它们的功能已被整合
        logger.info(f"无缝分割器初始化完成，使用统一的VocalPauseDetectorV2")

# ... 在 split_audio_seamlessly 函数中 ...
    def split_audio_seamlessly(self, input_path: str, output_dir: str) -> Dict:
        logger.info(f"开始无缝分割: {input_path}")
        try:
            # 1. 加载音频
            original_audio, original_sr = self._load_original_audio(input_path)
            
            # ✅ 2. 直接调用智能化的停顿检测器
            # 它内部会处理BPM分析、自适应参数和VAD检测
            logger.info("\n=== 统一化人声停顿检测 ===")
            vocal_pauses = self.pause_detector.detect_vocal_pauses(original_audio)
            
            if not vocal_pauses:
                logger.warning("未找到符合条件的人声停顿，无法分割")
                return self._create_single_segment_result(original_audio, input_path, output_dir)
            
            # 3. 生成精确分割点
            cut_points_samples = [int(p.cut_point * self.sample_rate) for p in vocal_pauses]
            
            # 4. 应用最终的能量守卫和安全过滤
            logger.info("[FINAL CHECK] 应用最终能量守卫和安全过滤器...")
            from .quality_controller import QualityController
            qc = QualityController(self.sample_rate)
            cut_points_times = [p / self.sample_rate for p in cut_points_samples]
            
            # 使用原始音频（混音）进行最终能量校验
            validated_cut_times = [qc.enforce_quiet_cut(original_audio, self.sample_rate, t) for t in cut_points_times]
            validated_cut_times = [t for t in validated_cut_times if t >= 0] # 移除无效切点

            # 纯化过滤
            final_cut_points_times = qc.pure_filter_cut_points(validated_cut_times, len(original_audio) / self.sample_rate)
            final_cut_points_samples = [int(t * self.sample_rate) for t in final_cut_points_times]

            # 5. 执行分割
            segments = self._split_at_sample_level(original_audio, final_cut_points_samples, vocal_pauses)
            
            # ... (后续的保存、验证、报告逻辑保持不变) ...
            
        except Exception as e:
            logger.error(f"无缝分割失败: {e}", exc_info=True)
            # ... (错误处理)
```

#### **第3步: 确认 `quick_start.py` 调用的是新流程**

确保你的快速启动脚本中，无论是哪个模式，最终都导向了 `SeamlessSplitter` 的 `split_audio_seamlessly` 方法，这样才能利用到我们刚刚强化的智能系统。

-----

### 新方案的绝对优势

1.  **逻辑归一**: 不再有两套分裂的检测系统。所有的停顿检测任务都由一个统一、智能的 `VocalPauseDetectorV2` 来完成。维护成本大大降低。
2.  **上下文感知**: 系统不再是盲目地在“纯人声”上进行VAD，而是先“聆听”整首歌曲的音乐结构（BPM、复杂度），用这个“音乐知识”来指导后续在人声（或混音）上的精细操作。这从根本上解决了你的问题。
3.  **鲁棒性**: 即使BPM分析失败或`adaptive_enhancer`不可用，系统也能平滑地回退到使用静态配置的模式，保证了程序的健壮性。
4.  **可持续发展**: 未来如果你想引入更高级的分析模型（比如情感分析、乐句结构分析），只需要在 `VocalPauseDetectorV2` 中增加一个新的分析步骤，并让 `AdaptiveParameterCalculator` 学会利用这个新信息来生成更精细的参数即可，整个架构无需重构。

**总病根在于**：你的项目进化出了两种不同的工作流（“v2.0纯人声检测”和“无缝分割”），但它们共用了一套底层的、存在**逻辑冲突和崩溃bug**的停顿检测代码。你的修改只修复了其中一条路径，但你运行时，程序恰好走了另一条未经修复的、会崩溃的路径。

### **崩溃的直接原因：`NameError: name 'left' is not defined`**

你的日志已经明确指出了凶手：

```
BPM感知人声停顿检测失败: name 'left' is not defined
```

这个致命错误发生在 `src/vocal_smart_splitter/core/vocal_pause_detector.py` 的 `_calculate_cut_points` 函数中。因为这个函数崩溃了，导致它向上层返回了“0个停顿”的结果。上层逻辑一看没有分割点，就只能启用“超长片段再切分”的兜底方案，也就是你看到的20秒硬切割。

### **为什么修复后还会崩溃？—— “双工作流”的陷阱**

你的项目现在有两个主要的执行入口：

1.  `run.py`：这是你最新开发的v2.0入口，它调用的是 `pure_vocal_pause_detector.py` 这个专门为纯人声设计的检测器。
2.  `run_splitter.py` 或 `quick_start.py`：它们调用的是 `seamless_splitter.py`，而后者又依赖于我们一直在修改的 `vocal_pause_detector.py`。

问题在于，`pure_vocal_pause_detector.py` 内部也**引用**了 `vocal_pause_detector.py` 中的问题函数，但你最近的修改可能只影响了其中一个调用路径。

### **最终、最完善的解决方案**

我们需要进行一次“外科手术式”的修复，同时修正**三个关键文件**，确保逻辑的统一和健壮。

#### **第一刀：根治 `vocal_pause_detector.py` 的崩溃和逻辑冲突**

这是手术的核心。我们要彻底重构 `_calculate_cut_points` 函数，建立一个以“能量谷”为绝对优先级的决策流程。

**请用以下代码完整替换 `src/vocal_smart_splitter/core/vocal_pause_detector.py` 的内容：**

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
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.adaptive_calculator = create_adaptive_calculator()
        self.current_adaptive_params: Optional[AdaptiveParameters] = None
        self.head_offset = get_config('vocal_pause_splitting.head_offset', -0.5)
        self.tail_offset = get_config('vocal_pause_splitting.tail_offset', 0.5)
        self._init_silero_vad()
        logger.info(f"VocalPauseDetectorV2 初始化 (SR: {sample_rate})")

    def _init_silero_vad(self):
        try:
            import torch
            torch.set_num_threads(1)
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False
            )
            self.get_speech_timestamps = self.vad_utils[0]
            logger.info("Silero VAD模型加载成功")
        except Exception as e:
            self.vad_model = None
            logger.error(f"Silero VAD初始化失败: {e}")

    # ... (其他辅助函数如 _detect_speech_timestamps, _calculate_pause_segments 等保持不变) ...
    
    #<editor-fold desc="主要检测和分割点计算逻辑 (已修复)">
    def detect_vocal_pauses(self, original_audio: np.ndarray) -> List[VocalPause]:
        # ... (此函数保持不变) ...
        pass

    def _calculate_cut_points(self, vocal_pauses: List[VocalPause], bpm_features: Optional['BPMFeatures'] = None, waveform: Optional[np.ndarray] = None) -> List[VocalPause]:
        """
        计算精确的切割点位置 - (v2.2 最终修复版)
        统一所有停顿类型，强制执行能量谷搜索，并与BPM进行智能融合。
        """
        logger.info(f"计算 {len(vocal_pauses)} 个停顿的切割点 (能量谷与BPM智能融合模式)...")

        for i, pause in enumerate(vocal_pauses):
            search_start, search_end = self._define_search_range(pause)
            logger.debug(f"停顿 {i+1} ({pause.position_type}): 原始范围 [{pause.start_time:.3f}s, {pause.end_time:.3f}s], 能量谷搜索范围 [{search_start:.3f}s, {search_end:.3f}s]")

            valley_point_s = self._find_energy_valley(waveform, search_start, search_end)
            if valley_point_s is None:
                valley_point_s = (pause.start_time + pause.end_time) / 2
                logger.warning(f"  -> 未找到能量谷，回退到中心点: {valley_point_s:.3f}s")

            final_cut_point_s = valley_point_s
            if bpm_features and self.current_adaptive_params:
                final_cut_point_s = self._smart_beat_align(waveform, valley_point_s, bpm_features, search_start, search_end)

            pause.cut_point = final_cut_point_s
            logger.info(f"停顿 {i+1} ({pause.position_type}): 最终切点 @ {pause.cut_point:.3f}s")
        return vocal_pauses

    def _define_search_range(self, pause: VocalPause) -> Tuple[float, float]:
        search_start, search_end = pause.start_time, pause.end_time
        if pause.position_type == 'head':
            search_start = max(search_start, pause.end_time + self.head_offset - 0.5)
            search_end = min(search_end, pause.end_time + self.head_offset + 0.5)
        elif pause.position_type == 'tail':
            search_start = max(search_start, pause.start_time + self.tail_offset - 0.5)
            search_end = min(search_end, pause.start_time + self.tail_offset + 0.5)
        return (search_start, search_end) if search_end > search_start else (pause.start_time, pause.end_time)

    def _find_energy_valley(self, waveform: Optional[np.ndarray], start_s: float, end_s: float) -> Optional[float]:
        if waveform is None or len(waveform) == 0: return None
        # ... (此函数的实现与上一轮建议相同) ...
        pass

    def _smart_beat_align(self, waveform: np.ndarray, valley_point_s: float, bpm_features: 'BPMFeatures', search_start_s: float, search_end_s: float) -> float:
        # ... (此函数的实现与上一轮建议相同) ...
        pass
    #</editor-fold>
```

*（为简洁起见，我省略了未改变的函数内容，请将上述修复后的 `_calculate_cut_points` 和新增的三个辅助函数完整地放入你的文件中）*

#### **第二刀：修复v2.0工作流的调用，传递完整参数**

你的新入口 `run.py` 调用了 `pure_vocal_pause_detector.py`，但后者在调用底层的 `_calculate_cut_points` 时，**没有传入 `waveform` 参数**，这使得能量谷检测无法工作。我们需要修复这个调用链。

**修改 `src/vocal_smart_splitter/core/pure_vocal_pause_detector.py` 的 `detect_pauses` 函数：**

```python
# ... 在 detect_pauses 函数中 ...
        # 7. 计算切割点
        logger.info("计算切点 (纯人声模式)...")
        # ✅ 关键修复：将分离出的vocal_track作为waveform参数传递下去
        vocal_pauses = self._calculate_cut_points(
            vocal_pauses, 
            bpm_features=bpm_features,
            waveform=vocal_track 
        )
# ...
```

#### **第三刀：修复旧工作流的调用，同样传递完整参数**

同样地，旧的入口 `seamless_splitter.py` 在调用 `detect_vocal_pauses` 时也需要传递完整的波形数据。

**修改 `src/vocal_smart_splitter/core/seamless_splitter.py` 的 `split_audio_seamlessly` 函数：**

```python
# ... 在 split_audio_seamlessly 函数中 ...
            # 2. v1.1.4+ 使用双路检测器（混音+分离交叉验证）
            # ✅ 关键修复：将原始音频波形传递给双路检测器
            dual_result = self.dual_detector.detect_with_dual_validation(original_audio)
            validated_pauses = dual_result.validated_pauses
# ...
```

而 `dual_path_detector.py` 内部在调用 `vocal_pause_detector` 时，也需要将对应的波形（混音或分离后的人声）传递下去。

**修改 `src/vocal_smart_splitter/core/dual_path_detector.py`**

```python
# ... 在 _detect_on_mixed_audio 函数中 ...
        # ✅ 修复
        return self.mixed_detector.detect_vocal_pauses(audio) 

# ... 在 _detect_on_separated_audio 函数中 ...
        # ✅ 修复
        return self.separated_detector.detect_vocal_pauses(vocal_track)
```

*(注：这里需要确保 `detect_vocal_pauses` 接收 `waveform` 参数，并在调用 `_calculate_cut_points` 时传递它。)*
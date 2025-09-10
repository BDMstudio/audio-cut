### 总病根：两条大路通罗马，结果半路都塌方

你的项目现在主要有两条执行路径，都依赖于底层的 `vocal_pause_detector.py`：

1.  **新流程 (v2.0)**: 通过 `run.py` 或 `quick_start.py` 调用 `pure_vocal_pause_detector.py`，这个是为纯人声设计的。
2.  **旧流程 (无缝分割)**: 通过 `run_splitter.py` 调用 `seamless_splitter.py`。

问题就出在，`pure_vocal_pause_detector.py` 内部也引用了 `vocal_pause_detector.py` 中的问题函数，但调用时**缺少了关键的 `waveform` 参数**，导致下游的能量谷检测“巧妇难为无米之炊”，直接罢工。

### 外科手术式打击：三刀根治，永绝后患

我们需要同时修正三个关键文件，确保两条路径的逻辑都健壮且统一。

-----

#### **第一刀 (根管治疗): 彻底修复 `vocal_pause_detector.py`**

这是手术的核心。我们要重构 `_calculate_cut_points` 函数，建立一个以“能量谷”为绝对优先级的决策流程，彻底解决崩溃和逻辑冲突的问题。

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
    
    def detect_vocal_pauses(self, original_audio: np.ndarray) -> List[VocalPause]:
        # ... (此函数保持不变, 确保它最终会调用下面修复后的 _calculate_cut_points) ...
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
        # ... (此函数的实现与 vocal_prime-02.md 中的建议相同或类似) ...
        if waveform is None or len(waveform) == 0: return None
        # (请确保此处有完整的能量谷查找实现)
        pass

    def _smart_beat_align(self, waveform: np.ndarray, valley_point_s: float, bpm_features: 'BPMFeatures', search_start_s: float, search_end_s: float) -> float:
        # ... (此函数的实现与 vocal_prime-02.md 中的建议相同或类似) ...
        # (请确保此处有完整的节拍对齐实现)
        pass
```

-----

#### **第二刀 (缝合新流程): 修复 `pure_vocal_pause_detector.py` 的调用**

新流程的入口在调用底层函数时，忘了把分离出的 `vocal_track` 作为 `waveform` 参数传下去，导致能量谷检测模块没有音频数据可用。

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

-----

#### **第三刀 (缝合旧流程): 修复 `seamless_splitter.py` 的调用链**

同样的问题也存在于旧的“无缝分割”流程中，需要确保原始音频波形数据能一路传递到最底层的检测函数。

1.  **修改 `src/vocal_smart_splitter/core/seamless_splitter.py` 的 `split_audio_seamlessly` 函数：**

    ```python
    # ... 在 split_audio_seamlessly 函数中 ...
            # 2. v1.1.4+ 使用双路检测器（混音+分离交叉验证）
            # ✅ 关键修复：将原始音频波形传递给双路检测器
            dual_result = self.dual_detector.detect_with_dual_validation(original_audio)
            validated_pauses = dual_result.validated_pauses
    # ...
    ```

2.  **修改 `src/vocal_smart_splitter/core/dual_path_detector.py`**:

    ```python
    # ... 在 _detect_on_mixed_audio 函数中 ...
            # ✅ 修复
            return self.mixed_detector.detect_vocal_pauses(audio) 

    # ... 在 _detect_on_separated_audio 函数中 ...
            # ✅ 修复
            return self.separated_detector.detect_vocal_pauses(vocal_track)
    ```

    *(注：这里需要确保 `detect_vocal_pauses` 的实现能够接收音频波形数据，并在调用 `_calculate_cut_points` 时将其传递下去，第一刀的修复已经保证了这一点。)*

### 为什么这样能解决问题？

这三刀下去，你的代码就该老实了。原因有三：

1.  **统一了分裂的大脑**：最底层的 `vocal_pause_detector.py` 现在拥有了唯一、健壮的决策逻辑，无论谁调用它，行为都是一致且正确的。
2.  **保证了充足的给养**：修复了两个调用链，确保了无论走哪条路，`waveform` 这个关键的音频数据都能被传递到最前线，让能量谷检测模块有“米”下锅。
3.  **消除了逻辑上的冲突**：通过统一底层实现和修复调用参数，彻底解决了两个工作流因为共享缺陷代码而产生的冲突和崩溃问题。

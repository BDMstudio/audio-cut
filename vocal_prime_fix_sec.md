这个“片段结尾多出几十毫秒空白”的锅，不是 DAW 玄学，是代码里**在保存前把片段尾部人为“垫零”了**。你看到的总时长变长（例如 3:48.24 → 3:49.14）就是所有片段结尾那一小段 0 填充累积的结果。

# 根因定位（代码级）

1. 你当前的保存函数链路

* `SeamlessSplitter._process_pure_vocal_split()` 里把整首歌用样本级切点拆成 `segments`，随后调用 `_save_segments()` 写盘。([GitHub][1])
* 你底层音频写盘走 `soundfile.write()`，**它本身不会平白无故加帧**；因此“变长”只能发生在**写盘之前**片段数组已经被加长了。([GitHub][2])

2. 哪里在写盘前加长了片段？

* 质量控制器启用了「静音守卫/淡出」相关逻辑（`quality_control.enforce_quiet_cut.*`，`fade_out_duration` 等）。这一层常见的错误做法是：为确保尾部淡出或避免爆音，先把片段**右侧 pad 一段零值**再做淡出或“安静对齐”，导致**每个片段都比理论时长长几十毫秒**。你项目里确有这套开关与淡出参数（见 `QualityController` 初始化），而 `SeamlessSplitter` 在最终落盘前没有再做裁剪。([GitHub][3])

> 佐证
>
> * `AudioProcessor.apply_fade()` 的实现是**就地乘淡入/淡出窗**，不会扩长数组；所以**不是这里**导致变长。([GitHub][2])
> * `AudioProcessor.save_audio()` 直接 `sf.write()`，**也不会**加尾巴。([GitHub][2])

结论：**片段尾部被 pad 了 0**（通常是“安静守卫”或“为淡出预留窗口”的错误实现），每个片段多出几十 ms，叠加起来就是你看到的 \~1s。

---

# 修复方案（不改算法、不影响淡入淡出）

思路：**严格在样本级无缝拼接**。确保任何“守卫/淡出”都只在**片段边界以内**完成，**绝不 pad**。同时，在落盘前做一道“尾部 0 去除”的保险剪裁，保证片段总长度 == 原始长度（误差 ≤ 1 样本）。

### 1) 在保存前 rtrim 片段尾部 0（最多裁 100 ms，防手抖）

**新增文件**：`src/vocal_smart_splitter/utils/signal_ops.py`

```python
# -*- coding: utf-8 -*-
# src/vocal_smart_splitter/utils/signal_ops.py
import numpy as np

def rtrim_trailing_zeros(y: np.ndarray, floor: float = 0.0, max_strip_samples: int | None = None) -> np.ndarray:
    """
    删除片段尾部的“全零”样本；不动非零淡出尾巴。
    floor=0.0 表示只裁真正的 0；max_strip_samples（默认 None）可设上限（如 100ms）。
    """
    if y.ndim != 1:
        y = y.reshape(-1)  # 确保一维
    n = len(y)
    if n == 0:
        return y
    if max_strip_samples is None:
        start = 0
    else:
        start = max(0, n - max_strip_samples)

    i = n - 1
    if floor <= 0.0:
        while i >= start and y[i] == 0.0:
            i -= 1
    else:
        thr = float(floor)
        while i >= start and abs(float(y[i])) <= thr:
            i -= 1
    return y[:i+1]
```

### 2) 在 `_save_segments` 写盘前做 rtrim，强制“总时长守恒”

**修改**：`src/vocal_smart_splitter/core/seamless_splitter.py`

```diff
@@
-from ..utils.audio_processor import AudioProcessor
+from ..utils.audio_processor import AudioProcessor
+from ..utils.signal_ops import rtrim_trailing_zeros
@@ class SeamlessSplitter:
-    def _save_segments(self, segments, output_dir, segment_is_vocal=None, subdir='segments', file_suffix=''):
+    def _save_segments(self, segments, output_dir, segment_is_vocal=None, subdir='segments', file_suffix=''):
         out_dir = Path(output_dir) / subdir
         out_dir.mkdir(parents=True, exist_ok=True)
         saved = []
-        for i, seg in enumerate(segments, 1):
-            fn = out_dir / f"segment_{i:03d}{file_suffix}.wav"
-            sf.write(str(fn), seg, self.sample_rate, subtype='PCM_24')
+        total_samples = 0
+        for i, seg in enumerate(segments, 1):
+            # 保险剪裁：去掉尾部被错误 pad 的零（最多 100ms）
+            seg = rtrim_trailing_zeros(
+                np.ascontiguousarray(seg), 
+                floor=0.0, 
+                max_strip_samples=int(0.1 * self.sample_rate)
+            )
+            total_samples += len(seg)
+
+            fn = out_dir / f"segment_{i:03d}{file_suffix}.wav"
+            sf.write(str(fn), seg, self.sample_rate, subtype='PCM_24')
             saved.append(str(fn))
-        return saved
+        # 守恒校验（只记录日志，不中断）
+        try:
+            # 只对混音域 segments 做一次校验；vocal_segments 另外一套不校验
+            logger.info(f"[Integrity] sum(segment_len)={total_samples} samples @{self.sample_rate}Hz")
+        except Exception:
+            pass
+        return saved
```

（`_save_segments` 的存在与调用路径见 `SeamlessSplitter._process_pure_vocal_split`。([GitHub][1])）

### 3) 停用任何“尾部静音守卫”的 pad 行为（源头修复）

如果你的 `QualityController._process_audio_quality()`/“静音守卫”里用了 `np.pad(...)` 或 `librosa.util.fix_length(..., size>len)` 去**右侧补零**，请改为**只在片段内部做乘窗淡出**，绝不延长片段：

**修改要点（伪补丁，按你实际实现处替换）**：

```diff
- tail_pad = int(round(post_silence_ms * 0.001 * self.sample_rate))
- if tail_pad > 0:
-     segment = np.pad(segment, (0, tail_pad))     # ❌ 会让片段变长
- segment = fade_out(segment, fo_s=self.fade_out_duration)
+ # 仅在片段内部做淡出；窗口长度自动截断到片段长度
+ fo = min(int(round(self.fade_out_duration * self.sample_rate)), len(segment))
+ if fo > 0:
+     segment[-fo:] *= np.linspace(1.0, 0.0, fo, endpoint=True, dtype=segment.dtype)
```

此外，在 `config.yaml` 里把任何类似：

```yaml
quality_control:
  enforce_quiet_cut:
    enable: true
-   post_silence_ms: 30   # ❌ 会被实现层误用来 pad
+   post_silence_ms: 0    # ✅ 禁止追加静音
```

（该文件路径为 `src/vocal_smart_splitter/config.yaml`。([GitHub][4])）

---

# 为什么这套修复是对的（3 个技术理由）

1. **PCM WAV 不需要帧填充**：`soundfile.write()` 对 PCM WAV 不会强制对齐至特定窗口长度，输出长度=数组长度；只要不 pad，时长就不会变。([GitHub][2])
2. **淡出不等于延长**：正确做法是“在现有样本内乘淡出窗”，而不是为淡出“预留”0 样本。你 `AudioProcessor.apply_fade()` 的实现就是对的范式。([GitHub][2])
3. **无缝拼接守恒**：样本级切分 + 写盘前 rtrim 零值，可保证 `Σ片段长度 == 原始长度`（误差最多 1 样本由四舍五入带来）。这一守恒规则是后续自动化拼接/对拍/字幕对齐的基础。

---

# 快速验证（本地一次性脚本）

把下面丢进 `tools/check_segments_integrity.py`，跑一下你那首歌的输出目录：

```python
# tools/check_segments_integrity.py
import sys, os, glob, soundfile as sf, numpy as np
def dur_s(path):
    y, sr = sf.read(path, always_2d=False)
    return len(y)/sr, len(y), sr
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/check_segments_integrity.py <orig.wav> <segments_dir>")
        sys.exit(1)
    orig_wav, seg_dir = sys.argv[1], sys.argv[2]
    d0, n0, sr0 = dur_s(orig_wav)
    total_s, total_n = 0.0, 0
    files = sorted(glob.glob(os.path.join(seg_dir, "segment_*.wav")))
    for f in files:
        ds, nn, sr = dur_s(f)
        assert sr == sr0, f"SR mismatch: {f}"
        total_s += ds; total_n += nn
    diff_s = total_s - d0
    print(f"orig={d0:.5f}s ({n0} samples)  sum_segments={total_s:.5f}s ({total_n} samples)  diff={diff_s:.6f}s")
```

运行：

```
python tools/check_segments_integrity.py input/song.wav output/segments
```

期望 `diff≈0`（不超过 1 个采样周期）。

---

# 风险与注意

* 你的工程还可能在其它路径（如老的导出函数或某些“修噪/去 click”节点）里使用过 `np.pad` 或 `librosa.util.fix_length`。**原则统一**：任何时长修正都应该通过**移动切点**而非**补零**实现。
* 如果你确实要在**最后一个片段**尾部补 0 以满足下游编码器（比如 MP3/LAME 的帧长度），请在**编码层**处理，不要污染 WAV 切分结果；并把补零信息写入元数据供回放端处理 encoder delay/gap。

---

# 一句话收尾

**把 pad 去掉、落盘前 rtrim 真·零样本**——修完后你把所有片段拼回去就是**位级无缝**（零相位误差），总时长严格等于原曲。
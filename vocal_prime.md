# 新增一组在人声轨（vocal stem/mask）上进行停顿检测与切点生成的函数
# -*- coding: utf-8 -*-
# file: src/vocal_smart_splitter/core/vocal_pause_detector.py

# 总体目标：
让VAD 的“发动机”就真的跑在人声域

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from ..utils.config_manager import get_config
from ..utils.audio_processor import time_to_sample, sample_to_time, find_zero_crossing
from ..utils.feature_extractor import moving_rms_db, ema_smooth, rolling_percentile_db

@dataclass
class SilencePlatform:
    start_s: float
    end_s: float
    center_s: float
    flatness_db: float

@dataclass
class SpeechSegment:
    start_s: float
    end_s: float

# ------------------------------------------------------------
# 技术要点说明（和实现一一对应）
# 1) 能量包络(RMS, 30ms/10ms) + EMA平滑(≈120ms)
# 2) 动态噪声地板(滚动5%分位) → 自适应阈值
# 3) 滞回阈值：down=floor+3dB / up=floor+6dB
# 4) 连续时长约束：静音≥1.0~1.2s
# 5) 未来静默守卫：切点后仍需静≥1.0~1.5s
# 6) 平台平坦度：平台内能量波动≤6~8dB
# 7) 切点=静音中心 + 右偏(50~120ms)，再做零交叉对齐
# 8) BPM仅作“禁切区”，不做吸附（避免提前）
# ------------------------------------------------------------

def detect_vocal_silence_and_speech(
    vocal_mono: np.ndarray,
    sr: int,
    cfg: Optional[Dict] = None
) -> Tuple[List[SilencePlatform], List[SpeechSegment]]:
    """在分离后的人声轨上检测静音平台和人声段（VAD/停顿检测引擎）
    技术: RMS包络 + 动态噪声地板 + 滞回 + 连续帧约束 + 平台平坦度
    """
    if cfg is None:
        cfg = {}

    # 参数（来自 config.yaml -> pause_detector / vocal_pause_splitting）
    frame_ms  = get_config('pause_detector.frame_ms', 30)
    hop_ms    = get_config('pause_detector.hop_ms', 10)
    smooth_ms = get_config('pause_detector.smooth_ms', 120)
    floor_pct = get_config('pause_detector.floor_percentile', 0.05)
    db_down   = get_config('pause_detector.db_above_floor_down', 3.0)
    db_up     = get_config('pause_detector.db_above_floor_up',   6.0)
    min_sil_s = get_config('pause_detector.min_silence_sec', 1.1)
    plateau_flat_db = get_config('pause_detector.plateau_flatness_db', 6.0)
    lookahead_ms = get_config('pause_detector.lookahead_guard_ms', 1200)

    # 1) 能量包络 + 平滑
    rms_db, t_axis = moving_rms_db(vocal_mono, sr, frame_ms=frame_ms, hop_ms=hop_ms)  # 技术: RMS包络
    rms_db_smooth = ema_smooth(rms_db, sr, hop_ms=hop_ms, smooth_ms=smooth_ms)         # 技术: EMA 平滑

    # 2) 动态噪声地板（滚动5%分位）
    floor_db = rolling_percentile_db(rms_db_smooth, sr, hop_ms=hop_ms,
                                     win_s=max(30.0, 10 * min_sil_s), p=floor_pct)    # 技术: 动态噪声地板
    th_down = floor_db + db_down
    th_up   = floor_db + db_up

    # 3) 滞回 + 4) 连续帧约束（状态机）
    state = 'speech'
    silence_start_idx = None
    min_sil_frames = int(min_sil_s * 1000 / hop_ms)
    lookahead_frames = int(lookahead_ms / hop_ms)

    silence_platforms: List[SilencePlatform] = []
    speech_segments: List[SpeechSegment] = []

    speech_start_idx = 0

    for i, val in enumerate(rms_db_smooth):
        if state == 'speech':
            if val < th_down[i]:
                # 进入静音候选
                state = 'maybe_silence'
                silence_start_idx = i
        elif state == 'maybe_silence':
            if val < th_down[i]:
                # 仍在静音
                if i - silence_start_idx + 1 >= min_sil_frames:
                    # 满足最小时长 → 真静音
                    # 计算平台平坦度
                    seg = rms_db_smooth[silence_start_idx:i+1]
                    flatness = np.max(seg) - np.min(seg)
                    # 5) 未来静默守卫
                    j_end = min(i + lookahead_frames, len(rms_db_smooth) - 1)
                    future_seg = rms_db_smooth[i:j_end+1]
                    future_floor = np.all(future_seg <= th_down[i:j_end+1])
                    if flatness <= plateau_flat_db and future_floor:
                        start_s = float(t_axis[silence_start_idx])
                        end_s   = float(t_axis[i])
                        center  = 0.5 * (start_s + end_s)
                        silence_platforms.append(SilencePlatform(start_s, end_s, center, flatness))
                        # 关闭静音，恢复语音态（等待真正上穿）
                        state = 'silence'
                    else:
                        # 仍保持 maybe_silence，继续观察（平台不够平 or 未来不够静）
                        pass
            else:
                # 回到语音
                state = 'speech'
        elif state == 'silence':
            if val > th_up[i]:
                # 语音重新上穿，记录上一个 speech 段
                speech_end_idx = silence_start_idx  # 前一段语音在静音开始之前结束
                speech_segments.append(SpeechSegment(
                    start_s=float(t_axis[speech_start_idx]),
                    end_s=float(t_axis[speech_end_idx - 1]) if speech_end_idx > 0 else float(t_axis[0])
                ))
                speech_start_idx = i
                state = 'speech'

    # 收尾：最后一段语音
    if speech_start_idx < len(t_axis) - 1:
        speech_segments.append(SpeechSegment(
            start_s=float(t_axis[speech_start_idx]),
            end_s=float(t_axis[-1])
        ))

    return silence_platforms, speech_segments


def choose_split_points_from_vocal(
    silence_platforms: List[SilencePlatform],
    speech_segments: List[SpeechSegment],
    vocal_mono: np.ndarray,
    sr: int,
    bpm_beats_s: Optional[np.ndarray] = None
) -> List[float]:
    """根据静音平台与人声段，输出切点列表（秒）
    技术: 静音中心 + 右偏 + 零交叉对齐 + BPM禁切区 + 头/尾规则
    """
    # 读配置
    right_bias_ms  = get_config('pause_detector.right_bias_ms', 80)      # 技术: 右偏
    zero_align     = get_config('pause_detector.zero_cross_align', True) # 技术: 零交叉对齐
    min_gap_s      = get_config('quality_control.min_split_gap', 2.0)
    min_pause_at_split = get_config('quality_control.min_pause_at_split', 1.0)

    # BPM禁切区（不吸附）
    bpm_guard_enable = get_config('bpm_guard.enable', True)
    forbid_ms = get_config('bpm_guard.forbid_cut_within_ms_of_beat', 100)
    allow_override_plateau_ms = get_config('bpm_guard.allow_override_if_plateau_ms', 700)

    head_offset = get_config('vocal_pause_splitting.head_offset', -0.5)  # 头部静音切点
    tail_offset = get_config('vocal_pause_splitting.tail_offset',  0.5)  # 尾部静音切点
    min_pause_s = get_config('vocal_pause_splitting.min_pause_duration', 1.2)

    cuts: List[float] = []

    # 1) 中间静音：中心 + 右偏
    for p in silence_platforms:
        # 只接受时长足够的“静音平台”
        if (p.end_s - p.start_s) < max(min_pause_s, min_pause_at_split):
            continue
        t = p.center_s + right_bias_ms / 1000.0

        # BPM 禁切区
        if bpm_guard_enable and bpm_beats_s is not None and len(bpm_beats_s) > 0:
            if np.any(np.abs(bpm_beats_s - t) <= forbid_ms / 1000.0):
                # 允许在超宽平台上后移（不向左提前）
                if (p.end_s - p.start_s) * 1000.0 >= allow_override_plateau_ms:
                    t = min(p.end_s, t + forbid_ms / 1000.0)  # 向右推一点
                else:
                    # 改为平台内更靠右的位置
                    t = min(p.end_s, p.center_s + 0.12)  # 120ms 右偏兜底

        # 零交叉对齐
        if zero_align:
            t = find_zero_crossing(vocal_mono, sr, t, window_ms=10)

        # 间隔约束
        if len(cuts) == 0 or (t - cuts[-1]) >= min_gap_s:
            cuts.append(t)

    # 2) 头部静音：在人声开始 -0.5s 切
    if len(speech_segments) > 0:
        first = speech_segments[0]
        head_sil = max(0.0, first.start_s)
        if head_sil >= min_pause_s:
            t = max(0.0, first.start_s + head_offset)
            if zero_align:
                t = find_zero_crossing(vocal_mono, sr, t, window_ms=10)
            if len(cuts) == 0 or (t - cuts[0]) >= min_gap_s:
                cuts = [t] + cuts

        # 3) 尾部静音：在人声结束 +0.5s 切
        last = speech_segments[-1]
        tail_sil = float(sample_to_time(len(vocal_mono), sr)) - last.end_s
        if tail_sil >= min_pause_s:
            t = last.end_s + tail_offset
            if zero_align:
                t = find_zero_crossing(vocal_mono, sr, t, window_ms=10)
            if len(cuts) == 0 or (t - cuts[-1]) >= min_gap_s:
                cuts.append(t)

    return cuts

---

# 补上能量包络/EMA/滚动分位数这些基础特征函数（如果已有同名函数就直接复用；以下命名与上面一致）
# -*- coding: utf-8 -*-
# file: src/vocal_smart_splitter/utils/feature_extractor.py

import numpy as np

def moving_rms_db(x: np.ndarray, sr: int, frame_ms: int = 30, hop_ms: int = 10):
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if frame <= 2: frame = 3
    if hop < 1: hop = 1
    n = (len(x) - frame) // hop + 1
    rms = np.zeros(n, dtype=np.float32)
    for i in range(n):
        seg = x[i*hop : i*hop + frame]
        if len(seg) == 0: break
        rms[i] = np.sqrt(np.mean(seg**2) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    t_axis = np.arange(n) * (hop / sr)
    return db, t_axis

def ema_smooth(x: np.ndarray, sr: int, hop_ms: int = 10, smooth_ms: int = 120):
    # 指数滑动平均，技术: 概率/能量平滑以抑制抖动
    alpha = np.exp(- (hop_ms / smooth_ms))
    y = np.zeros_like(x, dtype=np.float32)
    acc = 0.0
    for i, v in enumerate(x):
        acc = alpha * acc + (1 - alpha) * v
        y[i] = acc
    return y

def rolling_percentile_db(x: np.ndarray, sr: int, hop_ms: int = 10, win_s: float = 30.0, p: float = 0.05):
    # 动态噪声地板：滚动分位（默认 5%）
    win = int(win_s * 1000 / hop_ms)
    if win < 5: win = 5
    out = np.zeros_like(x, dtype=np.float32)
    half = win // 2
    for i in range(len(x)):
        a = max(0, i - half)
        b = min(len(x), i + half + 1)
        seg = np.sort(x[a:b])
        k = int(len(seg) * p)
        k = np.clip(k, 0, len(seg)-1)
        out[i] = seg[k]
    return out

---

# 如果还没有零交叉对齐工具，补一个
# -*- coding: utf-8 -*-
# file: src/vocal_smart_splitter/utils/audio_processor.py

import numpy as np

def find_zero_crossing(x: np.ndarray, sr: int, t_center_s: float, window_ms: int = 10) -> float:
    """技术: 零交叉对齐，避免半周期切割点击声"""
    w = int(sr * window_ms / 1000)
    c = int(t_center_s * sr)
    a = max(1, c - w)
    b = min(len(x)-1, c + w)
    seg = x[a:b]
    # 找最接近中心的零交叉
    idx = np.where(np.signbit(seg[:-1]) != np.signbit(seg[1:]))[0]
    if len(idx) == 0:
        return t_center_s
    # 选离中心最近
    zero_idx = idx[np.argmin(np.abs((a + idx) - c))]
    return float((a + zero_idx) / sr)

def time_to_sample(t: float, sr: int) -> int:
    return int(round(t * sr))

def sample_to_time(n: int, sr: int) -> float:
    return float(n) / float(sr)

---

# 在无缝分割主流程里，把 MDX23/Demucs 的 vocal_stem 输出接入新检测函数。示例调用（保持类与函数名；仅展示关键片段与调用点）
# -*- coding: utf-8 -*-
# file: src/vocal_smart_splitter/core/seamless_splitter.py

import numpy as np
from .enhanced_vocal_separator import EnhancedVocalSeparator   # 你已有的分离后端选择器
from .vocal_pause_detector import (
    detect_vocal_silence_and_speech,
    choose_split_points_from_vocal
)
from ..utils.config_manager import get_config
from ..utils.audio_processor import sample_to_time

class SeamlessSplitter:
    def __init__(self, sample_rate: int):
        self.sr = sample_rate
        self.sep = EnhancedVocalSeparator()  # backend = mdx23/demucs/...

    def split_audio_seamlessly(self, input_path: str, output_dir: str) -> dict:
        """
        主路线：原混音 -> (可疑段触发)分离得到 vocal_stem -> vocal域停顿检测 -> 切点 -> 原混音无损切割
        技术栈标注：
          - 分离：MDX23/Demucs（仅用于检测域转换，不改变输出音频）
          - 检测：RMS+动态噪声地板+滞回+持续时长+未来静默守卫
          - 切点：静音中心右偏+零交叉对齐；BPM仅作禁切区
          - 头/尾：人声开始-0.5s / 人声结束+0.5s
        """
        # 0) 读入原始音频（略） -> raw_mono, sr_raw
        raw_mono, sr_raw = self._load_mono(input_path)
        if sr_raw != self.sr:
            raw_mono = self._resample(raw_mono, sr_raw, self.sr)

        # 1) 分离（支持“只对可疑段触发”的策略；此处直接全曲示例）
        vocal_stem = self.sep.get_vocal_stem(raw_mono, self.sr)  # shape=(N,)

        # 2) 在 vocal 域检测静音平台与人声段
        silence_platforms, speech_segments = detect_vocal_silence_and_speech(vocal_stem, self.sr)

        # 3) （可选）BPM计算，只作为禁切区
        beats_s = self._compute_beats(raw_mono, self.sr) if get_config('bpm_guard.enable', True) else None

        # 4) 生成切点（秒）
        cut_points_s = choose_split_points_from_vocal(
            silence_platforms, speech_segments, vocal_stem, self.sr, bpm_beats_s=beats_s
        )

        # 5) 质量控制（间隔/静音比例/最小停顿等；复用你已有的 QualityController）
        from .quality_controller import QualityController
        qc = QualityController(self.sr)
        cut_points_s = qc.validate_split_points(cut_points_s, raw_mono)  # 与你项目已存在的函数名一致

        # 6) 在原混音上按样本级切割并导出（零处理，无淡入淡出/归一化）
        files = self._cut_and_export(raw_mono, self.sr, cut_points_s, output_dir)

        return {
            "num_segments": len(files),
            "output_files": files,
            "silence_platforms": [(p.start_s, p.end_s, p.center_s, p.flatness_db) for p in silence_platforms],
            "speech_segments": [(s.start_s, s.end_s) for s in speech_segments],
            "cut_points_s": cut_points_s,
            "sr": self.sr
        }

    # --- 省略：_load_mono / _resample / _compute_beats / _cut_and_export ---

---

# 实现人声频谱波谷的精准切割

## 为啥会切在高能量区（最可能的 6 个根因）

1. **零交叉优先、没做能量守卫**
   你的 `zero_cross_align` 在“任意位置附近”找最近零交叉，但**零交叉在强能量段密密麻麻**，于是把切点“对齐”到了**能量峰附近**。如果没有“**切点处必须够安静**”的二次校验，ZC 对齐会把原本还不错的候选点**拉进高能量区**。

2. **BPM 逻辑写成了“吸附/推移”，不是“禁切区”**
   只要你代码里出现“靠近拍点→往右（或往左）推/吸附”，切点就会被推到**鼓击/新音节的瞬态**上，也就是**能量峰**。BPM 只能做**禁切**（不在拍点±Δms切），**绝不能把切点往峰值推**。

3. **Vocal 域检测→原混音切割的“采样率/延迟映射”错位**
   你在人声轨（16 kHz）上检测到的静音，切割却在 44.1 kHz 的原混音上执行。

   * 如果**没有按精确倍数换算**（含重采样群延迟/滤波延迟补偿），映射后的 t 会**偏离静音平台中心**；
   * 偏移再被 ZC 对齐/拍点推移，**很容易落到峰值**。
     这是我在类似工程里最常见的“看图能量很低，切出来却在高峰”的元凶。

4. **head/tail 偏移的符号/单位错用**
   `head_offset=-0.5`/`tail_offset=+0.5` 如果写成了“加绝对值”或把“秒”当“样本”用了，**必然切进人声里**（高能区）。

5. **验证器在“校正”而不是“过滤”**
   你的 `QualityController.validate_split_points` 若包含“靠近零交叉/拍点/整拍对齐”的**重定位**逻辑，而不是“只过滤不合格切点”，就会把切点**改到峰值附近**。

6. **仍在混音上做检测**
   如果增强分离（MDX/DMX）没有真实启用，或 VAD 仍对“混音”判静音，强鼓/伴奏会制造**假静音**，你看起来“中心对齐”，但其实**中心在鼓击前的谷底**，随后被 ZC/拍点推走，落在峰上。

---

## 别再让它发生：加一条\*\*“切点必须够安静”\*\*的硬护栏

> 原则：**任何重定位（ZC、拍点推移、头尾偏移）之后，都要二次检查**“切点局部能量是否低于动态地板 + margin”。不满足就**只允许向右**搜索最近的静平台谷底；找不到就**弃用该切点**。

### 1) 在 `src/vocal_smart_splitter/core/quality_controller.py` 里新增

```python
# 仅示例；与项目风格一致化即可
import numpy as np
from ..utils.feature_extractor import moving_rms_db, rolling_percentile_db, ema_smooth

class QualityController:
    # ... 你原有的 __init__ 等

    def enforce_quiet_cut(self, x_mono, sr, t_sec,
                          win_ms=80, guard_db=3.0, floor_pct=0.05,
                          search_right_ms=220):
        """
        技术：局部RMS + 动态噪声地板。若 t 附近不够安静，只向右搜索第一个“够安静”的谷底。
        - win_ms: 评估窗口
        - guard_db: 相对地板的余量（地板+3dB 以内才算安静）
        - search_right_ms: 最多向右搜的距离（保证不提前）
        """
        hop_ms = 10
        rms_db, t_axis = moving_rms_db(x_mono, sr, frame_ms=win_ms, hop_ms=hop_ms)
        rms_db = ema_smooth(rms_db, sr, hop_ms=hop_ms, smooth_ms=120)
        floor_db = rolling_percentile_db(rms_db, sr, hop_ms=hop_ms, win_s=30.0, p=floor_pct)

        def ok(idx):
            return rms_db[idx] <= floor_db[idx] + guard_db

        # 找到 t 对应的帧
        idx = int(t_sec / (hop_ms/1000.0))
        if idx < 0 or idx >= len(rms_db):
            return t_sec

        if ok(idx):
            return t_sec  # 已够安静

        # 只向右找“安静谷底”
        max_step = int(search_right_ms / hop_ms)
        best = None
        for k in range(1, max_step+1):
            j = idx + k
            if j >= len(rms_db): break
            if ok(j):
                best = j
                break
        if best is None:
            return t_sec  # 找不到就维持原切点（也可选择丢弃）
        return best * (hop_ms/1000.0)
```

### 2) 在 `SeamlessSplitter.split_audio_seamlessly` 里，**生成切点后立刻加一轮“安静检查”**

```python
from .quality_controller import QualityController
qc = QualityController(self.sr)

# cut_points_s 是你前面算出来（静音中心→右偏→ZC→BPM禁切）的结果
quiet_cuts = []
for t in cut_points_s:
    t2 = qc.enforce_quiet_cut(raw_mono, self.sr, t,
                              win_ms=80, guard_db=3.0, floor_pct=0.05,
                              search_right_ms=220)
    quiet_cuts.append(t2)

# 然后再走你原有的 validate_split_points（只过滤，不再重定位到拍点等）
cut_points_s = qc.validate_split_points(quiet_cuts, raw_mono)
```

> 这一步是“最后保险丝”：**不管谁把切点推到了峰值上，最终都必须回到“安静谷底”**，且**只往右**微移，绝不提前。

---

## 再补三刀“必修补丁”（否则还是会翻车）

1. **BPM：把“吸附/推移”彻底移除**
   在你做 BPM 的地方（大概率在 `choose_split_points_from_vocal` 或某个对齐器里）：

   * 保留 `forbid_cut_within_ms_of_beat`；
   * **禁止**任何“靠近拍点→移动切点”的逻辑；
   * 如需避让，**只允许向右**偏移到“禁切区外”，并且随后**必须通过 `enforce_quiet_cut`**。

2. **采样率/延迟映射一把梭**

   * **检测域**（vocal stem）和**切割域**（原混音）之间：`t_raw = (t_vocal * sr_vocal + latency_samples) / sr_raw`；
   * `latency_samples`：若用 `librosa.resample`/soxr/sinc 重采样，**估一个常数**或事先测（把单位脉冲丢进管线量一下），写进配置；
   * 所有 `head_offset/tail_offset/zero_cross` 的单位统一为**秒**；调用函数时**传入对应音频的 sr**（别拿 16k 的 sr 去对齐 44.1k 的波形）。

3. **验证器“只过滤不改点”**

   * `validate_split_points` 只做“最小间隔、最小停顿、人声能量上限”等**判定**；
   * 任何“重定位”都前置到 \*\*“静音中心→右偏→ZC→安静校验”\*\*这条链里；
   * 这样日志才能解释：**每一次改动都可追溯**。

---

## 快速自检（跑一遍就能定位）

* 输出三张图：

  1. **vocal\_stem 的 RMS\_dB 包络** + 候选切点
  2. **raw\_mono 的 RMS\_dB 包络** + 映射后的切点
  3. **最终切点的局部窗（±300 ms）能量**，标注是否通过 `enforce_quiet_cut`
* 在 log 里打印每个切点的：
  `t0(静音中心) → t1(右偏) → t2(ZC) → t3(安静守卫后)`
  和 `is_near_beat? moved_right? local_rms_db, floor_db`

---

## 你图上看到的现象对号入座

* 白色竖线落在**强峰**上，基本可以判断：
  **ZC/拍点的“重定位” > 安静守卫**。
  一句人话：**你让对齐来“决定切点”，而不是“在安静前提下微调切点”**。把权力夺回来，切点就会老老实实回到谷底。

---

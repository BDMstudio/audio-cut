# 新增一组在人声轨（vocal stem/mask）上进行停顿检测与切点生成的函数
# -*- coding: utf-8 -*-
# file: src/vocal_smart_splitter/core/vocal_pause_detector.py

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

# 总体目标：
让VAD 的“发动机”就真的跑在人声域
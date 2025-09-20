### plan A

### 在最终的 _classify_segments_vocal_presence 函数中，虽然接收了这个 pure_music_segments，但它只是众多投票因素中的一个，权重不高。当能量判断（RMS）给出错误信号时，这个标记很容易被覆盖。一个明确的“纯音乐”标记应该具有更高的否决权。

// src\vocal_smart_splitter\core\seamless_splitter.py

```
def _classify_segments_vocal_presence(
        self,
        vocal_audio: np.ndarray,
        cut_points: List[int],
        marker_segments: Optional[List[Dict]] = None,
        pure_music_segments: Optional[List[Dict]] = None,
        instrumental_audio: Optional[np.ndarray] = None,
        original_audio: Optional[np.ndarray] = None,
    ) -> List[bool]:
        """
        [v2.9.1 精准分类修复版]
        综合 MDX23 分离结果、能量占比和置信标记判断片段是否包含人声。

        修复思路:
        1. 标记优先: 明确的无人声标记(pure_music_segments)拥有一票否决权。
        2. 动态阈值: 使用噪声地板(noise_floor)而非固定阈值来判断人声活跃度，适应不同动态范围。
        3. 多维投票: 综合“标记重叠率”、“能量比”、“人声活跃度”三个维度进行投票，逻辑更鲁棒。
        """
        num_segments = max(len(cut_points) - 1, 0)
        self._last_energy_ratio  = []
        if num_segments == 0:
            return []

        sr = self.sample_rate
        if sr <= 0 or vocal_audio is None or getattr(vocal_audio, 'size', 0) == 0:
            # 输入无效时的回退逻辑
            return [True] * num_segments

        # --- 从config加载参数 ---
        overlap_music_ratio = float(get_config('quality_control.segment_music_overlap_ratio', 0.55))
        energy_music_ratio = float(get_config('quality_control.segment_music_energy_ratio', 0.4))
        presence_vocal_ratio = float(get_config('quality_control.segment_vocal_presence_ratio', 0.35))
        noise_margin_db = float(get_config('quality_control.segment_noise_margin_db', 6.0))
        
        # --- 辅助函数 ---
        def _get_music_intervals(segments: Optional[List[Dict]]) -> List[Tuple[float, float]]:
            intervals = []
            if not segments: return intervals
            for seg in segments:
                try:
                    start, end = float(seg['start']), float(seg['end'])
                    if end > start and not seg.get('is_vocal', True):
                        intervals.append((start, end))
                except (ValueError, TypeError, KeyError):
                    continue
            return intervals

        def _total_overlap(seg_start: float, seg_end: float, intervals: List[Tuple[float, float]]) -> float:
            total = 0.0
            for start, end in intervals:
                overlap = min(seg_end, end) - max(seg_start, start)
                if overlap > 0:
                    total += overlap
            return total
            
        # --- 准备标记和噪声地板 ---
        music_intervals = _get_music_intervals(marker_segments)
        if pure_music_segments:
             music_intervals.extend(_get_music_intervals(pure_music_segments))

        noise_floor_db = None
        if music_intervals:
            noise_samples = []
            for start_s, end_s in music_intervals:
                start_idx, end_idx = int(start_s * sr), int(end_s * sr)
                segment = vocal_audio[start_idx:end_idx]
                if segment.size > 0:
                    rms = np.sqrt(np.mean(np.square(segment)) + 1e-12)
                    noise_samples.append(20 * np.log10(rms))
            if noise_samples:
                noise_floor_db = float(np.median(noise_samples))

        # --- 逐段分类 ---
        flags: List[bool] = []
        for i in range(num_segments):
            start_idx, end_idx = cut_points[i], cut_points[i+1]
            seg_start_s, seg_end_s = start_idx / sr, end_idx / sr
            seg_duration = seg_end_s - seg_start_s
            
            vocal_segment = vocal_audio[start_idx:end_idx]
            
            # 1. 标记优先判断 (Marker First)
            music_overlap = _total_overlap(seg_start_s, seg_end_s, music_intervals)
            if (music_overlap / seg_duration) >= overlap_music_ratio:
                flags.append(False)
                self._last_segment_classification_debug.append({'index': i, 'decision': False, 'reason': 'marker_music_overlap'})
                continue

            # 2. 能量比判断 (Energy Ratio)
            if instrumental_audio is not None and original_audio is not None:
                inst_segment = instrumental_audio[start_idx:end_idx]
                vocal_energy = np.mean(np.square(vocal_segment))
                inst_energy = np.mean(np.square(inst_segment))
                if (vocal_energy + inst_energy > 1e-12) and (vocal_energy / (vocal_energy + inst_energy)) <= energy_music_ratio:
                    flags.append(False)
                    self._last_segment_classification_debug.append({'index': i, 'decision': False, 'reason': 'energy_music_ratio'})
                    continue

            # 3. 人声活跃度判断 (Vocal Presence)
            if vocal_segment.size > 0:
                rms_db = 20 * np.log10(np.sqrt(np.mean(np.square(vocal_segment))) + 1e-12)
                
                # 使用动态噪声地板或备用静态阈值
                threshold = noise_floor_db + noise_margin_db if noise_floor_db else get_config('quality_control.segment_vocal_threshold_db', -50.0)
                
                if rms_db < threshold:
                    flags.append(False)
                    self._last_segment_classification_debug.append({'index': i, 'decision': False, 'reason': f'rms_below_threshold ({rms_db:.1f} < {threshold:.1f})'})
                    continue

            # 4. 如果以上都未判断为 music，则认为是 human
            flags.append(True)
            self._last_segment_classification_debug.append({'index': i, 'decision': True, 'reason': 'default_human'})
            
        return flags
```


### plan B

### 绝对阈值/单通道能量判定
不同歌的整体响度、动态范围差异极大，绝对阈值（比如 -50 dB）在弱混音段会把噪底当人声，在强混音段又把真唱段压下去。应该转为相对噪底 + **人声/伴奏能量差（ΔLUFS 或 ΔRMSdB）**的联合判定。你的 README 里有“动态噪声地板、滚动分位数”等思路，但后缀判定若没复用，就会漂移

只看混音，不看分离结果的人声占优程度
若后缀是基于原混音做 RMS/LUFS，副歌里鼓/合成器的能量会盖住人声，造成 _music 误标；桥段/间奏的人声残响又会把 _human 误开。应至少比较分离后的人声轨 vs 伴奏轨在该片段的能量差（ΔdB）。

没有“有声帧比例”（voiced ratio）约束
人声的特征是持续的有声帧（元音、带基频/谐波）。如果只看平均能量，嘶擦音、和声器垫、吉他泛音都可能“像人声”。应要求片段内有声帧占比 ≥ 阈值（例如 ≥30%），否则即便能量高也判 _music。

统计口径用“均值”而非“稳健统计”
用全段均值/最大值很容易被边界、呼吸音或一次爆音拉偏。建议用75 分位 RMSdB / LUFS作为代表值，并结合滚动 5% 分位噪底。你的 README 里强调了分位数阈值与滞回，但后缀打标若没用到，会导致口径不一致

片段太短或靠近切点
切点右推守卫虽能保音质，但会把跨越切点的元音尾部压进下一个片段，导致前后片段能量统计失真。应对**<5s短段或距离切点 <150ms的区域做特判/加权。README 的守卫参数已给出（win_ms / guard_db / search_right_ms / floor_percentile），后缀判定要沿用同一套守卫口径**

立体声到单声道的 6 dB 偏差 & 采样率/正则化不一致
有些能量计算把 L/R 简单相加或没对齐规范化，会带来固定偏移。确保通道汇聚采用均值而非求和，并在相同采样率与相同增益基准下比较。

// src\vocal_smart_splitter\config.yaml

```
quality_control:
  segment_vocal_threshold_db: -50.0    # 保留：老阈值，仍用于兜底
  delta_human_db: 6.0                  # 新：人声/伴奏能量差阈值（dB）
  delta_music_db: 3.0                  # 新：音乐偏向阈值（dB）
  min_voiced_ratio: 0.30               # 新：_human 的最小有声帧比例
  maybe_voiced_ratio: 0.15             # 新：_music 的上限（以下强判 music）
  suffix_hysteresis: 1                  # 新：是否启用邻接/历史平滑
```

// src/vocal_smart_splitter/core/quality_controller.py

```
# === 文件: src/vocal_smart_splitter/core/quality_controller.py ===
# [INSERT POINT A]：在文件顶部 imports 后添加
import numpy as np

def _rms_db(x: np.ndarray, eps=1e-12) -> float:
    """Return RMS in dBFS of mono signal."""
    if x.ndim == 2:  # [C, T] or [T, C]
        x = x.mean(axis=-1) if x.shape[0] != 1 else x[0]
    rms = np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps)
    return 20.0 * np.log10(rms + eps)

def _robust_db(x: np.ndarray, q=0.75) -> float:
    """75th percentile RMS dB for robustness against silence/peaks."""
    if x.ndim > 1:
        x = x.mean(axis=-1)
    # 分帧求 RMS，避免整段被边界拉偏
    fsz = max(1024, min(len(x)//64, 8192))
    hop = fsz // 2
    frames = np.lib.stride_tricks.sliding_window_view(x, fsz)[::hop]
    if len(frames) == 0:
        return _rms_db(x)
    vals = [ _rms_db(f) for f in frames ]
    return float(np.quantile(vals, q))

def _voiced_ratio_from_floor(vocal: np.ndarray, floor_db: float, down=3.0, up=6.0) -> float:
    """
    用“动态噪底 + 滞回双阈”统计有声帧比例。
    floor_db 来自你已有的滚动 5% 分位噪底估计（与 README 一致）。
    """
    if vocal.ndim > 1:
        vocal = vocal.mean(axis=-1)
    fsz = 1024
    hop = 512
    frames = np.lib.stride_tricks.sliding_window_view(vocal, fsz)[::hop]
    if len(frames) == 0:
        return 0.0
    voiced = 0
    state = False
    for f in frames:
        db = _rms_db(f)
        if state:
            # 低于 floor+down 退出有声
            state = db >= (floor_db + down)
        else:
            # 高于 floor+up 进入有声
            state = db >= (floor_db + up)
        voiced += int(state)
    return voiced / len(frames)

def classify_segment_suffix(vocal_seg: np.ndarray,
                            music_seg: np.ndarray,
                            floor_db: float,
                            delta_human_db: float = 6.0,
                            delta_music_db: float = 3.0,
                            min_voiced_ratio: float = 0.30,
                            maybe_voiced_ratio: float = 0.15) -> str:
    """
    返回 'human' 或 'music'。
    - 采用 75 分位稳健 RMSdB
    - 结合有声帧比例（与纯人声检测同一口径）
    - 双门限 + 滞回设计避免抖动
    """
    v_db = _robust_db(vocal_seg)
    m_db = _robust_db(music_seg) if music_seg is not None else (v_db - 12.0)  # 没有伴奏轨时的保守兜底
    delta_db = v_db - m_db
    vr = _voiced_ratio_from_floor(vocal_seg, floor_db)

    if delta_db >= delta_human_db and vr >= min_voiced_ratio:
        return "human"
    if delta_db <= delta_music_db or vr <= maybe_voiced_ratio:
        return "music"
    # 中间地带：交给调用处做邻接平滑/历史平滑
    return "human" if vr >= (min_voiced_ratio * 0.8) else "music"
```
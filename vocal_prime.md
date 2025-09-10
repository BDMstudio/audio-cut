# audio-cut：人声停顿分割失准问题技术复盘与修复方案（MDX23 + VAD）

> 适用仓库：`https://github.com/BDMstudio/audio-cut.git`
> 目标：为什么“纯人声域切割仍落在能量峰值”，以及**如何一刀切回静音谷底**。
> 风格：直接可落地，函数名与当前工程保持一致。

---

## 1. 现象（Symptoms）

* **切点明显落在能量高峰**（波形/频谱峰值处），而不是静音/低能量谷底。
* **三类切点都会命中**：中间停顿、头部、尾部。
* **耗时接近实时**（≈1×音频时长），推测仍存在“逐帧循环/重复重采样/未启用批处理/只在CPU做部分环节”。

---

## 2. 根因（Root Causes）

> 命中率从高到低；多因子叠加会放大错误。

1. **零交叉（ZC）对齐无“能量守卫”**
   `zero_cross_align` 在任意位置附近拉最近零交叉；**强能量段零交叉密集**，若没有“切点处必须足够安静”的二次校验，就会把切点**拉进峰值**。

2. **BPM 逻辑被实现为“吸附/推移”，而不是“禁切区”**
   任何“靠近拍点→把切点推到拍点附近（鼓击/瞬态）”都会**把切点推到能量峰**。BPM 只能**禁止**在拍点±Δms切，**绝不能吸附**。

3. **“人声域检测 → 原混音切割”的时间映射错位**
   Vocal stem（常 16 kHz）上找到的 t，如果**没正确映射**到原混音（44.1k/48k）——包括**重采样群延迟/滤波延迟**——落点会整体偏移；再叠加 ZC/BPM 推移，**极易落峰**。

4. **头/尾偏移符号或单位错误**
   `head_offset=-0.5s / tail_offset=+0.5s` 如被当“绝对值”或“样本”使用，必然切进人声。

5. **验证器在“重定位”而非“过滤”**
   `QualityController.validate_split_points` 若包含“挪到拍点/整拍/最近ZC”的重定位逻辑，等于**改写切点**，常把切点带进峰值。

6. **仍在混音上做检测 / 分离增强未启用**
   若增强分离（MDX23/Demucs）未实际生效或 VAD 输入不是 vocal stem，则鼓/伴奏会制造**假静音平台**——中心看似对齐，ZC/拍点一推就进峰。

---

## 3. 诊断清单（One-pass Checklist）

在 `quick_start.py` 跑一次，打印并核对：

* **后端状态**：最终分离后端（mdx23/demucs/hpss）、权重路径/哈希、`torch.cuda.is_available()`、ONNX providers。
* **检测域**：确认 VAD/停顿检测**使用 vocal\_stem**（不是混音）。
* **切点轨迹日志（每个切点必须输出）**：
  `t0(静音中心) → t1(右偏) → t2(ZC) → t3(安静守卫后)`，以及
  `local_rms_db, floor_db, nearest_beat_dist_ms, moved_by_bpm?, moved_by_zc?`。
* **跨采样率映射**：打印 `sr_vocal→sr_raw 的映射公式 + latency_samples`。
* **二验次数**：`mask 二验调用次数` 与 `拒绝/接受`统计。

若看不到以上信息，先修日志与前置检查，再谈算法。

---

## 4. 解决方案（Fixes）

### 4.1 **加“安静守卫（Quiet-Cut Guard）”，压住一切重定位**

在 `src/vocal_smart_splitter/core/quality_controller.py` 中新增（或合并）：

```python
import numpy as np
from ..utils.feature_extractor import moving_rms_db, rolling_percentile_db, ema_smooth

class QualityController:
    # ... 你的 __init__ 等保持不变

    def enforce_quiet_cut(self, x_mono, sr, t_sec,
                          win_ms=80, guard_db=3.0, floor_pct=0.05,
                          search_right_ms=220):
        """
        局部RMS + 动态噪声地板：若 t 附近不够安静，只向右搜索第一个“够安静”的谷底。
        - win_ms: 评估窗口
        - guard_db: 安静阈值余量（floor+3dB 以内才算安静）
        - search_right_ms: 只允许向右移动（避免提前）
        """
        hop_ms = 10
        rms_db, _ = moving_rms_db(x_mono, sr, frame_ms=win_ms, hop_ms=hop_ms)
        rms_db = ema_smooth(rms_db, sr, hop_ms=hop_ms, smooth_ms=120)
        floor_db = rolling_percentile_db(rms_db, sr, hop_ms=hop_ms, win_s=30.0, p=floor_pct)

        def ok(idx): return rms_db[idx] <= floor_db[idx] + guard_db
        idx = int(t_sec / (hop_ms/1000.0))
        if idx < 0 or idx >= len(rms_db): return t_sec
        if ok(idx): return t_sec

        max_step = int(search_right_ms / hop_ms)
        for k in range(1, max_step+1):
            j = idx + k
            if j >= len(rms_db): break
            if ok(j): return j * (hop_ms/1000.0)
        return t_sec  # 找不到就保留（或上层丢弃）
```

**使用原则**：**任何**重定位（右偏、ZC、BPM避让、头尾偏移）之后，**必须**调用一次 `enforce_quiet_cut()`；不安静就**只向右**挪到最近安静谷底。

---

### 4.2 **把 BPM 从“吸附/推移”改成“禁切区”**

在 `choose_split_points_from_vocal(...)` 中：

* **禁止**把切点向拍点“拉近/吸附”；
* 仅当切点落入拍点±`forbid_cut_within_ms_of_beat` 时，**向右**移出禁切区（不可向左）；
* 移出后**立即**调用 `enforce_quiet_cut()`。

伪代码（你已有该函数，改成下述逻辑即可）：

```python
if bpm_guard_enable and beats is not None:
    if any(abs(beats - t) <= forbid_ms):
        t = t + (forbid_ms / 1000.0)  # 只向右避让
t = qc.enforce_quiet_cut(raw_mono, sr_raw, t)  # 避让后立刻安静守卫
```

---

### 4.3 **修正“人声域 → 原混音”的时间映射（含群延迟）**

**核心**：不要直接用 `t_raw = t_vocal * (sr_raw/sr_vocal)`。
加入可配置的重采样延迟 `latency_samples`（测一次写入配置）。

在 `utils/audio_processor.py` 添加：

```python
def map_time_between_domains(t_src_sec: float, sr_src: int, sr_dst: int,
                             latency_samples: int = 0) -> float:
    """
    把 vocal 域时间映射回原混音域；latency_samples>0 表示 src->dst 的群延迟（样本）。
    """
    return (t_src_sec * sr_src + latency_samples) / sr_dst
```

在 `SeamlessSplitter.split_audio_seamlessly(...)` 中使用它，将 **vocal 域切点**映射为**原混音切割时间**，再做 ZC 与安静守卫（ZC 的 `sr` 必须用 `sr_raw`）。

> `latency_samples` 获取：给分离/重采样链路输入一个单位脉冲，测量主峰位置差值即可；写到 `config.yaml: time_mapping.latency_samples`。

---

### 4.4 **头/尾偏移只允许“向外”，并做单位校验**

* 头部：人声开始前 `|head_offset|` 秒（**向更早**）；
* 尾部：人声结束后 `tail_offset` 秒（**向更晚**）；
* 确认偏移量单位为**秒**，并在主流程里统一转换；
* 偏移后立即 `enforce_quiet_cut()`。

---

### 4.5 **验证器只“过滤”，绝不“改点”**

`QualityController.validate_split_points(...)`

* 仅做：**最小间隔、最小停顿、人声能量上限**等布尔判定；
* **不要**在这里做 ZC/拍点/对齐类重定位；
* 保证“切点调整权”全部在**生成阶段**且都经过**安静守卫**。

---

### 4.6 **确保 VAD/停顿检测运行在 vocal 域**

在 `seamless_splitter.py` 的主流程：

1. `vocal_stem = EnhancedVocalSeparator.get_vocal_stem(raw_mono, sr_raw)`
2. `silence_platforms, speech_segments = detect_vocal_silence_and_speech(vocal_stem_16k, 16k)`
3. `cut_points_vocal = choose_split_points_from_vocal(..., vocal_stem_16k, 16k, ...)`
4. **映射**：`t_raw = map_time_between_domains(t_vocal, 16k → sr_raw, latency_samples)`
5. 在 `raw_mono` 上：ZC → **安静守卫** → 导出

**不要**把混音交给 VAD 做主判定；分离增强没启用时应**报错退出**，而不是静默降级。

---

## 5. 最小代码接入点（与项目命名一致）

* `core/vocal_pause_detector.py`

  * 使用你已有的 `detect_vocal_silence_and_speech()` + `choose_split_points_from_vocal()`（若缺失，参考我之前给的实现）。
* `core/quality_controller.py`

  * 新增 `enforce_quiet_cut()`（见 4.1），`validate_split_points()` 保留“只过滤”。
* `utils/audio_processor.py`

  * 新增 `map_time_between_domains()`（见 4.3），`find_zero_crossing()` 已有则复用。
* `core/seamless_splitter.py`

  * 在生成切点后**立刻**：映射 → ZC（raw域）→ `enforce_quiet_cut()` → `validate_split_points()`。
* `core/enhanced_vocal_separator.py`

  * `get_vocal_stem()` 返回声道/采样率信息；若内部重采样到 16k，请把 `latency_samples` 暴露到上层（或写入 `analysis_report.json`）。

---

## 6. 配置（YAML）关键项

```yaml
pause_detector:
  frame_ms: 30
  hop_ms: 10
  smooth_ms: 120
  floor_percentile: 0.05
  db_above_floor_down: 3.0
  db_above_floor_up: 6.0
  min_silence_sec: 1.1
  plateau_flatness_db: 6.0
  lookahead_guard_ms: 1200
  right_bias_ms: 80
  zero_cross_align: true

bpm_guard:
  enable: true
  forbid_cut_within_ms_of_beat: 100
  allow_override_if_plateau_ms: 700   # 仅在超宽静平台内“向右”避让，不吸附

quality_control:
  min_split_gap: 2.0
  min_pause_at_split: 1.0
  # 新：安静守卫参数（如果你放到 QC 层）
  quiet_guard_win_ms: 80
  quiet_guard_db: 3.0
  quiet_guard_search_right_ms: 220

time_mapping:
  sr_vocal: 16000
  sr_raw: 44100
  latency_samples: 0   # 实测后填写
```

---

## 7. 性能与可观测性

* **只对可疑区段 ±2s 跑分离二验**；全曲只解码一次；16k/mono 检测域。
* **批处理候选区块**送 MDX23/Demucs；GPU 上合适的 batch 能显著提速。
* 打开日志：

  * `selected_backend, weights_path, device/provider`
  * 每个切点的 `t0→t1→t2→t3` 与 `local_rms_db / floor_db`
  * `mapped from 16k to 44.1k: latency_samples=...`
* 导出两张**诊断图**（可选）：

  1. vocal 域 RMS\_dB 与候选切点；
  2. raw 域 RMS\_dB 与最终切点（标注“是否通过安静守卫”）。

---

## 8. 验收标准（DoD）

* **切点局部安静**：最终切点处 `local_rms_db ≤ floor_db + 3 dB`。
* **永不提前**：任何调整只允许**向右移动**。
* **BPM 不吸附**：仅存在“禁切区避让”的右移记录。
* **跨域一致**：vocal→raw 映射后误差 ≤ 10 ms（含群延迟校正）。
* **回归样本**：对 10 条样本输出 `analysis_report.json`，统计：

  * 切点在峰值±30ms 的比例 **< 5%**（目标 < 1%）。
  * 切点后 1.0–1.5s 持续静音的比例 **≈ 100%**。

---

## 9. 常见坑位与防御

* **“看起来用了MDX23，其实没用上”**：权重缺失/导入失败被吞、auto 回退 HPSS。→ 前置检查+失败即报错。
* **ZC 抢权**：ZC 只能微调，**安静守卫**才是“一票否决”。
* **BPM 左移**：严格禁止；只向右避让。
* **单位混乱**：所有偏移/窗口统一“秒”，只在信号层转换为样本。
* **群延迟**：分离/重采样必有延迟；测一次写死配置。

---

## 10. 下一步（行动清单）

1. 合入 `enforce_quiet_cut()` 与 `map_time_between_domains()`。
2. 改 `choose_split_points_from_vocal()`：BPM→禁切区，重定位后立即安静守卫。
3. 在 `seamless_splitter.py`：**vocal→raw 映射** + ZC(raw) + 安静守卫 + QC过滤。
4. 打开**详尽日志**并保存诊断图。
5. 用 3 条问题样本回归：对比修改前后“切点是否落在能量谷底”。

---

**一句话版**：
切点落峰，不是参数玄学，是**控制流**问题。把“**安静守卫**”放在所有重定位之后，BPM只做**禁切区**，修好**跨域映射**与**头尾偏移**，你就能稳定地把切点钉在**静音谷底**，而不是被拍点/零交叉“拉进峰值”。

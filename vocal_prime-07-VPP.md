## VocalPrime v2.x · VPP（Vocal Pause Profile）自适应与加权选点方案（方案2）

### 目标
- 在 BPM、MDD 之外加入“人声停顿画像 VPP”，让阈值与选点更贴合“我们切的是人声停顿本身”。
- 采用方案2（加权NMS在终筛阶段执行）：在相邻时间邻域内仅保留“更长、且更安静”的候选切点。

### 关键约束与注意事项
- 避免把“间奏/整段空白”计入停顿统计（会把全曲误判为“慢停顿”）。
- VPP 仅用于自适应倍率，不直接决定切点；最终切点仍需通过“右推安静守卫 + 零交叉吸附 + 时长治理”。
- 方案2要求在 `_finalize_and_filter_cuts_v2` 里支持 `(time, score)` 的加权去重，但不改变已存在的“合并短段/强拆长段”策略。
- 性能要求：VPP 计算在整曲仅执行一次；特征全基于 vocal_track，避免引入大模型或多次分离。

### 总体思路
- 三路自适应：`mul_final = clamp(mul_bpm × mul_mdd × mul_pause, clamp_min, clamp_max)`
- 其中 `mul_pause` 来自 VPP 分类（pause-slow/medium/fast）。
- 终筛改造：把候选切点带上 `score`，在 `_finalize_and_filter_cuts_v2` 里做“加权NMS”（邻域内只保留分最高的切点），随后再执行守卫校正与时长治理。

### 一、VPP（Vocal Pause Profile）
1) singing_blocks 过滤（避免计入间奏）
- 在 vocal_track 上计算 `e_db(t)` 与 `floor_db(t)`（滚动5%分位或全局5%分位）。
- `voice_active = e_db(t) > floor_db(t) + Δ_db`（默认 `Δ_db=3.0dB`）。
- 形态学闭运算（`morph_close_ms=150`）填充词内微空洞，再开运算（`morph_open_ms=50`）去掉孤立毛刺。
- 连通域中时长 `< sing_block_min_s`（默认 2.0s）的段落丢弃；剩余为 `singing_blocks`。
- 如果存在 `duration ≥ interlude_min_s`（默认 4.0s）、且周围±2s 内 `voice_active` 覆盖率 `< 10%` 的大段静默，标记为 `interlude`，不计入统计（也不作为候选来源）。

2) 统计度量（仅在 `singing_blocks` 内）
- `MPD`：停顿时长中位数；`P95`：95分位；
- `PR`：每分钟停顿次数；`RR`：停顿总时长/块内时长；`PV`：std/median（区分“稳定长停顿”vs“碎停顿”）。

3) 分类与倍率（可配）
- `pause-slow`：`MPD ≥ 0.60s` 或 `P95 ≥ 1.20s` 或 `RR ≥ 0.35` → `mul_pause=1.10`
- `pause-fast`：`MPD ≤ 0.25s` 且 `PR ≥ 18/min` 且 `RR ≤ 0.15` → `mul_pause=0.85`
- 其他 → `pause-medium` → `mul_pause=1.00`
- 钳制：`clamp_min=0.75`, `clamp_max=1.25`（与 BPM/MDD 同一钳制范围）。

### 二、加权选点（方案2：在终筛执行）
1) 候选打分（在能量谷检测阶段产生）
- 时长分：`len_score = norm(duration, 0.2..1.5s)`（线性或σ型映射到 0..1）。
- 安静度分：`quiet_score = norm(depth_db, 0..20dB)` 或 `quiet_score = 1 - pause_energy/threshold`。
- 频谱提示（可选）：`flatness_hint` 高则加微分。
- 综合：`score = w_len·len_score + w_quiet·quiet_score + w_flat·flatness_hint`
  - 建议权重：`w_len=0.6, w_quiet=0.4, w_flat=0.1`。
- 输出候选格式：`[(time_sec, score), ...]`。

2) 加权NMS（在 `_finalize_and_filter_cuts_v2` 内执行）
- 输入：`[(time_sec, score)]`
- 步骤：
  - 对候选按 `score` 降序；
  - 依次取未抑制的最高分点，抑制其±`min_split_gap` 范围内的其它点；
  - 输出通过 NMS 的 `time_sec` 列表；
- 然后继续现有流程：右推安静守卫 → 最小间隔过滤（安全）→ 合并短段/强拆长段。

### 配置新增（建议写入 `config.yaml`）
```yaml
pure_vocal_detection:
  pause_stats_adaptation:
    enable: true
    delta_db: 3.0
    morph_close_ms: 150
    morph_open_ms: 50
    sing_block_min_s: 2.0
    interlude_min_s: 4.0
    classify_thresholds:
      slow: { mpd: 0.60, p95: 1.20, rr: 0.35 }
      fast: { mpd: 0.25, pr: 18, rr: 0.15 }
    multipliers: { slow: 1.10, medium: 1.00, fast: 0.85 }
    clamp_min: 0.75
    clamp_max: 1.25

  valley_scoring:
    w_len: 0.6
    w_quiet: 0.4
    w_flat: 0.1
    use_weighted_nms: true
```

### 伪代码
- VPP 估计（在 `pure_vocal_pause_detector.detect_pure_vocal_pauses` 的相对能量分支内，BPM/MDD 之后）：
```python
# ---- VPP: 只统计 singing_blocks 内的停顿 ----
mask = (e_db(audio) > floor_db(audio) + delta_db)
mask = morph_close(mask, ms=150)
mask = morph_open(mask, ms=50)
blocks = connected_components(mask, min_len_s=2.0)

pauses = []
for block in blocks:
    pauses += find_rest_segments_within(block, threshold=floor+delta_db)
    # 过滤 interlude：duration>=4s 且 局部voice_active覆盖率<10%
    pauses = [p for p in pauses if not is_interlude(p)]

mpd = median([p.len for p in pauses]); p95 = percentile([p.len], 95)
pr = len(pauses) / (audio_len/60); rr = sum(p.len)/sum(block.len)
cls = classify(mpd, p95, pr, rr)
mul_pause = {slow:1.10, medium:1.00, fast:0.85}[cls]

# 融合自适应倍率
mul_final = clip(mul_bpm * mul_mdd * mul_pause, clamp_min, clamp_max)
peak_ratio *= mul_final; rms_ratio *= mul_final
```

- 能量谷打分（在 `_detect_energy_valleys` 内，产生 `(time, score)`）
```python
for valley in valleys:
    duration = valley.end - valley.start
    depth_db = threshold_db - mean_db(valley.range)
    len_score = map_to_0_1(duration, 0.2, 1.5)
    quiet_score = map_to_0_1(depth_db, 0, 20)
    flat_score = estimate_flatness_hint(valley)  # 可选
    score = w_len*len_score + w_quiet*quiet_score + w_flat*flat_score
    candidates.append((valley.center_time, score))
```

- 加权NMS（在 `_finalize_and_filter_cuts_v2`，候选传入时带 `score`）
```python
# 输入: candidates = [(t, s), ...]
keep = []
for t, s in sorted(candidates, key=lambda x: x[1], reverse=True):
    if all(abs(t - kt) >= min_split_gap for kt, _ in keep):
        keep.append((t, s))
final_times = [t for t, _ in sorted(keep)]
# 后续：右推安静守卫 → 最小间隔过滤 → 合并短段/强拆长段（存在则覆盖）
```

### 验证计划
- 合成序列：长停顿/短停顿/碎停顿三组 + 两种 BPM；断言：VPP 分类、mul_pause 与候选量变化方向正确。
- 真实样本：慢曲、快曲、快曲但长停顿、慢曲但碎停顿、强副歌/弱主歌；指标：
  - E2E 段长分布更接近目标范围（<30s 且无 0s）。
  - 切点主观自然度上升；邻域内切点倾向更长更安静（得分更高）。

### 风险与回退
- VPP 误分类：保留 clamp 范围，倍率变化 ≤ ±25%，并保留 BPM/MDD 两路共同制衡。
- 性能：全曲一次 RMS/flatness + 形态学，CPU 代价极低；如出现性能问题，可降采样至 22.05kHz 做 VPP 估计。

### 日志与可观测性
- 输出：`VPP{cls=slow|medium|fast, mpd, p95, pr, rr}`、`mul_bpm/mul_mdd/mul_pause/mul_final`、`NMS_before/after`、候选最高分等。

结论：VPP + 方案2（加权NMS）能在不破坏现有终筛与守卫逻辑的前提下，稳定地把切点往“更长且更安静”的停顿处集中，同时避免被间奏误导，整体提升切割自然度与一致性。


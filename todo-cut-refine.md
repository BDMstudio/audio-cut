# todo-cut-refine — TODO（碎片段合并与时长均衡）

> 目的：解决切割结果中出现的 1–2s 碎片与过长片段，**不牺牲停顿自然度**，**不明显增加运行时长**。
> 原则：保持“**先 NMS 后守卫**”的主流程不变，**只在段落层**做轻量后处理；仍以**切点表**为唯一事实来源，保持**可逆性=0**。

---

## 0. 术语与约定

* **Segment**：由相邻两个切点定义的区间，带 `kind∈{human,music}`。
* **微碎片**：`dur < micro_merge_s`，默认 2.0s。
* **软最小/软最大**：`soft_min_s`（默认 6.0s），`soft_max_s`（默认 18.0s）。不强拆/强并，按**成本函数**做“温和合并/再切”。
* **救援切点**：超长段内部选出的新切点；优先选用被 NMS 抑制的候选，其次能量谷。
* **features_cache**：全曲一次 STFT/缓存的 RMS、MDD、BPM、VAD 等序列（已有）。
* **suppressed_cuts**：NMS 时被抑制的候选切点（需从 `refine.finalize_cut_points` 返回备用）。

---

## 1. 交付目标（Definition of Done）

* [ ] **碎片率**：`dur < 2s` 的段占比 **= 0**；`dur < soft_min_s` 的段占比 **≤ 2%**。
* [ ] **自然度**：跨越“强停顿”（`pause_strength ≥ 0.6`）的合并比例 **≤ 5%**。
* [ ] **可逆性**：保持 0（拼接误差=0）。
* [ ] **节拍一致性（可选）**：启用节拍吸附时，cut-to-beat 偏差中位数 **≤ 60ms**。
* [ ] **性能**：端到端耗时增加 **< 3%**。
* [ ] **参数化**：新增配置项与 CLI 开关；文档与调参指南完善。
* [ ] **CI 守门**：基于基准集的碎片率/时长分布/强停顿跨越率指标，未达标阻断合并。

---

## 2. 配置与开关（新增）

```yaml
segment_layout:
  enable: true

  # 双阈值
  micro_merge_s: 2.0          # <2s 直接合并
  soft_min_s: 6.0             # 短于此按成本合并（可多次级联）
  soft_max_s: 18.0            # 长于此尝试“救援切点”分割一次

  # BPM 自适应（高置信度时生效）
  bpm_adapt:
    enable: true
    beats_min: 8              # soft_min_s ≈ beats_min * 60 / bpm, clamp(5,8)
    beats_max: 32             # soft_max_s ≈ beats_max * 60 / bpm, clamp(12,24)
    conf_threshold: 0.8
    human_offset_s: 0.5       # 人声段阈值上调
    music_offset_s: 0.0

  # 代价函数权重
  weights:
    pause:   1.0              # 跨越切点强度惩罚
    kind:    1.2              # 人声/伴奏跨类惩罚
    energy:  0.6              # 能量上跳(dB)惩罚
    overmax: 0.8              # 合并后超过 soft_max 的惩罚
    pos:     0.5              # 救援切点距离段中心的惩罚

  # 其他
  min_gap_s: 1.0              # 最终相邻切点最小间隔保障
  beat_snap_ms: 80            # （可选）节拍吸附窗口，0=关闭
```

CLI：

* `--segment-layout [on|off]`（默认 on）
* `--segment-layout-profile {default, vocal-heavy, bgm-only}`（预设不同权重与阈值）

---

## 3. 模块与接口（新增文件）

**文件**：`src/audio_cut/cutting/segment_layout_refiner.py`

```python
# ==== 数据结构 ====
@dataclass
class Segment:
    start: float
    end: float
    kind: Literal["human", "music"]
    @property
    def dur(self) -> float: return self.end - self.start

@dataclass
class LayoutConfig:
    micro_merge_s: float
    soft_min_s: float
    soft_max_s: float
    min_gap_s: float
    beat_snap_ms: float
    weights: dict
    bpm_adapt: dict

# ==== 入口 ====
def refine_layout(
    segments: list[Segment],
    suppressed_cuts: list[tuple[float, float]],  # (t, score)
    features,                                     # TrackFeatureCache
    cfg: LayoutConfig
) -> list[Segment]:
    ...
```

**集成点**：在 `SeamlessSplitter` 产生初始 `segments` 之后、导出音频之前调用：

```python
if cfg.segment_layout.enable:
    segments = segment_layout_refiner.refine_layout(
        segments, suppressed_cuts, ctx.features, cfg.segment_layout
    )
```

> 注：需要把 `refine.finalize_cut_points(...)` 的返回更新为 `(final_cuts, suppressed_cuts)` 或提供 `get_suppressed()` 旁路。

---

## 4. 实施步骤（按优先级执行）

### P0：打基础（必做）

* [x] **返回 suppressed_cuts**

  * 修改 `refine.finalize_cut_points`：在 NMS 选择集之外的候选（含 `t, score, kind`）以列表返回给调用方。

* [x] **实现 LayoutConfig 派生**

  * 根据 `features.bpm` 与 `bpm_conf`、`bpm_adapt.conf_threshold` 调整 `soft_min_s/soft_max_s`；对人声段加 `human_offset_s`。
  * 若 `bpm_conf < 阈值` → 回落到固定 6/18（并记录日志）。

* [x] **A：微碎片合并**

  * 遍历 `segments`，对 `dur < micro_merge_s` 的段执行**无条件合并**：计算左右两侧 **`merge_cost`**，选择代价更小的一侧合并。
  * 边界条件：若只有一侧可合并（首尾段），则合并该侧。

* [ ] **B：软最小合并（堆驱动）**

  * 构建小根堆 `(dur, idx)`，循环：

    * 取最短段 `S_i`，若 `dur ≥ soft_min_s` 跳出；
    * 用 `merge_cost(i→左)`、`merge_cost(i→右)` 选较小者；
    * **只有当总代价 < 阈值**（可用 `pause_strength < 0.35` 且 `energy_jump_db_pos < 6dB` 的布尔门限）才执行合并；
    * 合并后更新受影响的邻居到堆；支持级联直至所有段 `≥ soft_min_s` 或不可合并。

* [ ] **C：软最大救援切分**

  * 对 `dur > soft_max_s` 的段，调用 `pick_rescue_cut`：

    * 优先在本段内的 `suppressed_cuts` 中选择 `split_cost` 最低者；
    * 若无，回退到能量谷搜索（限制在段中部 ±25% 的区间）；
    * 保障 `min_gap_s`：新切点与两侧切点间隔必须 ≥ `min_gap_s`；
    * 一次救援完成后可继续检查右段是否仍 > `soft_max_s`。

* [ ] **D：最终 min-gap 校验**

  * 统一跑一次 `enforce_min_gap`，确保守卫右推后的边界也不触碰。

* [ ] **（可选）E：节拍吸附**

  * 若 `beat_snap_ms > 0` 且 BPM 可信，将最终切点向最近的节拍网格吸附（限制在 ±`beat_snap_ms` 内），吸附后再次执行 `min-gap` 校验。

### P1：监控与调参

* [ ] **bench 指标**：在 `scripts/bench/run_bench.py` 增加

  * `fragment_rate_<2s>`、`rate_<soft_min_s>`、`segment_dur_p50/p95/max`、`strong_pause_cross_rate`、`beat_snap_median_ms`、`runtime_overhead%`。
* [ ] **日志**：为每次合并/再切打印一条理由（代价各项、左/右选择、新切点来源等）。
* [ ] **Profile**：记录本模块耗时（目标 < 3%）。

### P2：预设与回滚

* [ ] 预设 `segment-layout-profile`：

  * `default`（上面默认值）、`vocal-heavy`（`human_offset_s=+1.0,w_kind↑`）、`bgm-only`（`soft_min_s=4, soft_max_s=16`）。
* [ ] 回滚开关：`segment_layout.enable=false` 或 CLI `--segment-layout off` 立即恢复旧行为。
* [ ] 文档：`docs/tuning_segment_layout.md`（每个参数的作用、单调影响、调参建议）。

---

## 5. 成本函数定义（实现细节）

> 所有量均由 `features_cache` 与切点元信息提供，**不做频域/模型的二次计算**。

### 5.1 合并成本 `merge_cost(i→neighbor)`

跨越 `S_i` 与相邻 `S_j` 的切点 `c(i,j)`：

* `pause_strength = 1 - normalize(score_c) ∈ [0,1]`（也可以直接用 `1-score` 或经过温度拉伸）；
* `kind_mismatch_penalty = 1`（同类=0，跨类=1）；
* `energy_jump_db_pos = max(0, rms_db(j.start±ε) - rms_db(i.end±ε))`；
* `over_max_penalty_after_merge = max(0, (dur_i+dur_j - soft_max_s) / soft_max_s)`。

综合：

```
merge_cost = w_pause*pause_strength + w_kind*kind_mismatch_penalty
           + w_energy*energy_jump_db_pos + w_len*over_max_penalty_after_merge
```

阈值：`pause_strength < 0.35` 且 `energy_jump_db_pos < 6 dB` 时**允许**合并；否则拒绝（除非 `micro_merge` 强制）。

### 5.2 救援切点成本 `split_cost(t|S_long)`

* `pausedness`：在 t±W（W=150–200ms）窗口的静音/能量谷程度（越像停顿越大）；
* `pos_penalty = |t - mid(S)| / dur(S)`；
* `kind_transition_penalty`：切点两侧 kind 不一致时惩罚（倾向在同类内部切）。
* `local_energy_db`：t 附近的 RMS dB（越低越好）。

综合：

```
split_cost = w_pause*(1 - pausedness) + w_pos*pos_penalty
           + w_kind*kind_transition_penalty + w_energy*local_energy_db
```

---

## 6. 关键函数伪代码（可直接实现）

```python
def merge_into_better_neighbor(segs, i, features, cfg, force=False) -> int | None:
    left_ok  = (i-1) >= 0
    right_ok = (i+1) < len(segs)
    if not left_ok and not right_ok: return None

    def cost(to_left: bool) -> float:
        j = i-1 if to_left else i+1
        c = boundary_cutpoint(segs[i], segs[j])     # 含 score/kind/t
        energy_jump = max(0.0, rms_db_at(segs[j].start, features) - rms_db_at(segs[i].end, features)) \
                      if to_left else \
                      max(0.0, rms_db_at(segs[i].start, features) - rms_db_at(segs[j].end, features))
        overmax = max(0.0, (segs[i].dur + segs[j].dur - cfg.soft_max_s) / cfg.soft_max_s)
        kind_pen = 1.0 if segs[i].kind != segs[j].kind else 0.0
        pause_strength = 1.0 - c.score
        W = cfg.weights
        return W['pause']*pause_strength + W['kind']*kind_pen + W['energy']*energy_jump + W['overmax']*overmax

    choose_left = left_ok and (not right_ok or cost(True) <= cost(False))
    chosen_cost = cost(True) if choose_left else cost(False)
    if (not force) and not is_merge_allowed(chosen_cost, segs[i], cfg, features):
        return None

    # 执行合并
    j = i-1 if choose_left else i+1
    new_seg = Segment(min(segs[i].start, segs[j].start), max(segs[i].end, segs[j].end),
                      kind=majority_kind(segs[i], segs[j]))
    L, R = (j, i) if choose_left else (i, j)
    segs[L:R+1] = [new_seg]
    return L  # 返回新段索引

def pick_rescue_cut(seg, suppressed_cuts, features, cfg) -> float | None:
    candidates = [t for (t, s) in suppressed_cuts if seg.start < t < seg.end]
    if not candidates:
        candidates = find_energy_valleys(seg, features, win_ms=180.0)
    # 只考虑中部 ±25% 区间
    mid = 0.5*(seg.start+seg.end); span = 0.25*seg.dur
    candidates = [t for t in candidates if (mid-span) <= t <= (mid+span)]
    if not candidates: return None
    # 计算 split_cost 选最优
    best_t, best_cost = None, 1e9
    for t in candidates:
        cost = split_cost(t, seg, features, cfg)
        if cost < best_cost: best_t, best_cost = t, cost
    return best_t

def enforce_min_gap(segs, min_gap_s):
    # 如果两个段间隔 < min_gap_s，则以“向能量更低的一侧”微调边界或合并其中一个
    ...
    return segs
```

---

## 7. 指标与基准（脚本改动）

* [ ] `scripts/bench/run_bench.py` 新增输出：

  * `fragment_rate_lt2s`、`rate_lt_softmin`、`dur_p50/p95/p99/max`
  * `strong_pause_cross_rate`（合并跨越 score≥0.6 的切点比例）
  * `beat_snap_median_ms`（启用吸附时）
  * `segment_layout_overhead_ms`、`runtime_overhead_pct`
* [ ] 报表含 before/after 对比，默认写入 `output/bench/segment_layout_report.json/md`.

---

## 8. 测试（单元 + 合约）

**单元**

* [ ] `test_micro_merge`: 构造一短段位于弱停顿，验证必合并到较低代价侧。
* [ ] `test_softmin_heap_merge`: 多个连串小段，验证堆驱动级联合并正确、稳定。
* [ ] `test_softmax_rescue`: 段内存在多个候选，验证救援切点选择最优且满足 `min_gap_s`。
* [ ] `test_min_gap_enforce`: 守卫右推后贴边情况得到修复。
* [ ] `test_beat_snap`: 启用吸附时，切点偏差不超过窗口。

**合约**

* [ ] `test_no_quality_regression`: 合并/再切后，**强停顿跨越率 ≤ 5%**；**可逆性=0**。
* [ ] `test_performance_budget`: Overhead < 3%。

---

## 9. 风险与回滚

* **过度合并**：成本阈值过宽 → 增大 `w_pause`、收紧 `energy_jump_db_pos` 上限或提高 `soft_min_s` 的保守门限。
* **新生碎片**：救援切点靠近原边界 → 增大 `min_gap_s` 或调整 `pos` 权重。
* **性能抖动**：堆操作或日志过多 → 限制打印、仅在 DEBUG 输出。
* **回滚**：`segment_layout.enable=false` 或 CLI `--segment-layout off`；保留旧切点直接导出。

---

## 10. 调参指南（速记）

* **碎片还多**：`micro_merge_s ↑`、`soft_min_s ↑`；或降低 `w_pause`（允许跨弱停顿）。
* **听感变硬**：`w_pause ↑`、`energy_jump_db_pos 上限 ↓`；`beat_snap_ms` 设为 0（暂关吸附）。
* **段太长**：`soft_max_s ↓` 或提升 `beats_max`（若 BPM 可信）。
* **人声被合并进伴奏**：`w_kind ↑` + `human_offset_s ↑`。

---

## 11. 提交清单（合并前必须完成）

* [ ] 新模块与配置、CLI 开关就绪；
* [ ] `refine.finalize_cut_points` 返回 `suppressed_cuts`；
* [ ] 单元/合约测试全部通过；
* [ ] 基准报告：碎片率=0、性能预算达标；
* [ ] 文档：`docs/tuning_segment_layout.md` 与变更日志更新。

---

**收尾**
这份 TODO 走完，1–2s 小屑片段会被消灭，段时长落在 6–18s（或 BPM 自适应区间）为主，且不会破坏真正的停顿点。运行开销微小、听感稳定，符合你“保留 Silero + GPU 主路径”的总体策略。

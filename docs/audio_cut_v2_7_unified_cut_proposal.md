<!-- File: docs/audio_cut_v2_7_unified_cut_proposal.md -->
<!-- AI-SUMMARY: v2.7 统一切点引擎方案建议：修复 hybrid_mdd 切人声问题、参数智能化（AutoProfile）、自然切点融合（长停顿/气口/节拍/ASR 歌词加权）。 -->

# audio-cut v2.7 方案建议：统一切点引擎

> 目标读者：项目维护者 / coding agent
> 基线版本：v2.6-dev（commit `8271984`，VPBD + FireRedASR soft-prior 已落地，人工验收未完成）
> 文档日期：2026-06-09
> 性质：**方案建议**，不是实施记录。所有结论均附代码证据（文件:行号）。

---

## 0. 核心判断

```text
【核心判断】
✅ 值得做，但不是"再加一个模式"，而是"收敛成一个引擎"。

【关键洞察】
- 数据结构：v2.6 已经把数据结构做对了——CutCandidate（统一候选）+
  PhraseBoundaryScorer（统一打分）+ GlobalCutPlanner（全局规划）。
  这是正确的抽象。错的是 hybrid_mdd 完全没走这条链，而 vpbd_asr
  只让"声学候选"进了候选池，气口、ASR 句尾、节拍全被挡在门外。
- 复杂度：项目里同时存在 4 套参数自适应机制、至少 6 个死配置项、
  3 个互相矛盾的"最大片段时长"定义。参数调不出来不是因为参数不够多，
  是因为参数太多且互相打架。
- 风险点：hybrid_mdd 的 vad_protection 是个谎言——配置写着 true，
  代码里算了 VAD 区域却从来不用，副歌区注释明说"禁用 VAD 保护"。
  这就是你"切在人声上"的直接原因。

【Linus 式结论】
"Bad programmers worry about the code. Good programmers worry about
data structures." —— CutCandidate 候选池就是正确的数据结构。
把所有证据（长停顿、气口、节拍、MDD 谷、ASR 词间隙/句尾/mVAD）都变成
候选，把所有偏好（卡点感、风格、BPM）都变成打分权重，让 DP 规划器
全局选路，最后统一过静音守卫。特殊情况消失，三个问题一起解决。
```

---

## 1. 现状盘点（截至 2026-06-09）

### 1.1 版本与进展

| 项 | 状态 |
|---|---|
| 已发布 | v2.5.1（hybrid_mdd 三策略 + Strategy 模式重构） |
| 开发中 | v2.6 VPBD + FireRedASR2S soft-prior（commit `8271984`） |
| v2.6 完成度 | TODO 清单 A–K、M 全部勾完（97 个快速测试通过，真实 FireRed CLI smoke 通过）；仅剩 **L 节人工验收**（20 首曲目 playlist、boundary F1、主观自然度评分）未执行 |
| 验收基建 | `scripts/vpbd_asr_acceptance.py` + 评分表 + playlist 模板已就绪，本地 5 首实测 `cut_inside_word_rate=0.0`、`segment_5_15_pass_rate=0.96` |

### 1.2 五种模式的真实关系

```text
                     ┌─ v2.2_mdd ──────────► PureVocalPauseDetector → finalize_cut_points(守卫) → refine_layout
                     │
SeamlessSplitter ────┼─ hybrid_mdd ────────► 内嵌跑一遍 v2.2_mdd 拿 MDD 切点
                     │                        → SnapToBeat/BeatOnly 策略改写切点
                     │                        → 微碎片合并 → 直接导出        ← 改写后【不再过守卫】
                     │
                     ├─ librosa_onset ─────► 纯节拍/能量分割
                     │
                     └─ vpbd_acoustic /    ► VocalPhraseBoundaryDetector
                        vpbd_asr             （候选→打分→DP 规划）
                                             → finalize_cut_points(守卫)
                                             → 词保护恢复 → refine_layout(带 ASR 边界)
                                             → 局部谷值精调(带词保护)        ← 这条链是对的
```

关键事实：**"Hybrid MDD + 节拍卡点" 和 "FireRedASR" 目前是两条互不相通的路**。
hybrid_mdd 有卡点感但没有 ASR、没有守卫闭环；vpbd_asr 有 ASR 词保护和全局规划，
但没有节拍卡点行为（节拍只是 0.10 权重的加分项）。你被迫二选一，这就是
"效果不理想"的结构性原因。

---

## 2. 三个问题的根因分析

### 2.1 问题一：切在人声部分

逐条代码证据（按严重程度排序）：

| # | 根因 | 证据位置 |
|---|------|---------|
| R1 | **副歌区 VAD 保护被故意禁用**。注释原文："In chorus, we want beat-aligned cuts even if cutting through vocals. VAD protection is disabled in high-energy regions" | `snap_to_beat_strategy.py:140-145` |
| R2 | **`vad_protection` 是死代码**。`_compute_vad_active_regions()` 算出了 VAD 区域（109-114 行），但 `_would_cut_active_vocal()`（310 行）从未被调用；`snap_stats['vad_blocked']` 永远是 0 | `snap_to_beat_strategy.py:109-155, 310-318` |
| R3 | **即使 VAD 被调用，它分析的也是混音不是人声**。策略上下文传入的是 `original_audio`，而分离好的 `vocal_track` 就在同一个函数作用域里没人用 | `seamless_splitter.py:1218`（`audio=original_audio`）vs `1177` |
| R4 | **吸附后不再过静音守卫**。MDD 基础切点在内嵌 v2.2_mdd 里被 `finalize_cut_points` 守卫过，但吸附把切点移走了（最多 500ms），移动后直接进微碎片合并→导出，守卫成果作废 | `seamless_splitter.py:1248-1324`（无 finalize 调用） |
| R5 | **`snap_tolerance_ms: 500` 过大**。120 BPM 时节拍间隔正好 500ms——容差=节拍间隔意味着副歌区**每个** MDD 切点都必然吸附，等价于 beat_only。一个中文字的演唱时长约 200–500ms，移 500ms 必然横穿字 | `config/unified.yaml:332` vs 代码默认 300（`snap_to_beat_strategy.py:59`） |
| R6 | **high 密度模式直接添加小节起始拍作为切点，无任何人声检查** | `snap_to_beat_strategy.py:160-179` |

结论：这不是调参问题，是 hybrid_mdd 的设计把"卡点感"和"不切人声"当成了
互斥项，并选择了前者。注释写得很诚实，配置项却在撒谎。

### 2.2 问题二：参数冗杂、调不出最优

**根因 A：四套自适应机制并存，互相叠乘**

| 机制 | 位置 | 实际状态 |
|------|------|---------|
| ① `relative_threshold_adaptation`（BPM 乘数 + MDD 增益） | `unified.yaml:125-135`，`pure_vocal_pause_detector` 内生效 | **活跃** |
| ② `pause_stats_adaptation`（VPP 停顿统计，自带一套 slow/fast 分类阈值和乘数） | `unified.yaml:137-151` | **活跃**，与 ① 叠乘 |
| ③ `AdaptiveParameterCalculator`（`bpm_adaptive_core` 节拍参数，70/100/140 BPM 分档） | `adaptive_parameter_calculator.py` | **孤儿**：只被旧版 `vocal_pause_detector.py` 和 `quality_controller.py` 引用，主流程（v2.2_mdd/vpbd）不经过它，但配置段还占着 `unified.yaml:492-513` |
| ④ Schema v3 + `derive.py` + Profile（ballad/pop/edm/rap，90/140 BPM 分档） | `audio_cut/config/derive.py`、`profiles/*.yaml` | 已实现，**但需要用户手动 `--profile`**，不会自动判断 |

③ 和 ④ 各有一套 BPM 分类边界（70/100/140 vs 90/140），同一首 95 BPM 的歌在
两套体系里属于不同档位。一首歌的最终阈值 = 基础值 × ①BPM乘数 × ①MDD乘数 ×
②VPP乘数 × ④profile覆盖，clamp 两次。**你调任何一个旋钮，都被另外三层
稀释或反向修正——这就是"始终调不出理想参数"的数学原因。**

**根因 B：死配置项让人调了个寂寞**（已逐一验证零引用 / 零效果）

| 配置项 | 为什么是死的 |
|--------|------------|
| `hybrid_mdd.vad_protection` | 见 2.1 R2 |
| `phrase_boundary.min_score: 0.35` | 全 src/ 零引用（grep 验证） |
| `phrase_boundary.weights.mdd_affinity: 0.10` | `BoundaryFeatureExtractor` 构造时从不传 `mdd_times`（`vocal_phrase_boundary_detector.py:202`），该特征恒为 0 |
| `global_planner` 的 `vocal_risk_weight` / `beat_conflict_weight` | 规划器读 `features["vocal_cut_risk"]` / `["beat_conflict"]`（`global_cut_planner.py:167-168`），但 `BoundaryFeatures` 根本没有这两个字段（`boundary_features.py:14-39`），恒为 0 |
| `bpm_adaptive_core.*` 全节 | 见根因 A ③ |
| `vocal_pause_splitting.*`、`vocal_separation.*`（HPSS 旧参） | 旧版兼容段，主流程不读大部分字段 |

**根因 C：三个互相矛盾的"片段时长"真相**

| 配置 | 值 | 谁在用 |
|------|----|----|
| `quality_control.segment_max_duration` | 8.0s | v2.2_mdd 布局回退 |
| `segment_layout.soft_max_s` | 15.0s | 布局精炼救援阈值 |
| `global_planner.hard_max_s / target_max_s` | 18.0 / 12.0s | VPBD DP 规划 |

同一个意图（"片段别太长"）三处定义、三个数值。用户改了一处，另两处把行为拉回去。

### 2.3 问题三：切点不够自然流畅

用户期望的优先级是：**人声长停顿 > 气口 > 节拍/MDD，再用 ASR 歌词加权**。
当前实现离这个目标差在哪：

| # | 缺陷 | 证据 |
|---|------|------|
| N1 | **气口被整体丢弃**。`_classify_and_filter()` 把 `pause_type='breath'` 的停顿直接过滤，永远成不了候选。密集唱段里没有长停顿时，本该退而求其次切气口，现在只能硬切或交给布局救援的 RMS 谷搜索 | `pure_vocal_pause_detector.py:844-848` |
| N2 | **ASR 候选生成了又扔了**。`LyricsBoundaryCandidateGenerator` 产出 word-gap / 句尾 / mVAD 边界候选（`lyrics/candidates.py`），但 `vocal_phrase_boundary_detector.py:119-123` 只把**声学候选**送进打分和规划，`lyrics_candidates` 仅进了 metadata 计数。声学漏检、ASR 看得见的边界（例如伴奏持续轰鸣下的句尾）永远选不上 |
| N3 | **节拍不是候选**，只是 0.10 权重、±120ms 容差的加分项（`boundary_features.py:105-113`）。想要"卡点感"时引擎给不出节拍切点，逼用户退回 hybrid_mdd |
| N4 | **特征容差对 ASR 时间戳抖动过敏**。`sentence_tolerance_s=0.08`：FireRed 词级时间戳常见 ±100–200ms 抖动，真句尾 100ms 偏差就得 0 分；同时一个落在词尾内 20ms 的候选拿满额 `inside_word_penalty=1.0`——悬崖式打分 | `boundary_features.py:49-52, 71-77, 94-103` |
| N5 | 布局救援的二次切分虽已实现"优先声学谷 + ASR 边界加权"（PLAN.md 落地），但因为 N1/N2，候选池本身贫血，救援常常无谷可用 | `PLAN.md`、`segment_layout_refiner.py` |

---

## 3. 方案总览：一个引擎，四个阶段

设计原则（按优先级）：

1. **Never break userspace**：`separate_and_segment()` 签名不变；旧模式名继续可用；Manifest 旧字段不动。
2. **消灭特殊情况**：吸附、救援、气口、卡点全部表达为"候选 + 权重"，不再有后置改写切点的旁路。
3. **先修谎言，再做能力**：让每个配置项要么生效、要么删除。
4. **复用 v2.6 资产**：CutCandidate/Scorer/Planner/守卫链/验收脚本全部复用，不推倒重写。

```text
v2.7 目标架构（唯一切点链路，所有模式都是它的预设）：

原音频 → 分离(vocal/instrumental) → TrackFeatureCache(BPM/MDD/RMS/beat)
                                          │
                              ┌───────────▼────────────┐
                              │   AutoProfile 风格判定   │ ← 阶段三
                              │ (BPM/MDD/能量CV/人声覆盖) │
                              └───────────┬────────────┘
                                          ▼
        ┌─────────────────── 统一候选池 (CutCandidate) ────────────────────┐
        │ acoustic_pause(长停顿)  breath(气口,降权)   beat(高能量段,弱候选)  │ ← 阶段二
        │ mdd_valley             lyrics_gap/句尾     mVAD singing 边界     │
        └───────────────────────────────┬──────────────────────────────────┘
                                        ▼
                    PhraseBoundaryScorer（风格感知权重，修复死特征）
                                        ▼
                    GlobalCutPlanner（DP 全局选路，时长奖惩）
                                        ▼
                    finalize_cut_points（过零 + 静音守卫，人声轨）
                                        ▼
                    词保护恢复 → refine_layout → 局部谷值精调
                                        ▼
                    导出 + SegmentManifest(含片段歌词、切点 features)
```

---

## 4. 阶段零：今天就能做的止血（纯配置，零代码）

**给当前 hybrid_mdd 用户的临时配置**（治标，降低切字概率 ≈70%）：

```yaml
hybrid_mdd:
  beat_cut_density: low        # high 密度的小节硬切是无保护的，先关掉
  snap_tolerance_ms: 150       # 500 → 150：只吸附本来就贴着节拍的切点
  lib_alignment: snap_to_beat  # 别用 beat_only
```

**更推荐：直接切换到 vpbd_asr 模式**（v2.6 已交付，本地 5 首实测切字率 0%）：

```yaml
lyrics_alignment:
  enabled: true
  provider: auto               # sidecar → cli → 降级 vpbd_acoustic
fire_red:
  endpoint: http://127.0.0.1:8765
global_planner:
  target_min_s: 5.0
  target_max_s: 12.0           # 与 MV 镜头节奏对齐
```

```bash
python run_splitter.py input/song.mp3 --mode vpbd_asr \
  --gpu-device cuda:0 --lyrics-provider sidecar \
  --firered-endpoint http://127.0.0.1:8765
```

mvagent 侧请改读 `SegmentManifest.json` 的 `segments[*].lyrics`（v2.6 已写入），
不要再逐片段跑 ASR——整曲时间轴已经是一等公民。

---

## 5. 阶段一：修复 hybrid_mdd 的谎言（1–2 天，bugfix 性质）

> 定位：短期止血 + 兑现已有配置承诺。阶段二完成后此模式退化为引擎预设，
> 但修复本身独立有价值且立即可发版（v2.6.1）。

| 任务 | 改动点 | 验收 |
|------|--------|------|
| F1. 让 `vad_protection` 真的生效 | `SegmentationContext` 增加 `vocal_track` 字段（`seamless_splitter.py:1217` 处传入已有的 `vocal_track`）；吸附决策前检查目标节拍 ±80ms 窗口内**人声轨** RMS 是否低于地板+guard_db；不安全则在容差内找下一个安静节拍，找不到则保留 MDD 原点 | 新增单测：构造"节拍落在人声内"场景，断言不吸附；`snap_stats['vad_blocked'] > 0` |
| F2. 吸附后强制守卫 | `_process_hybrid_mdd_split` 在策略输出后调用与 v2.2_mdd 相同的 `finalize_cut_points`（人声轨守卫），lib 标记按最近邻映射回守卫后切点 | 复用 `test_cutting_consistency.py` 模式；拼接精度测试保持 ≤1e-12 |
| F3. high 密度小节切点过同一安全检查 | `snap_to_beat_strategy.py:160-179` 增加 F1 的安静检查 | 单测断言 high 密度下无切点落于人声活跃区 |
| F4. 副歌"宁卡点不护声"改为可配置 | 新增 `hybrid_mdd.chorus_force_snap: false`（默认关）。想要旧行为的用户显式打开——配置不再撒谎 | 配置契约测试 |
| F5. 默认 `snap_tolerance_ms` 500 → 200，并按 BPM 上限约束（≤ 0.4 × 节拍间隔） | `unified.yaml` + 策略内 clamp | 契约测试 |

兼容性声明：F1–F3 会改变 hybrid_mdd 输出——这是**把行为修正为配置宣称的行为**，
属于 bugfix；`chorus_force_snap: true` + `vad_protection: false` 可完整还原旧行为。

---

## 6. 阶段二：统一候选池（核心，1–2 周）

> 这是方案的心脏。完成后 vpbd_asr 同时具备：长停顿优先、气口兜底、
> 节拍卡点能力、ASR 词保护——hybrid_mdd 的存在理由消失。

### 6.1 候选源补全

| 任务 | 改动点 |
|------|--------|
| C1. **气口入池**。`PureVocalPauseDetector._classify_and_filter` 不再丢弃 breath，改为打 `pause_type='breath'` 标签随候选输出；`candidate_adapters` 将其映射为 `CandidateSource.BREATH`，基础分 = 置信度 × `breath_score_scale`（默认 0.6） | `pure_vocal_pause_detector.py:826-862`、`candidate_adapters.py` |
| C2. **ASR 候选入池**。`vocal_phrase_boundary_detector.detect()` 把 `lyrics_candidates` 与声学候选合并后一起进打分（修掉 119-123 行只送声学的问题）；±120ms 内近重复候选去重，保留最高分，`meta.sources` 记录融合来源 | `vocal_phrase_boundary_detector.py:111-125` |
| C3. **节拍弱候选入池**。高能量段（复用 `_detect_chorus_regions` 的多特征副歌判定，迁移到 `audio_cut/analysis/`）内，每 N 小节的小节起始拍生成 `CandidateSource.BEAT` 候选，基础分低（0.3）且强制带 `vocal_cut_risk` 特征 | 新增 `audio_cut/cutting/beat_candidates.py` |

### 6.2 修复死特征，打通安全闭环

| 任务 | 改动点 |
|------|--------|
| C4. **`vocal_cut_risk` 真实计算**：候选时刻 ±80ms 窗口的人声轨 RMS 相对全轨分位数归一化（数据就在 `TrackFeatureCache.rms` 里，O(1) 查询）。规划器的 `vocal_risk_weight` 从此生效——这是防切人声的**第二道闸**（第一道是 inside_word_penalty，第三道是静音守卫） | `boundary_features.py` 增字段；`vocal_phrase_boundary_detector.py` 传入 rms 序列 |
| C5. **`mdd_affinity` 复活**：从 MDD 序列提取谷值时刻传入 extractor | `vocal_phrase_boundary_detector.py:201-203` |
| C6. **特征容差抗抖动**：`sentence_tolerance_s` 0.08 → 0.25（线性衰减）；`inside_word_penalty` 增加词边缘软化——距词边界 < 60ms 时按距离比例衰减，消除悬崖 | `boundary_features.py:49-52, 71-77` |
| C7. 删除 `phrase_boundary.min_score`（零引用）或让规划器真用它做预过滤——二选一，不许保留谎言 | `unified.yaml:398` |

### 6.3 切点优先级 = 权重，不是 if/else

```yaml
# 阶段二后的 phrase_boundary.weights（natural 风格示例）
weights:
  acoustic_pause: 0.40   # 长停顿——最高优先
  asr_gap:        0.25   # ASR 词间隙
  sentence_end:   0.20   # 句尾（容差放宽后才敢提权）
  breath:         0.12   # 气口——新增，密集段兜底
  beat_affinity:  0.08   # natural 风格下节拍只是微弱偏好
  mdd_affinity:   0.10
  inside_word_penalty: 0.80   # 提高：词内禁切是底线
  singing_penalty:     0.50
```

DP 规划器天然会实现你要的行为：有长停顿用长停顿；没有就用"气口+句尾"
组合分；副歌密集段在 rhythmic 风格下节拍候选浮出水面。**没有任何一行
新增的 if 分支。**

### 6.4 验收

- 新增 `tests/unit/test_candidate_pool_fusion.py`：气口/ASR/节拍候选入池、去重、来源追踪。
- 跑通 v2.6 既有验收器：`cut_inside_word_rate ≤ 1%`、`segment_5_15_pass_rate ≥ 90%` 不回退。
- 新增指标进 QA report：`breath_cut_ratio`（气口切点占比）、`beat_aligned_ratio`（卡点占比）。

---

## 7. 阶段三：AutoProfile——参数跟着歌走（1 周）

> 解决"参数应该根据歌曲风格、BPM 智能判断"。不训练模型，纯规则 + 插值，
> 因为判定特征（BPM/MDD/能量CV/人声覆盖率）在检测开始前就已经在
> `TrackFeatureCache` 里算好了——**数据早就有，只是没人用它做决策**。

### 7.1 设计

```text
TrackFeatureCache ──► StyleEstimator（新增，~150 行）
  bpm                   │  规则打分 → style ∈ {ballad, pop, edm, rap} + 置信度
  global_mdd            │  连续插值：不取离散档位，对相邻锚点 profile 线性插值
  能量CV(副歌判定复用)    │  （消除 89 vs 91 BPM 的行为悬崖）
  人声覆盖率(Silero VAD)  ▼
                 SchemaV3Config（复用 derive.py 全套派生）
                        ▼
                 set_runtime_config(...)  + Manifest 写入 auto_profile 元数据
```

### 7.2 任务

| 任务 | 说明 |
|------|------|
| A1. 新增 `audio_cut/config/auto_profile.py`：`estimate_style(cache) -> (profile, confidence, features)`，规则版（BPM 区间 + MDD + 能量CV + 人声覆盖率加权投票） |
| A2. 锚点插值：profile 不再四选一，而是在 Schema v3 的 12 个参数空间内对两个最近锚点插值（如 95 BPM ⇒ 0.7×pop + 0.3×ballad） |
| A3. **统一 BPM 分类学**：删除 `AdaptiveParameterCalculator` 的 70/100/140 私有分档，全项目只认 `derive.py` 一套；`bpm_adaptive_core` 配置段标记 deprecated |
| A4. CLI：`--profile auto`（v2.7 默认），`--profile ballad` 等手动值仍然有效（兼容） |
| A5. 可观测性：Manifest 新增 optional `auto_profile: {style, confidence, bpm, mdd, applied_overrides}`——调不好时能看见引擎"想了什么" |
| A6. 风格联动打分权重：`phrase_boundary.weights` 按风格预设（rap/edm 提 beat_affinity 与 breath，ballad 提 acoustic_pause 与 sentence_end），与 A2 同机制插值 |

### 7.3 用户面参数收敛到 4 个

```yaml
# v2.7 用户只需要碰这一段（其余全部自动派生）
smart_cut:
  profile: auto                 # auto / ballad / pop / edm / rap
  cut_style: natural            # natural(MV叙事) / rhythmic(卡点) / dense(短视频)
  target_duration_s: [5, 12]    # 唯一的时长真相，派生 hard/soft/target 三层
  lyrics: auto                  # auto / off / strict
```

`target_duration_s` 成为单一事实源，`global_planner.hard_*`、`segment_layout.soft_*`、
`quality_control.segment_max_duration` 全部由它派生——三个矛盾时长定义合并为一个。

---

## 8. 阶段四：unified.yaml 瘦身（伴随阶段二/三完成）

| 处理 | 配置段 | 理由 |
|------|--------|------|
| **删除** | `bpm_adaptive_core.*`、`vocal_pause_splitting.bpm_adaptive_settings`、`phrase_boundary.min_score`（若不实现） | 死配置/孤儿（2.2 节证据） |
| **合并** | `pause_stats_adaptation` 的乘数体系并入 `relative_threshold_adaptation`，只保留一处 clamp | 两层叠乘不可调（2.2 根因 A） |
| **降级到 expert 层** | `valley_scoring`、`advanced_vad`、`ort`、`enforce_quiet_cut` 细参 | 用户不该碰；移到 `config/expert.yaml`，主文件只留 `smart_cut` + 路径/输出/GPU 三段，目标 **513 行 → ≤120 行** |
| **保留** | `VSS__` 环境变量覆盖、`set_runtime_config`、Schema v3 迁移工具 | userspace 契约 |
| **文档** | 每个保留参数注明"生效模式 + 生效代码位置" | 杜绝再次出现"调了个寂寞" |

迁移路径：v2.7 读旧配置时打 deprecation warning 并自动映射（扩展现有
`migrate_v2_to_v3.py`），v2.8 才移除——两个版本的缓冲期。

---

## 9. 里程碑与验收门

| 里程碑 | 内容 | 发版 | 验收门 |
|--------|------|------|--------|
| M1（1–2 天） | 阶段一 hybrid_mdd 修复 | v2.6.1 | 既有快速回归 97 项全绿；新增 VAD 保护单测；人工抽测 3 首副歌歌曲无切字 |
| M2（1–2 周） | 阶段二候选池统一 | v2.7.0-beta | `cut_inside_word_rate ≤ 1%`、`boundary_f1_500ms ≥ 0.82`、`segment_5_15_pass_rate ≥ 90%`（复用 v2.6 验收器与 20 首 playlist——**顺手把 v2.6 欠的 L 节人工验收一起做掉**） |
| M3（1 周） | 阶段三 AutoProfile | v2.7.0 | 20 首 playlist 上 auto 风格判定与人工标签一致率 ≥ 85%；auto 模式各项指标不劣于人工最优 profile 的 95% |
| M4（伴随） | 阶段四配置瘦身 | v2.7.0 | 配置契约测试覆盖新旧映射；`unified.yaml ≤ 120` 行 |
| 回归铁律 | 每个 M 合并前：`pytest -m "not slow and not gpu and not firered"` + 契约测试 + 拼接精度 ≤1e-12 + 旧模式三连跑（v2.2_mdd / hybrid_mdd / librosa_onset）输出命名与 Manifest 旧字段逐项 diff | | |

主观验收（MV 场景特有，自动指标测不出来）：每个 M2/M3 候选版本用
7 类曲目（playlist 已建好分类目录 `input/acceptance/`）各抽 1 首，
人工听切点：`subjective_naturalness ≥ 4.2/5`、`manual_recutter_rate`
较 v2.5.1 降 ≥ 40%（沿用 v2.6 既定指标）。

---

## 10. 风险与回滚

| 风险 | 缓解 |
|------|------|
| 阶段一改变 hybrid_mdd 输出引发下游不适 | `chorus_force_snap` + `vad_protection: false` 可逐项还原旧行为；v2.6.1 release notes 显式声明 |
| 气口候选导致碎片化 | 气口只在"无长停顿可用"时被 DP 选中（分数低 + 时长惩罚兜底）；`breath_score_scale` 可调，极端情况设 0 = 完全还原现状 |
| AutoProfile 误判风格 | 判定结果只是初始参数，置信度 < 0.6 时回退 pop 中性锚点；`--profile` 手动值永远优先；Manifest 留痕可追责 |
| FireRed 不可用 | 既有降级链完备（sidecar→cli→vpbd_acoustic），不动 |
| ASR 时间戳系统性偏移（个别歌曲混响重） | C6 的容差软化 + 守卫仍以声学为最终裁决（ASR 永远是 soft prior，v2.6 设计原则不变） |
| 整体不达标 | 每阶段独立 flag 控制：`vpbd.candidate_pool=legacy`、`smart_cut.profile=pop` 即回到 v2.6 行为；旧模式代码路径在 v2.8 前不删 |

---

## 11. 不做什么（Non-Goals）

- 不训练边界检测模型（规则融合未到天花板，先把死配置修了再谈 ML）。
- 不引入新的分离后端。
- 不做 GUI。
- 不在 v2.7 删除任何旧模式（hybrid_mdd 退化为引擎预设别名，行为由统一引擎提供）。
- 不把 FireRed 依赖并入主环境（sidecar 边界已被验证是对的）。

---

## 附录 A：本方案引用的关键代码证据索引

| 论断 | 位置 |
|------|------|
| 副歌禁用 VAD 保护（注释自述） | `src/vocal_smart_splitter/core/strategies/snap_to_beat_strategy.py:140-145` |
| VAD 算了不用 / vad_blocked 恒 0 | 同上 `:109-155`，`_would_cut_active_vocal :310` 零调用 |
| 策略拿到的是混音不是人声 | `src/vocal_smart_splitter/core/seamless_splitter.py:1218` |
| hybrid 吸附后无守卫 | `seamless_splitter.py:1248-1324` |
| ASR 候选生成后被丢 | `src/vocal_smart_splitter/core/vocal_phrase_boundary_detector.py:99,119-123` |
| 气口被硬过滤 | `src/vocal_smart_splitter/core/pure_vocal_pause_detector.py:844-848` |
| vocal_cut_risk / beat_conflict 恒 0 | `src/audio_cut/cutting/global_cut_planner.py:167-168` vs `src/audio_cut/analysis/boundary_features.py:14-39` |
| mdd_affinity 恒 0 | `vocal_phrase_boundary_detector.py:202`（未传 mdd_times） |
| min_score 零引用 | `grep -rn "min_score" src/` 无命中 |
| 孤儿自适应计算器 | `AdaptiveParameterCalculator` 仅被 `vocal_pause_detector.py:10,60` 与 `quality_controller.py:12,27` 引用 |
| 两套 BPM 分类学冲突 | `derive.py:297-301`（90/140） vs `adaptive_parameter_calculator.py:81-86`（70/100/140） |
| 三个时长定义矛盾 | `unified.yaml:198`（8s）/ `:240`（15s）/ `:411-413`（18/12s） |

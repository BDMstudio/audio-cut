<!-- File: docs/audio_cut_v2_7_unified_cut_todo.md -->
<!-- AI-SUMMARY: audio-cut v2.7 统一切点引擎的阶段化 TODO 清单（hybrid_mdd 修复 / 统一候选池 / AutoProfile / 配置瘦身）。 -->

# audio-cut v2.7 Unified Cut Engine TODO List

> 来源方案文档：`docs/audio_cut_v2_7_unified_cut_proposal.md`
> 基线版本：v2.6-dev（commit `8271984`）
> 勾选完成项时，在括号内补充验证证据：测试命令、日志路径、Manifest 字段、commit 或 PR。
> 铁律：每节合并前必须通过本节"验收命令"，且旧模式回归（H 节）不允许跳过。

---

## A. 阶段零：基线固定与止血配置

> 对应方案 §4。目标：固定 v2.6 行为基线，先用纯配置降低切人声概率，不动代码。

- [x] 运行并记录快速回归基线：`pytest -m "not slow and not gpu and not firered"`，记录通过数与耗时。（证据：`venv/bin/python -m pytest -s -m "not slow and not gpu and not firered"` -> 108 passed, 1 deselected in 11.54s）
- [x] 运行配置契约基线：`pytest tests/contracts/test_config_contracts.py`。（证据：`venv/bin/python -m pytest -s tests/contracts/test_config_contracts.py` -> 2 passed in 0.83s）
- [x] 运行拼接精度基线：`pytest tests/unit/test_cpu_baseline_perfect_reconstruction.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_cpu_baseline_perfect_reconstruction.py` -> 1 passed in 2.39s）
- [x] 选 2–3 首副歌密集的样本歌曲，跑 `--mode hybrid_mdd` 留存输出与 Manifest，作为 B 节修复前后的对照组。（证据：`output/v2_7_a4_before/sample_01_hybrid_mdd/SegmentManifest.json` -> 29 segments, max 15.16s；`sample_02_hybrid_mdd/SegmentManifest.json` -> 12 segments, max 21.10s；`sample_03_hybrid_mdd/SegmentManifest.json` -> 15 segments, max 54.53s）
- [ ] 应用止血配置并人工抽听对比：`hybrid_mdd.beat_cut_density: low`、`snap_tolerance_ms: 150`。（已生成匿名对比输出：`output/v2_7_a5_stopgap/sample_01_hybrid_mdd_stopgap`、`sample_02_hybrid_mdd_stopgap`、`sample_03_hybrid_mdd_stopgap`；待人工抽听。自动汇总显示 stopgap 对超长段改善有限：max 15.16s / 20.95s / 54.53s）
- [x] 验证 vpbd_asr 替代路径可用：`python run_splitter.py <样本> --mode vpbd_asr --lyrics-provider sidecar --firered-endpoint http://127.0.0.1:8765`，确认 Manifest 含 `segments[*].lyrics`。（证据：sidecar health `curl -fsS http://127.0.0.1:8765/health` -> connection refused；改用 FireRed CLI strict 路径验证，`output/v2_7_a6_vpbd_asr_cli/sample_02_vpbd_asr_cli/SegmentManifest.json` -> provider `firered_cli`, fallback_reason null, 12 segments, 9 segments with lyrics）
- [ ] 通知 mvagent 侧改读 `SegmentManifest.json` 的 `segments[*].lyrics`，停止逐片段重复 ASR。
- [x] 新建开发分支：`codex/v2.7-unified-cut` 或用户指定分支。（证据：`git switch -c codex/v2.7-unified-cut` 成功，当前分支 `codex/v2.7-unified-cut`）

当前缺口：A4 before 已用匿名输出目录留存；A5 与 B 节对照组仍需人工抽听；A6 的 sidecar 未启动，但 CLI strict 替代路径已验证。

验收命令：

```bash
pytest -m "not slow and not gpu and not firered"
pytest tests/contracts/test_config_contracts.py
pytest tests/unit/test_cpu_baseline_perfect_reconstruction.py
```

---

## B. 阶段一：hybrid_mdd 修复（M1 → v2.6.1）

> 对应方案 §5（F1–F5）。性质：bugfix——让配置宣称的行为成为真实行为。

### B1. F1 让 `vad_protection` 真的生效

- [x] `strategies/base.py`：`SegmentationContext` 新增 `vocal_track: Optional[np.ndarray]` 字段（默认 None，旧调用不破坏）。（证据：`tests/unit/test_snap_to_beat_vad_guard.py` -> 6 passed）
- [x] `seamless_splitter.py:_process_hybrid_mdd_split`：构造 context 时传入已有的 `vocal_track`。（证据：同上；after Manifest guard_count 非 0）
- [x] `snap_to_beat_strategy.py`：新增 `_is_quiet_at(t)` 安静检查——目标时刻 ±80ms 窗口内**人声轨** RMS 低于地板 + guard_db 才允许落点。（证据：`test_snap_to_beat_keeps_mdd_cut_when_nearest_beat_has_active_vocal`）
- [x] 吸附决策接入安静检查：目标节拍不安静 → 在容差内找下一个安静节拍 → 找不到则保留 MDD 原点；`snap_stats['vad_blocked']` 真实计数。（证据：新增单测断言 `vad_blocked == 1`）
- [x] 删除或接通死代码 `_would_cut_active_vocal` / `_compute_vad_active_regions`（二选一，不留僵尸）。（证据：旧 helper 已移除，策略共用 `is_quiet_vocal_window`）
- [x] 新增 `tests/unit/test_snap_to_beat_vad_guard.py`：构造"节拍落在人声活跃区"场景，断言不吸附且 `vad_blocked > 0`。（证据：`venv/bin/python -m pytest -s tests/unit/test_snap_to_beat_vad_guard.py -q` -> 6 passed）

### B2. F2 吸附后强制守卫

- [x] `_process_hybrid_mdd_split`：策略输出后调用与 v2.2_mdd 相同的 `finalize_cut_points`（人声轨守卫，复用 `seamless_splitter.py:1575-1604` 的配置读取）。（证据：after 输出 `output/v2_7_b_after_fix/sample_01_hybrid_mdd/SegmentManifest.json` guard_count=28；sample_02 guard_count=11；sample_03 guard_count=14）
- [x] lib 标记按最近邻映射回守卫后的切点，映射逻辑带单测。（证据：`test_hybrid_lib_flags_remap_to_guarded_cut_boundaries`）
- [x] 确认守卫后拼接精度不回退：`pytest tests/unit/test_cpu_baseline_perfect_reconstruction.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_cpu_baseline_perfect_reconstruction.py -q` -> passed）

### B3. F3–F5 其余修复

- [x] F3：high 密度小节起始拍切点（`snap_to_beat_strategy.py:160-179`）过同一安静检查。（证据：`test_high_density_inserted_beat_cuts_respect_vocal_guard`）
- [x] F4：新增 `hybrid_mdd.chorus_force_snap`（默认 false）；true 时还原"副歌强吸附"旧行为，配置契约测试覆盖。（证据：`test_chorus_force_snap_restores_legacy_snap_even_when_vocal_is_active`；`tests/contracts/test_config_contracts.py` -> passed）
- [x] F5：`unified.yaml` 默认 `snap_tolerance_ms: 500 → 200`；策略内按 BPM clamp（≤ 0.4 × 节拍间隔）。（证据：`test_snap_tolerance_is_clamped_to_fraction_of_beat_interval`）
- [x] `beat_only_strategy.py` 同步接入 B1 的安静检查（该策略文档已承认"可能切断人声"，同样要兑现保护）。（证据：`test_beat_only_bar_cuts_respect_vocal_guard`）
- [ ] 对照组复测：A 节留存的 2–3 首样本重跑，人工确认副歌切字消失、卡点感仍在。（已生成自动 after 输出：`output/v2_7_b_after_fix/sample_01_hybrid_mdd`、`sample_02_hybrid_mdd`、`sample_03_hybrid_mdd`；待人工抽听。自动指标显示 guard 已生效，但 sample_03 仍存在超长段，需后续 v2.7 候选池/布局阶段继续处理。）
- [x] 更新 `README.md` / `development.md` 中 hybrid_mdd 行为描述与新配置项。（证据：README/development 已描述 `vad_protection`、`chorus_force_snap`、200ms clamp 与 guard 链）
- [x] release notes：`docs/release_notes_v2_6_1.md`，显式声明行为变化与还原开关。（证据：`docs/release_notes_v2_6_1.md`）

验收命令：

```bash
pytest tests/unit/test_snap_to_beat_vad_guard.py
pytest tests/unit/test_cutting_consistency.py
pytest tests/unit/test_legacy_mode_regression.py
pytest -m "not slow and not gpu and not firered"
python run_splitter.py input/<sample>.mp3 --mode hybrid_mdd   # 人工抽听
```

---

## C. 阶段二：统一候选池——候选源补全（M2 之一）

> 对应方案 §6.1（C1–C3）。目标：气口、ASR、节拍全部成为一等候选。

### C1. 气口入池

- [ ] `cut_candidate.py`：`CandidateSource` 新增 `BREATH`。
- [ ] `pure_vocal_pause_detector.py:_classify_and_filter`：breath 不再丢弃，保留 `pause_type='breath'` 标签随结果输出（true_pause 行为不变）。
- [ ] `candidate_adapters.py`：breath 停顿映射为 `CandidateSource.BREATH`，基础分 = 置信度 × `vpbd.breath_score_scale`（新增配置，默认 0.6；设 0 时等价旧行为）。
- [ ] 确认旧模式（v2.2_mdd/hybrid_mdd）路径仍按旧逻辑过滤 breath——只有 VPBD 候选池消费 breath，旧输出零变化。
- [ ] 新增 `tests/unit/test_breath_candidates.py`：密集唱段无长停顿时，规划器选中气口候选；`breath_score_scale=0` 时不选。

### C2. ASR 候选入池

- [ ] `vocal_phrase_boundary_detector.py:detect`：`lyrics_candidates` 与声学候选合并后一起进 `_score_candidates`（修掉 119-123 行只送声学的问题）。
- [ ] 近重复去重：±120ms 内保留最高分候选，`meta.sources` 记录全部融合来源。
- [ ] `boundary_detection.candidate_counts` 维持旧字段语义，新增 `merged` 计数（只增不改）。
- [ ] 新增 `tests/unit/test_candidate_pool_fusion.py`：声学漏检 + ASR 句尾存在时，句尾候选可被选中；去重与来源追踪断言。

### C3. 节拍弱候选入池

- [ ] 将 `_detect_chorus_regions` 多特征副歌判定从 `snap_to_beat_strategy.py` 迁移到 `audio_cut/analysis/chorus_regions.py`（策略侧改为引用，行为等价单测保护）。
- [ ] 新增 `audio_cut/cutting/beat_candidates.py`：高能量段内每 N 小节的小节起始拍生成 `CandidateSource.BEAT` 候选，基础分 0.3，强制携带 `vocal_cut_risk` 特征。
- [ ] 新增配置 `vpbd.beat_candidates: {enable, bars_per_cut, base_score}` 并进配置契约测试。
- [ ] 新增 `tests/unit/test_beat_candidates.py`：仅高能量段产出；落在人声活跃区的节拍候选 `vocal_cut_risk` 高。

验收命令：

```bash
pytest tests/unit/test_breath_candidates.py
pytest tests/unit/test_candidate_pool_fusion.py
pytest tests/unit/test_beat_candidates.py
pytest tests/contracts/test_config_contracts.py
```

---

## D. 阶段二：死特征修复与打分闭环（M2 之二）

> 对应方案 §6.2（C4–C7）。原则：每个配置项要么生效、要么删除。

- [ ] C4：`BoundaryFeatures` 新增 `vocal_cut_risk` 字段；提取器用 `TrackFeatureCache.rms` 计算候选时刻 ±80ms 人声能量分位数归一化值；`global_planner.vocal_risk_weight` 从此真实生效。
- [ ] C5：从 MDD 序列提取谷值时刻，`vocal_phrase_boundary_detector.py:202` 传入 `mdd_times`，`mdd_affinity` 复活。
- [ ] C6a：`sentence_tolerance_s` 0.08 → 0.25，命中分按距离线性衰减。
- [ ] C6b：`inside_word_penalty` 词边缘软化——距词边界 < 60ms 按距离比例衰减，消除悬崖；新增 `word_edge_tolerance_ms` 配置。
- [ ] C7：`phrase_boundary.min_score` 二选一——在规划器预过滤中实装，或从 `unified.yaml` 删除；契约测试同步。
- [ ] `beat_conflict` 特征实装（rhythmic 风格用：高能量段内远离节拍的候选记冲突分）或删除 `beat_conflict_weight` 配置——同样二选一。
- [ ] 新增 `tests/unit/test_boundary_features_tolerance.py`：时间戳抖动 ±150ms 场景下句尾仍得分；词尾 20ms 处惩罚显著低于词中心。
- [ ] 扩展 `tests/unit/test_global_cut_planner.py`：`vocal_cut_risk` 高的候选在同分情况下被避开。

验收命令：

```bash
pytest tests/unit/test_boundary_features.py
pytest tests/unit/test_boundary_features_tolerance.py
pytest tests/unit/test_phrase_boundary_scorer.py
pytest tests/unit/test_global_cut_planner.py
```

---

## E. 阶段二：权重体系与 M2 验收（M2 之三 → v2.7.0-beta）

> 对应方案 §6.3–6.4。切点优先级 = 权重，不是 if/else。

- [ ] `unified.yaml` 更新 `phrase_boundary.weights`：新增 `breath`，`inside_word_penalty` 0.60 → 0.80，按方案 §6.3 natural 风格示例配齐。
- [ ] 权重和校验：正向权重合计与惩罚上限写入配置契约测试（防止后续改出 >1 的失衡）。
- [ ] QA report 新增指标：`breath_cut_ratio`、`beat_aligned_ratio`（扩展 `audio_cut/qa_report.py` 与 `tests/unit/test_qa_report.py`）。
- [ ] 候选调试 JSON 确认包含新 source 与新特征（`vpbd.candidate_debug_json: true` 路径）。
- [ ] 新增回退开关 `vpbd.candidate_pool: unified | legacy`（默认 unified；legacy = 仅声学候选，行为等价 v2.6）。
- [ ] 集成测试：`tests/integration/test_pipeline_vpbd_asr_fake_provider.py` 扩展——fake timeline 含句尾/气口场景，断言选点优先级"长停顿 > 气口+句尾 > 节拍"。
- [ ] 用 v2.6 验收器跑 M2 门槛：`cut_inside_word_rate ≤ 1%`、`segment_5_15_pass_rate ≥ 90%`、`boundary_f1_500ms ≥ 0.82` 不回退。

验收命令：

```bash
pytest tests/integration/test_pipeline_vpbd_asr_fake_provider.py
pytest tests/unit/test_qa_report.py
pytest -m "not slow and not gpu and not firered"
venv/bin/python scripts/vpbd_asr_acceptance.py --playlist docs/vpbd_asr_acceptance_playlist.local.json --output-dir output/v2_7_m2_acceptance
```

---

## F. 阶段三：AutoProfile（M3 → v2.7.0）

> 对应方案 §7（A1–A6）。数据已在 TrackFeatureCache，缺的只是决策模块。

- [ ] A1：新增 `audio_cut/config/auto_profile.py`：`estimate_style(cache) -> StyleEstimate(profile, confidence, features)`，规则版（BPM 区间 + global_mdd + 能量 CV + 人声覆盖率加权投票）。
- [ ] A2：锚点插值——在 Schema v3 的 12 参数空间对两个最近锚点 profile 线性插值（如 95 BPM ⇒ 0.7×pop + 0.3×ballad），消除档位边界悬崖。
- [ ] A3：统一 BPM 分类学——删除 `AdaptiveParameterCalculator` 的 70/100/140 私有分档逻辑依赖，全项目只认 `derive.py` 一套；`bpm_adaptive_core` 配置段标记 deprecated（读到时 warning）。
- [ ] A4：CLI 支持 `--profile auto`（v2.7 默认）；`--profile ballad` 等手动值优先级高于 auto；`quick_start.py` 菜单同步。
- [ ] A5：Manifest 新增 optional `auto_profile: {style, confidence, bpm, mdd, applied_overrides}`；schema/契约测试覆盖 optional 行为。
- [ ] A6：`phrase_boundary.weights` 按风格预设联动（rap/edm 提 beat_affinity 与 breath，ballad 提 acoustic_pause 与 sentence_end），与 A2 同机制插值。
- [ ] 置信度兜底：confidence < 0.6 时回退 pop 中性锚点并在 Manifest 留痕。
- [ ] 新增用户面配置节 `smart_cut: {profile, cut_style, target_duration_s, lyrics}`；`target_duration_s` 成为单一时长真相，派生 `global_planner.hard_*/target_*`、`segment_layout.soft_*`、`quality_control.segment_max_duration`。
- [ ] 新增 `tests/unit/test_auto_profile.py`：四类典型特征向量判定正确；插值单调；低置信回退。
- [ ] 新增 `tests/unit/test_smart_cut_duration_derivation.py`：三处时长配置由 `target_duration_s` 一致派生，手改子项时 warning。
- [ ] M3 验收：20 首 playlist 上 auto 判定与人工风格标签一致率 ≥ 85%；auto 模式各项指标 ≥ 人工最优 profile 的 95%。

验收命令：

```bash
pytest tests/unit/test_auto_profile.py
pytest tests/unit/test_smart_cut_duration_derivation.py
pytest tests/contracts/test_config_contracts.py
python run_splitter.py input/<sample>.mp3 --mode vpbd_asr --profile auto
```

---

## G. 阶段四：unified.yaml 瘦身与迁移（M4，伴随 E/F 合并）

> 对应方案 §8。目标：513 行 → ≤120 行，删谎言、并叠乘、降细参。

- [ ] 删除死配置：`bpm_adaptive_core.*`、`vocal_pause_splitting.bpm_adaptive_settings`、（若 D 节选择删除）`phrase_boundary.min_score`、`global_planner.beat_conflict_weight`。
- [ ] 合并叠乘：`pause_stats_adaptation` 乘数体系并入 `relative_threshold_adaptation`，全链路只保留一处 clamp；合并前后阈值等价性单测。
- [ ] 细参降级：`valley_scoring`、`advanced_vad`、`gpu_pipeline.ort`、`enforce_quiet_cut` 细项迁入 `config/expert.yaml`（缺省自动加载，主文件不再展示）。
- [ ] 主文件目标结构：`smart_cut` + `audio/output/logging` + `gpu_pipeline` 基础三项 + `lyrics_alignment/fire_red`，行数 ≤120。
- [ ] 扩展 `migrate_v2_to_v3.py`：旧配置键自动映射 + deprecation warning（v2.8 才移除，两版本缓冲）。
- [ ] 每个保留参数注释标注"生效模式 + 生效代码位置"，杜绝再次出现死配置。
- [ ] `VSS__` 环境变量覆盖与 `set_runtime_config` 行为不变（契约测试守住）。
- [ ] 同步更新 `README.md`、`development.md`、`audio-cut封装为模块.md` 的配置说明。

验收命令：

```bash
pytest tests/contracts/test_config_contracts.py
pytest tests/unit/test_config_migration.py   # 如无则新增
wc -l config/unified.yaml                    # ≤ 120
```

---

## H. 回归与发布门（每个 M 合并前强制）

- [ ] 快速回归：`pytest -m "not slow and not gpu and not firered" --cov=src --cov-report=term-missing` 全绿。
- [ ] 配置契约：`pytest tests/contracts/test_config_contracts.py`。
- [ ] 拼接精度：`pytest tests/unit/test_cpu_baseline_perfect_reconstruction.py`（≤1e-12）。
- [ ] 旧模式三连跑输出对比：`v2.2_mdd` / `hybrid_mdd` / `librosa_onset`，文件命名与 Manifest 旧字段逐项 diff（hybrid_mdd 在 B 节后允许的差异 = 切点位置，命名/字段结构零变化）。
- [ ] vpbd 回退开关验证：`vpbd.candidate_pool=legacy` + `--profile pop` 输出与 v2.6 基线一致。
- [ ] FireRed 可选性：FireRed 不可用时降级链（sidecar→cli→vpbd_acoustic）不被新代码破坏：`pytest tests/integration/test_pipeline_vpbd_acoustic_fallback.py`。
- [ ] 真实 FireRed smoke（有环境时）：`pytest -m firered tests/integration/test_firered_cli_provider_real.py --rungpu`。

---

## I. 人工验收（M2/M3 候选版本执行，顺带补完 v2.6 L 节欠账）

> 复用 `scripts/vpbd_asr_acceptance.py`、`docs/vpbd_asr_acceptance_playlist.*.json`、`input/acceptance/{category}` 七类目录与人工评分表。

- [ ] 备齐 20 首验收素材：中文流行慢歌 3 / 中文快歌・rap 3 / 英文流行 3 / 民谣低动态 3 / 强节奏副歌 3 / 和声 ad-lib 3 / 长器乐 intro-outro 2。
- [ ] M2 候选版本跑完整 playlist：`scripts/vpbd_asr_acceptance.py --playlist docs/vpbd_asr_acceptance_playlist.filled.json --review-csv manual_review_sheet.csv`。
- [ ] 人工标注边界并统计 `boundary_f1_500ms ≥ 0.82`。
- [ ] 统计 `cut_inside_word_rate ≤ 1%`。
- [ ] 统计 `cut_inside_high_conf_singing_rate ≤ 3%`。
- [ ] 统计 `segment_5_15_pass_rate ≥ 90%`。
- [ ] 主观自然度评分 `subjective_naturalness ≥ 4.2 / 5`（每类抽 1 首人工听切点）。
- [ ] `manual_recutter_rate` 较 v2.5.1 降低 ≥ 40%。
- [ ] M3 增项：auto 风格判定与人工标签一致率 ≥ 85%；记录每首 `auto_profile` Manifest 元数据。
- [ ] 新指标人工复核：`breath_cut_ratio` 合理（气口切点听感自然，无碎片化）、rhythmic 风格下 `beat_aligned_ratio` 提升且零切字。

---

## J. 文档与发布

- [ ] 所有新增模块含 `# File:` 与 `# AI-SUMMARY:` 头注释，公开类/函数有 docstring。
- [ ] 无新增 `TODO` / `FIXME` 残留（`rg -n "TODO|FIXME" src tests` 零命中）。
- [ ] v2.6.1 发布：B 节完成后出 `docs/release_notes_v2_6_1.md`（hybrid_mdd bugfix + 还原开关说明）。
- [ ] v2.7.0-beta 发布：E 节门槛达标后，release notes 标注统一候选池为默认、`candidate_pool=legacy` 为回退项。
- [ ] v2.7.0 发布：F+G+I 全部达标后，release notes 含 AutoProfile 说明与配置迁移指引。
- [ ] `development.md` 更新 v2.7 架构 SSOT（统一引擎数据流图、模式=预设的映射表）。
- [ ] `CLAUDE.md` / `README.md` 同步新 CLI 参数与 `smart_cut` 配置节。
- [ ] FireRed 依赖确认未进入 base `requirements.txt`（`rg -n "firered" requirements.txt setup.py` 零命中）。

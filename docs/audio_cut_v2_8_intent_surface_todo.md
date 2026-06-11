# audio-cut v2.8 意图门面 TODO

- 关联方案: `docs/audio_cut_v2_8_intent_surface_proposal.md`
- 惯例: 与 v2.6/v2.7 todo 相同——勾掉一项时在括号里补证据（测试命令/日志/commit），每节末尾是该节的验收命令块
- 铁律: 每节合并前 F 节回归门必须全绿；`alignment` 缺省 = 0.5 = 恒等变换，任何一项不得破坏此前提

---

## A. 设计冻结与基线

- [x] A1 评审并冻结两极表初值（方案 §3.2 表格），落为 `audio_cut/config/auto_profile.py` 代码常量 `LYRIC_POLE` / `BEAT_POLE`，支持 `expert.yaml` `phrase_boundary.alignment_poles` 覆盖（证据: `venv/bin/python -m pytest -s -q tests/unit/test_alignment_overrides.py ...` 目标组通过；root/package `config/expert.yaml` 均包含 `phrase_boundary.alignment_poles`）
- [x] A2 留存卡点感对照组: 2–3 首高能量歌跑 `--mode hybrid_mdd`（snap_to_beat, density medium/high），输出与 `SegmentManifest.json`（含 qa_report）存档 `output/baselines/v2_8_ab/`，不进 git（证据: `output/baselines/v2_8_ab/sample_{01,02}/hybrid_mdd_{medium,high}/SegmentManifest.json` 齐全；`output/baselines/v2_8_ab/v2_8_ab_report.json` 汇总匿名 sample 指标）
- [x] A3 留存恒等性基线: 同批歌跑 `--mode vpbd_asr` 现状输出存档，供 F2/F5 比对（证据: `output/baselines/v2_8_ab/sample_{01,02}/vpbd_asr_default/SegmentManifest.json` 齐全；F2 复跑报告 `output/baselines/v2_8_ab/f2_identity_diff_report.json` -> `status=pass`）
- [x] A4 建分支 `codex/v2.8-intent-surface`（证据: `git status --short --branch` -> `## codex/v2.8-intent-surface`）

```bash
# A 节验收
ls output/baselines/v2_8_ab/   # hybrid_mdd 与 vpbd_asr 两组基线 + manifest 齐全
git branch --show-current      # codex/v2.8-intent-surface
```

---

## B. 引擎侧：alignment 推导（接线点 `seamless_splitter.py:744`）

- [x] B1 `auto_profile.py` 新增 `ALIGNMENT_STOPS = {lyric:0.0, lyric_lean:0.25, balanced:0.5, beat_lean:0.75, beat:1.0}` 与 `resolve_alignment(value) -> float`：接受档位名 / float / None（→0.5），越界 clamp，非法值报错带可读信息（证据: `tests/unit/test_alignment_overrides.py::test_resolve_alignment_accepts_stops_float_and_none` / `::test_resolve_alignment_rejects_unknown_stop`；快速回归 `184 passed, 1 deselected`）
- [x] B2 新增 `derive_alignment_overrides(a, style_weights) -> Dict[str, Any]`：两段 lerp（a≤0.5 走 LYRIC_POLE→基点，a>0.5 走基点→BEAT_POLE）；**a=0.5 返回空 dict**；惩罚项随表插值但两极取值高位（方案 §3.2）（证据: `tests/unit/test_alignment_overrides.py::test_alignment_midpoint_is_identity_override`、`::test_alignment_endpoints_match_poles`、`::test_alignment_weight_monotonicity`；快速回归 `184 passed, 1 deselected`）
- [x] B3 节拍候选耦合: 由 `a` 派生 `vpbd.beat_candidates.base_score`（a<0.3→0.0；0.5→0.3 现状；1.0→0.65，线性）与 `global_planner.beat_conflict_weight`（0.0→0.15→0.30）；`bars_per_cut` 不动（密度轴管）（证据: `tests/unit/test_alignment_overrides.py::test_alignment_endpoints_match_poles`、`tests/unit/test_seamless_splitter_intent_runtime.py::test_splitter_applies_alignment_after_auto_profile`；快速回归 `184 passed, 1 deselected`）
- [x] B4 接线 `_apply_smart_cut_runtime`: 读 `smart_cut.alignment` → B1 解糖 → B2/B3 覆盖叠加在 AutoProfile 风格权重**之后**；`meta.alignment`（解析后 float 与原始输入）写入返回 meta 与 manifest `auto_profile` 同级（证据: `tests/unit/test_seamless_splitter_intent_runtime.py` -> 通过；快速回归 `184 passed, 1 deselected`）
- [x] B5 `cut_style` 拆轴映射 + `DeprecationWarning`: natural→0.5（恒等）、rhythmic→0.7、dense→segments many（[3,8]）；显式 `alignment` 存在时 cut_style 被忽略并告警（证据: `tests/unit/test_alignment_overrides.py::test_cut_style_maps_to_axes_and_warns`、`tests/unit/test_seamless_splitter_auto_profile.py::test_seamless_splitter_applies_auto_profile_runtime_overrides`；快速回归 `184 passed, 1 deselected`）
- [x] B6 新增单测 `tests/unit/test_alignment_overrides.py`:
  - a=0.5 → 空覆盖（恒等）
  - 端点 0.0/1.0 命中两极表
  - 单调性: beat_affinity 随 a 单调不减，asr_gap 单调不增
  - 档位解糖与 float 等价（`beat_lean` ≡ 0.75）
  - cut_style 映射与告警断言（证据: `venv/bin/python -m pytest -s -q tests/unit/test_alignment_overrides.py ...` 目标组通过；快速回归 `184 passed, 1 deselected`）

```bash
# B 节验收
pytest tests/unit/test_alignment_overrides.py -v
pytest tests/unit/test_auto_profile.py -v   # AutoProfile 既有行为不回归
```

---

## C. 配置面

- [x] C1 `config/unified.yaml` `smart_cut` 增 `segments: medium` 与 `alignment: balanced`（带注释: 档位/数值双轨）；`cut_style` 标 `# deprecated, v3.0 移除`；`global.default_mode` 与意图面默认统一（修方案 E3 打架）（证据: root/package `config/unified.yaml` 同步；`wc -l quick_start.py config/unified.yaml` -> `248 quick_start.py`, `64 config/unified.yaml`；`tests/contracts/test_config_contracts.py` 通过）
- [x] C2 `segments` 档位解糖: few→[10,18] / medium→[5,12] / many→[3,8] 写入 `target_duration_s` 派生链（`derive_smart_cut_overrides` 入口前）；与 `target_duration_s` 同时给出时数值轨赢并告警（证据: `tests/unit/test_alignment_overrides.py::test_segment_density_stops_drive_duration_derivation`、`::test_target_duration_wins_over_segments_with_warning`；快速回归 `184 passed, 1 deselected`）
- [x] C3 `tests/contracts/test_config_contracts.py` 更新: 新键存在、双轨等价、N3/N4/N6 三条兼容承诺断言（证据: `venv/bin/python -m pytest -s -q ... tests/contracts/test_config_contracts.py ...` 目标组通过；快速回归 `184 passed, 1 deselected`）
- [x] C4 `VSS__smart_cut__alignment` / `VSS__smart_cut__segments` 环境变量覆盖验证（沿用 ConfigManager 既有机制，补一条契约用例即可）（证据: `tests/contracts/test_config_contracts.py::test_vpbd_asr_config_supports_vss_env_override` 覆盖 `VSS__SMART_CUT__SEGMENTS` / `VSS__SMART_CUT__ALIGNMENT`；快速回归 `184 passed, 1 deselected`）

```bash
# C 节验收
pytest tests/contracts/test_config_contracts.py -v
VSS__smart_cut__alignment=0.8 python -c "from vocal_smart_splitter.utils.config_manager import get_config; print(get_config('smart_cut.alignment'))"
```

---

## D. 三个入口

- [x] D1 `quick_start.py` 重写为「文件选择 + 三问」（方案 §3.1 mock）：删除模式 5 选 1、风格 5 选 1、hybrid 密度/对齐策略、vpbd 歌词来源/失败策略、输出格式共六组问答；全程零算法名词；行数预算 ≤250（现 446）（证据: `tests/unit/test_quick_start_vpbd.py` 通过；`wc -l quick_start.py config/unified.yaml` -> `248 quick_start.py`）
- [x] D2 `run_splitter.py` 增 `--segments few|medium|many|MIN-MAX` 与 `--align <stop|FLOAT>`：给意图旗标且未给 `--mode` 时路由统一引擎（vpbd + lyrics auto）；显式 `--mode` 永远赢；help 文案去算法名词；既有 14 个旗标全部原样保留（证据: `tests/unit/test_intent_routing.py::test_cli_intent_flags_route_to_unified_engine_when_mode_is_omitted`、`::test_cli_explicit_mode_wins_over_intent_flags`、`tests/unit/test_run_splitter_cli.py`；快速回归 `184 passed, 1 deselected`）
- [x] D3 `api.separate_and_segment` 增可选参数 `segments=None, alignment=None`：解析→runtime_overrides；`mode` 缺省值改为 `None` 哨兵（`None` 且无意图参数→沿用 `v2.2_mdd` 旧默认，对既有调用行为零变化；`None` 且有意图参数→统一引擎；显式 `mode` 永远赢）；manifest 增 `intent` 回显节 `{target_duration_s, alignment, lyrics, profile}`；既有参数与返回结构零变化（证据: `tests/unit/test_intent_routing.py::test_public_api_intent_parameters_route_to_unified_engine`、`::test_public_api_without_intent_keeps_legacy_default`、`tests/unit/test_legacy_mode_regression.py::test_public_api_default_v22_manifest_shape_is_unchanged`；快速回归 `184 passed, 1 deselected`）
- [x] D4 新增 `tests/unit/test_intent_routing.py`: 意图参数→引擎路由矩阵（有/无 mode × 有/无意图参数 四象限）、`intent` 回显正确性（证据: `tests/unit/test_intent_routing.py` 通过；快速回归 `184 passed, 1 deselected`）

```bash
# D 节验收
python run_splitter.py input/<样例>.wav --segments medium --align beat_lean
python run_splitter.py input/<样例>.wav --mode hybrid_mdd   # 旧别名原样可用
pytest tests/unit/test_intent_routing.py -v
```

---

## E. Agent 契约与文档

- [x] E1 更新 `audio-cut封装为模块.md` 为一页契约: 新签名、intent 参数语义、Manifest 字段保证（`intent` 回显 / `segments[].lyrics` / `qa_report` 三件套）、降级语义（lyrics auto 失败→声学；GPU 失败→CPU；可选依赖缺失不抛崩）、可复现性说明（固定 profile 复跑）（证据: 已重写为 v2.8 agent 调用契约；`tests/contracts/test_agent_intent_contract.py` 通过）
- [x] E2 新增 `tests/contracts/test_agent_intent_contract.py`: `separate_and_segment(segments='medium', alignment=0.75)` 冒烟——manifest 含 `intent` 节、旧字段全在、`qa_report` 含 `beat_aligned_ratio`/`breath_cut_ratio`（用 fixture 音频 + fake lyrics provider，不依赖 GPU/模型）（证据: `venv/bin/python -m pytest -s -q tests/contracts/test_agent_intent_contract.py` -> 通过；快速回归 `184 passed, 1 deselected`）

```bash
# E 节验收
pytest tests/contracts/test_agent_intent_contract.py -v
```

---

## F. 验收门（每节合并前全绿，发布前整体复跑）

- [ ] F1 卡点感 A/B: `alignment=1.0 × segments many` vs A2 基线——`beat_aligned_ratio` 差 ≤10pp，`cut_inside_word_rate`=0，主观听感不低于对照组；不达标则校准 BEAT_POLE / base_score 插值曲线后复测（证据: 自动指标已生成 `output/baselines/v2_8_ab/sample_{01,02}/intent_beat_many/SegmentManifest.json` 与 `output/baselines/v2_8_ab/v2_8_ab_report.json`，`f1_auto.status=pass`、两首 sample 的 `beat_delta<=0.10` 且 `cut_inside_word_rate=0.0`；人工 A/B 评审包已生成 `output/baselines/v2_8_ab/f1_manual_review_sheet.csv` 与 `output/baselines/v2_8_ab/f1_manual_review_instructions.md`，仍待人工抽听确认“主观听感不低于对照组”）
- [x] F2 恒等性: 缺省配置跑 `vpbd_asr`，输出与 A3 基线逐样本 diff 为零（证据: `output/baselines/v2_8_ab/f2_identity_diff_report.json` -> `status=pass`，`sample_01 max_abs_diff=0.0 files_compared=64`，`sample_02 max_abs_diff=0.0 files_compared=58`）
- [x] F3 旧模式三连跑 diff: 六个 `--mode` 各跑三次，输出（含文件命名、`_lib` 后缀）diff 为零（证据: `output/baselines/v2_8_ab/f3_mode_three_run/f3_mode_three_run_report.json` -> `status=pass`；6 个显式 mode × 3 run，相对文件名一致，导出音频 `max_abs_diff=0.0`；仅忽略 `manifest_path`/timing/GPU telemetry/debug path/`separation.confidence` 等非输出运行元数据）
- [x] F4 快速回归 + 契约 + 拼接精度: `pytest -m "not slow and not gpu" --cov=src` 全绿；`test_cpu_baseline_perfect_reconstruction.py` 误差 ≤1e-12（证据: `venv/bin/python -m pytest -s -m "not slow and not gpu and not firered" --cov=src --cov-report=term-missing` -> `184 passed, 1 deselected`; `venv/bin/python -m pytest -s tests/unit/test_cpu_baseline_perfect_reconstruction.py -v` -> `1 passed`）
- [x] F5 三问可用性走查: quick_start 从启动到出片 ≤4 次输入，记录实际问答序列（证据: `venv/bin/python quick_start.py` 交互实跑，输入序列 `2 -> 1 -> 2 -> 3`，共 4 次输入，批量处理 5 个文件均完成；匿名记录见 `output/baselines/v2_8_ab/f5_quick_start_walkthrough.json`）

```bash
# F 节验收
pytest -m "not slow and not gpu" --cov=src --cov-report=term-missing
pytest tests/unit/test_cpu_baseline_perfect_reconstruction.py -v
python scripts/vpbd_asr_acceptance.py --help   # A/B 复用 v2.6 验收基建
```

---

## G. 发布

- [x] G1 同步文档: `README.md`（用户面三问/双轨说明）、`CLAUDE.md`（常用命令换意图旗标示例）、`development.md`（意图层→引擎覆盖链架构图）（证据: 已更新三份文档；文档不包含真实临时测试歌曲名；快速回归 `184 passed, 1 deselected`）
- [x] G2 `docs/release_notes_v2_8_draft.md`: 减法清单（6 问→3 问）、双轨语义、兼容承诺 N1–N6、cut_style 弃用公告（证据: 新增 `docs/release_notes_v2_8_draft.md`）
- [x] G3 版本号 `2.8.0-beta`（`global.version` 与打包元数据）（证据: root/package `config/unified.yaml` 与 `config/expert.yaml` 为 `2.8.0-beta`; `setup.py` 为 PEP 440 `2.8.0b0` 且 package data 包含 `expert.yaml`）

---

## 推进顺序

```
A（基线先行，没有 before 就没有 after）
→ B（引擎推导 + 单测，恒等性是第一个要绿的测试）
→ C（配置双轨）∥ D（三入口，可与 C 并行）
→ E（agent 契约）
→ F（验收门，F1 不过则回 B3 调曲线）
→ G（发布）
```

依赖说明: 本清单不依赖 v2.7 todo I 节（20 首人工验收）完成；F1 的 A/B 可与 I 节共用 playlist 与 `scripts/vpbd_asr_acceptance.py` 基建。

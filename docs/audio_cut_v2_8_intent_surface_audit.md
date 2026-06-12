# audio-cut v2.8 意图门面代码审计报告

- 审计对象: 分支 `codex/v2.8-intent-surface`（`97bff0e` 功能提交 + 3 个证据文档提交），对照 `docs/audio_cut_v2_8_intent_surface_todo.md`
- 审计日期: 2026-06-10
- 审计范围: 24 个文件，+1316/-798；引擎推导、三入口、配置双轨、agent 契约、测试与验收门证据
- 总体结论: **通过，建议合并**。代码无阻塞缺陷；唯一发布前置项是 F1 人工抽听（TODO 已明示 `pending_manual_listening`，属验收流程项而非代码问题）

---

## 1. 复跑验证（审计者独立执行）

| 项 | 命令 | 结果 |
| --- | --- | --- |
| 快速回归 | `venv/bin/python -m pytest -m "not slow and not gpu and not firered" -q` | exit 0，184 通过（与 TODO 宣称一致；pytest.ini `addopts=-q` 叠加 `-q` 抑制了摘要行，以进度点 72+72+40 与退出码核对） |
| 拼接精度 | `pytest tests/unit/test_cpu_baseline_perfect_reconstruction.py` | 1 passed |
| CLI 旗标保全 | main vs HEAD 旗标集合 diff | 原 13 旗标 + `input_file` 全保留，仅新增 `--segments` / `--align` |
| 配置双副本同步 | `diff config/*.yaml src/audio_cut/config/*.yaml` | 仅 `# File:` 头部路径行不同（符合预期） |
| F2 恒等证据 | `f2_identity_diff_report.json` | `status=pass`，sample_01/02 `max_abs_diff=0.0`，files_compared 64/58 |
| F3 旧模式证据 | `f3_mode_three_run_report.json` | `status=pass`，6 模式 × 3 run |
| F1 证据 | `v2_8_ab_report.json` / `f1_manual_review_status.json` | 自动指标 `pass`（beat_delta≤0.10、cut_inside_word_rate=0），主观抽听 `pending_manual_listening` |

## 2. 核心正确性确认

1. **恒等性数值自洽**（铁律项）: `derive_alignment_overrides(0.5)` 返回空 dict；`expert.yaml` 的 `vpbd.beat_candidates.base_score: 0.3` 与 `global_planner.beat_conflict_weight: 0.15` 恰为派生曲线在 a=0.5 的取值，0.5 两侧连续、无跳变。`_beat_candidate_base_score` 分段线性与 TODO B3 规格逐点一致（0/0.3→0.0，0.5→0.3，1.0→0.65）。
2. **两极表三处一致**: `LYRIC_POLE`/`BEAT_POLE` 代码常量、root `config/expert.yaml`、package `src/audio_cut/config/expert.yaml` 的 `phrase_boundary.alignment_poles` 数值逐键相同。
3. **旧模式无泄漏**: `_apply_smart_cut_runtime` 由 `is_vpbd_mode`（仅 `vpbd_acoustic`/`vpbd_asr`）门控，非 vpbd 模式显式清空 `_last_intent_meta`，旧模式 result/manifest 不会出现 `intent` 字段，F3 零 diff 成立有结构保证。
4. **API 哨兵零破坏**: `mode: Optional[str] = None`，新参数 `segments`/`alignment` 为 keyword-only；无意图参数时 `effective_mode='v2.2_mdd'`，行为与 v2.7 完全一致（`test_legacy_mode_regression` 同步收紧断言）。用户 `runtime_overrides` 在意图覆盖之后合并，显式覆盖优先级正确。
5. **CLI 优先级正确**: `apply_intent_runtime_overrides` 用 `setdefault` 写 lyrics 三键，随后 `apply_asr_runtime_overrides` 的显式 `--lyrics-provider` 等旗标可覆盖；显式 `--mode` 永远赢（`resolve_effective_mode`）。
6. **接线顺序符合 B4**: alignment 覆盖在 AutoProfile/手动 profile 权重之后叠加；`_phrase_weights_from_overrides` 三个 profile 分支都会产出全部 8 个权重键，基点读取无跨文件污染（批处理复用同一 splitter 时每文件重算）。
7. **profile 键空间不冲突**: `src/audio_cut/config/profiles/*.yaml` 不覆盖 `base_score`/`beat_conflict_weight`/时长键（grep 验证），alignment 派生不会与风格预设打架。
8. **打包修复到位**: `setup.py` 升至 PEP 440 `2.8.0b0`，并把 `expert.yaml` 加入 package_data 与 data_files——缺它则安装版 ConfigManager 启动即缺省值断层，属关键修复。
9. **quick_start 达标**: 248 行（预算 ≤250），交互文案零算法名词（测试以 forbidden-token 断言锁定），最多 4 次输入。
10. **`global.default_mode: vpbd_asr` 无行为风险**: 代码中无任何消费者（grep 验证），纯文档语义统一（修方案 E3 打架项）。
11. **红线检查**: 仓库文档无真实歌曲名（基线证据匿名为 sample_01/02 且整体在 `output/*` gitignore 下）；新文件头部含 `# File:` 与 `# AI-SUMMARY:`；核心实现无 TODO/FIXME/调试 print 残留。
12. **测试质量**: 新增 4 个测试文件均为真实断言——四象限路由矩阵、恒等点、两极命中、单调性、双轨告警、agent Manifest 契约（含 `segments[*].lyrics` 挂接与 `qa_report` 键）、VSS 环境变量双轨覆盖。`.gitignore` 已为新测试文件加白名单（仓库根有 `test_*.py` 忽略规则）。

## 3. 发现的问题（均不阻塞合并）

| # | 等级 | 问题 | 建议 |
| --- | --- | --- | --- |
| 1 | P3 | `alignment_poles` 配置覆盖路径（`_alignment_poles_from_config`）无专门单测；TODO A1 证据只验证了配置键存在，未验证覆盖生效 | 补一条单测：传入自定义 poles 断言端点权重随之变化 |
| 2 | P3 | `manifest['intent']` 两条路径形状略异：splitter 路径含 `applied_overrides`，api fallback 路径（contract test 所测）不含 | 在 `audio-cut封装为模块.md` 注明 `applied_overrides` 为可选字段，或统一两处产出 |
| 3 | nit | API 路径 `resolve_smart_cut_intent` 被调用两次（api echo + splitter 接线），deprecated `cut_style` 告警可能重复发出 | 可忽略；如在意可让 api 层复用 splitter 回显 |
| 4 | nit | YAML 中显式写 `alignment: 0.5` 与"未设置"不可区分（无 explicit_keys），此时 `cut_style: rhythmic` 仍映射到 0.7；runtime/CLI 路径不受影响 | 语义可辩护，文档现状已足够 |
| 5 | 流程 | F1 主观抽听 `pending_manual_listening`，评审包已就绪（`f1_manual_review_sheet.csv` + 边界蒙太奇 WAV） | 发布（G 节收尾/合 main）前完成人工抽听并回填 CSV |

## 4. 有意行为变化（合理，但应知晓）

1. **YAML `target_duration_s` 现在生效**: v2.7 仅 runtime key 触发时长派生，直接编辑 YAML 不生效；v2.8 `should_apply_duration_overrides` 修正为非默认值即生效。默认值路径恒等（F2 通过），属修复 v2.7 不一致。
2. **`cut_style=rhythmic/dense` 用户输出会变**: 从微调风格权重改为映射到 alignment=0.7 / segments=many 轴。已在 release notes 声明弃用映射，v3.0 移除。
3. **unknown profile 回退分支补齐风格权重**: 现在与 auto/manual 分支一致地应用 pop 权重（此前仅 profile overrides），使 `_phrase_weights_from_overrides` 基点完备。

## 5. 结论

- TODO A–G 各节勾选项与代码/证据逐一对得上，验收命令可独立复现。
- 兼容承诺（恒等点、旧默认、显式 mode 优先、Manifest 只增不删、双轨优先级、cut_style 弃用映射）均有测试或证据锚定。
- **合并条件**: 完成 F1 人工抽听并回填 `f1_manual_review_status.json` 为 pass；其余无阻塞项。问题 #1/#2 建议在合并前或合并后小补，不构成门槛。

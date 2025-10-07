<!-- File: todo.md -->
<!-- AI-SUMMARY: 项目任务状态看板，展示已完成事项、进行中任务、待办与 Codex 行动。 -->

# todo.md — 项目开发进度与任务清单（更新于 2025-10-04）

## 0. 说明
- 状态标记：`[x] 已完成` / `[/] 进行中` / `[ ] 未开始`
- 原则：不破坏兼容性；涉及行为变更必须同步测试；默认所有条目可回滚。

## 1. 已完成（Done）
- [x] 切分策略收敛：`_finalize_and_filter_cuts_v2(...)` 保留加权 NMS + 守卫 + 最小间隔，移除二次补刀流程。
- [x] 守卫参数可配置化：`quality_control.enforce_quiet_cut.{win_ms, guard_db, search_right_ms, floor_percentile}` 支持覆盖默认值。
- [x] README 补充 “调参指南（VPP + BPM）”，清理二次检测文案与多余日志路径说明。
- [x] 输出结构统一：`segments_vocal/` 产出 24-bit 人声片段，与伴奏对齐；主目录保留完整人声/伴奏。
- [x] 文档对齐：重写 README / development.md / PRD 交叉验证细节与插图，并同步 chunk vs full 基线说明（更新至 2025-10-04）。
- [x] Chunk vs full 真实模型基准：`tests/benchmarks/test_chunk_vs_full_equivalence.py` 已补真实模型路径并生成 `chunk_vs_full_real.{json,md}`。
- [x] Silero/VAD/特征跨块测试矩阵：`tests/unit/test_silero_chunk_vad.py`、`test_chunk_feature_builder_gpu.py`、`test_chunk_feature_builder_stft_equivalence.py` 覆盖短隙合并与帧一致性。
- [x] GPU 性能报表与文档：`scripts/bench/run_gpu_cpu_baseline.py` + `scripts/bench/README_gpu_pipeline.md` 输出字段解释与 PR 基线模板。
## 2. 进行中（Doing）

- [/] 同类型母带回放：收集长句、说唱、电音、现场、对白等基线素材，验证守卫与判定稳健性。

## 3. 待办（Backlog）

- [ ] 汇总 `segment_classification_debug` 样本，检验 presence/energy 阈值在不同风格的适配情况。

- [ ] 多 GPU 与 `--strict-gpu` 模式：验证一机多卡分配、失败策略与监控字段。

- [ ] 建立 VPP 统计与对标数据集，覆盖自动调参 (slow/medium/fast) 的真实样本分布。

- [ ] 准备一套端到端测试集（快歌/抒情/电子/直播/对白），记录片段数量与跨度基线。

- [ ] BPM 自适应的 clamp/multipliers 回归：验证在极端节拍下的鲁棒性并产出图表。

- [ ] 质量日志强化：记录守卫右移、边界缩进、被最小间隔过滤的候选数量等指标。

- [ ] README 扩充 “常见素材调参示例”，覆盖 BPM 驱动而非 profile 预设的调优路径。

- [ ] 修复 `run_splitter --validate-reconstruction` KeyError：对齐 `split_audio_seamlessly` 返回结构并更新 `tests/test_seamless_reconstruction.py`。

- [ ] 重写 `tests/test_pure_vocal_detection_v2.py`，替换 legacy Tester，接入 SeamlessSplitter v2.3 并纳入 pytest。

## 4. 验收标准
- 默认配置：无 >10s 片段被遗漏；单次检测产出稳定；`test_cpu_baseline_perfect_reconstruction` 通过。
- 静音守卫开启：守卫位移随参数调整可预测，无异常回跳（需人工试听不少于 10 首样本）。

## 5. Codex 当前任务（2025-09-20）
- [x] 阅读现有文档，梳理项目技术路线。
- [x] 拆解核心模块与数据流向。
- [x] 分析 `quick_start.py` 与 `src/vocal_smart_splitter/core` 的调用关系。
- [x] 输出分析结论与后续建议。
- [x] 排查智能命名管线在输出后可能出现的异常。
- [x] P0：验证拼接重建（原始音频 vs 人声）并生成回归报告。
- [x] P0：检查守卫策略在切点合并/重建流程中的表现。
- [x] P1：校准上线前误判样本并复核 presence/energy 分布。
- [x] P1：梳理权限边界与配置校验项，补充检测脚本。
- [x] P2：完善人声片段命名规范（文件名/目录/配置开关），同步更新文档。
- [x] P2：保持人声片段导出流程与伴奏切分的端到端一致性。

## 6. Codex 当前任务（2025-09-21）
- [x] 检视项目代码与文档，对齐输出结构与使用路径。
- [x] 梳理系统架构与核心算法流程，形成可复用描述。
- [x] 汇总项目进展、待办与风险清单。
- [ ] 生成质量守则对照表（AGENTS.md 衍生），待与产品确认术语。

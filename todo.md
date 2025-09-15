# todo.md · 项目开发进展与任务清单（更新于 2025‑09‑12）

## 0. 说明
- 状态：`[x] 完成` / `[/] 进行中` / `[ ] 未开始`
- 原则：不破坏兼容；重要变更配套测试；默认行为可回退。

## 1. 已完成（Done）
- 终筛改造V2：`_finalize_and_filter_cuts_v2(...)`（SeamlessSplitter）
  - 采纳守卫校正时间；最小间隔过滤；最小时长合并/最长强拆；
  - v2.1/v2.2 与 smart_split 两路径均已切换；
- 守卫可配置化（QualityController）
  - `quality_control.enforce_quiet_cut.{win_ms, guard_db, search_right_ms, floor_percentile}` 支持覆盖默认；
- 修复“停顿丰富仍出现整段未切 / 0s 与 30s+ 片段”的问题；
- 清理冗余：`tests/tmp_valley_smoke.py`、`core/vocal_pause_detector.py.backup`、`.pytest_cache/*`、历史临时说明文件等。

## 2. 进行中（Doing）
- [/] 文档对齐（README/development/PRD 小节精炼与截图补充）
- [/] 典型样本集回归（多风格：民谣/流行/电子/摇滚/说唱）

## 3. 待办（Backlog）
- [ ] VPP 方案（vocal_prime-07-VPP.md）落地（方案2）
  - [ ] config.yaml 新增 `pure_vocal_detection.pause_stats_adaptation` 与 `valley_scoring` 键位
  - [ ] 在 `pure_vocal_pause_detector.py` 相对能量分支：实现 VPP 估计与 `mul_pause` 计算，并与 BPM/MDD 融合
  - [ ] 在 `_detect_energy_valleys` 产生 `(time, score)` 候选（时长/安静度/可选平坦度）
  - [ ] 在 `_finalize_and_filter_cuts_v2` 增加“加权NMS”（邻域内保留最高分），再执行守卫与时长治理
  - [ ] 日志与可观测性：打印 VPP 指标、倍率分解、NMS 前后候选数、最高分
  - [ ] 单元测试（合成波形）：VPP 分类与 `mul_pause`、加权NMS 方向正确
  - [ ] 集成测试：快歌但长停顿/慢歌但碎停顿样本上自然度提升，且无 0s/30s+ 片段
- [ ] BPM 禁切区（VocalPrime 分支）契约补全与参数推荐
- [ ] quick_start 增加 v2 引擎切换（silero | vocal_prime）
- [ ] 质量日志增强：守卫右推距离/兜底次数/强拆次数等统计
- [ ] README 与示例音频的最小复现脚本（CI演示）

## 4. 验收（针对本轮变更）
- 默认配置下：
  - 片段不存在 0s；
  - 单段不超过 `quality_control.segment_max_duration`；
  - 拼接误差近似 0；
- 开启守卫覆盖参数后：
  - 结果随参数变化可解释，且无退化（人工抽检 10+ 首样本）。

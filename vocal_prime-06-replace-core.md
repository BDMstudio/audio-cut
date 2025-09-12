# VocalPrime v2.x 核心改造方案（深谷优先 + 时长治理）

本文基于近期线上反馈与波形/频谱观察，给出“长段未切”“出现 <1s 碎片”的根因分析与系统性解决方案。目标是在不破坏默认行为的前提下，稳定命中真实“深谷”切点，并保证最终片段时长合理（无 <1s、无 30s+）。

## 1. 现象与目标
- 现象：在 quick_start 的“纯人声检测 v2.1/ v2.2 MDD”路径下，部分歌曲出现只切前奏、主副歌合并成超长段；有时又出现 <1s 的小碎片。
- 实际需求：理想切点处的能量常接近“峰值/均值的 1%”（即 ≥20 dB 谷深），应优先在这些“深谷”落点，兼顾节拍禁切与声学平滑；同时，强约束最终片段时长的上下界。

## 2. 根因分析
1) 安静守卫过于刚性：`_finalize_and_filter_cuts()` 仅以 `enforce_quiet_cut(...) >= 0` 判断是否保留切点，但不采纳“守卫校正后的时间”。右推窗口较短（~220ms）时找不到“足够安静”便直接丢弃 → 中后段候选大量消失，留下超长片段。
2) 相邻停顿合并过度：`_merge_adjacent_pauses(0.3s)` 在快歌/密集段合并了过多候选。
3) 缺少统一的“片段时长治理”：无“最小时长合并/最长时长强拆”的终端修复，导致出现 <1s 和 30s+ 极端结果。
4) 相对能量阈值在强动态场景鲁棒性不足：仅用“峰/均值比例”易受局部峰污染或底噪抬升影响，错过真实深谷。

## 3. 解决方案总览
在保持默认行为零破坏的前提下，新增/改造如下模块：

### 3.1 CandidateGenerator（已有，增强约束）
- 基于纯人声轨（vocal_track）生成候选：相对能量深谷 + VocalPrime 平台 +（可选）Silero 端点。
- 深谷判定（联合约束）：
  - e_db(t) ≤ floor_db(t) + Δ_db（Δ_db∈[1,3]，保证处于平台内）；
  - 局部峰对比：`p95_db(±200–300ms) − e_db(t) ≥ 20–30 dB`（满足“≈1% 能量”的谷深）；
  - 局部均值对比：`e(t) / mean_local(t) ≤ 0.5%–2%`；
  - 形态约束：谷宽 ≥120ms，左右上坡斜率达标。

### 3.2 QuietGuardRefiner（守卫软校正）
- 对每个候选切点先做零交叉吸附，再执行“安静守卫软校正”：
  - 右推搜索 300–400ms，判定标准为“相对地板 + guard_db（2–3dB）”；
  - 若搜索失败，退而求其次：在窗口内取“局部 RMS 最小值”（而不是直接丢弃切点）。
- 采纳“校正后的时间”进入后续过滤（避免丢点）。

### 3.3 LengthGovernor（统一时长治理）
- 终筛后做两步修复：
  - 最小时长合并：合并 <`segment_min_duration`（建议 1.0s）的片段到相邻片段，循环直到无短片段；
  - 最长时长强拆：对 >`segment_max_duration`（建议 18–22s）片段，在该片段的纯人声子窗内再次扫“深谷”；若仍无，则落中点兜底；新增切点同样需要通过守卫与最小间隔校验。

### 3.4 BeatGuard（可选）
- 强拍±`forbid_ms` 禁切，仅“向右推”，再夹紧在停顿区间内。

### 3.5 BoundaryPolicy（已有）
- 零交叉吸附后做能量验证，若偏差>50ms 则回退原切点。

## 4. 配置建议（默认零破坏，可运行时覆盖）
- 相对能量深谷（极低阈值）：
  - `pure_vocal_detection.enable_relative_energy_mode: true`
  - `peak_relative_threshold_ratio: 0.01`（峰值×1%）
  - `rms_relative_threshold_ratio: 0.01–0.02`（均值×1%–2%）
- 地板与守卫：
  - `vocal_pause_splitting.silence_floor_percentile: 1–3`
  - `vocal_pause_splitting.lookahead_guard_ms: 300–400`
  - `quality_control.enforce_quiet_cut.{guard_db:2.0–3.0, search_right_ms:300–400, win_ms:80, floor_percentile:5}`（若开放配置读取）
- 减少过度合并：`pure_vocal_detection.merge_adjacent_threshold_s: 0.15–0.20`
- 统一时长治理：
  - `quality_control.min_split_gap: 1.0–1.2`
  - `quality_control.segment_min_duration: 1.0`
  - `quality_control.segment_max_duration: 20.0`

## 5. 具体代码落点（实现指引）
- `src/vocal_smart_splitter/core/seamless_splitter.py`
  - 新增：`_finalize_and_filter_cuts_v2(...)`（采纳守卫校正时间 + 最小时长合并 + 最长强拆 + 二次基础过滤）。
  - 替换调用：
    - v2.1/v2.2 流水线：`_process_pure_vocal_split()` 中用 `original_audio` + `pure_vocal_audio=vocal_track` 调用新函数；
    - smart_split 流水线：无纯人声轨，`pure_vocal_audio=None` 调用新函数。
- `src/vocal_smart_splitter/core/quality_controller.py`
  - `enforce_quiet_cut(...)` 支持从配置读取 `win_ms/guard_db/search_right_ms/floor_percentile`，保持默认值不变；
  - 允许回传“校正时间”，上层直接采纳。
- `src/vocal_smart_splitter/core/pure_vocal_pause_detector.py`
  - `_merge_adjacent_pauses(...)` 增加配置覆盖 `merge_adjacent_threshold_s`；
  - `_detect_energy_valleys(...)` 增强：加入“局部峰/局部均对比”和“谷深 dB”硬约束（满足 20–30 dB 深度），并保留谷宽/坡度约束。
- `src/vocal_smart_splitter/config.yaml`
  - 新增：
    - `quality_control.segment_min_duration`、`quality_control.segment_max_duration`
    - `quality_control.enforce_quiet_cut.*`（可选）
    - `pure_vocal_detection.merge_adjacent_threshold_s`

## 6. 验收标准
- 切点准确：在“无平台但有换气/摩擦”的样本上，切点集中于 RMS 深谷（≥20 dB）且经守卫校正；
- 时长合理：终筛后无 <1.0s 片段、无 >`segment_max_duration` 片段；
- 体验一致：在默认配置下不破坏已通过样例；启用深谷与时长治理后，能将“主副歌大段未切”的样本切开为自然段；
- 日志可验证：输出候选数、守卫校正次数、合并/强拆动作计数。

## 7. 风险与回退
- 若某些风格对“局部峰/均对比”的阈值不适配，可仅启用“相对地板 + 谷宽/坡度 + 守卫右推”的最小方案，或回退到现状配置。
- `_finalize_and_filter_cuts_v2` 以新增函数形式接入，可通过配置开关或参数安全回退到旧逻辑。

## 8. 建议的调参顺序
1) 先开启极低阈值（1%）与右推守卫（300–400ms），观察候选量与被接受的切点数；
2) 逐步加严“谷深 dB 与局部对比”以提升自然度；
3) 根据风格收敛 `segment_max_duration` 到 18–22s；
4) 打开 BPM 禁切仅“右推”，避免强拍破坏，说唱/快歌建议适度放宽 forbid_ms。


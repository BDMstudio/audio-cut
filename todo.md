# todo.md 项目开发进展与任务清单

## 0. 使用说明
- 任务状态符号：
  - [x] 已完成（COMPLETE）
  - [/] 进行中（IN_PROGRESS）
  - [ ] 未开始（NOT_STARTED）
- 变更规则：不破坏兼容；所有改动需配套自动化测试（单元/集成/契约）。

## 1. 已完成（Done）
- 纯人声检测 v2.0 主流程接线（quick_start 选项3，先分离再检测）。
- Silero VAD 在纯人声 stem 上执行；样本级分割与零交叉吸附。
- BPM 自适应回退健壮性修复（无特征时优雅回退）。
- cut_at_speech_end 可选模式（人声消失即切割）。
- keep_short_tail_segment=true（尾段短于最小时长亦保留）。
- 输入文件重复列举修复（Windows 大小写重复）。
- 快速启动交互：后端选择与配置应用（FORCE_SEPARATION_BACKEND）。

## 2. 进行中（Doing）
- [/] 文档重构归档（本次）：
  - 新增 development.md（技术路线与模块索引）。
  - 新增 todo.md（本文件）。
  - 清理并归档 PRD.md（需求文档，去除历史噪声段落）。
  - [/] 深谷优先 + 时长治理 方案文档（vocal_prime-06-replace-core.md 已新增）。

## 3. 待办（Backlog）

### 3.1 无静音平台“谷值切割”（Valley-based）原子任务
- [x] 设计落地与配置开关（不破坏兼容）。
  - [x] 新增并接入配置：
    - vocal_pause_splitting.enable_valley_mode=false
    - vocal_pause_splitting.auto_valley_fallback=true
    - vocal_pause_splitting.local_rms_window_ms=25
    - vocal_pause_splitting.silence_floor_percentile=5
    - vocal_pause_splitting.min_valley_width_ms=120
    - vocal_pause_splitting.lookahead_guard_ms=120
    - bpm_guard.forbid_ms=100
- [x] 特征与地板：短时 RMS 包络 + rolling 分位动态地板。
- [ ] 检测器改造：
  - [x] 无平台时触发 valley 分支；
  - [x] 候选谷满足谷宽与上坡约束；
  - [x] lookahead 守卫；
  - [x] 样本级零交叉细化与 20ms 边界保护；
  - [x] valley 强制路径优先于零交叉。
- [/] 第二阶段评分（可选加权）：
  - [x] 接入 spectral_flatness、spectral_centroid；
  - [x] 简易 voicing/HNR；
  - [x] bpm_guard（禁切右推）单测通过，契约与集成已补。
- [/] 测试（CI 必须）：
  - [x] 单元：valley_cut、bpm_guard 等；
  - [x] 契约：valley_no_silence.yaml；
  - [x] 集成：pipeline_v2_valley；
  - [x] 性能与回退：valley_perf、valley_fallback；
  - [x] 默认值验证：defaults_guard。

### 3.2 其它（既有）
- [ ] VocalPrime 补齐 BPM 禁切区（bpm_guard）。
  - 规则：打点不得落在拍点 ±forbid_ms；超宽平台仅向右推避开。
  - 测试：禁切开时断言，禁切关时不作断言。
- [ ] VocalPrime 参数配置化：min_silence_sec、lookahead_guard_ms、plateau_flatness_db、right_bias_ms 进入 get_config。
- [ ] quick_start 增加 v2.engine 选择开关（silero | vocal_prime）。
- [ ] 契约测试补齐：拍点禁切、尾段保留（VocalPrime 分支）。
- [ ] README 对齐 development.md / todo.md / PRD.md。

## 4. 技术债与清理
- [ ] 命名一致性：PureVocal* 与 Silero VAD 分支统一。
- [ ] 过时文档清理：PRD.md 历史模板/无关内容。
- [ ] 配置键去魔数：集中在 config/default.yaml 暴露，代码仅读取。

## 5. 版本里程碑
- v2.1（当前）
  - 目标：VocalPrime 集成、BPM 禁切区、契约测试。
- v2.1.1
  - 目标：参数配置化、engine 切换、文档与 README 对齐。
- v2.2
  - 目标：更多风格测试集、性能画像与参数推荐器。

## 6. 验收标准（针对上述 Backlog）
- VocalPrime 禁切区：开启时所有切点距拍点 >= forbid_ms；关闭时不作此断言。
- 尾段保留：末段 < min_segment_duration 且 keep_short_tail=true 时必须存在。
- 谷值切割（Valley-based）：
  - 无静音平台样本上，切点应落在气声/摩擦噪声窗口内（±20ms）。
  - 切点到高 voicing 区中心距离的分布较仅零交叉方案整体右移。
  - 默认配置下不改变现有行为（仅在无静音平台时自动兜底）。
- 引擎切换：silero 与 vocal_prime 在相同输入下流程可跑通，输出格式一致。
- CI：新增契约测试在干净环境稳定通过。

## 7. 参考
- README.md（运行与结构概览）
- development.md（技术路线与模块索引）
- PRD.md（用户需求与验收）
- MDX23_SETUP.md（安装）

## 8. 深谷优先 + 时长治理（新增工作项）

- 文档
  - [/] 新增方案文档：vocal_prime-06-replace-core.md（已提交，待评审）。

- 终筛改造（SeamlessSplitter）
  - [ ] 新增 `_finalize_and_filter_cuts_v2(...)`：采纳守卫校正时间；统一“最小时长合并/最长强拆”；长段内优先在纯人声子窗扫深谷；二次基础过滤。
  - [ ] 替换两处调用：v2.1/v2.2 与 smart_split 路径（pure_vocal_audio 传参）。

- 守卫可配置化（QualityController）
  - [ ] `enforce_quiet_cut` 读取 `quality_control.enforce_quiet_cut.{win_ms, guard_db, search_right_ms, floor_percentile}`，默认不变，存在则覆盖。
  - [ ] 增加调试日志：校正次数、右推距离、局部最小兜底次数。

- 纯人声深谷检测增强（PureVocalPauseDetector）
  - [ ] `_merge_adjacent_pauses` 支持 `pure_vocal_detection.merge_adjacent_threshold_s` 配置。
  - [ ] `_detect_energy_valleys` 增加：局部峰/均对比与“谷深 dB（>=20–30 dB）”硬约束；保留谷宽/坡度与 lookahead/零交叉。

- 配置与文档
  - [ ] 在 `src/vocal_smart_splitter/config.yaml` 新增/暴露：
        `quality_control.segment_min_duration`、`quality_control.segment_max_duration`、
        `quality_control.enforce_quiet_cut.*`、`pure_vocal_detection.merge_adjacent_threshold_s`。
  - [ ] README 与 development.md 对应更新；
  - [ ] 在 vocal_prime-06-replace-core.md 标注默认值与推荐范围。

- 测试与验收
  - [ ] 单元：`test_length_governor_merge_short`、`test_length_governor_split_long`、`test_guard_time_adoption`、`test_deep_valley_constraints`。
  - [ ] 集成：v2.2_mdd 典型样本不得出现 <1s、>20s；“只切前奏”样本需切开主/副歌且不落强拍禁切。
  - [ ] 契约：最长片段 <= segment_max_duration；无 < segment_min_duration 片段。


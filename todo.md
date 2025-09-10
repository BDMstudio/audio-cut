# todo.md — 开发进展与任务清单

## 0. 使用说明
- 本清单以 ✅/[/]/[ ] 标记任务状态：
  - ✅ 已完成（COMPLETE）
  - [/] 进行中（IN_PROGRESS）
  - [ ] 未开始（NOT_STARTED）
- 变更规则：不破坏兼容；所有改动需配套自动化测试（单元/集成/契约）。

## 1. 已完成（Done）
- ✅ 纯人声检测 v2.0 主流程接线（quick_start 选项3，先分离再检测）
- ✅ Silero VAD 在纯人声 stem 上执行；样本级分割与零交叉吸附
- ✅ BPM 自适应回退健壮性修复（无特征时优雅回退）
- ✅ cut_at_speech_end 可选模式（“人声消失即切割”）
- ✅ keep_short_tail_segment=true（尾段短于 min_segment_duration 亦保留）
- ✅ 输入文件重复列举修复（Windows 大小写重复）
- ✅ 快速启动交互：后端选择 & 配置应用（FORCE_SEPARATION_BACKEND）

参见：DEVELOPMENT_LOG.md、FINAL_v2_fixes.md、HOTFIX_v2_method_names.md

## 2. 进行中（Doing）
- [/] 文档重构归档（本次）：
  - 新增 development.md（技术路线/模块索引）
  - 新增 todo.md（本文件）
  - 清理并归档 PRD.md（需求文档，去除历史噪声片段）

## 3. 待办（Backlog）

### 3.1 无静音平台“谷值切割”（Valley-based）原子任务
- [x] 设计落地与配置开关（不破坏兼容）
  - [x] 新增配置键并接入 config_manager/get_config：
    - [x] vocal_pause_splitting.enable_valley_mode=false
    - [x] vocal_pause_splitting.auto_valley_fallback=true
    - [x] vocal_pause_splitting.local_rms_window_ms=25
    - [x] vocal_pause_splitting.silence_floor_percentile=5
    - [x] vocal_pause_splitting.min_valley_width_ms=120
    - [x] vocal_pause_splitting.lookahead_guard_ms=120
    - [x] bpm_guard.forbid_ms=100（沿用/对齐已有结构）
  - [x] default.yaml 暴露上述键，文档注释默认值与推荐范围
- [x] 特征与地板
  - [x] 短时 RMS 包络（最小实现位于 detector 内；窗≈25ms）
  - [x] rolling percentile（5%）动态地板（局部窗口近似）
- [ ] 检测器改造
  - [x] vocal_pause_detector._calculate_cut_points：无平台时触发 valley 分支（受 enable/auto 控制）
  - [x] 在候选区扫描“谷”：满足谷宽≥min_valley_width_ms、两侧上坡>阈值（最小实现：±(min_valley_width_ms/2) 边带均高于谷底×1.15）（待补）
  - [x] 未来静默守卫：lookahead_guard_ms（兜底 valley 时可禁用以切入谷心）
  - [x] 样本级零交叉细化：±窗口吸附；同时 20ms 边界保护
  - [x] valley 强制路径（enable 时）优先于零交叉，确保切在“谷”内
- [/] 第二阶段评分（可选加权）
  - [x] 接入 spectral_flatness、spectral_centroid（valley 强制路径打分）
  - [x] 简化 voicing/HNR 已接入；bpm_guard（禁切右推）已实现并通过单测；契约/集成已补
- [/] 测试（CI 必须）
  - [x] 单元：tests/unit/test_valley_cut.py, tests/unit/test_valley_cut_more.py, tests/unit/test_bpm_guard.py 通过
  - [x] 契约：tests/contracts/valley_no_silence.yaml（切点距边界≥20ms；启用 bpm_guard 不落强拍禁切区）
  - [x] 集成：tests/integration/test_pipeline_v2_valley.py（验证 valley 全部满足≥20ms 边界保护，且零交叉至少一例<20ms）
- [x] 性能与回退（本轮完成）
  - [x] 性能日志：tests/performance/test_valley_perf.py 记录端到端耗时（不做断言）
  - [x] 回退策略验证：tests/unit/test_valley_fallback.py（40ms 窄谷仍稳定、边界≥20ms）
  - [x] 默认值验证：tests/unit/test_defaults_guard.py（默认关闭 valley，不崩溃且切点位于平台区间内）

### 3.2 其它（既有）
- [ ] VocalPrime 补齐 BPM 禁切区（bpm_guard）
  - 规则：打点不得落在拍点 ±forbid_ms；超宽平台允许仅“向右推”避让
  - 测试：契约测试断言与拍点距离 ≥ forbid_ms（禁切开）/ 不断言（禁切关）
- [ ] VocalPrime 参数配置化
  - min_silence_sec、lookahead_guard_ms、plateau_flatness_db、right_bias_ms → get_config()
  - 默认值保持当前行为，避免破坏用户空间
- [ ] quick_start 增加 v2.engine 选择开关
  - 值：silero | vocal_prime（默认 silero）
  - 影响：仅切换检测引擎，输出/格式不变
- [ ] 契约测试补齐
  - 拍点禁切（VocalPrime）
  - 尾段保留（VocalPrime 分支同 v2.0 逻辑）
- [ ] README.md 小幅对齐（指向 development.md / todo.md / PRD.md）

## 4. 技术债与清理
- [ ] 名称一致性：PureVocal* 与 Silero VAD 分支命名与导入路径统一
- [ ] 过时文档段落清理：PRD.md 中的历史模板/注入内容
- [ ] 配置键去魔数：集中在 config/default.yaml 暴露，代码仅读取

## 5. 版本里程碑
- v2.1（当前开发分支）
  - 目标：VocalPrime 集成、BPM 禁切区、契约测试
- v2.1.1
  - 目标：参数配置化、engine 切换、文档与 README 对齐
- v2.2
  - 目标：更多风格测试集、性能画像与参数推荐器

## 6. 验收标准（针对上述 Backlog）
- VocalPrime 禁切区：开启时所有切点距拍点 ≥ forbid_ms；关闭时不做此断言
- 尾段保留：末段 < min_segment_duration 但 keep_short_tail=true 时必须存在
- 谷值切割（Valley-based）：
  - 无静音平台样本上，切点应落在气声/摩擦噪声谷 ±20ms；
  - 切点到高-voicing 区中心距离的分布较仅零交叉方案整体右移；
  - 默认配置下不改变现有行为（仅在无静音平台时自动兜底）。
- 引擎切换：silero 与 vocal_prime 在相同输入下流程可跑通，输出格式不变
- CI：新增契约测试在干净环境稳定通过

## 7. 参考
- README.md（运行与结构概览）
- development.md（技术/模块索引）
- PRD.md（用户需求与验收）
- MDX23_SETUP.md（安装）


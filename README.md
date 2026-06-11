<!-- File: README.md -->
<!-- AI-SUMMARY: Vocal Smart Splitter 的用户手册，涵盖特性、快速开始、配置与调参建议。 -->

# 智能人声分割器（Vocal Smart Splitter）

Vocal Smart Splitter 支持高保真声部拆分、纯人声检测和智能切分。v2.8 起用户面收敛为两个意图轴：片段密度（少/中/多）与切点风格（歌词到节拍的连续偏好）。旧 `--mode` 仍完整保留为专家兼容入口，但 `quick_start.py`、CLI 新参数和 Python API 默认都面向意图而不是算法名词。

## 核心能力
- **双通道分离**：默认使用 MDX23 ONNX 输出人声/伴奏，失败时自动回退 Demucs v4（可配置关闭）。
- **GPU 多流分块流水线**：`audio_cut.utils.gpu_pipeline` 负责 chunk 规划、CUDA streams、pinned 缓冲与背压；`EnhancedVocalSeparator` 会记录 `gpu_meta`，并在 GPU 失败时安全回退 CPU。
- **纯人声检测**：`PureVocalPauseDetector` 结合 F0、共振峰、RMS 能量、MDD/BPM 自适应完成停顿判定，仅执行一次检测。
- **VPBD 统一候选池**：`vpbd_acoustic` 使用声学停顿、VPBD 专属气口和高能量段弱节拍候选；`vpbd_asr` 可接入 fake、FireRed sidecar 或 FireRed CLI provider，并把 word gap、sentence end、mVAD 边界作为候选入池。ASR/节拍只按权重加分，词区间与 `vocal_cut_risk` 仍用于降权和 guard 避让。
- **守卫与补刀**：`audio_cut.cutting.finalize_cut_points` 在人声/混音轨执行过零吸附与静音守卫，同时统计守卫位移（`guard_shift_stats`）。
- **段落布局精炼**：`segment_layout_refiner` 负责微碎片合并、软最小合并与软最大救援，`segment_layout_applied` 字段可用于调试。
- **片段标注**：生成 `segment_{###}_{human|music}` 文件，并在 `segment_classification_debug` 中记录活跃度、能量阈值等判定依据。
- **高精度输出**：导出 24-bit WAV；`tests/unit/test_cpu_baseline_perfect_reconstruction.py` 约束拼接误差 ≤ 1e-12；`tests/test_seamless_reconstruction.py` 将在适配 v2.3 后恢复。
- **特征缓存复用**：`TrackFeatureCache` 在分离阶段构建一次，供检测、MDD 增强与守卫流程复用 BPM/MDD/RMS 序列。
- **分块 VAD 与焦点窗口**：`SileroChunkVAD` 合并跨块语音段，并为检测构建 ±pad 焦点窗口，仅在关键区域运行昂贵特征。

## 快速开始
1. 将待处理音频（`mp3/wav/flac/m4a`）放入 `input/` 目录。
2. 交互模式：
   ```bash
   python quick_start.py
   ```
   除文件选择外只问三件事：
   - 要切片，还是只做人声/伴奏分离；
   - 片段密度：少（10-18s）、中（5-12s）、多（3-8s）；
   - 切点风格：歌词优先、偏歌词、均衡、偏节拍、强卡点。

   歌词来源、风格估计、输出格式和 provider 回退链都走配置默认；常规用户和 agent 不需要选择内部实现。
3. 命令行意图参数：
   ```bash
   python run_splitter.py input/track.mp3 --segments medium --align beat_lean
   python run_splitter.py input/track.mp3 --segments 6-14 --align 0.8
   ```
   `--segments` 支持 `few|medium|many|MIN-MAX`；`--align` 支持 `lyric|lyric_lean|balanced|beat_lean|beat` 或 `0.0-1.0`。
4. 旧模式仍可显式调用：
   ```bash
   python run_splitter.py input/track.mp3 --mode vocal_separation
   python run_splitter.py input/track.mp3 --mode v2.2_mdd
   python run_splitter.py input/track.mp3 --mode hybrid_mdd
   python run_splitter.py input/track.mp3 --mode librosa_onset
   python run_splitter.py input/track.mp3 --mode vpbd_asr --lyrics-provider fake --lyrics-fixture tests/fixtures/lyrics/simple_song_timeline.json
   ```
   显式 `--mode` 永远优先；未显式传 `--mode` 但给了意图参数时，自动走统一候选池路径。
5. 输出目录统一为 `output/<日期>_<时间>_<原音频名>/`，单文件与批处理遵循同一规则。

## 模块化调用（作为子模块嵌入）
- `audio_cut.api.separate_and_segment(...)` 是上层 agent 的稳定入口；既有参数保持兼容，新意图参数全部可选。
- 调用示例：
  ```python
  from audio_cut.api import separate_and_segment

  manifest = separate_and_segment(
      input_uri="input/track.mp3",
      export_dir="output/job",
      segments="medium",
      alignment=0.75,
      export_types=("vocal", "human_segments", "music_segments"),
      export_manifest=True,
  )
  ```
- 不传 `segments/alignment/mode` 时沿用旧默认 `v2.2_mdd`；传意图参数且未显式传 `mode` 时走统一候选池；显式 `mode` 永远优先。
- Manifest 会增量回显 `intent`，并继续包含旧字段、`segments[*].lyrics`（可选）和 `qa_report`。详见 `audio-cut封装为模块.md`。
- `export_types` 控制导出资产：`vocal`、`instrumental`、`human_segments`、`music_segments`。未指定时默认导出全部。

## 输出结构
- `segment_###_{human|music}_*.wav`：混音片段，文件名追加 `_X.X`（秒，保留 1 位小数）表示片段时长。
- `segment_###_{human|music}_lib_*.wav`：**节拍卡点片段**（hybrid_mdd 模式），结束切点对齐小节边界。
- `segments_vocal/segment_###_{human|music}_vocal_*.wav`：对应人声片段，同样追加 `_X.X` 时长后缀。
- `<stem>_v2.2_mdd_vocal_full_*.wav` / `<stem>_v2.2_mdd_instrumental_*.wav`：全长人声/伴奏文件。
- `segment_classification_debug`：调试信息（活跃度、阈值、投票），CLI 可自行持久化为 JSON。
- `segment_lib_flags`：标记哪些片段是节拍卡点片段（hybrid_mdd 模式）。
- 结果字典包含 `guard_shift_stats`、`guard_adjustments`、`gpu_meta` 等诊断信息。
- `vpbd_asr` Manifest 可选包含 `lyrics_alignment`、`boundary_detection`、`segments[*].lyrics` 与 `cuts.final[*].features`；QA report 额外统计 `breath_cut_ratio` 与 `beat_aligned_ratio`。
- 其他字段：`cut_points_samples/sec`、`guard_adjustments`、`suppressed_cut_points_sec` 等，用于验证切点一致性。

## 配置总览
`config/unified.yaml` 是 v2.8 用户面配置，保持在 120 行以内；`config/expert.yaml` 存放高级默认值并由 `ConfigManager` 自动先加载。优先级从低到高为：`expert.yaml` -> `unified.yaml` -> `VSS_EXTERNAL_CONFIG_PATH` -> 显式配置文件 -> `VSS__...` 环境变量。

用户通常只需要改 `smart_cut`：
- `segments`：`few|medium|many`，分别解析为 `[10,18]`、`[5,12]`、`[3,8]` 秒；也可用 `target_duration_s` 直接给数值轨。
- `alignment`：`lyric|lyric_lean|balanced|beat_lean|beat` 或 `0.0-1.0`；`0.5` 是兼容恒等点。
- `lyrics=auto`：自动 provider 回退链；失败时降级声学候选继续处理。
- `profile=auto`：按 BPM/MDD/能量 CV/人声覆盖率估计风格；手动 `ballad/pop/edm/rap` 是专家逃生口。
- `cut_style` 已废弃，读取时会映射到 `alignment/segments`，计划在 v3.0 移除。

高级参数仍在 `config/expert.yaml`：`pure_vocal_detection.*`、`quality_control.*`、`segment_layout.*`、`hybrid_mdd.*`、`vpbd.*`、`phrase_boundary.*`、`global_planner.*`、`gpu_pipeline.ort.*` 等。需要覆盖时继续使用原路径或 `VSS__...`。

示例：
```bash
VSS__smart_cut__segments=many VSS__smart_cut__alignment=0.8 python run_splitter.py input/track.mp3
```

`vpbd_asr` 的长段二次分割遵循软约束：优先选择声学低谷，并用歌词句/唱段边界加权；找不到可信低谷时保留稍长片段，不使用 midpoint 硬切。

## 调参指引
- **切点过少/片段过长**：优先调整 `smart_cut.target_duration_s`；必要时在 `config/expert.yaml` 或 `VSS__...` 中降低 `pure_vocal_detection.peak_relative_threshold_ratio` / `rms_relative_threshold_ratio`、减小 `quality_control.min_split_gap` 或调节 `valley_scoring.merge_close_ms`。
- **切点过多/片段碎化**：优先收窄 `smart_cut.target_duration_s` 的下限或使用更保守 profile；必要时提升 expert 阈值、增大 `min_split_gap`，通过 `segment_min_duration` 限制最短片段。
- **静音守卫不稳定**：在 `config/expert.yaml` 中调整 `quality_control.enforce_quiet_cut.guard_db` / `search_right_ms`；检查输入是否被提前归一化。
- **VPBD ASR 切到歌词内部**：先检查 `boundary_detection.selected[*].source`、`meta.sources`、`features.inside_word_penalty` 与 `planner.final_time_by_raw_time`。自然风格下优先级应来自权重：长停顿最高，气口+句尾次之，节拍只是弱候选；`score=0` 的候选不会进入 rescue fallback；若 guard 把词外 raw cut 推入 ASR word interval，`vpbd_asr` 会恢复到 raw cut。
- **判定错误（伴奏被标成 human）**：查看 `segment_classification_debug` 中的活跃度；适当提升 `segment_vocal_activity_ratio`。

## 测试
- 快速回归：
  ```bash
  pytest -m "not slow and not gpu and not firered" --cov=src --cov-report=term-missing
  ```
- v2.7 H 发布门禁补充：
  ```bash
  python scripts/legacy_mode_diff_gate.py \
    --baseline-ref 8271984 \
    --input input/<local-smoke-audio>.mp3 \
    --output-dir output/v2_7_h_legacy_diff

  python scripts/vpbd_rollback_diff_gate.py \
    --baseline-ref 8271984 \
    --input input/<local-smoke-audio>.mp3 \
    --lyrics-fixture tests/fixtures/lyrics/simple_song_timeline.json \
    --output-dir output/v2_7_h_vpbd_rollback_diff
  ```
  `legacy_mode_diff_gate.py` 比对 `v2.2_mdd` / `hybrid_mdd` / `librosa_onset` 的 Manifest 字段与文件命名契约；`vpbd_rollback_diff_gate.py` 验证 `vpbd.candidate_pool=legacy` + `--profile pop` 相对 v2.6 基线不漂移。脚本要求调用方提供本地 smoke 音频，仓库不记录真实歌曲名。
- 重点单测：
  - `tests/unit/test_cpu_baseline_perfect_reconstruction.py`：样本级重建误差；
  - `tests/unit/test_cutting_refiner.py` / `test_cutting_consistency.py`：守卫与 NMS 行为；
  - `tests/unit/test_gpu_pipeline.py`：chunk 调度与 GPU 回退；
  - `tests/unit/test_chunk_feature_builder_*`：STFT 等价与跨块拼接；
  - `tests/unit/test_silero_chunk_vad.py`、`test_pure_vocal_focus_windows.py`：分块 VAD 与焦点窗口；
  - `tests/benchmarks/test_chunk_vs_full_equivalence.py`：chunk/full 误差报告；
  - `tests/integration/test_pipeline_v2_valley.py`：MDD 主流程；
  - `tests/contracts/test_config_contracts.py`：配置兼容契约；
  - `tests/unit/test_alignment_overrides.py`、`tests/unit/test_intent_routing.py`、`tests/unit/test_seamless_splitter_intent_runtime.py`、`tests/contracts/test_agent_intent_contract.py`：v2.8 意图解析、入口路由、runtime 接线与 agent Manifest 契约；
  - `tests/unit/test_firered_*`、`tests/unit/test_run_splitter_cli.py`、`tests/unit/test_quick_start_vpbd.py`：provider 与入口契约。
- 待补测试：`tests/test_seamless_reconstruction.py` 需适配 v2.3 结果结构；批量处理路径需要新的集成测试。

## 性能基线
- **分离阶段**：MDX23 GPU 目标 ≥0.7x 实时；CPU 回退 ≈3.5x 实时；记录 `h2d_ms/dtoh_ms/compute_ms/peak_mem_bytes`。
- **检测 + 守卫**：10 分钟素材单核约 12s；开启静音守卫额外耗时约 8%。
- **Chunk vs Full**：dummy 模型误差 <1e-6，真实模型断言 `L∞<5e-3`、`SNR>60 dB`。
- **拼接误差**：`test_cpu_baseline_perfect_reconstruction` 要求最大绝对误差 ≤1e-12。
- **性能脚本**：
  ```bash
  python scripts/bench/run_gpu_cpu_baseline.py input/your_song.wav --write-markdown
  python scripts/bench/run_multi_gpu_probe.py input/your_song.wav --mode v2.2_mdd
  ```

## 更新记录
- **2026-06-11 (v2.8 draft)**
  - 产品面减法：`quick_start.py` 从多层算法菜单收敛为文件选择 + 三问。
  - 新增意图参数：CLI `--segments/--align` 与 API `segments/alignment`，Manifest 增量回显 `intent`。
  - `alignment` 滑块在 AutoProfile 后叠加权重；`0.5` 为恒等点，词内和高人声风险惩罚保持高位。
  - `smart_cut.segments/alignment` 进入用户配置；`cut_style` 标记废弃并映射到新双轴。
  - agent 契约明确 `intent + segments[*].lyrics + qa_report` 三件套，旧 `--mode` 路径继续保留。
- **2026-06-10 (v2.7 draft)**
  - VPBD 候选池开始接收气口候选，默认按 `vpbd.breath_score_scale=0.6` 降权；旧模式仍按原逻辑过滤 breath。
  - `vpbd_asr` 的 lyrics gap / sentence end / mVAD 边界与声学候选合并后统一打分规划，近重复候选在 ±120ms 内融合并在 `meta.sources` 留痕。
  - VPBD 新增高能量段弱节拍候选，默认每 2 小节生成一次，候选携带 `vocal_cut_risk` 供后续规划降权。
  - 修复 VPBD 死特征：`vocal_cut_risk`、`mdd_affinity`、`beat_conflict` 现在进入打分/规划链；句尾容差放宽到 250ms，词边缘惩罚按距离软化。
  - 权重体系进入 v2.7 beta 形态：新增 `breath` 权重，`inside_word_penalty` 提升到 0.80；breath 不再冒充长停顿，merged breath+sentence 会保留 breath 特征，beat 候选不再绕过权重强行抬分。
  - 新增 `vpbd.candidate_pool=legacy` 回退开关和 `vpbd.candidate_debug_json` 候选调试路径；QA report 新增 `breath_cut_ratio`、`beat_aligned_ratio`。
  - 新增 AutoProfile：`smart_cut.profile=auto` 基于 BPM/MDD/能量 CV/人声覆盖率估计风格，插值已有 profile anchor，低置信回退 pop；`--profile auto` 和 quick_start 默认 auto，手动 profile 优先。
  - `smart_cut.target_duration_s` 成为 v2.7 时长单一入口，派生 `global_planner`、`segment_layout` 和 `quality_control.segment_max_duration`。
  - `config/unified.yaml` 瘦身为用户面配置，`config/expert.yaml` 自动加载高级默认值；删除 `bpm_adaptive_core.*` 与 `vocal_pause_splitting.bpm_adaptive_settings`，VPP 乘数并入 `relative_threshold_adaptation.pause_stats_multipliers`。
  - 删除未生效且会误伤低权重 soft prior 的 `phrase_boundary.min_score`。
  - `boundary_detection.candidate_counts` 新增 `merged` / `lyrics_pooled` 计数，旧字段语义保持不变。
- **2026-06-10 (v2.6.1 draft)**
  - 修复 `hybrid_mdd.vad_protection` 过去未真正参与副歌吸附决策的问题：策略层现在使用分离后人声轨判断目标节拍是否安静。
  - `snap_to_beat` 默认容差收紧为 200ms，并按 BPM clamp 到 ≤0.4 个 beat；新增 `hybrid_mdd.chorus_force_snap` 作为旧版强吸附回退开关。
  - `hybrid_mdd` 策略输出后重新进入统一 guard/refine 链，Manifest 暴露 `guard_shift_stats`/`guard_adjustments`，`_lib` 标记按最近原始切点映射到守卫后的边界。
- **2026-06-09 (v2.6 draft)**
  - 新增 `vpbd_acoustic` / `vpbd_asr` 规划路径：声学候选仍为主控，ASR 歌词时间轴仅作为 soft prior。
  - 修正 VPBD ASR 切点职责边界：rescue fallback 过滤 `score=0` 候选，布局救援使用声学低谷 + ASR 句/唱段边界，词区间仅用于降权与 guard/local refine 避让。
  - 本地临时中文歌曲 FireRed CLI smoke：导出 12 段，最长约 15.0s，最终切点 `inside_word_count=0`；测试素材不进入仓库。
  - 新增 FireRed sidecar/CLI provider 协议适配，真实 FireRed 测试通过 `firered` + `gpu` marker 保护。
  - CLI 新增 `--lyrics-provider`、`--firered-endpoint`、`--asr-chunk-s`、`--asr-overlap-s`、`--asr-strict`、`--lyrics-fixture`。
- **2026-01-18 (v2.5.1)**
  - **移除 `mdd_start` 策略**：保留 `beat_only` 和 `snap_to_beat` 两种策略，简化用户选择
  - **多特征副歌检测**：实现 RMS能量 + 频谱质心 + 频谱带宽融合算法
    - 根据能量变异系数(CV)自适应调整特征权重（低动态歌曲侧重频谱，高动态歌曲侧重能量）
    - 民谣/爵士等低动态歌曲副歌识别准确度提升60-70%（低动态测试样本：39/104→12/104副歌小节）
    - 保持流行歌曲准确度不受影响
  - **交互式策略选择**：`quick_start.py` 新增 lib_alignment 策略选择菜单
  - **连续性检测增强**：要求至少连续4小节高能量才识别为副歌，过滤主歌零散高点
  - 完成 SeamlessSplitter 重构：BeatAnalyzer/SegmentExporter/ResultBuilder 接入
  - 新增重构记录：`docs/SeamlessSplitter 重构记录.md`
- **2026-01-17 (v2.5.0)**
  - 新增 `hybrid_mdd` 模式：MDD 人声分割 + librosa 节拍卡点增强
  - `_lib` 后缀标记节拍对齐的片段，适合 MV 剪辑
  - 密度控制 (low/medium/high) 通过 `unified.yaml` 或 quick_start 交互配置
  - 预过滤算法：节拍切点添加前检查是否会产生短片段
  - **方案 C (snap_to_beat)**：仅在副歌/高能量段将 MDD 切点吸附到最近节拍（卡点感），主歌保持 MDD 原切点；`_lib` 标记仅出现在副歌。
  - 设计文档: `docs/hybrid_mdd_design.md`, `docs/hybrid_mdd_refactor_evaluation.md`
- **2026-01-17 (v2.4.1)**
  - 代码清理：删除未生效的 `enable_bpm_adaptation` 和 `interlude_coverage_check` 算法
  - 删除 `unified.yaml` 中对应的冗余配置项
  - 验证并确认 22 个切点辅助算法中 13 个生效，9 个因配置禁用
- **2025-10-12**
  - 新增 `audio_cut.api.separate_and_segment` 统一 API，可生成标准 Manifest 并在外部工程内复用。
  - `SeamlessSplitter` 增强导出计划控制，补充 `full_vocal_file`/`full_instrumental_file` 等元数据。
- **2025-10-10**
  - quick_start 增加批处理模式；输出目录统一使用 `<日期>_<时间>_<原音频名>` 命名。
  - 导出文件名统一附带 `_X.X`（秒）后缀，便于 QA 对照。
- **2025-10-04**
  - 补充 chunk vs full 基线说明，修正 dummy/真实模型断言阈值。
  - 增加 `audio_cut.cutting.metrics` 使用说明。
- **2025-09-27**
  - 完成 GPU 流水线与分块 VAD 能力，更新调试字段。
- **2025-09-26**
  - README 全面重写，适配 v2.3 架构与配置默认值。

## 使用声明
本项目仅用于技术研究与个人实验；涉及商用或分发，请自行确认版权与法律合规。

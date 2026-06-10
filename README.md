<!-- File: README.md -->
<!-- AI-SUMMARY: Vocal Smart Splitter 的用户手册，涵盖特性、快速开始、配置与调参建议。 -->

# 智能人声分割器（Vocal Smart Splitter）

Vocal Smart Splitter 支持高保真声部拆分、纯人声检测，以及带 MDD（Musical Dynamic Density）守卫的一站式处理能力。自 v2.5 起新增 `hybrid_mdd` 模式，支持 MDD 人声分割 + librosa 节拍卡点增强，片段带 `_lib` 后缀标记，适合 MV 剪辑场景。v2.6.1 起 `hybrid_mdd` 的节拍吸附会使用分离后人声轨做安静度保护，并在吸附后重新进入统一守卫链；v2.7 draft 起 `vpbd_asr` 会把 FireRedASR/FireRedASR2S 歌词时间轴、声学低谷、气口与高能量段弱节拍点统一放入候选池，仍由权重和规划器决定最终切点。

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
   - 第一步选择处理范围（单文件或批量处理）。
   - 第二步选择处理模式：
     - `1` 仅分离
     - `2` Pure Vocal v2.2 MDD
     - `3` librosa_onset 节拍分割
     - `4` **Hybrid MDD**（MDD + 节拍卡点，推荐 MV 剪辑）
     - `5` **VPBD + FireRedASR2S**（实验模式，歌词时间轴 soft prior）
   - 选择 Hybrid MDD 后可配置：
     - **卡点密度**（少/中/多）
     - **对齐策略**（beat_only 强制节拍切割 / snap_to_beat MDD智能吸附，推荐）
   - 选择 VPBD + FireRedASR2S 后可选择 sidecar、CLI、fake fixture 或 auto provider。
3. 命令行模式：
   ```bash
   # MDD 模式（默认）
   python run_splitter.py input/your_song.mp3 --mode v2.2_mdd
   
   # Hybrid MDD 模式（MDD + 节拍卡点）
   python run_splitter.py input/your_song.mp3 --mode hybrid_mdd
   
   # librosa_onset 模式（纯节拍分割）
   python run_splitter.py input/your_song.mp3 --mode librosa_onset

   # VPBD + fake lyrics fixture（测试/回归）
   python run_splitter.py input/your_song.mp3 --mode vpbd_asr \
     --lyrics-provider fake \
     --lyrics-fixture tests/fixtures/lyrics/simple_song_timeline.json

   # VPBD + FireRed sidecar
   python run_splitter.py input/your_song.mp3 --mode vpbd_asr \
     --lyrics-provider sidecar \
     --firered-endpoint http://127.0.0.1:8765
   ```
   可按需追加 `--validate-reconstruction`、`--gpu-device cuda:1|cpu`、`--strict-gpu`、`--profile ...`、`--compat-config v2`、`--asr-chunk-s`、`--asr-overlap-s`、`--asr-strict`。
4. FireRed provider 部署：
   - **Sidecar**：外部常驻 worker 提供 `GET /health` 与 `POST /analyze`；请求 JSON 包含 `audio_path`、`duration_s`、`sample_rate`、`strict`、`meta`，响应 JSON 输出 `words`、`sentences`、可选 `mvad`。
   - **CLI**：外部 worker 接收 `--input-json - --output <lyrics_timeline.json>`，返回码非 0、超时或非法 JSON 会按 strict 配置报错或降级。
   - FireRed 依赖不进入 base requirements；实际 worker 可放在 `/home/ubuntu/asr_test` 或独立环境。
5. 输出目录统一为 `output/<日期>_<时间>_<原音频名>/`（例如 `20241010_153045_song`），单文件与批处理遵循同一规则。

## 模块化调用（作为子模块嵌入）
- 通过 `audio_cut.api.separate_and_segment(...)` 可在上层项目中直接调用整套流水线（分离→切分→布局→导出→Manifest）。
- 调用示例：
  ```python
  from audio_cut.api import separate_and_segment

  manifest = separate_and_segment(
      input_uri="input/song.mp3",
      export_dir="output/song_job",
      mode="v2.2_mdd",
      device="cuda:0",
      export_types=("vocal", "human_segments", "music_segments"),
      layout={"micro_merge_s": 2.0, "soft_min_s": 6.0, "soft_max_s": 18.0},
      strict_gpu=True,
      export_manifest=True,
  )
  ```
- Manifest 默认写入 `SegmentManifest.json`，包含音频哈希/时长、导出计划、切点与守卫统计、片段列表及导出资产相对路径（详见 `audio-cut封装为模块.md`）。
- `export_types` 控制导出资产：`vocal`（全量人声）、`instrumental`（全量伴奏）、`human_segments`（人声片段）、`music_segments`（混音片段）。未指定时默认导出全部。
- 返回结果中的 `export_plan` 字段会列出实际执行计划；如模型缺少伴奏分离会自动剔除 `instrumental` 并写入日志。

## 输出结构
- `segment_###_{human|music}_*.wav`：混音片段，文件名追加 `_X.X`（秒，保留 1 位小数）表示片段时长。
- `segment_###_{human|music}_lib_*.wav`：**节拍卡点片段**（hybrid_mdd 模式），结束切点对齐小节边界。
- `segments_vocal/segment_###_{human|music}_vocal_*.wav`：对应人声片段，同样追加 `_X.X` 时长后缀。
- `<stem>_v2.2_mdd_vocal_full_*.wav` / `<stem>_v2.2_mdd_instrumental_*.wav`：全长人声/伴奏文件。
- `segment_classification_debug`：调试信息（活跃度、阈值、投票），CLI 可自行持久化为 JSON。
- `segment_lib_flags`：标记哪些片段是节拍卡点片段（hybrid_mdd 模式）。
- 结果字典包含 `guard_shift_stats`、`guard_adjustments`、`gpu_meta` 等诊断信息。
- `vpbd_asr` Manifest 可选包含 `lyrics_alignment`、`boundary_detection`、`segments[*].lyrics` 与 `cuts.final[*].features`。
- 其他字段：`cut_points_samples/sec`、`guard_adjustments`、`suppressed_cut_points_sec` 等，用于验证切点一致性。

## 配置总览
主配置位于 `config/unified.yaml`（唯一配置入口），可通过 `VSS__...` 或 `AUDIOCUT_*` 环境变量覆盖。常用条目：
- `audio.*`：采样率（默认 44.1 kHz）、声道等。
- `gpu_pipeline.*`：Chunk 长度、overlap、halo、CUDA streams、inflight 限流、`strict_gpu` 等。
- `pure_vocal_detection.*`：
  - `peak_relative_threshold_ratio` 默认 0.26；
  - `rms_relative_threshold_ratio` 默认 0.30；
  - `relative_threshold_adaptation` / `pause_stats_adaptation` 基于 BPM/MDD/VPP 保持阈值稳定。
- `quality_control.*`：
  - `min_split_gap`：片段最小间隔；
  - `segment_vocal_activity_ratio`：阈值上调可减少误判；
  - `enforce_quiet_cut.*`：静音守卫参数（`guard_db`, `search_right_ms`）。
- `segment_layout.*`：`micro_merge_s`/`soft_min_s`/`soft_max_s`/`min_gap_s`/`beat_snap_ms` 控制微碎片合并与节拍吸附。
- `lyrics_alignment.*`：控制 `vpbd_asr` 是否启用、provider（disabled/fake/auto/sidecar/cli）、strict、ASR chunk 与 overlap、fake fixture。
- `fire_red.*`：控制 provider 顺序、sidecar endpoint/path、CLI executable/model_dir/timeout。
- `vpbd.*` / `phrase_boundary.*` / `global_planner.*`：控制 VPBD 模式开关、候选打分权重和全局切点规划约束；`vpbd.breath_score_scale` 只影响 VPBD 候选池，设为 `0` 可关闭气口候选；`vpbd.beat_candidates` 控制高能量段弱节拍候选；`phrase_boundary.word_edge_tolerance_ms` 软化词边缘惩罚，`global_planner.vocal_risk_weight` / `beat_conflict_weight` 控制风险降权。
- `vpbd_asr` 的长段二次分割遵循软约束：优先选择声学低谷，并用 ASR 句/唱段边界加权；找不到可信低谷时保留稍长片段，不使用 midpoint 硬切。Hybrid legacy helper 仍保留 midpoint fallback 以兼容原有节拍卡点行为。
- `hybrid_mdd` 参数：
  - `lib_alignment`：节拍对齐策略（推荐使用 quick_start.py 交互式选择）：
    - `beat_only`：强制节拍分割（副歌每小节切割，适合强节奏卡点）
    - `snap_to_beat`：MDD智能吸附到节拍（平衡方案，推荐）
  - `density`：卡点密度控制（low/medium/high）
  - `snap_tolerance_ms`：吸附容差（默认 200ms，仅 snap_to_beat 使用；运行时上限为 0.4 个 beat，避免横跨整拍吸附）
  - `vad_protection`：是否开启人声轨安静度保护；开启后 snap_to_beat、high 密度插入切点和 beat_only 的高能量小节切点都会避开活跃人声
  - `chorus_force_snap`：默认 `false`；设为 `true` 时恢复旧版副歌强吸附行为，适合只追求卡点但可能切入人声的回退场景
  - `energy_percentile`：副歌识别能量百分位（40=高密度，60=中密度，70=低密度）
- **副歌检测（v2.5.1）**：使用多特征融合算法（RMS能量 + 频谱质心 + 频谱带宽），根据歌曲动态范围自适应调整权重，有效提升民谣/爵士等低动态歌曲的准确度（误判率降低60-70%）。
- `output.*`：默认 `format: wav`；`wav.subtype`, `mp3.bitrate` 可单独配置；其他格式可在 `audio_export` 注册扩展。

## 调参指引
- **切点过少/片段过长**：降低 `pure_vocal_detection.peak_relative_threshold_ratio` 与 `rms_relative_threshold_ratio`；减小 `quality_control.min_split_gap`；调节 `valley_scoring.merge_close_ms`。
- **切点过多/片段碎化**：提升上述阈值；增大 `min_split_gap`；通过 `segment_min_duration` 限制最短片段。
- **静音守卫不稳定**：开启 `enforce_quiet_cut` 并逐步调整 `guard_db`/`search_right_ms`；检查输入是否被提前归一化。
- **VPBD ASR 切到歌词内部**：先检查 `boundary_detection.selected[*].source`、`meta.sources`、`features.inside_word_penalty` 与 `planner.final_time_by_raw_time`。`score=0` 的候选不会进入 rescue fallback；若 guard 把词外 raw cut 推入 ASR word interval，`vpbd_asr` 会恢复到 raw cut。
- **判定错误（伴奏被标成 human）**：查看 `segment_classification_debug` 中的活跃度；适当提升 `segment_vocal_activity_ratio`。

## 测试
- 快速回归：
  ```bash
  pytest -m "not slow and not gpu and not firered" --cov=src --cov-report=term-missing
  ```
- 重点单测：
  - `tests/unit/test_cpu_baseline_perfect_reconstruction.py`：样本级重建误差；
  - `tests/unit/test_cutting_refiner.py` / `test_cutting_consistency.py`：守卫与 NMS 行为；
  - `tests/unit/test_gpu_pipeline.py`：chunk 调度与 GPU 回退；
  - `tests/unit/test_chunk_feature_builder_*`：STFT 等价与跨块拼接；
  - `tests/unit/test_silero_chunk_vad.py`、`test_pure_vocal_focus_windows.py`：分块 VAD 与焦点窗口；
  - `tests/benchmarks/test_chunk_vs_full_equivalence.py`：chunk/full 误差报告；
  - `tests/integration/test_pipeline_v2_valley.py`：MDD 主流程；
  - `tests/contracts/test_config_contracts.py`：配置兼容契约；
  - `tests/unit/test_firered_*`、`tests/unit/test_run_splitter_cli.py`、`tests/unit/test_quick_start_vpbd.py`：v2.6 VPBD ASR provider 与入口契约。
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
- **2026-06-10 (v2.7 draft)**
  - VPBD 候选池开始接收气口候选，默认按 `vpbd.breath_score_scale=0.6` 降权；旧模式仍按原逻辑过滤 breath。
  - `vpbd_asr` 的 lyrics gap / sentence end / mVAD 边界与声学候选合并后统一打分规划，近重复候选在 ±120ms 内融合并在 `meta.sources` 留痕。
  - VPBD 新增高能量段弱节拍候选，默认每 2 小节生成一次，候选携带 `vocal_cut_risk` 供后续规划降权。
  - 修复 VPBD 死特征：`vocal_cut_risk`、`mdd_affinity`、`beat_conflict` 现在进入打分/规划链；句尾容差放宽到 250ms，词边缘惩罚按距离软化。
  - 删除未生效且会误伤低权重 soft prior 的 `phrase_boundary.min_score`。
  - `boundary_detection.candidate_counts` 新增 `merged` 计数，旧字段语义保持不变。
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

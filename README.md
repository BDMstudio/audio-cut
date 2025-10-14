<!-- File: README.md -->
<!-- AI-SUMMARY: Vocal Smart Splitter 的用户手册，涵盖特性、快速开始、配置与调参建议。 -->

# 智能人声分割器（Vocal Smart Splitter）

Vocal Smart Splitter 支持高保真声部拆分、纯人声检测，以及带 MDD（Musical Dynamic Density）守卫的一站式处理能力。自 v2.3 起统一使用 SeamlessSplitter 流水线，保持“一次检测、一次切分”的用户空间兼容性。

## 核心能力
- **双通道分离**：默认使用 MDX23 ONNX 输出人声/伴奏，失败时自动回退 Demucs v4（可配置关闭）。
- **GPU 多流分块流水线**：`audio_cut.utils.gpu_pipeline` 负责 chunk 规划、CUDA streams、pinned 缓冲与背压；`EnhancedVocalSeparator` 会记录 `gpu_meta`，并在 GPU 失败时安全回退 CPU。
- **纯人声检测**：`PureVocalPauseDetector` 结合 F0、共振峰、RMS 能量、MDD/BPM 自适应完成停顿判定，仅执行一次检测。
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
   - 第二步选择处理模式：`1` 仅分离；`2` Pure Vocal v2.2 MDD。
3. 命令行模式：
   ```bash
   python run_splitter.py input/your_song.mp3 --mode v2.2_mdd
   ```
   可按需追加 `--validate-reconstruction`、`--gpu-device cuda:1|cpu`、`--strict-gpu`、`--profile ...`、`--compat-config v2`。
4. 输出目录统一为 `output/<日期>_<时间>_<原音频名>/`（例如 `20241010_153045_song`），单文件与批处理遵循同一规则。

## 输出结构
- `segment_###_{human|music}_*.wav`：混音片段，文件名追加 `_X.X`（秒，保留 1 位小数）表示片段时长。
- `segments_vocal/segment_###_{human|music}_vocal_*.wav`：对应人声片段，同样追加 `_X.X` 时长后缀。
- `<stem>_v2.2_mdd_vocal_full_*.wav` / `<stem>_v2.2_mdd_instrumental_*.wav`：全长人声/伴奏文件。
- `segment_classification_debug`：调试信息（活跃度、阈值、投票），CLI 可自行持久化为 JSON。
- 结果字典包含 `guard_shift_stats`、`guard_adjustments`、`gpu_meta` 等诊断信息。
- 其他字段：`cut_points_samples/sec`、`guard_adjustments`、`suppressed_cut_points_sec` 等，用于验证切点一致性。

## 配置总览
主配置位于 `src/vocal_smart_splitter/config.yaml`，可通过 `VSS__...` 环境变量覆盖。常用条目：
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
- `output.*`：默认 `format: wav`；`wav.subtype`, `mp3.bitrate` 可单独配置；其他格式可在 `audio_export` 注册扩展。

## 调参指引
- **切点过少/片段过长**：降低 `pure_vocal_detection.peak_relative_threshold_ratio` 与 `rms_relative_threshold_ratio`；减小 `quality_control.min_split_gap`；调节 `valley_scoring.merge_close_ms`。
+- **切点过多/片段碎化**：提升上述阈值；增大 `min_split_gap`；通过 `segment_min_duration` 限制最短片段。
- **静音守卫不稳定**：开启 `enforce_quiet_cut` 并逐步调整 `guard_db`/`search_right_ms`；检查输入是否被提前归一化。
- **判定错误（伴奏被标成 human）**：查看 `segment_classification_debug` 中的活跃度；适当提升 `segment_vocal_activity_ratio`。

## 测试
- 快速回归：
  ```bash
  pytest -m "not slow and not gpu" --cov=src --cov-report=term-missing
  ```
- 重点单测：
  - `tests/unit/test_cpu_baseline_perfect_reconstruction.py`：样本级重建误差；
  - `tests/unit/test_cutting_refiner.py` / `test_cutting_consistency.py`：守卫与 NMS 行为；
  - `tests/unit/test_gpu_pipeline.py`：chunk 调度与 GPU 回退；
  - `tests/unit/test_chunk_feature_builder_*`：STFT 等价与跨块拼接；
  - `tests/unit/test_silero_chunk_vad.py`、`test_pure_vocal_focus_windows.py`：分块 VAD 与焦点窗口；
  - `tests/benchmarks/test_chunk_vs_full_equivalence.py`：chunk/full 误差报告；
  - `tests/integration/test_pipeline_v2_valley.py`：MDD 主流程；
  - `tests/contracts/test_config_contracts.py`：配置兼容契约。
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

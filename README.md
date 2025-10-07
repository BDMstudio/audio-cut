<!-- File: README.md -->
<!-- AI-SUMMARY: 智能人声分割器的用户说明，涵盖特性、快速开始、配置与调参建议。 -->

# 智能人声分割器（Vocal Smart Splitter）

支持高保真声部拆分、纯人声检测与 MDD (Musical Dynamic Density) 守卫的一站式工具。v2.3 统一使用 SeamlessSplitter 管线，保持「一次检测、一次切分」的用户空间兼容性。

## 核心能力
- **双通道分离**：默认使用 MDX23 ONNX 模型输出人声/伴奏；失败时自动回退 Demucs v4（可禁用）。
- **GPU 多流分块流水线**：`audio_cut.utils.gpu_pipeline` 调度 chunk 计划、CUDA streams、pinned 缓冲与背压；`EnhancedVocalSeparator` 自动记录 `gpu_meta` 并在 GPU 失败时回退 CPU。
- **纯人声检测**：PureVocalPauseDetector 结合 F0、共振峰、RMS 能量与 MDD/BPM 自适应，标定可切分区间。
- **守卫与补偿**：`audio_cut.cutting.refine.finalize_cut_points` 先在人声轨执行过零吸附+静音守卫，再在混音轨复用同一逻辑；SeamlessSplitter 汇总位移统计为 `guard_shift_stats`。
- **片段标注**：SeamlessSplitter 以 `segment_{###}_{human|music}` 命名片段，并在结果字典的 `segment_classification_debug` 中记录活跃度与判决原因；当前判决仅依赖 `vocal_activity_ratio` 阈值，marker/energy 投票字段暂作为调试占位输出 None。
- **高精度输出**：24-bit WAV，端到端验证 `tests/test_seamless_reconstruction.py` + 样本级基线 `tests/unit/test_cpu_baseline_perfect_reconstruction.py` 均约束拼接误差 <= 1e-12。
- **特征缓存复用**：`audio_cut.analysis.TrackFeatureCache` 在分离后构建一次，供 PureVocalPauseDetector、MDD 增强与守卫流程共享 BPM/MDD/RMS 序列。
- **分块 VAD 与焦点窗口**：`audio_cut.detectors.SileroChunkVAD` 合并跨块语音段并为 PureVocalPauseDetector 构建 ±pad 焦点窗口，只在关键区间运行昂贵特征分析。

## 快速开始
1. 将待处理音频（`mp3/wav/flac/m4a`）放入 `input/`。
2. 交互体验：
   ```bash
   python quick_start.py
   ```
   - `1`：仅分离人声/伴奏。
   - `2`：Pure Vocal v2.2 MDD（推荐）。
3. 批处理：
   ```bash
   python run_splitter.py input/your_song.mp3 --mode v2.2_mdd
   ```
  可加 `--validate-reconstruction` 请求拼接检查；但当前 `tests/test_seamless_reconstruction.py` 仍沿用旧返回结构，命令会抛出 KeyError，需先完成测试脚本重构。
  需要指定显卡时，可追加 `--gpu-device cuda:1`（或 `cpu`）；若希望 GPU 出错时直接失败，可追加 `--strict-gpu`。若需快速套用风格化调参，可附加 `--profile ballad|pop|edm|rap`。旧版兼容可使用 `--compat-config v2`（自动迁移 `config/default.yaml` 并带来 `meta.compat_config` 标记）。
4. 结果位于 `output/<job timestamp>/`。

> 初次运行前请安装依赖：`python -m venv env && env\Scripts\activate` → `pip install -e .[dev]`。

## 输出结构
- `segment_###_{human|music}.wav`：混音片段。
- `segments_vocal/segment_###_{human|music}_vocal.wav`：对应人声片段。
- `<stem>_v2.2_mdd_vocal_full.wav` / `<stem>_v2.2_mdd_instrumental.wav`：全长人声与伴奏。
- `segment_classification_debug` 以列表形式返回给调用者；CLI 如需 JSON 可自行落盘（当前脚本默认仅打印）。
- 结果字典包含 guard_shift_stats 汇总守卫位移统计，可按需打印或落盘分析。
- 额外返回 `cut_points_samples` / `cut_points_sec` 以及 `guard_adjustments`，便于校验块级与整段流程的一致性。
- 结果字典合并 `gpu_pipeline_*` 元数据（device/chunks/h2d_ms/dtoh_ms/peak_mem_bytes 等），失败时附带 `fallback_reason` 与 `gpu_pipeline_failures`。

## 配置总览
主配置：`src/vocal_smart_splitter/config.yaml`；ConfigManager 直接加载该 YAML，若需动态覆盖需通过 `set_runtime_config` 或自定义入口传参。

- Schema v3 精简方案位于 `src/audio_cut/config/schema_v3.yaml`，仅保留 `min_pause_s` / `min_gap_s` / `guard` / `threshold` / `adapt` / `nms` 等核心键；当传入 v3 文件或 Profile 时会自动派生全量 legacy 配置。
- 预设 Profile (`src/audio_cut/config/profiles/ballad|pop|edm|rap`) 可配合 `run_splitter --profile ...` 使用，处理结果会在 `meta.profile` 标记所选方案，便于回归记录。
- CPU 诊断兜底：`audio_cut.detectors.energy_gate.EnergyGateDetector` 仅用于缺少 GPU/Silero 时代替 VAD 的调试，不会在主路径自动启用。

- `audio.*`：采样率（默认 44.1 kHz）、单声道入参。
- `gpu_pipeline.*`：默认启用 CUDA 路径，控制 chunk/overlap/halo、pinned 缓冲数量、inflight 限制与 ORT Provider；`strict_gpu` 可让 GPU 失败直接报错。
- `pure_vocal_detection.*`
  - `enable_relative_energy_mode`: true 以 RMS/峰值比例快速过滤。
  - `relative_threshold_adaptation`: 基于 BPM + MDD 自动调整 `peak_relative_threshold_ratio`/`rms_relative_threshold_ratio`。
  - `pause_stats_adaptation`: VPP 统计，控制慢歌与快歌的闸值范围。
  - `valley_scoring.*`: 候选加权 NMS、`merge_close_ms`=300ms、防炸点。
- `quality_control.*`
  - `min_split_gap`: 1.0s（BPM 自适应可上调）。
  - `segment_vocal_activity_ratio`: 0.10 判定 `_human` 片段的活跃度下限。
  - `enforce_quiet_cut.enable`: 默认 false，开启后使用 `win_ms/guard_db/search_right_ms/floor_percentile` 执行静音守卫。
- `vocal_separation.*`: HPSS 预过滤参数；`enhanced_separation.backend` 的默认值在 `config_backup_original.yaml` 中可查。

基于不同分发环境，可通过 `FORCE_SEPARATION_BACKEND` 强制指定 `mdx23`/`demucs_v4`。

## 调参指引
- **切点过少 / 片段过长**
  - 降低 `pure_vocal_detection.peak_relative_threshold_ratio` (例如 0.22) 与 `rms_relative_threshold_ratio` (0.28)。
  - 下调 `quality_control.min_split_gap` 至 0.9s。
  - 调整 `pure_vocal_detection.valley_scoring.merge_close_ms` 到 150–200ms。
- **切点过多 / 片段过碎**
  - 提升上述阈值（峰值 0.30、RMS 0.34）。
  - 提高 `min_split_gap` 至 1.2–1.3s，并将 `segment_min_duration` 设为 6–7s。
- **静音切点不干净**
  - 启用 `enforce_quiet_cut.enable: true`，并根据素材调节 `guard_db` 2.0–3.0、`search_right_ms` 150–200。
  - 确认输入素材未被额外规范化；主流程在 `_load_and_resample_if_needed` 中以 `normalize=False` 调用 `AudioProcessor.load_audio`，而若独立使用该类其默认仍会归一化。
- **判定错误（伴奏被标成 human）**
  - 观察 `segment_classification_debug` 中 `vocal_activity_ratio` 与阈值差距。
  - 需要更严格判定时，可提升 `quality_control.segment_vocal_activity_ratio` 至 0.12–0.15。

## 测试
- 核心快速回归：
  ```bash
  pytest -m "not slow and not gpu" --cov=src --cov-report=term-missing
  ```


- 关键用例：
  - `tests/unit/test_cpu_baseline_perfect_reconstruction.py`：样本级零误差基线。
  - `tests/unit/test_cutting_refiner.py`：切点精炼器的 NMS/守卫/过零行为。
  - `tests/unit/test_cutting_consistency.py`：块级候选与整段基准的时间/守卫偏差约束。
  - `tests/unit/test_cut_alignment.py`：守卫与过零逻辑。
  - `tests/unit/test_gpu_pipeline.py`：验证 chunk 调度、背压与 GPU fallback。
  - `tests/unit/test_chunk_feature_builder_gpu.py`：GPU STFT 缓存与跨块拼接的一致性。
  - `tests/unit/test_chunk_feature_builder_stft_equivalence.py`：分块 STFT 与整段 STFT 数值等价。
  - `tests/unit/test_silero_chunk_vad.py`：Silero 分块合并与焦点窗口行为。
  - `tests/unit/test_pure_vocal_focus_windows.py`：VPP 焦点窗口聚合与阈值裁剪。
  - `tests/benchmarks/test_chunk_vs_full_equivalence.py`：chunk vs full 误差报告。
  - `tests/integration/test_pipeline_v2_valley.py`：v2.2 管线集成。
  - `tests/contracts/test_config_contracts.py`：配置兼容契约。
- 待补充：
  - `tests/test_seamless_reconstruction.py`：旧版验证脚本尚未对接 v2.3 结果结构，运行前需先完成重构。

### 性能基线
- GPU vs CPU 对比（吞吐、H2D/DtoH、显存峰值）：
  ```bash
  python scripts/bench/run_gpu_cpu_baseline.py input/your_song.wav --write-markdown
  ```
  结果写入 `output/bench/<timestamp>/gpu_cpu_baseline.json|.md`，可检视 `speedup_ratio` 与 `gpu_pipeline_*` 元数据。
- 多卡分配验证：
  ```bash
  python scripts/bench/run_multi_gpu_probe.py input/your_song.wav --mode v2.2_mdd
  ```
  生成 `multi_gpu_probe_*.json`，逐卡记录 `gpu_pipeline_*`、NVML/nvidia-smi 指标与运行耗时，便于多 GPU 环境验收。

## 更新记录

- **2025-10-04**
  - 更正 chunk vs full 基线描述，明确 dummy/真实模型断言阈值，并同步至 development/todo。
  - 补充 cutting.metrics 工具链说明，保持架构文档一致。

- **2025-10-03**
  - 补充 chunk vs full 真实模型基准、STFT 等价与焦点窗口单测说明，对齐 	odo.md 状态。
  - 新增 scripts/bench/README_gpu_pipeline.md 使用指引，并在 README/development.md 中同步引用。
- **2025-09-27**
  - 增补 GPU 流水线/分块 VAD 能力、结果元数据与配置说明。
  - 更新测试清单，纳入 GPU/分块相关用例。
- **2025-09-26**
  - 文档全面重写，修正编码问题，补充 v2.3 架构描述。
  - 增补调参建议与输出结构说明，对齐 `config.yaml` 默认值。
- **2025-09-14**
  - VPP 单次判决化：去除二次守卫/补刀流程，保持一次检测。
  - README 添加 BPM/MDD 调参示例，默认守卫窗口 80ms。

## 使用声明
本项目仅供技术研究与个人实验，涉及商用请自行确认版权与法律合规。

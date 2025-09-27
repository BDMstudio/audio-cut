<!-- File: README.md -->
<!-- AI-SUMMARY: 智能人声分割器的用户说明，涵盖特性、快速开始、配置与调参建议。 -->

# 智能人声分割器（Vocal Smart Splitter）

支持高保真声部拆分、纯人声检测与 MDD (Musical Dynamic Density) 守卫的一站式工具。v2.3 统一使用 SeamlessSplitter 管线，保持「一次检测、一次切分」的用户空间兼容性。

## 核心能力
- **双通道分离**：默认使用 MDX23 ONNX 模型输出人声/伴奏；失败时自动回退 Demucs v4（可禁用）。
- **纯人声检测**：PureVocalPauseDetector 结合 F0、共振峰、RMS 能量与 MDD/BPM 自适应，标定可切分区间。
- **守卫与补偿**：`audio_cut.cutting.refine.finalize_cut_points` 先在人声轨执行过零吸附+静音守卫，再在混音轨复用同一逻辑；SeamlessSplitter 汇总位移统计为 `guard_shift_stats`。
- **片段标注**：SeamlessSplitter 以 `segment_{###}_{human|music}` 命名片段，并在结果字典的 `segment_classification_debug` 中记录活跃度与判决原因。
- **高精度输出**：24-bit WAV，无损复原测试 `tests/test_seamless_reconstruction.py` 保证拼接误差 < 1e-8。
- **特征缓存复用**：`audio_cut.analysis.TrackFeatureCache` 在分离后构建一次，供 PureVocalPauseDetector、MDD 增强与守卫流程共享 BPM/MDD/RMS 序列。

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
   可加 `--validate-reconstruction` 触发拼接完整性校验。
4. 结果位于 `output/<job timestamp>/`。

> 初次运行前请安装依赖：`python -m venv env && env\Scripts\activate` → `pip install -e .[dev]`。

## 输出结构
- `segment_###_{human|music}.wav`：混音片段。
- `segments_vocal/segment_###_{human|music}_vocal.wav`：对应人声片段。
- `<stem>_v2.2_mdd_vocal_full.wav` / `<stem>_v2.2_mdd_instrumental.wav`：全长人声与伴奏。
- `segment_classification_debug` 以列表形式返回给调用者；CLI 如需 JSON 可自行落盘（当前脚本默认仅打印）。
- 控制台输出 `guard_shift_stats` 合并统计守卫位移情况。

## 配置总览
主配置：`src/vocal_smart_splitter/config.yaml`；ConfigManager 直接加载该 YAML，若需动态覆盖需通过 `set_runtime_config` 或自定义入口传参。

- `audio.*`：采样率（默认 44.1 kHz）、单声道入参。
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
  - 确认输入素材未被额外规范化；`AudioProcessor.load_audio` 默认不归一化。
- **判定错误（伴奏被标成 human）**
  - 观察 `segment_classification_debug` 中 `vocal_activity_ratio` 与阈值差距。
  - 需要更严格判定时，可提升 `quality_control.segment_vocal_activity_ratio` 至 0.12–0.15。

## 测试
- 核心快速回归：
  ```bash
  pytest -m "not slow and not gpu" --cov=src --cov-report=term-missing
  ```
- 关键用例：
  - `tests/unit/test_seamless_reconstruction.py`：拼接零损耗基线。
  - `tests/unit/test_cut_alignment.py`：守卫与过零逻辑。
  - `tests/integration/test_pipeline_v2_valley.py`：v2.2 管线集成。
  - `tests/contracts/test_config_contracts.py`：配置兼容契约。

## 更新记录
- **2025-09-26**
  - 文档全面重写，修正编码问题，补充 v2.3 架构描述。
  - 增补调参建议与输出结构说明，对齐 `config.yaml` 默认值。
- **2025-09-14**
  - VPP 单次判决化：去除二次守卫/补刀流程，保持一次检测。
  - README 添加 BPM/MDD 调参示例，默认守卫窗口 80ms。

## 使用声明
本项目仅供技术研究与个人实验，涉及商用请自行确认版权与法律合规。

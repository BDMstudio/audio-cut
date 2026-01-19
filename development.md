<!-- File: development.md -->
<!-- AI-SUMMARY: 记录 Vocal Smart Splitter 的架构、流程、测试矩阵与演进规划。 -->

# development.md — 技术路线与模块总览（更新于 2026-01-18）

本文档是工程事实的单一可信来源（SSOT），持续记录系统架构、实现约束与进度。所有涉及流程、参数或测试的改动，须同步更新此处。

## 1. 版本演进

- **v2.5.1（2026-01-18）**: 多特征副歌检测（能量+频谱融合，自适应权重），移除 `mdd_start` 策略，交互式策略选择，连续性检测增强。
- **v2.5.0（2026-01-17）**: 新增 `hybrid_mdd` 模式（MDD + 节拍卡点增强），支持 `_lib` 后缀标记、密度控制、预过滤短片段。
- **v2.4.1（2026-01-17）**: 删除未生效算法 (`enable_bpm_adaptation`, `interlude_coverage_check`)，清理冗余配置。
- **v2.4（2026-01-17）**: 统一配置入口 `config/unified.yaml`，新增 `librosa_onset` 模式。
- **v2.3（2025-09-26）**：SeamlessSplitter 成为唯一入口；结果调试（`segment_classification_debug`、`guard_shift_stats`）结构化。
- **v2.2（2025-09-12）**：Pure Vocal + MDD 合流，确立“一次检测 + NMS + 守卫”策略。

## 2. 目录职责
- `src/vocal_smart_splitter/core/`：主流程组件（SeamlessSplitter、PureVocalPauseDetector、EnhancedVocalSeparator、VocalPauseDetectorV2）；QualityController 保留为 legacy 兜底。
- `src/vocal_smart_splitter/core/utils/`：`SegmentExporter` 与 `ResultBuilder` 等编排辅助工具。
- `src/vocal_smart_splitter/utils/`：音频 IO、配置优先级、BPM 自适应参数、特征抽取。
- `src/audio_cut/analysis/`：`TrackFeatureCache` 及构建器，集中缓存 BPM/MDD/RMS 等特征。
- `src/audio_cut/utils/`：GPU 流水线（chunk 规划、CUDA streams、pinned buffer、inflight 限流）与 ORT Provider 注入。
- `src/audio_cut/api.py`：对外统一 API，封装 `SeamlessSplitter`，生成 Manifest 并管理导出计划。
- `src/audio_cut/detectors/`：Silero 分块 VAD 及兼容层。
- `src/audio_cut/cutting/`：CutPoint/CutContext 与切点精修，提供 NMS、过零、静音守卫、chunk vs full metric。
- `scripts/`：运行入口与诊断脚本（`quick_start.py`、`run_splitter.py`、bench 工具）。
- `tests/`：分层测试（unit / integration / contracts / performance / sanity）。
- `config/`：默认配置与 schema；严禁提交个人实验参数。

## 3. 核心流程
0. `audio_cut.api.separate_and_segment` 在上层项目中聚合资源配置、调用 `SeamlessSplitter` 并生成 Manifest。
1. `AudioProcessor.load_audio` 读取音频并默认归一化至 [-1, 1]，必要时重采样至 44.1 kHz。
2. `EnhancedVocalSeparator.separate_for_detection` 构造 `PipelineContext`，规划 chunk/overlap/halo，GPU 模式记录 `gpu_meta`，失败时回退 CPU。
3. `SileroChunkVAD.process_chunk` 进行分块 VAD 和 halo 裁剪；`ChunkFeatureBuilder` 在 GPU 缓存 STFT/RMS 供后续复用。
4. `PureVocalPauseDetector.detect_pure_vocal_pauses` 使用焦点窗口与特征缓存，在相对能量模式下结合 BPM/MDD/VPP 自适应判定停顿。
5. `audio_cut.cutting.finalize_cut_points` 对候选执行加权 NMS、静音守卫、最小间隔，输出守卫位移统计。
6. `SeamlessSplitter._classify_segments_vocal_presence` 根据 RMS 活跃度估计 `_human/_music` 标签并记录调试信息。
7. `segment_layout_refiner.refine_layout` 执行微碎片合并、软最小合并、软最大救援；如开启 `quality_control.local_boundary_refine`，再次细调边界。
8. `SegmentExporter` 统一调用 `audio_export` 模块导出文件，默认追加 `_X.X`（秒，保留一位小数）后缀；落盘目录按 `<日期>_<时间>_<原音频名>` 命名。

## 4. 核心模块要点
- **SeamlessSplitter**：统一调度入口，缓存 `segment_classification_debug`、`guard_shift_stats`、守卫调整明细，确保 GPU chunk 与整段流程可追踪。
- **EnhancedVocalSeparator**：封装 MDX23/Demucs 后端，记录 `h2d_ms/dtoh_ms/compute_ms/peak_mem_bytes`，提供 `fallback_reason`。
- **GPU Pipeline**：`PipelineConfig`/`PipelineContext` 管理 chunk 规划、CUDA stream、pinned buffer 与背压。
- **SileroChunkVAD**：分块推理 + halo 裁剪 + 焦点窗口构造，仅在关键区间运行昂贵特征。
- **ChunkFeatureBuilder/TrackFeatureCache**：集中管理 STFT/RMS/MDD 等特征，支持 GPU 批量计算与跨块拼接。
- **SegmentExporter/ResultBuilder**：统一导出与结果字典构建，减少重复逻辑与手工拼接字段。
- **segment_layout_refiner**：微碎片合并、软最小合并、软最大救援，复用 NMS 被抑制的 cut point；配合 `_last_suppressed_cut_points` 提升布局质量。
- **输出目录策略**：`quick_start.py` 与 `run_splitter.py` 均使用 `<日期>_<时间>_<原音频名>` 创建输出目录，便于批量回归与部署一致。

## 5. 配置与参数策略
- `config_manager.get_config 默认加载 `config/unified.yaml`（唯一配置入口），支持 `VSS__...` 环境变量重写。
- `pure_vocal_detection` 默认 `peak_relative_threshold_ratio=0.26`、`rms_relative_threshold_ratio=0.30`，BPM/MDD/VPP 自适应缩放范围 0.85–1.15。
- `quality_control` 提供 `min_split_gap`、`segment_vocal_activity_ratio` 等守护阈值；`enforce_quiet_cut` 注重静音守卫参数（`guard_db`, `search_right_ms`）。
- `segment_layout` 默认启用，`micro_merge_s` / `soft_min_s` / `soft_max_s` / `min_gap_s` 控制碎片合并策略；若更改需同步 doc/CLI。
- `output.format` 默认 `wav`，可通过 `output.mp3.bitrate` 调整 MP3 输出；`audio_export` 模块负责统一写入。

## 6. 测试矩阵
- **Unit**：`test_cpu_baseline_perfect_reconstruction`、`test_cutting_consistency`、`test_segment_labeling`、`test_gpu_pipeline`、`test_chunk_feature_builder_gpu/stft_equivalence` 等覆盖核心算法。
- `tests/unit/test_api_manifest.py`：校验模块化 API 的 Manifest 输出与导出计划控制。
- **Integration**：`tests/integration/test_pipeline_v2_valley.py` 验证 MDD 主路径；批处理场景计划新增。
- **Contracts**：`tests/contracts/test_config_contracts.py` 保证配置兼容；待补输出命名回归。
- **Performance**：`tests/performance/test_valley_perf.py` 监控检测+守卫耗时。
- **Benchmarks**：`tests/benchmarks/test_chunk_vs_full_equivalence.py` 分析 chunk vs full 误差。
- **Sanity**：`tests/sanity/ort_mdx23_cuda_sanity.py` 自检 GPU Provider。
- 标准要求：新增能力必须补齐相应测试层；`quick_start` 批处理逻辑需结合集成测试验证。

## 7. 性能与复杂度基线
- 分离阶段：MDX23 GPU 目标 ≥0.7x 实时，记录 `h2d_ms/dtoh_ms/compute_ms/peak_mem_bytes`；CPU 回退约 3.5x 实时。
- 检测 + 守卫：处理 10 分钟素材约 12s；启用静音守卫额外增加 ~8%。
- Chunk vs Full：dummy 模型误差 <1e-6，真实模型断言 `L∞<5e-3`、`SNR>60dB`。
- 拼接误差：`test_cpu_baseline_perfect_reconstruction` 要求最大绝对误差 ≤1e-12。
- 性能脚本：`python scripts/bench/run_gpu_cpu_baseline.py`、`python scripts/bench/run_multi_gpu_probe.py` 输出性能报告。

## 8. 当前进展与下一步
- **已完成 (v2.5.1 - 2026-01-18)**：
  - **多特征副歌检测算法**：
    - 实现 RMS能量 + 频谱质心 + 频谱带宽三特征融合
    - 基于能量变异系数(CV)的自适应权重机制（低动态侧重频谱，高动态侧重能量）
    - 连续性检测：要求至少连续4小节高能量才识别为副歌
    - 民谣/爵士等低动态歌曲准确度提升60-70%，流行歌曲保持稳定
  - **移除 `mdd_start` 策略**：保留 `beat_only` 和 `snap_to_beat` 两种策略，简化选择
  - **交互式策略选择**：`quick_start.py` 新增 lib_alignment 策略选择菜单（beat_only/snap_to_beat）
  - **BeatAnalyzer 增强**：新增 `bar_spectral_centroids` 和 `bar_spectral_bandwidths` 特征计算
  - **SegmentationContext 扩展**：支持传递频谱特征到策略层
- **已完成 (v2.5.0)**：
  - GPU 多流流水线（streams / pinned buffer / inflight limiter）
  - Silero 分块 VAD、ChunkFeatureBuilder GPU 缓存
  - `segment_layout_refiner` 接入主流程，并统一 `_X.X` 时长后缀
  - 输出目录统一为 `<日期>_<时间>_<原音频名>`
  - `hybrid_mdd` 模式实现：MDD + librosa 节拍卡点，`_lib` 后缀标记，密度控制
  - 预过滤算法：节拍切点添加前检查是否会产生短片段
  - Strategy 模式重构：新增 `strategies/` 目录，实现 `SegmentationStrategy` 基类
  - SeamlessSplitter 重构：BeatAnalyzer/SegmentExporter/ResultBuilder 接入
- **设计文档**：
  - `docs/hybrid_mdd_design.md` - 切点策略方案对比
  - `docs/SeamlessSplitter 重构记录.md` - 重构评估报告
- **待规划**：
  - 副歌检测阶段2：重复结构检测、MFCC变化率特征（提升至85-90%准确度）
  - `seamless_splitter.py` 进一步模块拆分（analyzers）
  - IO Binding / TensorRT / FP16 支持
  - `tests/test_seamless_reconstruction.py` 适配 v2.5 结果结构

## 9. 环境与工具
- Python 3.10+；核心依赖：PyTorch、librosa、numpy/scipy/soundfile、pydub（MP3 导出需 FFmpeg）。
- `pip install -e .[dev]` 安装开发依赖（pytest/black/flake8 等）。
- Windows + PowerShell 为默认环境，WSL / Linux 同样支持。
- 外部模型：`MVSEP-MDX23-music-separation-model/`，需确认路径。
- CLI 示例：
  ```bash
  python quick_start.py                   # 交互式单/批处理
  python run_splitter.py input/song.mp3   # CLI 模式
  ```

> 若修改输出结构、文件命名或调试字段，务必同步更新 README 与本文件，保持文档与实现一致。

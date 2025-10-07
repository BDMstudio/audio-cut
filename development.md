<!-- File: development.md -->
<!-- AI-SUMMARY: 记录 Vocal Smart Splitter 的架构、流程、测试矩阵与近期演进。 -->

# development.md — 技术路线与模块总览（更新于 2025-10-04）

本文是工程事实来源（SSOT），记录架构、流程与约束。

## 1. 版本演进
- v1.x：混音 -> BPM/动态范围 -> Silero VAD -> 阈值切分（遗留，仅用于比较）。
- v2.0：混音 -> MDX23/Demucs 分离 -> 人声静区检索 -> 切分；首次引入长音保护。
- v2.1：VocalPrime（VMS/EMA）实验版，提供更细粒度 F0/共振峰分析，仍保留旧守卫链路。
- v2.2（2025-09-12）：Pure Vocal + MDD 合流，统一成“一次检测 + NMS + 守卫”的策略。
- v2.3（2025-09-26）：SeamlessSplitter 收敛为单入口；守卫统计、片段标注落盘；`audio_cut.*` 拆出通用特征缓存与切点精修。

## 2. 目录与职责
- src/vocal_smart_splitter/core：主流程（SeamlessSplitter、PureVocalPauseDetector、EnhancedVocalSeparator、VocalPauseDetectorV2 等）；QualityController 现保留为 legacy 兜底（默认未接入主路径）。
- src/vocal_smart_splitter/utils：IO、配置优先级、BPM 自适应参数计算、特征抽取。
- src/audio_cut/analysis：TrackFeatureCache，缓存 BPM/MDD/RMS 等序列，主流程分离后构建一次并向检测/守卫复用。
- src/audio_cut/utils：GPU 流水线（chunk 计划、CUDA streams、pinned 缓冲、inflight 限流）与 ORT 依赖注入。
- src/audio_cut/detectors：Silero 分块 VAD、检测器适配层。
- src/audio_cut/cutting：CutPoint/CutContext/切点精修，提供过零、静音守卫、min-gap NMS 的独立实现，并新增 metrics 工具衡量 chunk vs full 时间误差，现为 `_finalize_and_filter_cuts_v2` 的唯一实现，便于复用与测试。
- scripts/：快速诊断脚本（quick_start、run_splitter、debug 工具）。
- tests/：unit / integration / contracts / performance 分层。
- config/：预置 YAML，避免在主配置中留下实验参数。

## 3. 核心流程
- (1) `AudioProcessor.load_audio` 读取并保持原始振幅，必要时重采样至 44.1 kHz。
- (2) `EnhancedVocalSeparator.separate_for_detection` 基于 `PipelineConfig` 构建 GPU/CPU `PipelineContext`：规划 chunk/overlap/halo，启用 CUDA streams、pinned 缓冲与 inflight 限流；GPU 失败时在 `gpu_meta` 中记录 `fallback_reason` 并回退 CPU。
- (3) `SileroChunkVAD.process_chunk` 按 chunk 计划执行 Silero 推理，剪裁 halo 区并合并跨块时间段；同时 `ChunkFeatureBuilder` 在 GPU 上缓存 STFT/RMS 序列以供后续检测复用。
- (4) `PureVocalPauseDetector.detect_pure_vocal_pauses` 复用焦点窗口与特征缓存运行相对能量 + MDD/BPM/VPP 自适应，必要时退回全特征评估。
- (5) `audio_cut.cutting.finalize_cut_points` 承接候选，先按权重执行 NMS，再在人声/混音轨应用静音守卫、最小间隔并生成守卫位移统计。
- (6) `SeamlessSplitter._classify_segments_vocal_presence` 以 RMS 活跃度估算 `_human/_music` 标签；marker/energy 投票字段保留占位用于后续扩展。
- (7) `_save_segments` 写出 24-bit WAV、完整人声/伴奏，并输出 `cut_points_*`、`guard_adjustments` 与 `gpu_meta` 诊断信息。
## 4. 核心模块要点
- SeamlessSplitter：统一入口；缓存 `segment_classification_debug`、`guard_shift_stats` 与每个 `guard_adjustment`，确保 GPU 块级与整段流程可量化对齐。
- EnhancedVocalSeparator：封装 MDX23/Demucs，解析 `PipelineConfig` 并在 GPU 模式下累计 `h2d_ms/dtoh_ms/compute_ms/peak_mem_bytes`，通过 `gpu_meta` 向外暴露诊断信息。
- audio_cut.utils.gpu_pipeline：`PipelineConfig`/`PipelineContext`/`Streams`/`PinnedBufferPool`/`InflightLimiter` 提供 chunk 规划、CUDA stream 管理、pinned 缓冲与背压。
- SileroChunkVAD：按 chunk 推理并合并跨块语音段，可生成 ±pad 焦点窗口供纯人声检测复用。
- ChunkFeatureBuilder/TrackFeatureCache：集中计算 STFT、RMS、MDD 等特征，支持 GPU 批量计算与跨块拼接。
- PureVocalPauseDetector：在相对能量模式下结合 BPM/MDD/VPP 自适应；复杂模式含 F0、共振峰、谱质心、谐波比率等特征。
- QualityController：legacy 守卫/过零吸附模块，当前未接入 SeamlessSplitter 主路径，仅在单元测试内维持兜底能力。
- AdaptiveVADEnhancer/BPMAnalyzer：抽取 BPM、动态密度、节拍指标，为 pause_stats 与 quality_controller 提供上下文。
- audio_cut.cutting/refine：NMS + 过零 + 静音守卫的割点库，是 `_finalize_and_filter_cuts_v2` 的唯一实现。
## 5. 配置与参数策略
- `config_manager.get_config` 直接读取 YAML；如需覆盖需调用 `set_runtime_config` 或注入自定义 ConfigManager。
- 相对能量阈值默认 0.26/0.32，并通过 BPM/MDD/VPP clamp 在 0.85–1.15 范围。
- 守卫默认关闭；当项目需要零爆音保障时，应同步开启 `quality_control.enforce_quiet_cut.enable` 与 `save_analysis_report` 便于验收。
- `segment_vocal_activity_ratio` 设 0.10，若测试集中出现误判，应通过单元测试验证新的阈值后再调整。
- Schema v3：`src/audio_cut/config/schema_v3.yaml` + `config/profiles/*` 只保留 6–8 个核心键，通过 `audio_cut.config.derive` 自动派生 legacy 配置并写入 `meta.schema_version/profile`，迁移脚本 `config/migrate_v2_to_v3.py` 用于将旧 YAML 映射到新格式。
- CPU 兜底：`audio_cut.detectors.energy_gate` 作为纯能量门控诊断工具保留，默认关闭，仅在 Silero 不可用或 CI CPU 验证时手动调用。
- 兼容模式：`run_splitter --compat-config v2` 会自动迁移 `config/default.yaml`，并在运行结果的 `meta.compat_config` 标记来源，确保旧部署可平滑过渡一个版本周期。

## 6. 测试矩阵
- unit：`test_cut_alignment`, `test_cutting_consistency`, `test_segment_labeling`, `test_pre_vocal_split`, `test_gpu_pipeline`, `test_chunk_feature_builder_gpu`, `test_chunk_feature_builder_stft_equivalence`, `test_silero_chunk_vad`, `test_pure_vocal_focus_windows`, `test_track_feature_cache`, `test_mdx23_path_resolution` 等覆盖算法细节与配置解析。
- integration：`test_pipeline_v2_valley.py`、`tests/test_pure_vocal_detection_v2.py` 验证全流程与主要模式。
- contracts：`test_config_contracts.py`、`tests/contracts/test_valley_contract.py` 保证配置契约。
- benchmarks：`tests/benchmarks/test_chunk_vs_full_equivalence.py` 输出 chunk vs full 误差报告。
- performance：`tests/performance/test_valley_perf.py` 监控 MDD+VPP 耗时。
- sanity：`tests/sanity/ort_mdx23_cuda_sanity.py` 自检 GPU Provider。
- 慢测试与 GPU case 使用 `@pytest.mark.slow`、`@pytest.mark.gpu` 标记，CI 默认跳过。
## 7. 性能与复杂度基线
- 分离阶段：MDX23 GPU 路径目标≥0.7x 实时，并记录 `h2d_ms/dtoh_ms/compute_ms/peak_mem_bytes`；CPU 回退约 3.5x 实时。
- 检测 + 守卫：10 分钟素材在单核下约 12s；开启静音守卫会额外增加 ~8%。
- Chunk vs Full：`tests/benchmarks/test_chunk_vs_full_equivalence.py` 对 dummy 后端约束波形误差 <1e-6、SNR >80 dB；真实模型用例断言 L∞<5e-3、L2<1e-3、相对 L1<5e-3、SNR >60 dB。
- 拼接误差：`test_cpu_baseline_perfect_reconstruction` 要求最大绝对误差 <= 1e-12。
- 性能脚本：`python scripts/bench/run_gpu_cpu_baseline.py` 对比吞吐/H2D/DtoH，`python scripts/bench/run_multi_gpu_probe.py` 记录逐卡指标。
## 8. 当前进展与下一步
- 已完成：GPU 多流流水线（streams/pinned/inflight）、Silero 分块 VAD、ChunkFeatureBuilder GPU 缓存、守卫统计，以及 chunk vs full 真实模型基准与文档同步；`audio_cut.cutting.finalize_cut_points` 接管精炼主路径。
- 进行中：同类型母带回放基线、`--strict-gpu` 策略与质量守则对照表（Milestone 2-G1/G2）。
- 待规划：IO Binding / TensorRT / FP16 优化与运行时参数监控工具。
## 9. 环境与工具
- Python 3.10+，PyTorch + librosa + numpy/scipy/soundfile。
- `pip install -e .[dev]` 提供开发依赖（pytest/black/flake8）。
- Windows + PowerShell/WSL 均可运行；外部模型位于 `MVSEP-MDX23-music-separation-model/`。


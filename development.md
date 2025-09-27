<!-- File: development.md -->
<!-- AI-SUMMARY: 记录 Vocal Smart Splitter 的架构、流程、测试矩阵与近期演进。 -->

# development.md — 技术路线与模块总览（更新于 2025-09-26）

本文是工程事实来源（SSOT），记录架构、流程与约束。

## 1. 版本演进
- v1.x：混音 -> BPM/动态范围 -> Silero VAD -> 阈值切分（遗留，仅用于比较）。
- v2.0：混音 -> MDX23/Demucs 分离 -> 人声静区检索 -> 切分；首次引入长音保护。
- v2.1：VocalPrime（VMS/EMA）实验版，提供更细粒度 F0/共振峰分析，仍保留旧守卫链路。
- v2.2（2025-09-12）：Pure Vocal + MDD 合流，统一成“一次检测 + NMS + 守卫”的策略。
- v2.3（2025-09-26）：SeamlessSplitter 收敛为单入口；守卫统计、片段标注落盘；`audio_cut.*` 拆出通用特征缓存与切点精修。

## 2. 目录与职责
- src/vocal_smart_splitter/core：主流程（SeamlessSplitter、PureVocalPauseDetector、QualityController、EnhancedVocalSeparator、VocalPauseDetectorV2 等）。
- src/vocal_smart_splitter/utils：IO、配置优先级、BPM 自适应参数计算、特征抽取。
- src/audio_cut/analysis：TrackFeatureCache，缓存 BPM/MDD/RMS 等序列，主流程分离后构建一次并向检测/守卫复用。
- src/audio_cut/cutting：CutPoint/CutContext/切点精修，提供过零、静音守卫、min-gap NMS 的独立实现，现为 `_finalize_and_filter_cuts_v2` 的唯一实现，便于复用与测试。
- scripts/：快速诊断脚本（quick_start、run_splitter、debug 工具）。
- tests/：unit / integration / contracts / performance 分层。
- config/：预置 YAML，避免在主配置中留下实验参数。

## 3. 核心流程
- (1) `AudioProcessor.load_audio` 读取并保持原始振幅，必要时重采样到 44.1 kHz。
- (2) `EnhancedVocalSeparator.separate_for_detection` 先行分离人声/伴奏，记录后端与置信度。
- (3) `PureVocalPauseDetector.detect_pure_vocal_pauses` 计算候选静区：先走相对能量模式（默认），再视需要回退到全特征评估；BPM/MDD/VPP 自适应在此执行，并优先从 `TrackFeatureCache` 读取全局特征。
- (4) `SeamlessSplitter._finalize_and_filter_cuts_v2` 调用 `audio_cut.cutting.finalize_cut_points`，先执行加权 NMS，再在人声/混音轨套用守卫并做最小间隔过滤。
- (5) `SeamlessSplitter._classify_segments_vocal_presence` 依据 RMS 活跃度、阈值和辅助标记判断 `_human/_music`。
- (6) `QualityController` 输出守卫位移统计；`_save_segments` 落盘 24-bit WAV 与调试信息。

## 4. 核心模块要点
- SeamlessSplitter：统一入口；缓存 `segment_classification_debug`、`guard_shift_stats`；负责落盘，并在分离后构建 `TrackFeatureCache`。
- EnhancedVocalSeparator：封装 MDX23/Demucs，支持环境变量强制后端、失败回退、keep-in-memory。
- PureVocalPauseDetector：在相对能量模式下使用峰值/RMS 比例、BPM/MDD/VPP 自适应补偿；复杂模式含 F0、共振峰、谱质心、谐波比率等特征。
- QualityController：保留静音守卫/过零吸附的兼容实现，用于 fallback 与质量度量；`AdaptiveParameterCalculator` 仍提供 BPM 级别自适应参数。
- AdaptiveVADEnhancer/BPMAnalyzer：抽取 BPM、动态密度、节拍信息，为 pause_stats 与 quality_controller 提供上下文。
- audio_cut.analysis/features_cache：集中计算 RMS、谱平坦度、onset 等序列，避免多次重复运算；已在 SeamlessSplitter → PureVocalPauseDetector → VocalPauseDetector 流程中落地复用。
- audio_cut.cutting/refine：NMS + 过零 + 静音守卫的割点库，已托管 `_finalize_and_filter_cuts_v2` 的主要实现，便于复用与测试。

## 5. 配置与参数策略
- `config_manager.get_config` 直接读取 YAML；如需覆盖需调用 `set_runtime_config` 或注入自定义 ConfigManager。
- 相对能量阈值默认 0.26/0.32，并通过 BPM/MDD/VPP clamp 在 0.85–1.15 范围。
- 守卫默认关闭；当项目需要零爆音保障时，应同步开启 `quality_control.enforce_quiet_cut.enable` 与 `save_analysis_report` 便于验收。
- `segment_vocal_activity_ratio` 设 0.10，若测试集中出现误判，应通过单元测试验证新的阈值后再调整。

## 6. 测试矩阵
- unit：`test_cut_alignment`, `test_segment_labeling`, `test_pre_vocal_split`, `test_cpu_baseline_perfect_reconstruction` 等，保证算法细节。
- integration：`test_pipeline_v2_valley.py` 跑通 v2.2 全流程。
- contracts：配置文件契约 (`test_config_contracts.py`)、valley 基准 (`valley_no_silence.yaml`)。
- performance：`test_valley_perf.py` 监控 MDD+VPP 的耗时。
- 慢测试与 GPU 相关 case 以 `@pytest.mark.slow`、`@pytest.mark.gpu` 标记，CI 默认跳过。

## 7. 性能与复杂度基线
- 分离阶段：MDX23 在 RTX 4090 上 ~0.6x 实时，CPU 回退 ~3.5x。
- 检测 + 守卫：10 分钟素材在单核下约 12s；开启静音守卫会额外增加 ~8%。
- 拼接误差：`test_cpu_baseline_perfect_reconstruction` 要求最大绝对误差 < 1e-8。

## 8. 当前进展与下一步
- 已完成：v2.2 单次判决合流、守卫统计、片段调试输出、文档重写；TrackFeatureCache 接入主线；`audio_cut.cutting.finalize_cut_points` 接管 `_finalize_and_filter_cuts_v2`。
- 进行中：合并重复过滤与 legacy 守卫残留，确保候选仅通过 `audio_cut.cutting.refine` 管线。
- 待规划：VPP/BPM 指标化、全链路性能采样脚本、配置迁移工具（v2→v3）。

## 9. 环境与工具
- Python 3.10+，PyTorch + librosa + numpy/scipy/soundfile。
- `pip install -e .[dev]` 提供开发依赖（pytest/black/flake8）。
- Windows + PowerShell/WSL 均可运行；外部模型位于 `MVSEP-MDX23-music-separation-model/`。

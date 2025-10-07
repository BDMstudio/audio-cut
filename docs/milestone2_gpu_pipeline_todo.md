<!-- File: docs/milestone2_gpu_pipeline_todo.md -->
<!-- AI-SUMMARY: Milestone 2 执行清单，按阶段列出 GPU 分块流水线落地步骤与 DoD。 -->

# Milestone 2 — GPU 流水线执行清单

> 基于 `docs/milestone2_gpu_pipeline_plan -improve.md`，按阶段拆解为可执行任务。所有条目须在完成后勾选，并补充验证证据（日志、基准、PR 链接等）。

---

## A. 现状确认与准备

- [x] `config.yaml` 中开启 `gpu_pipeline.enable` 默认值，并记录 chunk/overlap/halo 配置。(src/vocal_smart_splitter/config.yaml)
- [x] 更新 `requirements.txt` / 环境文档，明确 CUDA 12.x + cuDNN 9.x + onnxruntime-gpu 版本依赖及 PATH 设置。(requirements.txt, MDX23_SETUP.md)
- [x] 补充 `tests/sanity/ort_mdx23_cuda_sanity.py` 使用说明，确保 GPU Provider 自检脚本在 CI/本地均可运行。(MDX23_SETUP.md)

---

## B. GPU 三流流水线落地

- [x] 在 `gpu_pipeline` 模块实现并验证 `Streams / record_event / wait_event` 调度逻辑。(src/audio_cut/utils/gpu_pipeline.py:95-118)
- [x] `EnhancedVocalSeparator` 在 `SeamlessSplitter` 调用链中，按 **S1 分离 → S2 VAD → S3 特征/检测** 的顺序使用 CUDA streams 并行执行，同步依赖通过 events 管理。(src/vocal_smart_splitter/core/enhanced_vocal_separator.py:295-359)
- [x] 在 gpu_pipeline 工具层加入 环形 pinned 缓冲 与 inflight_chunks_limit 背压。(src/audio_cut/utils/gpu_pipeline.py:228-299)
- [x] 事件同步以 cudaEvent.record()/wait_event() 为唯一手段，严禁隐式同步。在 bench 中单列 H2D/DtoH 重叠率。(src/vocal_smart_splitter/core/enhanced_vocal_separator.py:330-338)
- [x] 在 GPU 路径中启用 pinned memory / inflight 限制，确保 H2D/DtoH 重叠与背压生效。(src/vocal_smart_splitter/core/enhanced_vocal_separator.py:305-339)
- [x] 异常回退机制：S1/S2/S3 任意阶段 GPU 失败时，自动切换到 CPU pipeline，并输出 `gpu_pipeline_used=false` 诊断信息。(src/vocal_smart_splitter/core/enhanced_vocal_separator.py:167-199)

---

## C. Silero VAD / 特征缓存分块化

- [x] Silero VAD 按块推理（含 halo），跨块合并时间戳（间隙 <120ms 合并）并输出全局时间轴。 (src/audio_cut/detectors/silero_chunk_vad.py)
- [x] “仅在语音段边界 ±200 ms 的窗口里跑贵计算（VPP/MDD）”，窗口外跳过；并把短隙 <120 ms 合并写成可配置阈值。 (src/vocal_smart_splitter/core/pure_vocal_pause_detector.py)
- [x] `PureVocalPauseDetector` 的昂贵检测步骤（能量谷、MDD、特征分析）仅在焦点窗口内执行，复用块级 VAD 结果。 (src/vocal_smart_splitter/core/pure_vocal_pause_detector.py)
- [x] ChunkFeatureBuilder 支持 GPU STFT，一次计算多段复用，仅写入有效帧并跨块拼接 (src/audio_cut/analysis/features_cache.py)
- [x] 校验分块 STFT 与整段 STFT 等价，误差 MAE < 1e-4。 (tests/unit/test_chunk_feature_builder_stft_equivalence.py)

---

## D. 精炼与切点一致性

- [x] `refine.finalize_cut_points` 与块级流程集成：先 NMS 后守卫，守卫 O(1) 跳转，最终 min-gap 校验。（`src/vocal_smart_splitter/core/seamless_splitter.py`, `src/audio_cut/cutting/metrics.py`）
- [x] 守卫/过零/静音逻辑在块间连续性无缝衔接，确保可逆性=0。（`tests/unit/test_cutting_consistency.py`）
- [x] 纯 VAD / Valley 结果在跨块边界的时间误差满足：均值 ≤10 ms、P95 ≤30 ms；守卫右推差均值 ≤15 ms。（`tests/unit/test_cutting_consistency.py`) 

---

## E. 性能基线与监控

- [x] 小样本集跑通 GPU 流水线与整段 CPU 基线，记录端到端耗时、吞吐（sec_audio/sec）、显存峰值、H2D/DtoH 时间。（`scripts/bench/run_gpu_cpu_baseline.py` 生成 `output/bench/<timestamp>/gpu_cpu_baseline.json`）
- [x] 满足目标：GPU 端到端平均耗时 ≥30% 提升；显存峰值 ≤ 基线 +10%；H2D/DtoH 时间下降 ≥15%。（脚本自动计算 `speedup_ratio` 并标记 `meets_target`）
- [x] 记录 `gpu_pipeline_used/gpu_meta` 并输出至日志或结果 JSON，便于后续监控。（`SeamlessSplitter` 结果追加 `gpu_pipeline_*` 字段，含 `h2d_ms/dtoh_ms/compute_ms/peak_mem_bytes` 等）

---

## F. 测试与文档

- [x] 扩充 `tests/benchmarks/test_chunk_vs_full_equivalence.py`，新增 `test_chunk_vs_full_equivalence_real_model` 对 MDX23 推理生成 `chunk_vs_full_real.{json,md}` 报告。
- [x] 增补 Silero/VAD/特征跨块单测：`tests/unit/test_silero_chunk_vad.py` 与 `tests/unit/test_chunk_feature_builder_gpu.py` 覆盖短隙合并、焦点窗口裁剪和帧索引单调。
- [x] 更新 `docs/milestone2_gpu_pipeline_plan -improve.md` 与 `todo-refine.md`，记录最新测试准则与验收指标。
- [x] 准备性能报表说明：在 `scripts/bench/README_gpu_pipeline.md` 描述 `run_gpu_cpu_baseline.py` 输出字段，便于 PR 附件直接引用。

---

## G. 多 GPU 与后续扩展（P0）

- [x] 多流并行：分离、VAD、特征三路真正并行运行（`src/vocal_smart_splitter/core/enhanced_vocal_separator.py:362`）
  - [x] S1：在 `EnhancedVocalSeparator._separate_with_pipeline` 中拆分 S_sep / S_vad / S_feat，并以 CUDA stream + `record_event`/`wait_event` 串联依赖（同上）
  - [x] S2：`MDX23OnnxBackend.infer_chunk` 支持 stream / non_blocking 调度，保持分块推理事件对齐（`src/audio_cut/separation/backends.py:276`）
  - [x] S3：`ChunkFeatureBuilder` 与 `SileroChunkVAD` 支持在 GPU stream 内执行（`src/audio_cut/analysis/features_cache.py:123`, `src/audio_cut/detectors/silero_chunk_vad.py:71`）
  - [x] 验证：更新 `tests/unit/test_silero_chunk_vad.py`、`tests/unit/test_chunk_feature_builder_gpu.py` 覆盖并行行为
- [x] Pinned 内存与 H2D/DtoH 重叠：引入持久化缓存与背压（`src/vocal_smart_splitter/core/enhanced_vocal_separator.py:375`）
  - [x] `PinnedBufferPool` 接入分离链路，统一 acquire/release pinned tensor（同上）
  - [x] 使用 `InflightLimiter` 控制 `_separate_with_pipeline` 内的并行 chunk 数量（`src/audio_cut/utils/gpu_pipeline.py:531`）
  - [x] 验证：`scripts/bench/run_gpu_cpu_baseline.py` 产出 `h2d_ms/dtoh_ms/peak_mem_bytes` 指标
- [x] 多 GPU 扩展：每卡一进程与指标固化
  - [x] `PipelineContext` 支持按设备维护 streams/pinned pool，并在异常时记录 `gpu_pipeline_failures` 并回退（`src/audio_cut/utils/gpu_pipeline.py:508`）
  - [x] `scripts/bench/run_multi_gpu_probe.py` 输出 per-device JSON（util%、mem_peak、processing_time、strict 模式）
  - [x] 文档同步：`docs/milestone2_gpu_pipeline_plan -improve.md`、`todo-refine.md` 已更新当前方案

---

> 勾选时请在括号中标注 PR、Commit 或报告链接，例如：`- [x] … (PR #123, bench_20251001.json)`。



<!-- File: docs/audio_cut_v2_6_vpbd_asr_todo.md -->
<!-- AI-SUMMARY: audio-cut v2.6 VPBD + FireRedASR2S 重构的阶段化 TODO 清单。 -->

# audio-cut v2.6 VPBD + ASR TODO List

> 基于 `docs/audio_cut_v2_6_vpbd_asr_refactor.md` 和 `docs/audio_cut_v2_6_vpbd_asr_plan.md`。  
> 勾选完成项时，在括号内补充验证证据：测试命令、日志路径、bench report、commit 或 PR。  

---

## A. Baseline Protection

- [x] 阅读 `development.md`，确认当前架构 SSOT。（证据：本轮已读取 `development.md` 1-220 行）
- [x] 阅读 `README.md`，确认 CLI/API/输出结构。（证据：本轮已读取 `README.md` 1-220 行）
- [x] 阅读 `audio-cut封装为模块.md`，确认 mvagent 进程内调用契约。（证据：本轮已读取该文档 1-220 行）
- [x] 运行旧模式快速测试并记录 baseline。（证据：`venv/bin/python -m pytest -s -m "not slow and not gpu"`；实现前 `14 passed in 2.35s`，本批后 `25 passed in 1.57s`）
- [x] 运行配置契约测试，确认当前配置状态。（证据：新增并运行 `tests/contracts/test_config_contracts.py`，见 `11 passed in 1.24s` 聚焦测试）
- [x] 运行拼接完整性测试，确认旧切分可逆性。（证据：新增 `tests/unit/test_cpu_baseline_perfect_reconstruction.py` 覆盖 `_split_at_sample_level` 样本级分段后 `np.concatenate(segments)` 与原始 buffer 严格相等；`venv/bin/python -m pytest -s tests/unit/test_cpu_baseline_perfect_reconstruction.py` -> `1 passed in 3.36s`）
- [x] 新建开发分支：`codex/v2.6-vpbd-asr` 或用户指定分支。（证据：`git switch -c codex/v2.6-vpbd-asr` 成功）

验收命令：

```bash
pytest -m "not slow and not gpu"
pytest tests/contracts/test_config_contracts.py
pytest tests/unit/test_cpu_baseline_perfect_reconstruction.py
```

---

## B. Config And Data Models

- [x] 新增 `src/audio_cut/lyrics/__init__.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_models.py tests/unit/test_fake_lyrics_provider.py tests/unit/test_boundary_data_models.py tests/contracts/test_config_contracts.py` -> `11 passed in 1.24s`）
- [x] 新增 `src/audio_cut/lyrics/models.py`：`Word`、`Sentence`、`VadRegion`、`LyricsTimeline`。（证据：`tests/unit/test_lyrics_models.py`）
- [x] 实现 timeline validation：时间合法性、排序、非法片段处理。（证据：`tests/unit/test_lyrics_models.py` 覆盖 strict 与 non-strict）
- [x] 新增 `src/audio_cut/cutting/cut_candidate.py`：`CutCandidate`、`CandidateSource`。（证据：`tests/unit/test_boundary_data_models.py`）
- [x] 新增 `src/audio_cut/analysis/boundary_features.py`：`BoundaryFeatures`。（证据：`tests/unit/test_boundary_data_models.py`）
- [x] 新增异常类型：`LyricsAlignmentUnavailable`、`FireRedProviderError`、`TimelineValidationError`、`GlobalCutPlanningError`。（证据：`src/audio_cut/exceptions.py`，provider/model 测试覆盖其中两个异常）
- [x] 扩展 `config/unified.yaml`：`vpbd`。（证据：`tests/contracts/test_config_contracts.py`）
- [x] 扩展 `config/unified.yaml`：`lyrics_alignment`。（证据：`tests/contracts/test_config_contracts.py`）
- [x] 扩展 `config/unified.yaml`：`fire_red`。（证据：`tests/contracts/test_config_contracts.py`）
- [x] 扩展 `config/unified.yaml`：`phrase_boundary`。（证据：`tests/contracts/test_config_contracts.py`）
- [x] 扩展 `config/unified.yaml`：`global_planner`。（证据：`tests/contracts/test_config_contracts.py`）
- [x] 扩展配置 schema/contract tests，覆盖新配置默认值和环境变量覆盖。（证据：`tests/contracts/test_config_contracts.py`；未扩大 v3 schema）
- [x] 新增 `tests/unit/test_lyrics_models.py`。（证据：聚焦测试 `11 passed in 1.24s`）

验收命令：

```bash
pytest tests/contracts/test_config_contracts.py
pytest tests/unit/test_lyrics_models.py
```

---

## C. LyricsProvider Skeleton And Fake Provider

- [x] 新增 `src/audio_cut/lyrics/providers.py`。（证据：`tests/unit/test_fake_lyrics_provider.py`）
- [x] 定义 `LyricsProviderRequest`。（证据：`tests/unit/test_fake_lyrics_provider.py`）
- [x] 定义 `LyricsProvider` 抽象接口。（证据：`tests/unit/test_fake_lyrics_provider.py`）
- [x] 实现 `NullLyricsProvider`。（证据：`tests/unit/test_fake_lyrics_provider.py`）
- [x] 实现 `FakeLyricsProvider`。（证据：`tests/unit/test_fake_lyrics_provider.py`）
- [x] 新增 fixture：`tests/fixtures/lyrics/simple_song_timeline.json`。（证据：`tests/unit/test_fake_lyrics_provider.py`）
- [x] 实现 `build_lyrics_provider(cfg)`。（证据：`tests/unit/test_fake_lyrics_provider.py`）
- [x] 支持 provider 选择：`disabled`、`fake`、`auto`。（证据：`tests/unit/test_fake_lyrics_provider.py`）
- [x] 对旧模式传入 `lyrics_alignment` 时只 warning，不改变旧路径行为。（证据：`tests/unit/test_legacy_mode_regression.py::test_legacy_mode_warns_but_ignores_lyrics_alignment`；相关验证 `9 passed in 2.79s`）
- [x] 新增 `tests/unit/test_fake_lyrics_provider.py`。（证据：聚焦测试 `11 passed in 1.24s`）

验收命令：

```bash
pytest tests/unit/test_fake_lyrics_provider.py
```

---

## D. ASR Audio Prep, Chunker, Timeline Merge

- [x] 新增 `src/audio_cut/utils/audio_resample.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 实现 `ensure_16k_mono_pcm_wav(...)`。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 确认输出为 16kHz、mono、PCM_16 wav。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 确认检测副本进入 `intermediate/` 或 cache，不覆盖高质量导出音频。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 新增 `src/audio_cut/lyrics/chunker.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 实现默认 `chunk_s=35.0`、`overlap_s=1.0`。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 强制 chunk 不超过 `max_chunk_s`，并保留 FireRed AED `<=60s` 约束。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 为每个 chunk 记录 `chunk_id`、`global_t0`、`global_t1`、`path`、`duration_s`。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 新增 `src/audio_cut/lyrics/timeline.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 实现 chunk-local time 到 global time 转换。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 实现 overlap words 去重。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] confidence 缺失时按 overlap 中心距离稳定合并。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 禁止把 chunk boundary 自身作为候选切点。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 新增 `src/audio_cut/lyrics/cache.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 设计 cache key：audio hash、separator、mode、provider、FireRed 版本、chunk 参数、scorer/planner 配置。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 新增 `tests/unit/test_lyrics_chunker.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）
- [x] 新增 `tests/unit/test_lyrics_timeline_merge.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_lyrics_chunker.py tests/unit/test_lyrics_timeline_merge.py` -> `8 passed in 1.80s`；快速回归 `33 passed in 2.05s`）

验收命令：

```bash
pytest tests/unit/test_lyrics_chunker.py
pytest tests/unit/test_lyrics_timeline_merge.py
```

---

## E. Candidate Generation And Boundary Scoring

- [x] 实现 legacy acoustic candidate adapter。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 将旧候选转换为 `CutCandidate(source="acoustic_pause" | "mdd_valley")`。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 保留旧候选原始分数、RMS valley、MDD/VPP 参数到 `candidate.meta`。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 实现 `LyricsBoundaryCandidateGenerator`。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 支持 word gap 候选。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 支持 sentence end 候选。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 支持 punctuation end 弱加分。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 支持 mVAD singing boundary 候选。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 实现高置信 word 内部禁切。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 实现高置信连续 singing 区内部禁切。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 低置信 ASR 只降权，不硬禁止。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 复用现有 beat/MDD 结果作为 affinity 或弱候选。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 实现 `BoundaryFeatureExtractor`。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 实现 inside word penalty。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 实现 ASR gap score。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 实现 sentence end bonus。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 实现 high-confidence singing penalty。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 实现 beat/mdd affinity。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 实现 `PhraseBoundaryScorer`。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] phrase boundary 权重全部从配置读取。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 所有分项归一化到 `[0, 1]`。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 最终分数 clamp 到 `[0, 1]`。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 候选 `features`、`source`、`reasons` 保留用于 Manifest。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 输出候选调试 JSON。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 新增 `tests/unit/test_boundary_features.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）
- [x] 新增 `tests/unit/test_phrase_boundary_scorer.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_boundary_features.py tests/unit/test_phrase_boundary_scorer.py` -> `8 passed in 1.30s`；快速回归 `41 passed in 2.51s`）

验收命令：

```bash
pytest tests/unit/test_boundary_features.py
pytest tests/unit/test_phrase_boundary_scorer.py
```

---

## F. Global Cut Planner

- [x] 新增 `src/audio_cut/cutting/global_cut_planner.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 实现 dynamic programming planner。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 实现 `hard_min_s` / `hard_max_s` 约束。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 实现 `target_min_s` / `target_max_s` 奖励区间。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 实现 soft duration penalty。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 实现 candidate vocal cut risk。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 实现 beat conflict penalty。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 实现 rescue planning。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 实现 candidate pruning：`max_candidates_per_second`。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] planner 输出保留 selected/suppressed metadata。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 将 planner 输出串接到 `finalize_cut_points`。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 确保 `finalize_cut_points` 后仍可追踪 planner metadata。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 静音守卫移动切点后回写 `guard_shift_ms`。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）
- [x] 片段 lyrics 映射使用守卫后的最终 `t0/t1`。（证据：`tests/unit/test_manifest_vpbd_asr.py::test_manifest_uses_guard_shifted_segment_bounds_for_lyrics_and_cut_metadata`；同时修复 `cuts.final` raw metadata 映射到 `final_time_by_raw_time`；相关验证 `9 passed in 2.79s`）
- [x] 新增 `tests/unit/test_global_cut_planner.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_global_cut_planner.py tests/unit/test_cutting_consistency.py` -> `6 passed in 1.42s`；快速回归 `47 passed in 2.39s`）

验收命令：

```bash
pytest tests/unit/test_global_cut_planner.py
pytest tests/unit/test_cutting_consistency.py
```

---

## G. SeamlessSplitter Integration

- [x] 新增 `src/vocal_smart_splitter/core/vocal_phrase_boundary_detector.py`。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] 新 detector 复用现有分离、特征缓存、守卫、布局、导出模块。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] `mode="vpbd_acoustic"` 调用无 ASR 链路。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] `mode="vpbd_asr"` 调用 lyrics provider 链路。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] `lyrics_alignment.enabled=false` 时降级 `vpbd_acoustic`。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] provider unavailable 且 `strict=false` 时 warning + fallback `vpbd_acoustic`。（证据：`tests/integration/test_pipeline_vpbd_acoustic_fallback.py::test_vpbd_asr_unavailable_provider_falls_back_when_not_strict`；G 验证 `14 passed in 6.30s`）
- [x] provider unavailable 且 `strict=true` 时抛 `LyricsAlignmentUnavailable`。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] FireRed timeout 且 `strict=false` 时 fallback 并记录 reason。（证据：`tests/integration/test_pipeline_vpbd_acoustic_fallback.py::test_vpbd_asr_firered_timeout_falls_back_when_not_strict`；G 验证 `14 passed in 6.30s`）
- [x] FireRed timeout 且 `strict=true` 时抛 `FireRedProviderError` 并由顶层返回失败结果。（证据：`tests/integration/test_pipeline_vpbd_asr_strict_failure.py::test_vpbd_asr_strict_firered_timeout_returns_failed_result`；provider 单测验证 timeout 转 `FireRedProviderError`；G 验证 `14 passed in 6.30s`）
- [x] timeline 部分非法时 strict=false 尽量过滤并记录 warnings。（证据：`tests/unit/test_lyrics_models.py::test_timeline_filters_invalid_items_when_not_strict`；G 验证 `14 passed in 6.30s`）
- [x] timeline 部分非法时 strict=true 抛 `TimelineValidationError`。（证据：`tests/unit/test_lyrics_models.py::test_timeline_rejects_invalid_items_when_strict`；G 验证 `14 passed in 6.30s`）
- [x] planner 无可行路径时触发 rescue 或受控 fallback。（证据：`tests/unit/test_global_cut_planner.py::test_global_cut_planner_rescues_when_no_candidate_path_exists`；G 验证 `14 passed in 6.30s`）
- [x] 旧 `v2.2_mdd` 路径不改。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] 旧 `hybrid_mdd` 路径不改。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] 旧 `librosa_onset` 路径不改。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] 记录 `boundary_detection` meta。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] 记录 `lyrics_alignment` meta。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] 新增 `tests/integration/test_pipeline_vpbd_asr_fake_provider.py`。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] 新增 `tests/integration/test_pipeline_vpbd_acoustic_fallback.py`。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）
- [x] 新增 `tests/integration/test_pipeline_vpbd_asr_strict_failure.py`。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_asr_strict_failure.py` -> `4 passed in 5.76s`；快速回归 `51 passed in 5.89s`，最终 `51 passed in 5.89s`）

验收命令：

```bash
pytest tests/integration/test_pipeline_vpbd_asr_fake_provider.py
pytest tests/integration/test_pipeline_vpbd_acoustic_fallback.py
pytest tests/integration/test_pipeline_vpbd_asr_strict_failure.py
```

---

## H. Manifest And Segment Lyrics

- [x] `ResultBuilder` 增加 optional `lyrics_alignment` 字段。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] `ResultBuilder` 增加 optional `boundary_detection` 字段。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] `ResultBuilder` 增加 optional `cuts.final[*].features` 字段。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] `ResultBuilder` 增加 optional `cuts.final[*].guard_shift_ms` 字段。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] `ResultBuilder` 增加 optional `segments[*].lyrics` 字段。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] `SegmentExporter` 保持文件名兼容。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] 实现 `attach_lyrics_to_segments(...)`。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] 中文 lyrics text 拼接不加空格。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] 英文 lyrics text 拼接按空格。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] word 与片段交界重叠不足时默认排除。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] 无 words 时允许 `lyrics=null` 或空对象，不影响导出。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] Manifest schema/contract 测试覆盖新增字段 optional 行为。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] 新增 `tests/unit/test_manifest_vpbd_asr.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）
- [x] 更新 `audio-cut封装为模块.md` 的 mvagent 调用示例。（证据：`venv/bin/python -m pytest -s tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `4 passed in 1.66s`；G 集成 `4 passed in 6.45s`；快速回归 `55 passed in 5.77s`）

验收命令：

```bash
pytest tests/unit/test_api_manifest.py
pytest tests/unit/test_manifest_vpbd_asr.py
```

---

## I. FireRed CLI And Sidecar Providers

- [x] 新增 `src/audio_cut/lyrics/firered_cli_provider.py`。（证据：`venv/bin/python -m pytest -s tests/unit/test_firered_protocol.py tests/unit/test_firered_cli_provider.py tests/unit/test_firered_sidecar_provider.py tests/unit/test_firered_provider_selection.py` -> `10 passed in 2.15s`；快速回归 `65 passed, 1 deselected in 6.52s`）
- [x] `FireRedCliProvider` 通过 subprocess 调用外部 worker。（证据：`tests/unit/test_firered_cli_provider.py::test_cli_provider_invokes_worker_with_timeout_and_reads_output`）
- [x] CLI provider 设置 timeout。（证据：`tests/unit/test_firered_cli_provider.py::test_cli_provider_invokes_worker_with_timeout_and_reads_output` 与 `::test_cli_provider_raises_on_timeout`）
- [x] CLI provider 捕获 stderr。（证据：`tests/unit/test_firered_cli_provider.py::test_cli_provider_invokes_worker_with_timeout_and_reads_output`）
- [x] CLI provider 检查返回码。（证据：`tests/unit/test_firered_cli_provider.py::test_cli_provider_raises_with_stderr_on_nonzero_exit`）
- [x] CLI provider 输出标准 `lyrics_timeline.json`。（证据：`tests/unit/test_firered_cli_provider.py::test_cli_provider_invokes_worker_with_timeout_and_reads_output`）
- [x] 新增 `src/audio_cut/lyrics/firered_sidecar_provider.py`。（证据：`tests/unit/test_firered_sidecar_provider.py`）
- [x] Sidecar provider 实现 `/health`。（证据：`tests/unit/test_firered_sidecar_provider.py::test_sidecar_provider_uses_local_health_and_analyze_endpoints`）
- [x] Sidecar provider 实现 `/analyze`。（证据：`tests/unit/test_firered_sidecar_provider.py::test_sidecar_provider_uses_local_health_and_analyze_endpoints`）
- [x] Sidecar provider 支持 local endpoint。（证据：`tests/unit/test_firered_sidecar_provider.py::test_sidecar_provider_uses_local_health_and_analyze_endpoints` 使用本地 `127.0.0.1` HTTP server）
- [x] Sidecar provider 支持 worker 常驻。（证据：`FireRedSidecarProvider` 复用固定 endpoint；同一实例完成 `/health` 与 `/analyze`）
- [x] 定义 worker JSON 输入协议。（证据：`tests/unit/test_firered_protocol.py::test_build_worker_request_uses_standard_json_input_protocol`）
- [x] 定义 worker JSON 输出协议。（证据：`tests/unit/test_firered_protocol.py::test_parse_worker_response_converts_local_ms_to_global_seconds`）
- [x] FireRed 返回 chunk-local ms 时转换为全局秒。（证据：`tests/unit/test_firered_protocol.py::test_parse_worker_response_converts_local_ms_to_global_seconds`）
- [x] FireRed 缺失 confidence 时置为 `None`。（证据：`tests/unit/test_firered_protocol.py::test_parse_worker_response_converts_local_ms_to_global_seconds`）
- [x] FireRed 缺失 mVAD 时允许空列表。（证据：`tests/unit/test_firered_protocol.py::test_parse_worker_response_allows_missing_mvad`）
- [x] provider auto selection 支持 `sidecar -> cli -> in_process -> null`。（证据：`tests/unit/test_firered_provider_selection.py`；`in_process` 识别为未配置后继续降级）
- [x] 可选 `FireRedInProcessProvider` 本轮不实现，保留 CLI/sidecar 隔离边界。（设计决策：真实 FireRed smoke 已通过 CLI worker；in-process 会把 FireRed/Torch/CUDA 重依赖耦合进主进程，当前收益不足以抵消复杂度；README/development 以 CLI/sidecar 作为部署路径）
- [x] 新增 `tests/integration/test_firered_cli_provider_real.py`，标记 `firered` 和 `gpu`。（证据：`venv/bin/python -m pytest -s -m firered tests/integration/test_firered_cli_provider_real.py` -> `1 skipped in 0.83s`，默认受 `gpu`/环境变量保护）

验收命令：

```bash
pytest -m firered tests/integration/test_firered_cli_provider_real.py
```

---

## J. CLI, quick_start, Docs

- [x] `run_splitter.py` 增加 `--lyrics-provider`。（证据：`venv/bin/python -m pytest -s tests/unit/test_run_splitter_cli.py tests/unit/test_quick_start_vpbd.py` -> `4 passed in 1.36s`；快速回归 `69 passed, 1 deselected in 7.67s`）
- [x] `run_splitter.py` 增加 `--firered-endpoint`。（证据：`tests/unit/test_run_splitter_cli.py::test_run_splitter_endpoint_defaults_provider_to_sidecar`）
- [x] `run_splitter.py` 增加 `--asr-chunk-s`。（证据：`tests/unit/test_run_splitter_cli.py::test_run_splitter_accepts_vpbd_asr_lyrics_options`）
- [x] `run_splitter.py` 增加 `--asr-overlap-s`。（证据：`tests/unit/test_run_splitter_cli.py::test_run_splitter_accepts_vpbd_asr_lyrics_options`）
- [x] `run_splitter.py` 增加 `--asr-strict`。（证据：`tests/unit/test_run_splitter_cli.py::test_run_splitter_accepts_vpbd_asr_lyrics_options`）
- [x] `run_splitter.py` 增加 `--lyrics-fixture`，用于 fake provider。（证据：`tests/unit/test_run_splitter_cli.py::test_run_splitter_accepts_vpbd_asr_lyrics_options`）
- [x] `quick_start.py` 增加 VPBD + FireRedASR2S 菜单项。（证据：`tests/unit/test_quick_start_vpbd.py::test_quick_start_processing_menu_includes_vpbd_asr`）
- [x] README 新增 `vpbd_asr` 使用说明。（证据：已更新 `README.md` 快速开始、配置、输出结构与更新记录）
- [x] README 新增 FireRed sidecar/CLI 部署说明。（证据：已更新 `README.md` FireRed provider 部署段落）
- [x] `development.md` 更新 v2.6 架构 SSOT。（证据：已更新版本演进、目录职责、核心流程、配置与测试矩阵）
- [x] release notes 草稿。（证据：新增 `docs/release_notes_v2_6_draft.md`）

验收命令：

```bash
python run_splitter.py input/song.mp3 --mode vpbd_asr --lyrics-provider fake
python quick_start.py
```

---

## K. Regression And QA

- [x] 回归 `v2.2_mdd`：API、CLI、输出命名、Manifest 旧字段。（证据：`venv/bin/python -m pytest -s tests/unit/test_legacy_mode_regression.py` -> `4 passed in 1.42s`；快速回归 `76 passed, 1 deselected in 7.41s`）
- [x] 回归 `hybrid_mdd`：`_lib` 标记不变。（证据：`tests/unit/test_legacy_mode_regression.py::test_segment_exporter_preserves_duration_suffix_and_hybrid_lib_marker`）
- [x] 回归 `librosa_onset`：旧节拍路径不变。（证据：`tests/unit/test_legacy_mode_regression.py::test_librosa_onset_manifest_keeps_smart_segmentation_fields`）
- [x] 确认旧模式不需要新增必需参数。（证据：`tests/unit/test_legacy_mode_regression.py::test_legacy_cli_requires_no_new_asr_arguments`）
- [x] 确认输出文件名 `_X.X` 秒后缀不变。（证据：`tests/unit/test_legacy_mode_regression.py::test_segment_exporter_preserves_duration_suffix_and_hybrid_lib_marker`）
- [x] 确认 `SegmentManifest.json` 旧字段不删除。（证据：`venv/bin/python -m pytest -s tests/unit/test_qa_report.py tests/unit/test_api_manifest.py tests/unit/test_manifest_vpbd_asr.py` -> `6 passed in 1.52s`；快速回归 `72 passed, 1 deselected in 7.29s`）
- [x] fake provider 模式验证不切在 high-confidence word 内部。（证据：`tests/integration/test_pipeline_vpbd_asr_fake_provider.py` selected cuts 断言）
- [x] fake provider 模式验证片段 lyrics 正确附着。（证据：`tests/integration/test_pipeline_vpbd_asr_fake_provider.py` 通过 `_build_manifest` 断言 `segments[*].lyrics`）
- [x] strict=false fallback Manifest 记录 `fallback_reason`。（证据：`tests/integration/test_pipeline_vpbd_acoustic_fallback.py::test_vpbd_asr_unavailable_provider_falls_back_when_not_strict`；`tests/unit/test_qa_report.py` 验证 Manifest 派生 `fallback_reason`）
- [x] strict=true provider unavailable 抛异常。（证据：`tests/integration/test_pipeline_vpbd_asr_strict_failure.py`，快速回归覆盖）
- [x] QA report 输出 `segments_count`。（证据：`tests/unit/test_qa_report.py`）
- [x] QA report 输出 `median_segment_s`。（证据：`tests/unit/test_qa_report.py`）
- [x] QA report 输出 `segment_5_15_pass_rate`。（证据：`tests/unit/test_qa_report.py`）
- [x] QA report 输出 `cut_inside_word_rate`。（证据：`tests/unit/test_qa_report.py`）
- [x] QA report 输出 `cut_inside_singing_rate`。（证据：`tests/unit/test_qa_report.py`）
- [x] QA report 输出 `avg_boundary_score`。（证据：`tests/unit/test_qa_report.py`）
- [x] QA report 输出 `lyrics_coverage_ratio`。（证据：`tests/unit/test_qa_report.py`）
- [x] QA report 输出 `asr_avg_confidence`。（证据：`tests/unit/test_qa_report.py`）
- [x] QA report 输出 `guard_shift_p50_ms / p95_ms`。（证据：`tests/unit/test_qa_report.py`）
- [x] QA report 输出 `fallback_reason`。（证据：`tests/unit/test_qa_report.py`）

基础回归命令：

```bash
pytest -m "not slow and not gpu and not firered" --cov=src --cov-report=term-missing
```

旧模式回归命令：

```bash
python run_splitter.py input/song.mp3 --mode v2.2_mdd
python run_splitter.py input/song.mp3 --mode hybrid_mdd
python run_splitter.py input/song.mp3 --mode librosa_onset
```

新模式 fake provider 命令：

```bash
python run_splitter.py input/song.mp3 \
  --mode vpbd_asr \
  --lyrics-provider fake \
  --lyrics-fixture tests/fixtures/lyrics/simple_song_timeline.json
```

新模式 FireRed sidecar 命令：

```bash
python run_splitter.py input/song.mp3 \
  --mode vpbd_asr \
  --gpu-device cuda:0 \
  --lyrics-provider sidecar \
  --firered-endpoint http://127.0.0.1:8765
```

Manifest 检查命令：

```bash
python scripts/diagnostics/inspect_manifest.py output/.../SegmentManifest.json \
  --check-lyrics \
  --check-boundary-scores \
  --check-no-cut-inside-word
```

---

## L. Manual Acceptance Playlist

- [x] 新增可复现验收执行器、本地 seed playlist、完整 20 首模板和人工评分表。（证据：`scripts/vpbd_asr_acceptance.py`、`docs/vpbd_asr_acceptance_playlist.local.json`、`docs/vpbd_asr_acceptance_playlist.template.json`、`docs/vpbd_asr_manual_scoring_sheet.csv`；`venv/bin/python -m pytest -s tests/unit/test_vpbd_asr_acceptance.py` -> `6 passed in 1.96s`；完整模板 dry-run -> `status=incomplete`, `track_count=20`, `manifest_count=0`, `playlist_coverage=pass`, `track_statuses=[missing_audio]`，正确区分“覆盖足够但缺实际音频/人工指标”；本地 5 首 FireRed 真跑 -> `status=incomplete`, `manifest_count=5`, `cut_inside_word_rate=0.0` pass, `cut_inside_high_conf_singing_rate=0.0` pass, `segment_5_15_pass_rate=0.963414634146` pass；`venv/bin/python scripts/vpbd_asr_acceptance.py --playlist docs/vpbd_asr_acceptance_playlist.local.json --output-dir output/vpbd_asr_acceptance_local --review-csv manual_review_sheet.csv` 生成待人工标注 CSV：`output/vpbd_asr_acceptance_local/manual_review_sheet.csv`；`scripts/prepare_vpbd_asr_acceptance_assets.py` 生成 20 首素材清单：`docs/vpbd_asr_acceptance_audio_inventory.csv`、`docs/vpbd_asr_acceptance_audio_inventory.md`，并创建 `input/acceptance/{category}` 七类放歌目录；`scripts/sync_vpbd_asr_acceptance_playlist.py` 可将 `docs/vpbd_asr_manual_scoring_sheet.csv` 反写为 `docs/vpbd_asr_acceptance_playlist.filled.json`，filled dry-run -> `status=incomplete`, `track_count=20`, `manifest_count=0`, `playlist_coverage=pass`, `track_statuses=[missing_audio]`，仍需 20 首完整样本、人工边界、主观评分和返工率）
- [ ] 中文流行慢歌 3 首。
- [ ] 中文快歌/rap 3 首。
- [ ] 英文流行 3 首。
- [ ] 民谣/低动态 3 首。
- [ ] 强节奏副歌 3 首。
- [ ] 和声/ad-lib 明显 3 首。
- [ ] 器乐 intro/outro 长 2 首。
- [ ] 统计 `boundary_f1_500ms >= 0.82`。
- [ ] 统计 `cut_inside_word_rate <= 1%`。
- [ ] 统计 `cut_inside_high_conf_singing_rate <= 3%`。
- [ ] 统计 `5-15s segment pass rate >= 90%`。
- [ ] 统计 `subjective_naturalness >= 4.2 / 5`。
- [ ] 统计 `manual_recutter_rate` 相比 v2.5.1 降低 `>= 40%`。

---

## M. Final Release Checklist

- [x] 所有新增模块有文件头 `File` 和 `AI-SUMMARY`。（证据：脚本检查 20 个新增/核心模块，输出 `checked 20 files`）
- [x] 核心类/函数有 docstring。（证据：AST 脚本检查新增模块公开类/函数，输出 `checked public classes/functions in 20 files`）
- [x] 无新增 `TODO` / `FIXME` / `console.log` 残留。（证据：`rg -n "TODO|FIXME|console\.log" src run_splitter.py quick_start.py tests README.md development.md docs/release_notes_v2_6_draft.md audio-cut封装为模块.md` 无命中）
- [x] FireRed 依赖未进入 base `requirements.txt`。（证据：`rg -n "FireRed|firered|FireRedASR" requirements.txt setup.py` 无命中）
- [x] CPU-only 路径不会因 FireRed 不可用阻断旧流程。（证据：`venv/bin/python -m pytest -s -m "not slow and not gpu"` -> `97 passed, 1 deselected in 11.08s`；旧 CLI 默认 ASR 参数为空）
- [x] 旧模式输出契约测试通过。（证据：`venv/bin/python -m pytest -s tests/unit/test_legacy_mode_regression.py` -> `4 passed in 1.42s`）
- [x] 新模式 fake provider 集成测试通过。（证据：`venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_asr_fake_provider.py tests/integration/test_pipeline_vpbd_acoustic_fallback.py` -> `4 passed in 6.39s`）
- [x] 可选 FireRed 测试在 `/home/ubuntu/asr_test` 对应环境下通过。（证据：安装 `textgrid`、`kaldi_native_fbank==1.15`，下载 `FireRedASR2-AED`、`FireRedVAD`、`FireRedLID`、`FireRedPunc`；`venv/bin/python scripts/check_fireredasr2s_env.py --json` -> `ok=true`；`/home/ubuntu/asr_test/venv/bin/python -m fireredasr2s.fireredasr2s_cli --help` 通过；`FIRERED_CLI_WORKER=scripts/fireredasr2s_worker.py FIRERED_TEST_WAV=/home/ubuntu/asr_test/audio/test_synth.wav venv/bin/python -m pytest -s -m firered tests/integration/test_firered_cli_provider_real.py --rungpu` -> `1 passed in 20.62s`；provider/worker 回归 `7 passed in 1.21s`）
- [x] README、development、mvagent 调用文档完成。（证据：`rg -n "vpbd_asr|FireRed|FireRedASR2S|lyrics_alignment" README.md development.md audio-cut封装为模块.md docs/release_notes_v2_6_draft.md` 有命中）
- [x] release notes 标明 `vpbd_asr` 为新增/可选能力。（证据：`docs/release_notes_v2_6_draft.md`）


<!-- File: docs/audio_cut_v2_6_vpbd_asr_plan.md -->
<!-- AI-SUMMARY: audio-cut v2.6 VPBD + FireRedASR2S 重构的可执行实施计划。 -->

# audio-cut v2.6 VPBD + ASR Refactor Plan

> 来源设计文档：`docs/audio_cut_v2_6_vpbd_asr_refactor.md`  
> 目标：把设计方案拆成可执行、可验收、可回滚的工程计划。  
> 主路径：WSL2 Ubuntu 22.04 + Python 3.10+ + NVIDIA CUDA。  

---

## 1. Objective

将 audio-cut 从“声学停顿分割器”升级为可选的“声学 + 歌词语义联合分割器”：

- 新增 `mode="vpbd_asr"`：声学候选 + FireRedASR2S 歌词时间轴 + VPBD 打分 + 全局规划。
- 新增 `mode="vpbd_acoustic"`：无 ASR 时仍使用 VPBD scorer/planner。
- 保留旧模式：`v2.2_mdd`、`hybrid_mdd`、`librosa_onset` 的 API、CLI、输出命名和 Manifest 旧字段不变。
- FireRedASR2S 只作为 soft prior，不替代声学停顿、静音守卫、布局精炼。

## 2. Non-Goals

- 不推倒重写 `SeamlessSplitter`。
- 不把 FireRedASR2S 依赖加入 audio-cut base install。
- 不默认使用 FireRedASR2-LLM。
- 不训练新模型。
- 不做 GUI 或云端 ASR 服务。
- 不改变旧输出文件命名。
- 不要求 CPU-only 跑完整 ASR 联合分割。

## 3. Hard Constraints

- `audio_cut.api.separate_and_segment(...)` 必须保持向后兼容。
- `SeamlessSplitter`、`EnhancedVocalSeparator`、`TrackFeatureCache`、`finalize_cut_points`、`segment_layout_refiner`、`SegmentExporter`、`ResultBuilder` 必须复用。
- 新配置必须进入 `config/unified.yaml` 和配置契约测试。
- 新 Manifest 字段必须 optional，只能新增，不能删除或改变旧字段语义。
- 核心算法测试必须先通过 `FakeLyricsProvider`，不得依赖真实 FireRed、torch 大模型或 GPU。
- `vpbd_asr` 在 FireRed 不可用且 `strict=false` 时必须降级到 `vpbd_acoustic`。
- `strict=true` 时，FireRed 不可用必须显式失败。

---

## 4. Delivery Strategy

### 4.1 Minimal Integration Path

优先新增旁路能力，不改旧路径：

```text
audio_cut.api.separate_and_segment
  -> SeamlessSplitter
  -> if mode in {"vpbd_asr", "vpbd_acoustic"}:
       VocalPhraseBoundaryDetector
     else:
       legacy detector path
```

旧模式继续走当前 `PureVocalPauseDetector -> finalize_cut_points -> segment_layout_refiner` 链路。新模式在 `finalize_cut_points` 前新增候选融合、打分和全局规划。

### 4.2 Fake First, FireRed Last

实现顺序必须先保证无外部模型链路：

1. 数据模型、配置、fake provider。
2. chunker、timeline merge、candidate/scorer/planner。
3. `vpbd_acoustic` 和 `vpbd_asr + fake provider` 集成测试。
4. Manifest 和 mvagent 契约。
5. 最后接 `FireRedCliProvider` / `FireRedSidecarProvider`。

### 4.3 Keep ASR Optional

FireRedASR2S 独立部署为 CLI/sidecar worker：

```text
audio-cut main env
  - MDX23/Demucs/audio-cut dependencies
  - LyricsProvider interface

firered worker env
  - FireRedASR2S dependencies
  - 16k mono PCM wav input
  - JSON timeline output
```

这样避免 FireRed 固定的 `torch==2.1.0+cu118` 污染 audio-cut 主环境。

---

## 5. Milestones

### M0. Baseline Protection

目标：确认现状、固定旧行为基线。

交付：

- 阅读 `development.md`、`README.md`、`audio-cut封装为模块.md`。
- 记录当前快速测试结果。
- 确认旧模式 Manifest 和输出命名契约。
- 建议分支：`codex/v2.6-vpbd-asr` 或用户指定分支。

验收：

```bash
pytest -m "not slow and not gpu"
pytest tests/contracts/test_config_contracts.py
pytest tests/unit/test_cpu_baseline_perfect_reconstruction.py
```

### M1. Config, Models, Provider Skeleton

目标：建立新模式的最小类型系统和可测试 provider seam。

交付：

- `src/audio_cut/lyrics/models.py`
- `src/audio_cut/cutting/cut_candidate.py`
- `src/audio_cut/analysis/boundary_features.py`
- `src/audio_cut/lyrics/providers.py`
- `NullLyricsProvider`
- `FakeLyricsProvider`
- `config/unified.yaml` 新增 `vpbd`、`lyrics_alignment`、`fire_red`、`phrase_boundary`、`global_planner`
- 配置契约测试和模型序列化测试

验收：

```bash
pytest tests/contracts/test_config_contracts.py
pytest tests/unit/test_lyrics_models.py
pytest tests/unit/test_fake_lyrics_provider.py
```

### M2. ASR-Safe Timeline Pipeline

目标：把 vocal stem 安全转换为 FireRed 可接受的时间轴输入，并能合并 chunk 结果。

交付：

- `ensure_16k_mono_pcm_wav`
- ASR chunker：默认 `35s` chunk、`1s` overlap，硬性不超过 FireRed AED 限制
- chunk metadata：`chunk_id`、`global_t0`、`global_t1`、`path`、`duration_s`
- `merge_chunk_timelines`
- overlap words 去重
- timeline validation
- timeline cache key 设计

验收：

```bash
pytest tests/unit/test_lyrics_chunker.py
pytest tests/unit/test_lyrics_timeline_merge.py
```

### M3. Candidate, Feature, Scoring

目标：把旧声学候选、歌词候选、mVAD/beat/MDD 证据统一为可解释的 `CutCandidate`。

交付：

- legacy acoustic candidate adapter
- `LyricsBoundaryCandidateGenerator`
- FireRed VAD/mVAD boundary candidate adapter
- beat/MDD weak candidate or affinity adapter
- `BoundaryFeatureExtractor`
- `PhraseBoundaryScorer`
- 所有权重从配置读取
- candidate debug JSON

验收：

```bash
pytest tests/unit/test_boundary_features.py
pytest tests/unit/test_phrase_boundary_scorer.py
```

### M4. Global Cut Planner

目标：在 `finalize_cut_points` 前引入动态规划，减少局部 NMS 导致的碎片化。

交付：

- `src/audio_cut/cutting/global_cut_planner.py`
- duration penalty
- candidate pruning
- rescue planning
- planner metadata 保留
- guard shift metadata 回写策略

验收：

```bash
pytest tests/unit/test_global_cut_planner.py
pytest tests/unit/test_cutting_consistency.py
```

### M5. SeamlessSplitter Integration

目标：新增模式接入主流程，同时旧模式路径不变。

交付：

- `src/vocal_smart_splitter/core/vocal_phrase_boundary_detector.py`
- `mode="vpbd_acoustic"`
- `mode="vpbd_asr"`
- provider fallback
- strict failure
- `boundary_detection` / `lyrics_alignment` runtime meta

验收：

```bash
pytest tests/integration/test_pipeline_vpbd_asr_fake_provider.py
pytest tests/integration/test_pipeline_vpbd_acoustic_fallback.py
pytest tests/integration/test_pipeline_vpbd_asr_strict_failure.py
```

### M6. Manifest And mvagent Contract

目标：让下游直接复用整曲 ASR 时间轴和片段歌词，避免逐片段重复 ASR。

交付：

- optional `lyrics_alignment`
- optional `boundary_detection`
- optional `cuts.final[*].features`
- optional `segments[*].lyrics`
- `attach_lyrics_to_segments`
- `audio-cut封装为模块.md` 更新 mvagent 调用示例

验收：

```bash
pytest tests/unit/test_api_manifest.py
pytest tests/unit/test_manifest_vpbd_asr.py
```

### M7. FireRed Providers

目标：接入真实 FireRedASR2S，但保持可选、可降级、可超时。

交付：

- `FireRedCliProvider`
- `FireRedSidecarProvider`
- `/health` 协议
- `/analyze` 协议
- timeout、stderr、return code、HTTP error 处理
- `pytest.mark.firered` / `pytest.mark.gpu` 可选测试

验收：

```bash
pytest -m firered tests/integration/test_firered_cli_provider_real.py
```

默认 CI 不跑真实 FireRed 测试。

### M8. CLI, quick_start, Docs

目标：让用户能通过 CLI 和 quick_start 使用新模式。

交付：

- `run_splitter.py` 新增 ASR 参数
- `quick_start.py` 新增 VPBD + FireRed 菜单
- `README.md` 新增 v2.6 使用说明
- `development.md` 更新 SSOT
- release notes 草稿

验收：

```bash
python run_splitter.py input/song.mp3 --mode vpbd_asr --lyrics-provider fake
python quick_start.py
```

---

## 6. Regression Gates

每个阶段合并前至少跑对应单测。M5 之后必须固定旧模式回归：

```bash
pytest -m "not slow and not gpu and not firered"
pytest tests/contracts/test_config_contracts.py
pytest tests/unit/test_api_manifest.py
pytest tests/unit/test_cpu_baseline_perfect_reconstruction.py
pytest tests/unit/test_cutting_consistency.py
```

旧模式手工回归：

```bash
python run_splitter.py input/song.mp3 --mode v2.2_mdd
python run_splitter.py input/song.mp3 --mode hybrid_mdd
python run_splitter.py input/song.mp3 --mode librosa_onset
```

新模式 fake provider：

```bash
python run_splitter.py input/song.mp3 \
  --mode vpbd_asr \
  --lyrics-provider fake \
  --lyrics-fixture tests/fixtures/lyrics/simple_song_timeline.json
```

新模式 FireRed sidecar：

```bash
python run_splitter.py input/song.mp3 \
  --mode vpbd_asr \
  --gpu-device cuda:0 \
  --lyrics-provider sidecar \
  --firered-endpoint http://127.0.0.1:8765
```

---

## 7. Acceptance Metrics

最低质量目标：

- `boundary_f1_500ms >= 0.82`
- `cut_inside_word_rate <= 1%`
- `cut_inside_high_conf_singing_rate <= 3%`
- `5-15s segment pass rate >= 90%`
- `subjective_naturalness >= 4.2 / 5`
- `manual_recutter_rate` 相比 v2.5.1 降低 `>= 40%`

自动统计应进入 Manifest 或 QA report：

- `segments_count`
- `median_segment_s`
- `segment_5_15_pass_rate`
- `cut_inside_word_rate`
- `cut_inside_singing_rate`
- `avg_boundary_score`
- `lyrics_coverage_ratio`
- `asr_avg_confidence`
- `guard_shift_p50_ms / p95_ms`
- `fallback_reason`

---

## 8. Rollback Plan

如果 `vpbd_asr` 质量或稳定性不达标：

1. 保留代码。
2. 默认 `lyrics_alignment.enabled=false`。
3. CLI 菜单标记 experimental 或隐藏。
4. mvagent 回退 `mode="hybrid_mdd"` 或 `mode="v2.2_mdd"`。
5. 因旧模式未改动，旧输出兼容性不受影响。

---

## 9. Recommended Commit Order

```text
1. chore(config): add vpbd_asr config schema
2. feat(lyrics): add timeline models and fake provider
3. feat(lyrics): add asr chunker and timeline merger
4. feat(boundary): add cut candidate and boundary feature extraction
5. feat(boundary): add phrase boundary scorer
6. feat(cutting): add global cut planner
7. feat(core): integrate vpbd_acoustic mode
8. feat(core): integrate vpbd_asr mode with provider fallback
9. feat(manifest): add lyrics and boundary metadata
10. feat(firered): add cli and sidecar lyrics providers
11. test: add vpbd_asr fake-provider integration coverage
12. docs: update README, development, mvagent integration guide
```


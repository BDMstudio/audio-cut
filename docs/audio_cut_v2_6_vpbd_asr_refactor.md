# audio-cut v2.6 重构设计文档：VPBD + FireRedASR2S 歌词时间轴联合分割

> 目标读者：coding agent / audio-cut 维护者  
> 目标版本：v2.6.x  
> 主开发环境：WSL2 Ubuntu 22.04 + Python 3.10+ + NVIDIA CUDA  
> 文档日期：2026-06-09

---

## 0. 给 coding agent 的执行要求

1. **不要推倒重写。** 现有 `SeamlessSplitter`、`EnhancedVocalSeparator`、`TrackFeatureCache`、`finalize_cut_points`、`segment_layout_refiner`、`SegmentExporter`、`ResultBuilder` 都要复用。
2. **不要破坏旧模式。** `v2.2_mdd`、`hybrid_mdd`、`librosa_onset` 的外部 API、CLI 参数、输出命名和 Manifest 旧字段必须保持兼容。
3. **新增模式优先。** 新能力以 `mode="vpbd_asr"` 或 `mode="v2.6_vpbd_asr"` 接入，灰度稳定后再考虑设为默认。
4. **ASR 是 soft prior，不是分割主控。** FireRedASR2S 的歌词/词级时间戳只参与切点评分与禁切区域，不允许直接替代声学停顿、换气、静音守卫和布局规划。
5. **FireRed 依赖必须可选。** 没有 FireRedASR2S、没有 GPU、ASR worker 不可用时，audio-cut 必须自动降级到 `vpbd_acoustic` 或旧的 `v2.2_mdd/hybrid_mdd` 路径。
6. **先实现 fake provider 测试链路，再接真实 FireRed。** 所有核心算法单测不得依赖大模型或真实 GPU。
7. **所有新增配置必须进入 `config/unified.yaml` 和配置契约测试。** 不允许散落在脚本中的个人实验参数。
8. **所有新增输出字段必须进入 Manifest schema/测试。** 旧字段只能新增，不能删除或改语义。
9. **WSL2 Ubuntu 22.04 是主路径。** Windows 原生兼容不再作为本次重构的主目标，但不要移除已有 Windows/ORT 兜底逻辑。

---

## 1. 背景与现状

### 1.1 当前 audio-cut 基线

当前工程已经具备稳定的一站式链路：

```text
audio_cut.api.separate_and_segment
  -> SeamlessSplitter
  -> AudioProcessor.load_audio
  -> EnhancedVocalSeparator.separate_for_detection
  -> SileroChunkVAD / ChunkFeatureBuilder / TrackFeatureCache
  -> PureVocalPauseDetector.detect_pure_vocal_pauses
  -> audio_cut.cutting.finalize_cut_points
  -> SeamlessSplitter._classify_segments_vocal_presence
  -> segment_layout_refiner.refine_layout
  -> SegmentExporter / ResultBuilder / SegmentManifest.json
```

需要保留的事实：

- `audio_cut.api.separate_and_segment(...)` 是上层项目的统一入口。
- `SeamlessSplitter` 是主调度入口。
- `EnhancedVocalSeparator` 已封装 MDX23/Demucs，并记录 GPU 元数据。
- `TrackFeatureCache` 已缓存 BPM/MDD/RMS 等特征。
- `finalize_cut_points` 已负责 NMS、过零吸附、静音守卫、最小间隔和守卫位移统计。
- `segment_layout_refiner` 已负责微碎片合并、软最小、软最大救援。
- 输出目录和文件命名已经统一。

### 1.2 当前问题

现有方案主要依据人声分离后的声学停顿、RMS/MDD/BPM/VPP 规则和静音守卫进行切分。它对许多歌曲有效，但存在以下结构性限制：

| 问题 | 表现 | 原因 |
|---|---|---|
| 半字/半词切断 | 切在 word 内部或拖音中间 | 声学低谷不等于语义边界 |
| 拖音误切 | 长尾音、转音、混响尾部被误判 | VAD/RMS 对 singing tail 不稳 |
| 快歌/rap 碎片化 | 候选点过多，局部 NMS 不够 | 缺少全局最优规划 |
| 和声/垫音干扰 | 纯人声 stem 仍有人声残留 | 分离器不是边界检测器 |
| mvagent 重复计算 | 分割后再逐片段 ASR | ASR 结果未作为统一时间轴缓存 |

### 1.3 v2.6 的核心升级

把 mvagent 原本后置的歌词提取前移为 audio-cut 内部可选阶段，形成：

```text
原音频
  -> 人声分离
  -> 检测副本 16k mono PCM
  -> FireRedVAD / FireRedASR2-AED 生成歌词时间轴
  -> 声学停顿候选 + 歌词边界候选 + MDD/beat 候选
  -> Phrase Boundary Scorer
  -> Global Cut Planner
  -> finalize_cut_points 静音守卫
  -> segment_layout_refiner
  -> 导出音频片段 + 片段歌词 + Manifest
```

最终效果目标：从“声学停顿分割器”升级为“声学 + 歌词语义联合分割器”。

---

## 2. 外部 FireRedASR2S 约束

本次重构默认使用 **FireRedASR2-AED**，不要默认使用 FireRedASR2-LLM。

### 2.1 推荐模块

```text
FireRedASR2-AED:
  - 用于歌词/ASR
  - 需要 word-level timestamps
  - 需要 sentence-level timestamps
  - 需要 confidence scores，缺失时允许置空

FireRedVAD / mVAD:
  - 用于 speech/singing/music 时间段
  - 用于给切点增加 non-singing / singing 边界证据
```

### 2.2 官方约束需要编码进工程

coding agent 必须把以下约束落实到代码和配置中：

1. FireRedASR2S 官方代码主测试环境是 **Linux Ubuntu 22.04**。
2. FireRedASR2S 输入音频格式要求：**16kHz、16-bit、mono、PCM wav**。
3. FireRedASR2-AED 单段输入建议 **≤60s**；超过 60s 可能出现 hallucination，超过 200s 会触发 positional encoding error。
4. FireRedASR2-LLM 单段输入建议 **≤40s**，长输入行为未测试。
5. FireRedASR2S `requirements.txt` 固定了 `torch==2.1.0+cu118` 和 `torchaudio==2.1.0+cu118`，因此不要把它强塞进 audio-cut 主依赖。

### 2.3 部署策略

默认策略：**sidecar / CLI worker 优先，in-process 仅作为高级选项。**

```text
mvagent / audio-cut 主进程
  - 保持现有 audio-cut 依赖环境
  - 调用 audio_cut.api.separate_and_segment
  - 通过 provider 访问歌词时间轴

firered worker
  - 独立 venv / conda / Docker / WSL2 环境
  - 安装 FireRedASR2S 依赖
  - 暴露 CLI 或 HTTP local endpoint
```

推荐原因：

- 避免 audio-cut 的 ONNX Runtime、PyTorch、CUDA 版本被 FireRed 依赖污染。
- 避免 MDX23/Demucs 与 FireRedASR2-AED 同时占用显存。
- 8GB/12GB 中端 GPU 可以串行使用同一块 GPU。
- CPU-only 或 FireRed 不可用时能无损降级。

---

## 3. 新目标架构

### 3.1 模块图

```text
src/audio_cut/
  api.py
  lyrics/
    __init__.py
    models.py
    timeline.py
    chunker.py
    providers.py
    firered_cli_provider.py
    firered_sidecar_provider.py
    fake_provider.py
    cache.py
  analysis/
    boundary_features.py
    phrase_boundary.py
    singing_activity.py
  cutting/
    global_cut_planner.py
    cut_candidate.py
  utils/
    audio_resample.py

src/vocal_smart_splitter/core/
  seamless_splitter.py
  vocal_phrase_boundary_detector.py
```

### 3.2 新增职责

| 模块 | 职责 |
|---|---|
| `audio_cut.lyrics.models` | 定义 `Word`, `Sentence`, `VadRegion`, `LyricsTimeline` 等数据结构 |
| `audio_cut.lyrics.chunker` | 把 16k mono vocal stem 切成 ASR-safe chunks，管理 overlap |
| `audio_cut.lyrics.providers` | 定义 `LyricsProvider` 抽象接口 |
| `audio_cut.lyrics.firered_*` | 连接 FireRedASR2S CLI/HTTP/in-process worker |
| `audio_cut.lyrics.timeline` | 归一化、校验、去重、合并 overlap 区词级时间戳 |
| `audio_cut.lyrics.cache` | 以音频哈希 + 模型版本 + 配置生成缓存键 |
| `audio_cut.analysis.singing_activity` | 融合 RMS/F0/VAD/mVAD 得到 singing/non-singing 概率 |
| `audio_cut.analysis.boundary_features` | 为候选切点提取声学、歌词、节拍、布局特征 |
| `audio_cut.analysis.phrase_boundary` | 计算候选切点自然度分数 |
| `audio_cut.cutting.cut_candidate` | 定义候选切点数据结构 |
| `audio_cut.cutting.global_cut_planner` | 用动态规划选择全局最优切点序列 |
| `vocal_phrase_boundary_detector.py` | 把现有停顿检测和新 scoring/planning 串起来 |

---

## 4. 新流水线设计

### 4.1 总流程

```text
0. API 入口
   audio_cut.api.separate_and_segment(..., mode="vpbd_asr")

1. 加载与分离
   AudioProcessor.load_audio
   EnhancedVocalSeparator.separate_for_detection
   输出：mix_44k, vocal_44k, instrumental_44k?, feature_cache

2. 构造检测副本
   vocal_44k -> vocal_16k_mono_pcm.wav
   保持原始 44.1k/stereo 音频只用于最终导出

3. 歌词时间轴
   LyricsProvider.analyze(vocal_16k_mono_pcm.wav)
   输出：LyricsTimeline(words, sentences, vad_regions, mvads, meta)

4. 声学活动与候选
   SingingActivityDetector
   AcousticPauseCandidateGenerator / legacy PureVocalPauseDetector
   LyricsBoundaryCandidateGenerator
   BeatMddCandidateGenerator

5. 候选融合和打分
   BoundaryFeatureExtractor
   PhraseBoundaryScorer

6. 全局规划
   GlobalCutPlanner
   输出未精修 cut points

7. 终筛与布局
   finalize_cut_points
   segment_layout_refiner.refine_layout

8. 导出
   SegmentExporter
   ResultBuilder
   SegmentManifest.json 增加 lyrics_alignment / boundary_scores / segment.lyrics
```

### 4.2 模式和降级

| 模式 | 含义 | FireRed 依赖 | 用途 |
|---|---|---:|---|
| `vpbd_asr` | 声学 + FireRed 歌词联合分割 | 可选，auto | 推荐新模式 |
| `vpbd_acoustic` | 新 scorer/planner，但无 ASR | 否 | CPU/无 FireRed 降级 |
| `v2.2_mdd` | 旧 MDD 路径 | 否 | 兼容 |
| `hybrid_mdd` | 旧 MDD + beat 卡点 | 否 | 兼容 MV 卡点 |
| `librosa_onset` | 旧节拍路径 | 否 | 兼容 |

`vpbd_asr` 的降级规则：

```text
if lyrics_alignment.enabled == false:
    run vpbd_acoustic
elif FireRed provider unavailable and lyrics_alignment.strict == false:
    log warning
    run vpbd_acoustic
elif FireRed provider unavailable and lyrics_alignment.strict == true:
    raise LyricsAlignmentUnavailable
else:
    run vpbd_asr
```

---

## 5. 数据模型

### 5.1 LyricsTimeline

新增文件：`src/audio_cut/lyrics/models.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass(frozen=True)
class Word:
    t0: float
    t1: float
    text: str
    confidence: float | None = None
    lang: str | None = None
    source_chunk: int | None = None

@dataclass(frozen=True)
class Sentence:
    t0: float
    t1: float
    text: str
    confidence: float | None = None
    lang: str | None = None
    word_start: int | None = None
    word_end: int | None = None

@dataclass(frozen=True)
class VadRegion:
    t0: float
    t1: float
    label: Literal["speech", "singing", "music", "non_speech", "unknown"]
    confidence: float | None = None

@dataclass
class LyricsTimeline:
    engine: str
    source_audio: str
    words: list[Word] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)
    vad_regions: list[VadRegion] = field(default_factory=list)
    duration_s: float | None = None
    language: str | None = None
    avg_confidence: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
```

要求：

- 所有时间统一使用 **秒**，基于原始整首歌全局时间轴。
- `t0 <= t1`，非法时间段必须被过滤或抛出 `TimelineValidationError`。
- overlap 合并后 words 按 `t0` 排序。
- 允许 `confidence=None`，但 scorer 必须能处理。
- `source_audio` 指向用于 ASR 的 16k mono 检测副本或缓存路径。

### 5.2 CutCandidate

新增文件：`src/audio_cut/cutting/cut_candidate.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CandidateSource = Literal[
    "acoustic_pause",
    "lyric_word_gap",
    "lyric_sentence_end",
    "firered_vad_boundary",
    "mdd_valley",
    "beat_downbeat",
    "rescue",
]

@dataclass
class CutCandidate:
    t: float
    source: CandidateSource
    score: float = 0.0
    confidence: float | None = None
    window_t0: float | None = None
    window_t1: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    hard_forbidden: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
```

要求：

- 旧的候选切点可以适配为 `CutCandidate(source="acoustic_pause" | "mdd_valley")`。
- `hard_forbidden=True` 的候选不能进入 planner，除非处于 rescue 模式。
- 所有候选都要保留 source 和 reasons，供 Manifest 调试。

### 5.3 BoundaryFeatures

新增文件：`src/audio_cut/analysis/boundary_features.py`

```python
@dataclass
class BoundaryFeatures:
    t: float
    acoustic_pause_score: float = 0.0
    lyric_boundary_score: float = 0.0
    asr_gap_score: float = 0.0
    firered_non_singing_score: float = 0.0
    singing_tail_safety_score: float = 0.0
    vocal_leakage_safety_score: float = 0.0
    duration_fit_score: float = 0.0
    beat_affinity_score: float = 0.0
    mdd_affinity_score: float = 0.0
    zero_crossing_safety_score: float = 0.0
    inside_word_penalty: float = 0.0
    inside_high_conf_singing_penalty: float = 0.0
    onset_conflict_penalty: float = 0.0
```

---

## 6. 歌词时间轴生成

### 6.1 16k mono 检测副本

新增工具函数：`src/audio_cut/utils/audio_resample.py`

要求：

```python
def ensure_16k_mono_pcm_wav(
    input_audio: str | np.ndarray,
    sr: int,
    export_path: str,
) -> str:
    ...
```

实现规则：

- 输入优先使用 `vocal_44k`，不是原始混音。
- 输出必须是 16kHz、mono、PCM_16 wav。
- 不要覆盖原始高质量导出音频。
- 检测副本可进入 `intermediate/` 或 cache 目录。
- 若使用 soundfile 写入，显式指定 `subtype="PCM_16"`。

### 6.2 ASR chunker

新增文件：`src/audio_cut/lyrics/chunker.py`

默认参数：

```yaml
lyrics_alignment:
  chunk_s: 35.0
  overlap_s: 1.0
  max_chunk_s: 45.0
  min_chunk_s: 4.0
  pad_s: 0.2
```

规则：

- FireRedASR2-AED chunk 长度不得超过 60s。
- 默认 chunk 取 35s，overlap 1s。
- 不允许把原曲整段直接送给 ASR。
- 每个 chunk 需要记录：`chunk_id`, `global_t0`, `global_t1`, `path`, `duration_s`。
- overlap 区产生的重复 words，需要在 timeline merge 中去重。

### 6.3 overlap 去重

新增文件：`src/audio_cut/lyrics/timeline.py`

函数：

```python
def merge_chunk_timelines(chunks: list[LyricsTimeline]) -> LyricsTimeline:
    ...
```

去重规则：

1. 把 chunk 内时间转换为全局时间：`global_t = chunk_offset + local_t`。
2. 对相邻 chunk overlap 区的 words 做相似匹配：
   - 文本相同或高度相似。
   - 起止时间差小于 `dedup_tolerance_ms`，默认 250ms。
   - 保留 confidence 更高的 word；若 confidence 缺失，保留更靠近非 overlap 区中心的 word。
3. sentences 可基于合并后的 words 重建或保留 FireRed sentence 后再修正边界。
4. 严禁把 chunk boundary 自身当作切点候选。

---

## 7. LyricsProvider 抽象

### 7.1 接口

新增文件：`src/audio_cut/lyrics/providers.py`

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LyricsProviderRequest:
    vocal_wav_16k: str
    original_audio: str
    duration_s: float
    work_dir: str
    device: str = "auto"
    language: str | None = None
    return_timestamps: bool = True
    return_vad: bool = True

class LyricsProvider(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def analyze(self, req: LyricsProviderRequest) -> LyricsTimeline:
        raise NotImplementedError
```

### 7.2 Provider 实现

#### `NullLyricsProvider`

- 永远可用。
- 返回空 `LyricsTimeline`。
- 用于 `vpbd_acoustic` 和无 ASR 降级。

#### `FakeLyricsProvider`

- 从 JSON fixture 读取 timeline。
- 单测和 integration 测试默认使用。
- 不依赖 FireRed、torch、GPU。

#### `FireRedCliProvider`

- 通过 subprocess 调用外部 FireRed worker。
- 适合 WSL2 本地开发。
- 输入：16k mono wav path。
- 输出：标准 `lyrics_timeline.json`。
- 必须设置 timeout、stderr 捕获、返回码检查。

#### `FireRedSidecarProvider`

- 通过 local HTTP/gRPC 调用 FireRed worker。
- 适合 mvagent 长进程批量任务。
- 支持 `/health` 与 `/analyze`。
- 支持 worker 常驻，避免每首歌重复加载模型。

#### `FireRedInProcessProvider`

- 可选高级路径。
- 仅在明确安装 FireRedASR2S 依赖且版本兼容时启用。
- 不作为默认路径。

### 7.3 Provider 选择逻辑

```python
def build_lyrics_provider(cfg) -> LyricsProvider:
    provider = cfg.lyrics_alignment.provider
    if provider == "disabled":
        return NullLyricsProvider()
    if provider == "fake":
        return FakeLyricsProvider(cfg.fixture_path)
    if provider == "cli":
        return FireRedCliProvider(cfg)
    if provider == "sidecar":
        return FireRedSidecarProvider(cfg)
    if provider == "in_process":
        return FireRedInProcessProvider(cfg)
    if provider == "auto":
        try sidecar -> cli -> in_process -> null
```

---

## 8. 候选切点生成

### 8.1 Acoustic candidates

复用现有 `PureVocalPauseDetector.detect_pure_vocal_pauses`。

要求：

- 不要删除旧逻辑。
- 为新模式增加 adapter，把旧候选转换成 `CutCandidate`。
- 旧候选的原始分数、RMS valley、MDD/VPP 参数写入 `candidate.meta`。

### 8.2 Lyrics candidates

新增 `LyricsBoundaryCandidateGenerator`，来源：

1. **word gap**：相邻词之间的空隙。
2. **sentence end**：FireRed sentence end。
3. **punctuation end**：标点后，弱加分。
4. **mVAD singing boundary**：singing 结束/开始附近。

规则：

```text
word_gap = next_word.t0 - prev_word.t1
candidate_t = center(prev_word.t1, next_word.t0)

if word_gap >= min_word_gap_s:
    source = lyric_word_gap

if sentence_end exists:
    source = lyric_sentence_end
```

默认阈值：

```yaml
lyrics_alignment:
  min_word_gap_s: 0.12
  strong_word_gap_s: 0.30
  max_lyric_candidate_gap_s: 1.50
  sentence_end_bonus: 0.15
  punctuation_bonus: 0.05
```

禁切规则：

- 高置信 word 内部禁止切。
- 高置信连续 singing 区内部禁止切，除非 planner 进入 rescue。
- 低置信 ASR 只降权，不硬禁止。

### 8.3 Beat/MDD candidates

复用现有 MDD、BPM、hybrid_mdd、BeatAnalyzer 结果。

规则：

- beat/downbeat 不应该覆盖演唱语义。
- 仅当候选已通过声学或歌词安全检查时，beat/downbeat 作为加分或吸附目标。
- `hybrid_mdd` 旧模式保持原行为。
- `vpbd_asr` 中的 beat 只作为 `beat_affinity_score` 或弱候选。

---

## 9. PhraseBoundaryScorer

新增文件：`src/audio_cut/analysis/phrase_boundary.py`

### 9.1 默认打分公式

初版手写权重：

```text
score(c) =
  0.35 * acoustic_pause_score
+ 0.20 * lyric_boundary_score
+ 0.15 * asr_gap_score
+ 0.10 * firered_non_singing_score
+ 0.08 * singing_tail_safety_score
+ 0.07 * duration_fit_score
+ 0.03 * beat_affinity_score
+ 0.02 * zero_crossing_safety_score
- 0.60 * inside_word_penalty
- 0.40 * inside_high_conf_singing_penalty
- 0.25 * onset_conflict_penalty
```

要求：

- 权重必须来自配置，不能硬编码不可覆盖。
- 所有分项归一化到 `[0, 1]`。
- 最终分数建议 clamp 到 `[0, 1]`。
- 候选的 `features` 要完整保留，用于 Manifest 和调试。

### 9.2 歌词边界分数

```text
lyric_boundary_score:
  - sentence end: 0.85 ~ 1.0
  - word gap >= strong_word_gap_s: 0.70 ~ 0.90
  - word gap >= min_word_gap_s: 0.40 ~ 0.70
  - punctuation end: +0.05 ~ +0.10
  - ASR confidence low: multiply 0.5 ~ 0.8
```

### 9.3 inside word penalty

```text
if c.t inside any word [word.t0, word.t1]:
    if word.confidence is None:
        penalty = 0.6
    elif word.confidence >= high_confidence:
        penalty = 1.0
    else:
        penalty = 0.4
else:
    penalty = 0.0
```

默认：

```yaml
lyrics_alignment:
  high_word_confidence: 0.70
  forbid_inside_word: true
```

### 9.4 FireRed mVAD 分数

```text
firered_non_singing_score:
  - candidate 在 non-singing gap 内：高分
  - candidate 靠近 singing end/start：中高分
  - candidate 在 high-confidence singing 区内部：低分/penalty
```

---

## 10. GlobalCutPlanner

新增文件：`src/audio_cut/cutting/global_cut_planner.py`

### 10.1 目标

替换“局部选最高分 + NMS”的单阶段策略，在 `finalize_cut_points` 前先做全局规划。

约束目标：

```text
hard_min_s <= segment_duration <= hard_max_s
优先满足 target_min_s <= segment_duration <= target_max_s
切点优先选择高 boundary_score
避免切在 word/singing 内部
避免产生微碎片
无自然停顿时启用 rescue
```

默认配置：

```yaml
global_planner:
  enabled: true
  algorithm: dynamic_programming
  hard_min_s: 3.0
  soft_min_s: 5.0
  target_min_s: 6.0
  target_max_s: 12.0
  soft_max_s: 15.0
  hard_max_s: 22.0
  rescue_enabled: true
  rescue_step_s: 8.0
  max_candidates_per_second: 4
```

### 10.2 动态规划伪代码

```python
points = [0.0] + sorted(candidate_times) + [duration]

cost[0] = 0
prev[0] = None

for j in range(1, len(points)):
    cost[j] = INF
    for i in range(0, j):
        dur = points[j] - points[i]
        if dur < hard_min_s and points[j] != duration:
            continue
        if dur > hard_max_s:
            continue

        boundary_reward = candidate_score(points[j])
        duration_penalty = segment_duration_penalty(dur)
        vocal_risk = candidate_vocal_cut_risk(points[j])
        beat_conflict = beat_conflict_penalty(points[j])

        new_cost = (
            cost[i]
            + duration_penalty
            + vocal_risk
            + beat_conflict
            - boundary_reward
        )
        if new_cost < cost[j]:
            cost[j] = new_cost
            prev[j] = i

return backtrack(prev)
```

### 10.3 duration penalty

```text
0                           if target_min_s <= dur <= target_max_s
small linear penalty         if soft_min_s <= dur < target_min_s
small linear penalty         if target_max_s < dur <= soft_max_s
large penalty                if hard_min_s <= dur < soft_min_s
large penalty                if soft_max_s < dur <= hard_max_s
INF                          otherwise
```

### 10.4 与 finalize_cut_points 的关系

`GlobalCutPlanner` 输出的是“语义/声学规划切点”。之后仍必须进入：

```text
audio_cut.cutting.finalize_cut_points
  - weighted NMS
  - zero crossing refine
  - quiet guard
  - min gap
  - guard_shift_stats
```

注意：

- `finalize_cut_points` 不应删除所有 planner metadata。
- 如果静音守卫把切点移动，要把 `guard_shift_ms` 回写到对应 cut metadata。
- 片段 lyrics 映射应使用守卫后的最终 t0/t1。

---

## 11. API 设计

### 11.1 `separate_and_segment` 新参数

保持旧参数兼容，新增可选参数：

```python
manifest = separate_and_segment(
    input_uri="input/song.mp3",
    export_dir="output/song_job",
    mode="vpbd_asr",
    device="cuda:0",
    export_types=("vocal", "human_segments", "music_segments"),
    layout={"soft_min_s": 5.0, "soft_max_s": 15.0},
    lyrics_alignment={
        "enabled": "auto",
        "provider": "sidecar",
        "strict": False,
        "chunk_s": 35.0,
        "overlap_s": 1.0,
        "use_word_timestamps": True,
        "use_sentence_timestamps": True,
        "use_vad": True,
    },
    fire_red={
        "endpoint": "http://127.0.0.1:8765",
        "model": "FireRedASR2-AED",
        "device": "cuda:0",
        "batch_size": 1,
    },
    export_manifest=True,
)
```

兼容要求：

- 不传 `lyrics_alignment` 时，旧模式行为不变。
- 旧模式传 `lyrics_alignment` 时，应忽略或写 warning，不改变旧切分。
- `mode="vpbd_asr"` 时默认 `lyrics_alignment.enabled="auto"`。

### 11.2 CLI 新参数

`run_splitter.py` 新增：

```bash
python run_splitter.py input/song.mp3 \
  --mode vpbd_asr \
  --gpu-device cuda:0 \
  --lyrics-provider sidecar \
  --firered-endpoint http://127.0.0.1:8765 \
  --asr-chunk-s 35 \
  --asr-overlap-s 1 \
  --asr-strict false
```

`quick_start.py` 新增菜单：

```text
5. VPBD + FireRedASR2S 歌词联合分割（推荐：自然演唱分割）
   - provider: auto / sidecar / cli / fake / disabled
   - GPU: auto / cuda:0 / cpu
   - strict ASR: false / true
```

---

## 12. 配置设计

修改：`config/unified.yaml`

```yaml
vpbd:
  enabled: true
  acoustic_candidate_adapter: true
  use_legacy_pure_vocal_detector: true
  use_singing_activity_detector: true
  merge_candidate_tolerance_ms: 120
  candidate_nms_ms: 180

lyrics_alignment:
  enabled: auto              # auto | true | false
  strict: false
  provider: auto             # auto | sidecar | cli | in_process | fake | disabled
  engine: fireredasr2_aed
  use_vocal_stem: true
  input_sr: 16000
  input_channels: 1
  input_subtype: PCM_16
  chunk_s: 35.0
  overlap_s: 1.0
  max_chunk_s: 45.0
  min_chunk_s: 4.0
  dedup_tolerance_ms: 250
  min_word_gap_s: 0.12
  strong_word_gap_s: 0.30
  high_word_confidence: 0.70
  min_sentence_confidence: 0.50
  forbid_inside_word: true
  forbid_inside_high_conf_singing: true
  use_word_timestamps: true
  use_sentence_timestamps: true
  use_vad: true
  use_punctuation: false
  cache_enabled: true

fire_red:
  sidecar_endpoint: http://127.0.0.1:8765
  cli_path: null
  model: FireRedASR2-AED
  vad_model: FireRedVAD
  device: auto
  batch_size: 1
  timeout_s: 600
  env_name: fireredasr2s
  unload_after_job: false

phrase_boundary:
  weights:
    acoustic_pause: 0.35
    lyric_boundary: 0.20
    asr_gap: 0.15
    firered_non_singing: 0.10
    singing_tail_safety: 0.08
    duration_fit: 0.07
    beat_affinity: 0.03
    zero_crossing_safety: 0.02
  penalties:
    inside_word: 0.60
    inside_high_conf_singing: 0.40
    onset_conflict: 0.25
  min_score: 0.25
  strong_score: 0.70

global_planner:
  enabled: true
  algorithm: dynamic_programming
  hard_min_s: 3.0
  soft_min_s: 5.0
  target_min_s: 6.0
  target_max_s: 12.0
  soft_max_s: 15.0
  hard_max_s: 22.0
  rescue_enabled: true
  rescue_step_s: 8.0
  max_candidates_per_second: 4
  prefer_lyrics_boundary_when_close_ms: 160
  prefer_acoustic_pause_when_close_ms: 120
```

环境变量覆盖示例：

```bash
export VSS__lyrics_alignment__provider=sidecar
export VSS__fire_red__sidecar_endpoint=http://127.0.0.1:8765
export VSS__lyrics_alignment__strict=false
export VSS__global_planner__target_min_s=6.0
export VSS__global_planner__target_max_s=12.0
```

---

## 13. Manifest 扩展

旧字段全部保留。新增字段必须为 optional。

```json
{
  "version": "2.6_vpbd_asr",
  "audio": {
    "sr": 44100,
    "channels": 2,
    "duration": 257.2,
    "hash": "sha256:..."
  },
  "lyrics_alignment": {
    "enabled": true,
    "available": true,
    "strict": false,
    "engine": "FireRedASR2-AED",
    "provider": "sidecar",
    "source": "intermediate/vocal_16k_mono.wav",
    "input_format": {
      "sr": 16000,
      "channels": 1,
      "subtype": "PCM_16"
    },
    "chunks": [
      {"id": 0, "t0": 0.0, "t1": 35.0, "duration": 35.0},
      {"id": 1, "t0": 34.0, "t1": 69.0, "duration": 35.0}
    ],
    "words_count": 418,
    "sentences_count": 56,
    "avg_confidence": 0.82,
    "fallback_reason": null,
    "cache_hit": false,
    "timings_ms": {
      "prepare_audio": 230,
      "asr": 18500,
      "merge_timeline": 40
    }
  },
  "boundary_detection": {
    "algorithm": "vpbd_asr_v1",
    "candidate_sources": {
      "acoustic_pause": 42,
      "lyric_word_gap": 96,
      "lyric_sentence_end": 56,
      "firered_vad_boundary": 31,
      "beat_downbeat": 88
    },
    "planner": "dynamic_programming",
    "weights": {
      "acoustic_pause": 0.35,
      "lyric_boundary": 0.20,
      "asr_gap": 0.15
    }
  },
  "cuts": {
    "final": [
      {
        "t": 18.42,
        "sample": 812322,
        "score": 0.91,
        "reason": "acoustic_pause+lyric_sentence_end",
        "guard_shift_ms": 34,
        "features": {
          "acoustic_pause_score": 0.88,
          "lyric_boundary_score": 0.95,
          "asr_gap_score": 0.73,
          "inside_word_penalty": 0.0
        }
      }
    ],
    "suppressed": []
  },
  "segments": [
    {
      "id": "0004",
      "t0": 18.42,
      "t1": 26.83,
      "dur": 8.41,
      "kind": "human",
      "lyrics": {
        "text": "我终于明白你已远走",
        "confidence": 0.86,
        "word_start": 32,
        "word_end": 41,
        "words": [
          {"t0": 18.60, "t1": 18.82, "text": "我", "confidence": 0.91}
        ]
      },
      "boundary": {
        "start_reason": "guarded_previous_cut",
        "end_reason": "acoustic_pause+lyric_sentence_end",
        "end_score": 0.91,
        "beat_aligned": false
      }
    }
  ]
}
```

### 13.1 片段歌词映射

新增函数：

```python
def attach_lyrics_to_segments(
    segments: list[dict],
    timeline: LyricsTimeline,
    include_partial_words: bool = False,
    boundary_tolerance_s: float = 0.05,
) -> list[dict]:
    ...
```

规则：

- 默认只包含与片段 `[t0, t1]` 有充分重叠的 words。
- word 与片段交界重叠少于 `boundary_tolerance_s` 时可排除。
- `text` 由 words 拼接；中文不加空格，英文按空格。
- 如果无 words，则 `lyrics` 可为空对象或 `null`，不要影响片段导出。

---

## 14. mvagent 集成变更

### 14.1 新推荐调用

mvagent 不再对每个切片重复 ASR。改为：

```python
from audio_cut.api import separate_and_segment

manifest = separate_and_segment(
    input_uri=job.audio_path,
    export_dir=str(job.assets_dir / "audio"),
    mode="vpbd_asr",
    device="cuda:0",
    export_types=("vocal", "human_segments", "music_segments"),
    layout={"soft_min_s": 5.0, "soft_max_s": 15.0},
    lyrics_alignment={
        "enabled": "auto",
        "provider": "sidecar",
        "strict": False,
    },
    fire_red={
        "endpoint": "http://127.0.0.1:8765",
        "model": "FireRedASR2-AED",
    },
    export_manifest=True,
)

# 后续 mvagent 使用：
# manifest["segments"][i]["lyrics"]
# manifest["lyrics_alignment"]
# manifest["cuts"]["final"]
```

### 14.2 依赖方向

正确依赖方向：

```text
mvagent -> audio_cut.api -> optional FireRed provider
```

禁止：

```text
audio-cut import mvagent
```

如果 mvagent 已有 FireRed worker，可由 mvagent 传入 endpoint 或 provider 配置；audio-cut 不应该依赖 mvagent 内部包名。

### 14.3 缓存键

mvagent 的 job cache 应包含：

```text
audio_hash
separator_model + separator_version
mode
layout config
lyrics provider
FireRedASR2S model/version/path
ASR chunk_s / overlap_s
phrase_boundary weights
global_planner config
```

否则同一音频在不同 ASR/切分参数下可能错误复用旧 Manifest。

---

## 15. 性能与资源策略

### 15.1 中端 GPU 默认策略

| 硬件 | 默认策略 |
|---|---|
| 12GB+ NVIDIA GPU | `vpbd_asr` 可默认启用，FireRedASR2-AED 串行 chunk |
| 8GB NVIDIA GPU | `vpbd_asr` 可用，但 batch_size=1，分离与 ASR 不同时驻留 |
| 4GB NVIDIA GPU | 默认降级 `vpbd_acoustic`，ASR 需显式开启 |
| CPU-only | 默认 `vpbd_acoustic` 或旧模式，只允许 FireRedVAD/ASR 慢速手动模式 |

### 15.2 显存避免冲突

实现建议：

1. 分离完成后，释放 separator 模型或切换到 worker 进程执行 ASR。
2. ASR worker `batch_size=1`。
3. 每首歌处理完可选 `torch.cuda.empty_cache()`，但不要在 tight loop 中频繁调用。
4. sidecar worker 应提供 `/unload` 或 `unload_after_job` 选项。
5. 严格记录 `gpu_meta`、`lyrics_alignment.timings_ms`、`peak_mem_bytes`。

### 15.3 CPU-only 降级

CPU-only 不应阻断主流程：

```text
lyrics provider unavailable -> NullLyricsProvider -> vpbd_acoustic
```

如果用户显式要求 ASR：

```text
lyrics_alignment.strict=true
provider=cli/sidecar
```

则失败时抛异常，便于上层任务感知。

---

## 16. 测试计划

### 16.1 Unit tests

新增：

```text
tests/unit/test_lyrics_models.py
  - Word/Sentence/VadRegion 时间合法性
  - LyricsTimeline 序列化/反序列化

src/audio_cut/lyrics/test_chunker.py 或 tests/unit/test_lyrics_chunker.py
  - 35s chunk + 1s overlap
  - 不超过 max_chunk_s
  - 最后一段短音频处理

 tests/unit/test_lyrics_timeline_merge.py
  - overlap 重复词去重
  - chunk local time -> global time
  - confidence 缺失时稳定合并

 tests/unit/test_fake_lyrics_provider.py
  - fixture 加载
  - provider unavailable/available 行为

 tests/unit/test_boundary_features.py
  - inside word penalty
  - ASR gap score
  - sentence end bonus
  - high-confidence singing penalty

 tests/unit/test_phrase_boundary_scorer.py
  - 权重可配置
  - 分数 clamp
  - hard_forbidden candidate 行为

 tests/unit/test_global_cut_planner.py
  - 满足 hard_min/hard_max
  - 优先 5-15s/6-12s
  - 避免微碎片
  - rescue 模式

 tests/unit/test_manifest_vpbd_asr.py
  - lyrics_alignment 字段
  - segment.lyrics 字段
  - cuts.final.features 字段
```

### 16.2 Integration tests

新增：

```text
tests/integration/test_pipeline_vpbd_asr_fake_provider.py
  - 使用短 synthetic audio + fake lyrics timeline
  - mode=vpbd_asr
  - 不依赖 FireRed/GPU
  - 验证不切在 word 内部
  - 验证片段 lyrics 正确附着

 tests/integration/test_pipeline_vpbd_acoustic_fallback.py
  - FireRed unavailable
  - strict=false
  - 自动 fallback
  - Manifest 记录 fallback_reason

 tests/integration/test_pipeline_vpbd_asr_strict_failure.py
  - FireRed unavailable
  - strict=true
  - 抛出 LyricsAlignmentUnavailable
```

### 16.3 Optional FireRed tests

标记为 `pytest.mark.firered` 和 `pytest.mark.gpu`：

```text
tests/integration/test_firered_cli_provider_real.py
  - 需要 FIRE_RED_CLI_PATH 或 FIRE_RED_ENDPOINT
  - 输入 5-10s wav
  - 验证返回 words/sentences/vad
```

默认 CI 不跑真实 FireRed 测试。

### 16.4 Regression tests

必须确保：

```bash
pytest -m "not slow and not gpu and not firered"
pytest tests/contracts/test_config_contracts.py
pytest tests/unit/test_api_manifest.py
pytest tests/unit/test_cpu_baseline_perfect_reconstruction.py
pytest tests/unit/test_cutting_consistency.py
```

旧模式输出契约：

- `v2.2_mdd` 不新增必需参数。
- `hybrid_mdd` 的 `_lib` 标记不改变。
- 输出文件名 `_X.X` 秒后缀不改变。
- `SegmentManifest.json` 旧字段不删除。

---

## 17. 人工验收指标

### 17.1 质量指标

最低目标：

```text
boundary_f1_500ms >= 0.82
cut_inside_word_rate <= 1%
cut_inside_high_conf_singing_rate <= 3%
5-15s segment pass rate >= 90%
subjective_naturalness >= 4.2 / 5
manual_recutter_rate 相比 v2.5.1 降低 >= 40%
```

如果没有人工标注集，先做 20 首手工 QA playlist：

```text
- 中文流行慢歌 3 首
- 中文快歌/rap 3 首
- 英文流行 3 首
- 民谣/低动态 3 首
- 强节奏副歌 3 首
- 和声/ad-lib 明显 3 首
- 器乐 intro/outro 长 2 首
```

### 17.2 自动统计

Manifest 或 QA report 输出：

```text
segments_count
median_segment_s
segment_5_15_pass_rate
cut_inside_word_rate
cut_inside_singing_rate
avg_boundary_score
lyrics_coverage_ratio
asr_avg_confidence
guard_shift_p50_ms / p95_ms
fallback_reason
```

---

## 18. 分阶段任务清单

### Phase 0：仓库勘察与保护旧行为

- [ ] 阅读 `development.md`，确认当前 SSOT。
- [ ] 阅读 `README.md`，确认 CLI/API/输出结构。
- [ ] 阅读 `audio-cut封装为模块.md`，确认 mvagent 进程内调用契约。
- [ ] 运行现有快速测试，记录 baseline。
- [ ] 新建 feature branch：`feature/v2.6-vpbd-asr`。

提交建议：

```text
chore: add vpbd_asr refactor scaffold without behavior changes
```

### Phase 1：配置和数据模型

- [ ] 新增 `src/audio_cut/lyrics/models.py`。
- [ ] 新增 `src/audio_cut/cutting/cut_candidate.py`。
- [ ] 新增 `src/audio_cut/analysis/boundary_features.py`。
- [ ] 扩展 `config/unified.yaml`。
- [ ] 扩展配置 schema/contract tests。
- [ ] 新增序列化 helper。

验收：

```bash
pytest tests/contracts/test_config_contracts.py
pytest tests/unit/test_lyrics_models.py
```

### Phase 2：LyricsProvider 与 fake provider

- [ ] 新增 `LyricsProvider` 抽象。
- [ ] 实现 `NullLyricsProvider`。
- [ ] 实现 `FakeLyricsProvider`。
- [ ] 新增 fixture：`tests/fixtures/lyrics/simple_song_timeline.json`。
- [ ] 实现 provider auto selection。

验收：

```bash
pytest tests/unit/test_fake_lyrics_provider.py
```

### Phase 3：ASR chunker 与 timeline merge

- [ ] 实现 `ensure_16k_mono_pcm_wav`。
- [ ] 实现 `lyrics/chunker.py`。
- [ ] 实现 `merge_chunk_timelines`。
- [ ] 实现 overlap words 去重。
- [ ] 实现 timeline validation。

验收：

```bash
pytest tests/unit/test_lyrics_chunker.py
pytest tests/unit/test_lyrics_timeline_merge.py
```

### Phase 4：候选生成、特征、打分

- [ ] 实现旧 acoustic candidate adapter。
- [ ] 实现 lyrics candidate generator。
- [ ] 实现 FireRed VAD/mVAD boundary candidate generator。
- [ ] 实现 `BoundaryFeatureExtractor`。
- [ ] 实现 `PhraseBoundaryScorer`。
- [ ] 输出候选调试 JSON。

验收：

```bash
pytest tests/unit/test_boundary_features.py
pytest tests/unit/test_phrase_boundary_scorer.py
```

### Phase 5：GlobalCutPlanner

- [ ] 实现 DP planner。
- [ ] 实现 duration penalty。
- [ ] 实现 rescue candidate。
- [ ] 实现 candidate pruning。
- [ ] 与 `finalize_cut_points` 串接。

验收：

```bash
pytest tests/unit/test_global_cut_planner.py
pytest tests/unit/test_cutting_consistency.py
```

### Phase 6：SeamlessSplitter 接入

- [ ] 新增 `vocal_phrase_boundary_detector.py`。
- [ ] `mode="vpbd_asr"` 时调用新链路。
- [ ] `mode="vpbd_acoustic"` 时调用无 ASR 链路。
- [ ] 旧模式路径不改。
- [ ] 记录 `boundary_detection` 和 `lyrics_alignment` meta。
- [ ] 处理 fallback。

验收：

```bash
pytest tests/integration/test_pipeline_vpbd_asr_fake_provider.py
pytest tests/integration/test_pipeline_vpbd_acoustic_fallback.py
pytest tests/integration/test_pipeline_vpbd_asr_strict_failure.py
```

### Phase 7：Manifest、片段歌词、mvagent 契约

- [ ] `ResultBuilder` 增加 optional lyrics fields。
- [ ] `SegmentExporter` 保持文件名兼容。
- [ ] 实现 `attach_lyrics_to_segments`。
- [ ] `SegmentManifest.json` 增加 schema/contract 测试。
- [ ] 更新 `audio-cut封装为模块.md` 示例。

验收：

```bash
pytest tests/unit/test_api_manifest.py
pytest tests/unit/test_manifest_vpbd_asr.py
```

### Phase 8：FireRed CLI/sidecar provider

- [ ] 实现 `FireRedCliProvider`。
- [ ] 实现 `FireRedSidecarProvider`。
- [ ] 定义 worker JSON 输入输出协议。
- [ ] 增加 provider health check。
- [ ] 增加 timeout 和错误降级。
- [ ] 增加 `pytest.mark.firered` 测试。

验收：

```bash
pytest -m firered tests/integration/test_firered_cli_provider_real.py
```

### Phase 9：CLI/quick_start/文档

- [ ] `run_splitter.py` 增加 vpbd_asr 参数。
- [ ] `quick_start.py` 增加菜单项。
- [ ] README 新增 v2.6 使用说明。
- [ ] development.md 更新 SSOT。
- [ ] release notes 草稿。

验收：

```bash
python run_splitter.py input/song.mp3 --mode vpbd_asr --lyrics-provider fake
python quick_start.py
```

---

## 19. FireRed worker JSON 协议建议

### 19.1 `/health`

响应：

```json
{
  "ok": true,
  "engine": "FireRedASR2-AED",
  "device": "cuda:0",
  "loaded": true,
  "version": "unknown"
}
```

### 19.2 `/analyze`

请求：

```json
{
  "audio_path": "/abs/path/to/vocal_16k_mono.wav",
  "chunks": [
    {"id": 0, "path": "/abs/path/chunk_000.wav", "t0": 0.0, "t1": 35.0},
    {"id": 1, "path": "/abs/path/chunk_001.wav", "t0": 34.0, "t1": 69.0}
  ],
  "return_timestamps": true,
  "return_vad": true,
  "language": null
}
```

响应：

```json
{
  "engine": "FireRedASR2-AED",
  "duration_s": 257.2,
  "chunks": [
    {
      "id": 0,
      "t0": 0.0,
      "t1": 35.0,
      "words": [
        {"start_ms": 420, "end_ms": 780, "text": "我", "confidence": 0.91}
      ],
      "sentences": [
        {"start_ms": 420, "end_ms": 5120, "text": "我终于明白", "asr_confidence": 0.86, "lang": "zh"}
      ],
      "vad_segments_ms": [[300, 5400]],
      "mvads": [
        {"start_ms": 300, "end_ms": 5400, "label": "singing", "confidence": 0.94}
      ]
    }
  ],
  "timings_ms": {
    "asr": 18500,
    "vad": 210
  }
}
```

注意：

- Worker 可以返回 chunk-local ms，audio-cut 负责转全局秒。
- 如果 FireRed 返回的 word 没有 confidence，置为 `null`。
- 如果 FireRed 没有 mVAD，`mvads` 可为空。

---

## 20. 错误处理

新增异常：

```python
class LyricsAlignmentUnavailable(RuntimeError): ...
class FireRedProviderError(RuntimeError): ...
class TimelineValidationError(ValueError): ...
class GlobalCutPlanningError(RuntimeError): ...
```

错误策略：

| 场景 | strict=false | strict=true |
|---|---|---|
| FireRed provider 不可用 | fallback `vpbd_acoustic`，Manifest 记录 reason | 抛 `LyricsAlignmentUnavailable` |
| FireRed timeout | fallback，记录 timeout | 抛 `FireRedProviderError` |
| timeline 部分非法 | 尽量过滤非法项，记录 warnings | 抛 `TimelineValidationError` |
| planner 无可行路径 | rescue planning | 抛或 fallback 旧 layout，视配置 |
| segment lyrics 为空 | 正常输出 | 正常输出 |

---

## 21. 日志规范

新增 logger：

```text
audio_cut.lyrics
audio_cut.lyrics.firered
audio_cut.boundary
audio_cut.planner
```

关键日志：

```text
[VPBD] mode=vpbd_asr provider=sidecar enabled=true strict=false
[LYRICS] prepared vocal_16k path=... dur=257.2s
[LYRICS] chunks=8 chunk_s=35.0 overlap_s=1.0
[LYRICS] provider=sidecar cache_hit=false words=418 sentences=56 avg_conf=0.82
[BOUNDARY] candidates acoustic=42 lyric_gap=96 sentence=56 vad=31 beat=88 merged=173
[PLANNER] selected_cuts=28 median_segment=8.7s pass_5_15=0.93
[VPBD] fallback provider_unavailable -> vpbd_acoustic
```

---

## 22. 回滚策略

如果 `vpbd_asr` 出现严重问题：

1. 保留代码，但默认配置 `lyrics_alignment.enabled=false`。
2. CLI 菜单隐藏或标记 experimental。
3. mvagent 调用改回 `mode="hybrid_mdd"` 或 `mode="v2.2_mdd"`。
4. 因旧模式未改动，输出兼容性不受影响。

---

## 23. 不在 v2.6 做的事

- 不训练新深度学习模型。
- 不引入 FireRedASR2-LLM 作为默认。
- 不做 GUI。
- 不做云端 ASR 服务。
- 不改变旧输出文件命名。
- 不把 FireRedASR2S 依赖加入 audio-cut base install。
- 不要求 CPU-only 跑完整 ASR 联合分割。

---

## 24. 推荐提交顺序

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
10. feat(firered): add cli/sidecar lyrics providers
11. test: add vpbd_asr fake-provider integration coverage
12. docs: update README, development, mvagent integration guide
```

---

## 25. 最终验收命令

基础回归：

```bash
pytest -m "not slow and not gpu and not firered" --cov=src --cov-report=term-missing
```

旧模式回归：

```bash
python run_splitter.py input/song.mp3 --mode v2.2_mdd
python run_splitter.py input/song.mp3 --mode hybrid_mdd
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

Manifest 检查：

```bash
python scripts/diagnostics/inspect_manifest.py output/.../SegmentManifest.json \
  --check-lyrics \
  --check-boundary-scores \
  --check-no-cut-inside-word
```

---

## 26. 核心判断

v2.6 不应该把 audio-cut 改成“ASR 切歌器”。正确目标是：

```text
MDX23/Demucs 负责把人声 stem 变干净；
FireRedASR2S 负责提供歌词、词级时间戳和 singing/music 证据；
VPBD 负责判断哪里像人类剪辑师会切；
GlobalCutPlanner 负责在 5–15 秒目标下选择全局最优切点；
finalize_cut_points 和 segment_layout_refiner 继续负责工程安全。
```

这条路径可以在 WSL2 Ubuntu 22.04 + 中端 NVIDIA GPU 上落地，并且能在 FireRed 不可用或 CPU-only 时保持兼容降级。

# audio-cut agent 调用契约（v2.8）

目标：上层 agent 只表达剪辑意图，不理解内部算法名词。一次调用完成分离、切分、布局、导出和 `SegmentManifest.json`。

## 1. 稳定入口

```python
from audio_cut.api import separate_and_segment

manifest = separate_and_segment(
    input_uri="input/track.mp3",
    export_dir="output/job",
    segments="medium",      # few | medium | many，或 (min_s, max_s)
    alignment=0.75,          # lyric..beat 档位名，或 0.0-1.0
    export_types=("vocal", "human_segments", "music_segments"),
    export_manifest=True,
)
```

兼容承诺：既有参数保持可选且语义不变；不传 `segments/alignment/mode` 时仍沿用旧默认 `v2.2_mdd`。给出意图参数且未显式指定 `mode` 时，路由统一候选池路径；显式 `mode` 永远优先。

## 2. 意图参数

| 参数 | 人类档位 | 数值真值 |
| --- | --- | --- |
| `segments` | `few` / `medium` / `many` | `[10,18]` / `[5,12]` / `[3,8]` 秒 |
| `alignment` | `lyric` / `lyric_lean` / `balanced` / `beat_lean` / `beat` | `0.0` / `0.25` / `0.5` / `0.75` / `1.0` |

`alignment=0.5` 是恒等点；越靠左越偏歌词和自然停顿，越靠右越偏节拍，但词内和高人声风险惩罚不放松。纯节拍硬切仍属于专家旧模式，不是默认 agent 契约。

## 3. Manifest 增量字段

v2.8 只增字段，不删除旧字段。agent 应重点读取：

```json
{
  "intent": {
    "target_duration_s": [5.0, 12.0],
    "segments": "medium",
    "alignment": 0.75,
    "alignment_raw": 0.75,
    "lyrics": "auto",
    "profile": "auto",
    "applied_overrides": ["phrase_boundary.weights.beat_affinity"]
  },
  "segments": [
    {"id": "0001", "start": 0.0, "end": 5.2, "lyrics": {"text": "..."}}
  ],
  "qa_report": {
    "beat_aligned_ratio": 0.0,
    "breath_cut_ratio": 0.0,
    "cut_inside_word_rate": 0.0
  }
}
```

`intent.applied_overrides` 是可选诊断字段：真实统一引擎路径会回显 alignment 实际改动的运行时键；轻量封装、测试替身或旧路径可能不提供该字段。`segments[*].lyrics` 是可选字段；没有歌词覆盖时可能不存在或为 `null`。`qa_report` 是 agent 闭环检查入口，用于判断卡点比例、气口自然度和词内切割风险。

## 4. 降级语义

- `lyrics=auto`：provider 不可用或 ASR 失败时降级到声学候选继续处理，并在 Manifest 相关诊断字段记录原因。
- GPU 失败：默认回退 CPU；只有 `strict_gpu=True` 时失败即中止。
- 可选依赖缺失：默认走可用路径；不应让批量任务因单一 provider 缺失整体崩溃。
- 可复现：同输入、同 intent 数值、同配置与模型资产应产生稳定输出；AutoProfile 估计结果会随 Manifest 落盘，复跑时可固定 `profile`。

## 5. CLI 兜底

```bash
python run_splitter.py input/track.mp3 --segments medium --align beat_lean
python run_splitter.py input/track.mp3 --segments 6-14 --align 0.8
python run_splitter.py input/track.mp3 --mode v2.2_mdd
```

上层集成优先使用 Python API；CLI 仅作为进程隔离或应急兜底。

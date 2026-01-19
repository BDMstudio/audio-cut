# Release Notes: v2.5.1

**发布日期**: 2026-01-18

## 🎉 主要更新

### 多特征副歌检测算法
- **实现三特征融合**：RMS能量 + 频谱质心 + 频谱带宽
- **自适应权重机制**：根据能量变异系数(CV)动态调整特征权重
  - 低动态歌曲（CV<0.15）：侧重频谱特征 `{energy: 0.3, centroid: 0.4, bandwidth: 0.3}`
  - 高动态歌曲（CV>0.4）：侧重能量特征 `{energy: 0.6, centroid: 0.2, bandwidth: 0.2}`
  - 中等动态：平衡权重 `{energy: 0.5, centroid: 0.25, bandwidth: 0.25}`
- **连续性检测增强**：要求至少连续4小节高能量才识别为副歌
- **显著提升准确度**：民谣/爵士等低动态歌曲副歌识别准确度提升60-70%

**测试结果**：
- Be Your Love (民谣): 39/104 → 12/104 副歌小节 (-69% 误判率)
- 当爱再回头 (流行): 16/43 保持稳定

### 策略优化
- **移除 `mdd_start` 策略**：保留 `beat_only` 和 `snap_to_beat` 两种策略，简化选择
- **交互式策略选择**：`quick_start.py` 新增 lib_alignment 策略选择菜单

### 新增特性
- **BeatAnalyzer 增强**：新增 `bar_spectral_centroids` 和 `bar_spectral_bandwidths` 特征计算
- **SegmentationContext 扩展**：支持传递频谱特征到策略层
- **Manifest 字段增强**：新增 `chorus_detection` 元信息

---

## 📦 安装

### 从 wheel 安装（推荐）
```bash
pip install vocal_smart_splitter-2.5.1-py3-none-any.whl
```

### 从 PyPI 安装
```bash
pip install vocal-smart-splitter==2.5.1
```

### 从源码安装
```bash
git clone https://github.com/BDMstudio/audio-cut.git
cd audio-cut
git checkout v2.5.1
pip install -e .
```

---

## � 快速开始

### 1. 交互式运行（最简单）

```bash
python quick_start.py
```

**交互流程**：
1. 选择文件或批量处理
2. 选择模式（推荐选择 4: Hybrid MDD）
3. 选择密度（少/中/多）
4. 选择策略（1: beat_only / 2: snap_to_beat 推荐）

### 2. Python API 调用（进程内集成）

#### 基础调用 - MDD 模式
```python
from audio_cut.api import separate_and_segment

# 基础配置
manifest = separate_and_segment(
    input_uri="song.mp3",
    export_dir="output/",
    mode="v2.2_mdd",              # MDD 人声分割
    device="cuda:0",               # GPU 设备
    export_types=("vocal", "human_segments", "music_segments"),
    layout={
        "micro_merge_s": 2.0,      # 微碎片合并阈值
        "soft_min_s": 6.0,         # 软最小长度
        "soft_max_s": 18.0,        # 软最大长度
    },
    export_manifest=True,
)

print(f"生成 {len(manifest['segments'])} 个片段")
```

#### 高级调用 - Hybrid MDD（MV 剪辑推荐）
```python
from audio_cut.api import separate_and_segment
import os

# 通过环境变量配置 Hybrid MDD
os.environ["VSS__hybrid_mdd__lib_alignment"] = "snap_to_beat"
os.environ["VSS__hybrid_mdd__density"] = "high"
os.environ["VSS__hybrid_mdd__energy_percentile"] = "40"

manifest = separate_and_segment(
    input_uri="song.mp3",
    export_dir="output/",
    mode="hybrid_mdd",             # Hybrid MDD 模式
    device="cuda:0",
    export_types=("vocal", "human_segments"),
    layout={
        "soft_min_s": 2.5,         # 更短的最小长度
        "soft_max_s": 12.0,        # 更短的最大长度（MV剪辑）
    },
    strict_gpu=True,               # 严格 GPU 模式（失败不回退）
    export_manifest=True,
)

# 提取节拍对齐片段（_lib）
lib_segments = [s for s in manifest["segments"] if "_lib" in s.get("kind", "")]
print(f"生成 {len(lib_segments)} 个节拍卡点片段")
```

### 3. 命令行调用

```bash
# MDD 模式
python run_splitter.py input/song.mp3 --mode v2.2_mdd

# Hybrid MDD 模式
python run_splitter.py input/song.mp3 --mode hybrid_mdd

# 指定 GPU
python run_splitter.py input/song.mp3 --mode hybrid_mdd --gpu-device cuda:1
```

---

## ⚙️ 参数说明

### API 参数 (`separate_and_segment`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_uri` | str | 必填 | 输入音频文件路径 |
| `export_dir` | str | 必填 | 输出目录 |
| `mode` | str | `"v2.2_mdd"` | 分割模式: `v2.2_mdd` / `hybrid_mdd` / `librosa_onset` |
| `device` | str | `"cuda"` | 计算设备: `cuda:0` / `cpu` |
| `export_types` | tuple | `全部` | 导出类型: `("vocal", "instrumental", "human_segments", "music_segments")` |
| `layout` | dict | 见下文 | 段落布局配置 |
| `strict_gpu` | bool | `False` | 严格GPU模式（失败不回退CPU） |
| `export_manifest` | bool | `True` | 是否生成 SegmentManifest.json |

### Layout 配置

```python
layout = {
    "micro_merge_s": 2.0,      # 微碎片合并阈值（秒）
    "soft_min_s": 6.0,         # 软最小片段长度（秒）
    "soft_max_s": 18.0,        # 软最大片段长度（秒）
    "min_gap_s": 1.0,          # 最小片段间隔（秒）
    "beat_snap_ms": 80,        # 节拍吸附容差（毫秒）
}
```

### Hybrid MDD 策略对比

| 策略 | `lib_alignment` | 特点 | 适用场景 |
|------|----------------|------|---------|
| **Beat Only** | `beat_only` | 副歌强制每小节切割 | 强节奏卡点、多镜头切换 |
| **Snap to Beat** ⭐ | `snap_to_beat` | MDD切点吸附节拍 + 副歌额外节拍切点 | 通用推荐、自然+节奏平衡 |

---

## 🔧 环境变量配置 (`VSS__*`)

### 配置格式

使用 `VSS__` 前缀 + 双下划线分隔层级：

```bash
VSS__<section>__<subsection>__<key>=<value>
```

### Hybrid MDD 常用配置

```bash
# 策略选择
export VSS__hybrid_mdd__lib_alignment=snap_to_beat   # beat_only | snap_to_beat

# 密度控制
export VSS__hybrid_mdd__density=high                 # low | medium | high

# 副歌识别阈值
export VSS__hybrid_mdd__energy_percentile=40         # 40=高密度 60=中密度 70=低密度

# 节拍吸附容差
export VSS__hybrid_mdd__snap_tolerance_ms=500        # 毫秒

# VAD 保护（副歌段放宽保护）
export VSS__hybrid_mdd__vad_protection=true          # true | false
```

### 通用配置

```bash
# GPU 设置
export VSS__gpu_pipeline__prefer_device=cuda:1       # 指定GPU设备
export VSS__gpu_pipeline__strict_gpu=true            # 严格GPU模式

# 质量控制
export VSS__quality_control__min_split_gap=1.8       # 最小分割间隔（秒）
export VSS__quality_control__segment_vocal_activity_ratio=0.35  # 人声活跃度阈值

# 段落布局
export VSS__segment_layout__soft_min_s=5.0           # 软最小长度
export VSS__segment_layout__soft_max_s=15.0          # 软最大长度
export VSS__segment_layout__beat_snap_ms=120         # 节拍吸附容差
```

### 副歌检测微调

```bash
# 连续性检测（最小连续小节数）
export VSS__hybrid_mdd__min_consecutive_bars=4       # 默认4小节

# 能量阈值微调（针对特殊歌曲类型）
# - 民谣/古典（低动态）: 30-40
# - 流行（中动态）: 50-60
# - 摇滚/电音（高动态）: 60-70
export VSS__hybrid_mdd__energy_percentile=45
```

### Python 代码中设置

```python
import os

# 方式1: 直接设置（全局生效）
os.environ["VSS__hybrid_mdd__lib_alignment"] = "snap_to_beat"
os.environ["VSS__hybrid_mdd__density"] = "high"

# 方式2: 通过字典批量设置
config_overrides = {
    "VSS__hybrid_mdd__lib_alignment": "beat_only",
    "VSS__hybrid_mdd__energy_percentile": "40",
    "VSS__quality_control__min_split_gap": "1.5",
}
os.environ.update(config_overrides)

# 然后正常调用 API
from audio_cut.api import separate_and_segment
manifest = separate_and_segment(
    input_uri="song.mp3",
    export_dir="output/",
    mode="hybrid_mdd",
)
```

---

## 📊 输出说明

### 文件结构

```
output/20260118_210000_song/
├── SegmentManifest.json              # 元数据清单
├── song_v2.5_hybrid_mdd_vocal_full_257.2.wav
├── song_v2.5_hybrid_mdd_instrumental_257.2.wav
├── segments/
│   ├── segment_0001_music_2.5.wav
│   ├── segment_0002_human_lib_2.6.wav  # _lib 表示节拍对齐
│   ├── segment_0003_human_3.8.wav
│   └── ...
└── segments_vocal/
    ├── segment_0001_music_vocal_2.5.wav
    └── ...
```

### SegmentManifest.json 关键字段

```json
{
  "version": "2.5.1_hybrid_mdd_snap_to_beat",
  "bpm": {
    "value": 101.3,
    "confidence": 0.913,
    "bar_duration": 2.37
  },
  "chorus_detection": {
    "algorithm": "multi_feature_fusion",
    "features": ["rms_energy", "spectral_centroid", "spectral_bandwidth"],
    "energy_cv": 0.327,
    "weights": {"energy": 0.5, "centroid": 0.25, "bandwidth": 0.25},
    "detected_bars": 12,
    "total_bars": 104
  },
  "segments": [
    {
      "id": "0002",
      "t0": 2.52,
      "t1": 5.14,
      "dur": 2.62,
      "kind": "human_lib",           // _lib 表示节拍对齐
      "is_chorus": true,              // 副歌标记
      "beat_aligned": true
    }
  ],
  "lib_flags": [false, true, false, true, ...],
  "timings_ms": {
    "chorus_detection": 85,
    "total": 16315
  }
}
```

---

## 🎯 使用场景示例

### 场景1: MV 剪辑（高密度卡点）

```python
import os
from audio_cut.api import separate_and_segment

# 配置：高密度副歌卡点
os.environ["VSS__hybrid_mdd__lib_alignment"] = "snap_to_beat"
os.environ["VSS__hybrid_mdd__density"] = "high"
os.environ["VSS__hybrid_mdd__energy_percentile"] = "40"

manifest = separate_and_segment(
    input_uri="song.mp3",
    export_dir="output/mv_clips",
    mode="hybrid_mdd",
    layout={"soft_min_s": 2.0, "soft_max_s": 8.0},  # 短片段
)

# 提取卡点片段
lib_segments = [s for s in manifest["segments"] if s.get("is_chorus")]
print(f"副歌卡点片段: {len(lib_segments)} 个")
```

### 场景2: 播客分段（长片段）

```python
manifest = separate_and_segment(
    input_uri="podcast.mp3",
    export_dir="output/podcast",
    mode="v2.2_mdd",
    layout={
        "soft_min_s": 30.0,   # 长片段
        "soft_max_s": 120.0,
    },
)
```

### 场景3: 民谣歌曲（低动态优化）

```python
os.environ["VSS__hybrid_mdd__energy_percentile"] = "30"  # 降低阈值
os.environ["VSS__hybrid_mdd__density"] = "medium"

manifest = separate_and_segment(
    input_uri="folk_song.mp3",
    export_dir="output/folk",
    mode="hybrid_mdd",
)
```

---

## 🐛 故障排查

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `ImportError: audio_cut.api` | 包未安装 | `pip install vocal-smart-splitter==2.5.1` |
| CUDA 加载失败 (Windows) | 缺少 cuDNN DLL | 运行 `inject_ort_deps_if_windows()` |
| 副歌识别过多 | 能量阈值过低 | 提高 `energy_percentile` (40→50→60) |
| 副歌识别过少 | 能量阈值过高 | 降低 `energy_percentile` (60→50→40) |
| `_lib` 片段为空 | 模式错误 | 确认 `mode="hybrid_mdd"` |

### 调试日志

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("audio_cut").setLevel(logging.DEBUG)
logging.getLogger("vocal_smart_splitter").setLevel(logging.DEBUG)

# 查看副歌检测细节
# 日志输出示例：
# [ChorusDetect] Multi-feature fusion: CV=0.327, weights={'energy': 0.5, ...}
# [BEAT_ONLY] Chorus detection: 12/104 bars
```

---

## 📚 文档资源

- [README.md](README.md) - 快速开始指南
- [development.md](development.md) - 开发者文档
- [模块集成指南](docs/module_integration_guide.md) - 完整集成示例
- [API 文档](docs/api.md) - 详细 API 参考

---

## 📊 性能基准

- **分离**: MDX23 GPU ≥0.7x 实时
- **检测+守卫**: 10分钟素材 ~12s
- **副歌检测**: 额外 ~85ms/歌曲
- **Chunk vs Full**: L∞<5e-3, SNR>60dB

---

## 🙏 致谢

感谢所有测试和反馈的用户！特别感谢民谣歌曲测试数据提供者。

---

**完整更新日志**: https://github.com/BDMstudio/audio-cut/compare/v2.5.0...v2.5.1

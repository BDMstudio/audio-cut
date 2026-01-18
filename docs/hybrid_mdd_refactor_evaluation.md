# Hybrid MDD 重构评估报告

> **评估目的**：分析 `seamless_splitter.py` 的现状，评估实现方案 B/C 的难度，并提出重构建议。

---

## 一、现状分析

### 1.1 文件规模

| 指标 | 数值 | 评估 |
|-----|------|-----|
| 总行数 | **2,144 行** | 🔴 过大，需拆分 |
| 方法数 | 34 个 | 🟡 中等 |
| 最大方法 | `_process_hybrid_mdd_split` (409 行) | 🔴 严重过长 |
| 次大方法 | `_process_pure_vocal_split` (403 行) | 🔴 严重过长 |
| 第三大 | `_process_librosa_onset_split` (316 行) | 🟡 偏长 |

### 1.2 职责分析

当前 `SeamlessSplitter` 承担了**过多职责**：

```
SeamlessSplitter
├── 模式路由 (split_audio_seamlessly)
├── 人声分离 (vocal separation)
├── MDD 停顿检测调用
├── 节拍分析 (librosa beat tracking)
├── 能量分析 (RMS energy)
├── 切点合并与过滤
├── 片段时长约束
├── 人声/音乐分类
├── 文件导出
└── 质量控制 (PrecisionGuard)
```

**违反单一职责原则 (SRP)**

---

## 二、方案 B/C 实现难度评估

### 2.1 方案 B：纯节拍分割

```python
# 核心逻辑：完全忽略 MDD，只用节拍
cut_points = bar_times[::bars_per_cut]
```

| 评估项 | 难度 | 说明 |
|-------|-----|------|
| 算法复杂度 | ✅ 低 | 直接使用 librosa bar_times |
| 代码改动量 | ✅ 低 | ~50 行新增 |
| 测试覆盖 | ✅ 低 | 时长固定，易于验证 |
| **总体难度** | **⭐ 简单** | 1-2 小时可完成 |

### 2.2 方案 C：MDD 吸附到节拍

```python
# 核心逻辑：MDD 切点吸附到最近节拍（带 VAD 保护）
for mdd_cut in mdd_cuts:
    nearest_beat = find_nearest_beat(mdd_cut, bar_times)
    if abs(mdd_cut - nearest_beat) < snap_tolerance:
        if not would_cut_vocal(nearest_beat, vad_mask):
            snapped_cuts.append(nearest_beat)
        else:
            snapped_cuts.append(mdd_cut)  # 保留原位置
```

| 评估项 | 难度 | 说明 |
|-------|-----|------|
| 算法复杂度 | 🟡 中 | 需要 VAD 保护逻辑 |
| 代码改动量 | 🟡 中 | ~100-150 行新增 |
| 测试覆盖 | 🟡 中 | 需测试边界情况 |
| **总体难度** | **⭐⭐ 中等** | 3-5 小时可完成 |

---

## 三、重构建议

### 3.1 推荐的模块拆分

```
src/vocal_smart_splitter/core/
├── seamless_splitter.py      # 主编排器（~300行）
├── strategies/
│   ├── __init__.py
│   ├── base.py               # 抽象策略基类
│   ├── mdd_strategy.py       # MDD 模式 (方案 A)
│   ├── beat_only_strategy.py # 纯节拍模式 (方案 B)
│   └── hybrid_strategy.py    # 混合吸附模式 (方案 C)
├── analyzers/
│   ├── beat_analyzer.py      # 节拍分析 (librosa 封装)
│   └── energy_analyzer.py    # 能量分析
└── utils/
    ├── segment_merger.py     # 短片段合并
    └── cut_point_filter.py   # 切点过滤
```

### 3.2 策略模式设计

```python
# base.py
class SegmentationStrategy(ABC):
    @abstractmethod
    def generate_cut_points(
        self,
        audio: np.ndarray,
        mdd_cuts: List[int],
        beat_times: np.ndarray,
        bar_times: np.ndarray,
        config: Dict[str, Any],
    ) -> Tuple[List[int], List[bool]]:  # (cuts, lib_flags)
        pass
```

```python
# beat_only_strategy.py (方案 B)
class BeatOnlyStrategy(SegmentationStrategy):
    def generate_cut_points(self, ...):
        # 纯节拍切割逻辑
```

```python
# hybrid_strategy.py (方案 C)
class HybridSnapStrategy(SegmentationStrategy):
    def generate_cut_points(self, ...):
        # MDD 吸附到节拍逻辑
```

### 3.3 重构优先级

| 优先级 | 任务 | 工作量 | 收益 |
|-------|-----|-------|-----|
| **P0** | 提取 `BeatAnalyzer` | 2h | 复用于 B/C |
| **P1** | 抽象 `SegmentationStrategy` | 3h | 支持 A/B/C 切换 |
| **P2** | 实现方案 B | 2h | 最简单起步 |
| **P3** | 实现方案 C | 4h | 最佳用户体验 |
| **P4** | 重构主文件至 300 行 | 4h | 可维护性 |

---

## 四、实施路径建议

### 4.1 渐进式重构（推荐）

```
阶段 1: 不改动现有代码，新增 strategies/ 目录
        ↓
阶段 2: 实现方案 B（BeatOnlyStrategy），独立测试
        ↓
阶段 3: 实现方案 C（HybridSnapStrategy），独立测试
        ↓
阶段 4: 在 unified.yaml 添加 lib_alignment 配置
        ↓
阶段 5: 在 _process_hybrid_mdd_split 中路由到对应策略
        ↓
阶段 6: 逐步将公共逻辑提取到 analyzers/
```

### 4.2 配置驱动

```yaml
# unified.yaml
hybrid_mdd:
  lib_alignment: mdd_start        # 方案 A (当前默认)
  # lib_alignment: beat_only      # 方案 B
  # lib_alignment: snap_to_beat   # 方案 C
  
  snap_tolerance_ms: 300          # 方案 C 专用
  vad_protection: true            # 方案 C 专用
```

---

## 五、结论

| 问题 | 结论 |
|-----|------|
| 是否需要重构？ | ✅ **需要**，文件过大，职责过多 |
| 先实现还是先重构？ | 🟢 **先实现 B/C**，后续再重构 |
| 方案 B 难度 | ⭐ 简单 (1-2h) |
| 方案 C 难度 | ⭐⭐ 中等 (3-5h) |

**推荐路径**：
1. 先在现有文件中快速实现 B/C（避免大规模重构阻塞）
2. 通过 `lib_alignment` 配置切换
3. 后续迭代中逐步提取公共模块

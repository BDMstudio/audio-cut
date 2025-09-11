# 分割过少问题修复报告

**修复时间**: 2025-09-10 22:00  
**问题描述**: 歌曲只被分割成前奏和主副歌两个部分，明显的人声停顿没有被切割

## 问题根因分析

### 1. MDD（音乐动态密度）系统几乎失效
- **问题**: `threshold_multiplier = 0.05` 太小
- **影响**: MDD调整量仅为 3.5%，几乎不起作用
- **位置**: config.yaml 第89行

### 2. 配置路径错误
- **问题**: 代码读取错误的配置路径
- **影响**: 使用了0.05而不是预期的0.5
- **位置**: vocal_pause_detector.py 第231行

### 3. 切割点间隔过滤过于严格
- **问题**: 默认min_interval=2.0秒，过滤掉密集的切割点
- **影响**: 间隔小于2秒的停顿全部被丢弃
- **位置**: quality_controller.py 第1054行

### 4. 统计学阈值过高
- **问题**: base_threshold_ratio=0.5，阈值偏保守
- **影响**: 许多有效停顿被过滤
- **位置**: config.yaml 第259行

## 修复内容

### 配置文件修改 (config.yaml)
```yaml
# 1. MDD阈值调整倍数
threshold_multiplier: 0.05 → 0.3  # 提高6倍

# 2. 最小切割间隔
min_split_gap: 2.5 → 1.0  # 降低间隔要求

# 3. 统计学过滤参数
base_threshold_ratio: 0.5 → 0.4  # 更激进的检测
absolute_min_pause: 0.3 → 0.2  # 检测更短停顿
absolute_max_pause: 2.5 → 3.0  # 包容更长停顿
```

### 代码修改

#### 1. vocal_pause_detector.py
```python
# 修复默认值
mdd_threshold_multiplier = get_config('musical_dynamic_density.threshold_multiplier', 0.3)
```

#### 2. seamless_splitter.py
```python
# 添加可配置的最小间隔
min_interval = get_config('quality_control.min_split_gap', 1.0)
final_times = self.quality_controller.pure_filter_cut_points(
    validated_times, audio_duration_s, min_interval=min_interval
)
```

## 修复效果预期

1. **MDD系统恢复正常**: 高密度区域的阈值调整量从3.5%提升到21%
2. **切割点保留更多**: 间隔限制从2秒降到1秒
3. **检测更加敏感**: 可检测0.2秒以上的停顿
4. **动态阈值更合理**: 基础阈值降低20%

## 测试建议

重新运行分割命令：
```bash
python quick_start.py
# 选择模式3: 纯人声检测v2.1
```

观察是否在15秒附近的明显停顿处产生新的切割点。

## 参数调优指南

如需进一步调整：
- 想要更多切割：降低`base_threshold_ratio`（如0.3）
- 想要更少切割：提高`base_threshold_ratio`（如0.5）
- MDD影响太强：降低`threshold_multiplier`（如0.2）
- MDD影响太弱：提高`threshold_multiplier`（如0.4）
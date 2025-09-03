# Bug 修复记录 - 智能人声分割器

## 🐛 问题描述

### 问题1: KeyError: 'output_files'
**错误信息:**
```
KeyError: 'output_files'
File "run_splitter.py", line 134, in main
    logger.info(f"生成片段数: {len(result['output_files'])}")
```

**原因分析:**
- `vocal_smart_splitter/main.py` 返回的结果字典中使用的键名是 `saved_files`
- 而 `run_splitter.py` 中期望的键名是 `output_files`
- 键名不匹配导致 KeyError

### 问题2: 分割片段过少
**现象:**
- 256秒的音频只生成了1个片段
- 期望应该生成15-25个片段（按5-15秒/片段计算）

**原因分析:**
- `split_quality_threshold: 0.68` 阈值过高，导致大部分候选分割点被过滤
- `target_segment_length: 11` 目标长度偏大
- `allow_content_extension: true` 允许内容延伸，减少了分割机会
- `strict_time_limit: false` 没有严格的时间限制

### 问题3: 音质损失明显
**现象:**
- 分割后的音频文件音质明显下降
- 可能存在音频处理过程中的质量损失

## 🔧 修复方案

### 修复1: 解决 KeyError
**文件:** `vocal_smart_splitter/main.py`
**状态:** ✅ 已修复

在返回结果中同时提供两个键名以保持兼容性:
```python
'output_files': saved_files,  # 修复键名
'saved_files': saved_files,   # 保持兼容性
```

### 修复2: 调整分割参数
**文件:** `vocal_smart_splitter/config.yaml`
**状态:** ✅ 已调整

关键参数调整:
- `target_segment_length: 8` (从11降到8)
- `allow_content_extension: false` (禁用内容延伸)
- `strict_time_limit: true` (启用严格时间限制)
- `split_quality_threshold: 0.5` (从0.68降到0.5)
- `max_vocal_at_split: 0.2` (从0.1提高到0.2)

### 修复3: 音质优化
**当前设置:** 192kbps MP3, 22050Hz
**建议:** 考虑提升到 320kbps, 44100Hz

## 📊 预期效果

- ✅ 程序运行无错误
- 🎯 生成片段数预期: 20-32个
- 🎵 片段长度: 5-15秒范围内
- 🔊 音质: 控制损失在可接受范围

## 🧪 测试命令

```bash
# 基本测试
python run_splitter.py input/01.mp3 --verbose

# 音质测试
python test_audio_quality_fix.py
```

## 🎵 音质修复 (v1.0.2)

### 🚨 新发现的问题
运行成功但出现严重音质损失和破音问题：
- 明显的破音和失真
- 动态范围压缩严重
- 音质明显下降

### 🔍 根本原因分析
1. **采样率过低**: 22050Hz导致高频信息丢失
2. **过度标准化**: RMS标准化压缩动态范围
3. **渐入渐出过短**: 0.02秒导致破音
4. **多重处理链**: 人声分离+标准化+后处理累积失真
5. **人声分离过强**: 参数设置过于激进

### 🔧 音质修复方案

#### 1. 提升音频质量参数
```yaml
audio:
  sample_rate: 44100  # 22050 → 44100 (CD质量)
  quality: 320        # 192 → 320 (最高MP3质量)
```

#### 2. 优化人声分离参数
```yaml
vocal_separation:
  hpss_margin: 1.0    # 3.0 → 1.0 (降低分离强度)
  hpss_power: 1.0     # 2.0 → 1.0 (减少失真)
  mask_threshold: 0.3 # 0.15 → 0.3 (更保守分离)
  mask_smoothing: 3   # 5 → 3 (保持更多细节)
```

#### 3. 改进音频处理
```yaml
quality_control:
  fade_in_duration: 0.1    # 0.02 → 0.1 (避免破音)
  fade_out_duration: 0.1   # 0.02 → 0.1 (避免破音)
  normalize_audio: false   # 禁用标准化保持动态范围
  remove_click_noise: false # 禁用过度处理
  smooth_transitions: false # 保持原始音质
```

#### 4. 优化标准化算法
- 使用峰值标准化替代RMS标准化
- 软限制替代硬切割: `np.tanh(audio * 0.9) * 0.95`
- 更保守的增益限制: `np.clip(gain, 0.3, 3.0)`

#### 5. 高质量MP3编码
```python
parameters=[
    "-q:a", "0",  # 最高质量
    "-compression_level", "0"  # 最低压缩
]
```

### 📊 预期改进效果
- ✅ 消除破音和失真
- ✅ 保持原始动态范围
- ✅ 提升整体音质
- ✅ 减少处理链失真
- ✅ 44100Hz采样率保持高频细节

### 🧪 质量测试工具
创建了专门的音质测试脚本 `test_audio_quality_fix.py`：
- 分析原始音频质量
- 对比处理前后的音质指标
- 检测削波、动态范围等问题
- 提供详细的质量评估报告

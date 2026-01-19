# Release Notes: v2.5.1

**发布日期**: 2026-01-18

## 🎉 主要更新

### 多特征副歌检测算法
- **实现三特征融合**：RMS能量 + 频谱质心 + 频谱带宽
- **自适应权重机制**：根据能量变异系数(CV)动态调整特征权重
  - 低动态歌曲（CV<0.15）：侧重频谱特征
  - 高动态歌曲（CV>0.4）：侧重能量特征
  - 中等动态：平衡权重
- **连续性检测增强**：要求至少连续4小节高能量才识别为副歌
- **显著提升准确度**：民谣/爵士等低动态歌曲副歌识别准确度提升60-70%

**测试结果**：
- Be Your Love (民谣): 39/104 → 12/104 副歌小节 (-69% 误判率)
- 当爱再回头 (流行): 16/43 保持稳定

### 策略优化
- **移除 `mdd_start` 策略**：保留 `beat_only` 和 `snap_to_beat` 两种策略
- **交互式策略选择**：`quick_start.py` 新增 lib_alignment 策略选择菜单

### 新增特性
- **BeatAnalyzer 增强**：新增 `bar_spectral_centroids` 和 `bar_spectral_bandwidths` 特征计算
- **SegmentationContext 扩展**：支持传递频谱特征到策略层
- **Manifest 字段增强**：新增 `chorus_detection` 元信息

## 📦 安装

### 从 wheel 安装
```bash
pip install vocal_smart_splitter-2.5.1-py3-none-any.whl
```

### 从源码安装
```bash
git clone https://github.com/BDMstudio/audio-cut.git
cd audio-cut
git checkout v2.5.1
pip install -e .
```

## 🔧 配置示例

### Hybrid MDD 高精度配置
```python
from audio_cut.api import separate_and_segment

manifest = separate_and_segment(
    input_uri="song.mp3",
    export_dir="output/",
    mode="hybrid_mdd",
    device="cuda:0",
)
```

### 环境变量配置
```bash
export VSS__hybrid_mdd__lib_alignment=snap_to_beat
export VSS__hybrid_mdd__density=high
export VSS__hybrid_mdd__energy_percentile=40
```

## 📊 性能基准

- **分离**: MDX23 GPU ≥0.7x 实时
- **检测+守卫**: 10分钟素材 ~12s
- **Chunk vs Full**: L∞<5e-3, SNR>60dB
- **副歌检测**: 额外 ~85ms/歌曲

## 🐛 已知问题

- Windows 需要手动注入 ORT 依赖（参见文档）
- 极低动态歌曲（CV<0.1）可能需要手动调整 `energy_percentile`

## 📚 文档更新

- [README.md](README.md) - 新增副歌检测说明
- [development.md](development.md) - 更新当前进展
- [module_integration_guide.md](.gemini/antigravity/brain/.../module_integration_guide.md) - 完整集成指南

## 🙏 致谢

感谢所有测试和反馈的用户！

---

**完整更新日志**: https://github.com/BDMstudio/audio-cut/compare/v2.5.0...v2.5.1

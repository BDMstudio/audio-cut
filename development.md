<!-- File: development.md -->
<!-- AI-SUMMARY: Vocal Smart Splitter 工程侧技术演进、架构与配置准则的单一事实来源。 -->

# development.md · 技术路线与模块总览（更新于 2025‑09‑14）

本文作为工程侧的单一事实来源（SSOT），梳理技术路线、系统架构、关键模块、配置要点与最新变更。

## 1. 技术路线（演进）
- v1.x：原混音 → BPM/复杂度分析 → Silero VAD → 节拍对齐 → 样本级切割
- v2.0：原混音 → 人声分离（MDX23/Demucs）→ 多维特征/频谱感知/验证 → 切点 → 样本级切割
- v2.1：VocalPrime（RMS/EMA/动态地板/滞回/平台验证/未来静默守卫）
- v2.2：MDD（Musical Dynamic Density）主/副歌识别与阈值动态调整
- 2025‑09‑12：终筛改造V2 + 守卫可配置化
- 2025‑09‑14：VPP 一次判定落地（仅合并短段；移除二次插点/强拆）

关键铁律：
- 先分离（vocal stem）再在纯人声域做停顿/换气检测；
- 切点吸附零交叉避免爆音；
- 默认不破坏兼容，失败路径优雅回退。

## 2. 目录与模块
- `core/seamless_splitter.py`（统一引擎）
  - `split_audio_seamlessly(...)`：入口
  - `_finalize_and_filter_cuts_v2(...)`：终筛改造V2（采纳守卫校正时间 + 时长治理）
- `core/pure_vocal_pause_detector.py`：纯人声多维特征检测
- `core/vocal_pause_detector.py`：能量谷与 BPM 自适应工具（供纯人声检测内部调用）
- `core/enhanced_vocal_separator.py`：分离后端（MDX23/Demucs）选择与质量评估
- `core/quality_controller.py`：质量控制/零交叉/安静守卫/过滤
- `core/adaptive_vad_enhancer.py`：BPM/复杂度自适应增强
- `utils/*`：音频处理、特征提取、配置管理
- `quick_start.py` / `run_splitter.py`：交互式与参数化入口

## 3. 数据流
1) 加载原音频（必要时重采样至 `44100Hz`）；
2) 人声分离（MDX23 优先，失败降级 Demucs/HPSS）；
3) 检测停顿：纯人声路径（v2.2 MDD 一次检测）；
4) 终筛（VPP 一次判定）：采纳守卫校正时间 → 过滤最小间隔 → 合并短段（不再二次插点/强拆）；
5) 样本级切割、纯音乐能量切分与命名标注（_human/_music），同步导出原混音与纯人声两套 24-bit WAV 切片；
6)（可选）拼接完整性验证（误差 0）。

## 4. 切点与守卫策略
- 零交叉吸附 + 安静守卫（右推搜索“相对地板+余量”的安静点），必要时局部最小兜底；
- BPM 轻度自适应（可选）：按 BPM/VPP 轻微调整阈值与分割间隙；
- 长短段治理：仅 `segment_min_duration` 合并短段；`segment_max_duration` 作为目标参考，不强行打断；
- 过滤最小间隔：`min_split_gap`。

## 5. MDD（v2.2）概要
综合能量（RMS）、频谱平坦度、起始率三维指标，动态调整阈值并识别主/副歌，减少副歌段过切、提升自然度。

## 6. VPP 一次判定与守卫可配置化（2025‑09‑14）
变更：
- `SeamlessSplitter._finalize_and_filter_cuts_v2(...)` 简化为：NMS → 守卫校正 → 最小间隔 → 仅合并短段；移除“二次插点/强拆”。
- `QualityController.enforce_quiet_cut(...)` 支持从配置覆盖 `win_ms / guard_db / search_right_ms / floor_percentile`。

效果：
- 保持一次检测的自然度与稳定性，避免后处理“二次破坏”优选切点；
- 配合 README 的“调参口诀”，通过 VPP 阈值/权重与 BPM 自适应即可解决快歌/和声等场景的过长片段问题。

建议配置（可选，详见 README 调参口诀）：
```yaml
quality_control:
  min_split_gap: 0.9–1.2
  segment_min_duration: 4–6
  segment_max_duration: 15–20   # 目标参考，不强制打断
  enforce_quiet_cut:
    win_ms: 80
    guard_db: 2.0–3.0
    search_right_ms: 120–200
    floor_percentile: 5
```

## 7. 测试与验证
- 单元：valley_cut/bpm_guard/defaults_guard 等；
- 契约：`contracts/valley_no_silence.yaml`；
- 集成：`integration/test_pipeline_v2_valley.py`；
- 拼接验证：`test_seamless_reconstruction.py`（误差 0）。

## 8. 环境与依赖
- Python 3.10+
- librosa/numpy/scipy/soundfile/pydub
- demucs（可选）、PyTorch（Demucs/Silero 所需）

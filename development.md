# development.md — 技术路线与模块总览

## 0. 范畴说明
- 本文档归档技术路线、系统架构与模块职责，作为工程侧单一事实来源（SSOT）。
- 仅梳理“如何实现/为何如此设计”，不包含需求/计划（见 PRD.md / todo.md）。

## 1. 总体技术路线（演进）
- v1.0–v1.2：BPM 自适应 + Silero VAD（可选分离增强）
  - 路线：原混音 → BPM/复杂度分析 → Silero VAD → 节拍对齐 → 样本级切割
- v2.0：纯人声检测（MDX23/Demucs 分离 → 多维特征 + 频谱感知 + 验证 + BPM 优化）
  - 路线：原混音 → 人声分离(vocal stem) → 特征/分类/验证 → 切点 → 样本级切割
- v2.1：VocalPrime（纯人声域 RMS/EMA + 动态噪声地板 + 滞回 + 平台验证 + 守卫）
  - 路线：原混音 → 人声分离(vocal stem) → RMS/EMA/地板/滞回/平台 → 切点 → 样本级切割

关键铁律（已在实现中落实）：
- 先分离（MDX23/Demucs）→ 再在纯人声域进行“停顿/切点”检测；
- 切点对齐零交叉，避免爆音；
- 配置化的“人声消失即切割”与“保留短尾段”。

## 2. 核心模块与职责
- 分离（pure vocal 获取）
  - src/vocal_smart_splitter/core/enhanced_vocal_separator.py
    - 后端选择：mdx23 | demucs_v4 | auto；降级策略；分离质量评估
    - 入口：EnhancedVocalSeparator.separate_for_detection()
- v2.0 纯人声检测
  - src/vocal_smart_splitter/core/vocal_pause_detector.py（Silero VAD on vocal stem + 切点策略/零交叉）
  - （规划）pure_vocal_pause_detector.py、spectral_aware_classifier.py、bpm_vocal_optimizer.py、multi_level_validator.py
- v2.1 VocalPrime 检测
  - 规范：vocal_prime.md（RMS/EMA/动态地板/滞回/平台平坦度/未来静默守卫/BPM禁切区）
  - 实现：src/vocal_smart_splitter/core/vocal_prime_detector.py（存在，仍需补 BPM 禁切区）
- 无缝分割与验证
  - src/vocal_smart_splitter/core/seamless_splitter.py（主引擎，样本级分割，完美重构校验）
  - tests/test_seamless_reconstruction.py（拼接误差=0 验证）
- 配置与工具
  - src/vocal_smart_splitter/utils/config_manager.py（参数集中管理）
  - src/vocal_smart_splitter/utils/audio_processor.py / feature_extractor.py（零交叉/特征）
- 入口脚本
  - quick_start.py（交互式：分离/纯人声 v2.0/兼容模式/智能分割）
  - run_splitter.py（批处理/参数化入口）

## 3. 数据流（以 v2.0/2.1 为例）
1) 输入音频（44.1kHz 单声道内部处理）
2) EnhancedVocalSeparator → vocal_track（纯人声）
3) 检测：
   - v2.0：Silero VAD on vocal_track（已实现），或 VocalPrime（v2.1，规划接线）
4) 生成 pause/cut_point（零交叉吸附；可选 cut_at_speech_end）
5) 质量控制（最小间隔/最小停顿；尾段保留）
6) 样本级切割与导出（零处理，WAV/FLAC）

## 4. 切点策略（关键实现约束）
- 零交叉吸附：避免点击声
- 平台中心右偏 vs 人声消失即切割（cut_at_speech_end=true）
- BPM：
  - v1.x/v2.0：用于参数自适应（存在优雅回退）
  - VocalPrime：仅作“禁切区”，不吸附；超宽平台允许“向右推”避开拍点
- 尾段保留：keep_short_tail_segment=true 时，末段不达最小时长也保留

## 5. 关键配置（示例，详见 config/default.yaml 或 src/.../config.yaml）
- enhanced_separation.backend: auto|mdx23|demucs_v4
- vocal_pause_splitting.cut_at_speech_end: false|true
- vocal_pause_splitting.keep_short_tail_segment: true
- vocal_pause_splitting.enable_zero_crossing_align: true
- advanced_vad.*（Silero 窗、pad、最小语音时长等）
- bpm_adaptive.*（v1.x 自适应）/ bpm_guard.*（VocalPrime 禁切区）

## 6. 质量与测试
- 单元：tests/unit/*（检测/切点/对齐）
- 集成：tests/test_pure_vocal_detection_v2.py（v2.0 路径）
- 契约：
  - 禁切区（VocalPrime）：切点与拍点距离 ≥ forbid_ms（待补）
  - 尾段保留：末段 < min_segment_duration 仍应存在（v2.0 已在 quick_start 侧逻辑实现，VocalPrime 分支待补）
- E2E：run_splitter/quick_start 典型样例

## 7. 依赖与环境
- PyTorch 2.8.0 + CUDA 12.9（GPU）
- Silero VAD（torchscript）
- MDX23（CLI/外部项目）/ Demucs v4（PyTorch）
- librosa/soundfile/numpy/scipy
- 参考：MDX23_SETUP.md、README.md

## 8. 实现状态更新（2025-09-09 代码审查）

### ✅ 已实现并验证
- **VocalPrime 核心检测器**: `vocal_prime_detector.py` (362行) - 完整实现RMS包络+滞回状态机+平台验证
- **v2.0 处理流程**: `quick_start.py` 中 `split_pure_vocal_v2()` 函数 - 完整8步流水线
- **Valley-based切割**: 从 `todo.md` 状态显示已完成单元/集成/契约测试
- **GPU兼容性**: `pytorch_compatibility_fix.py` 修复PyTorch 2.8.0兼容性
- **测试覆盖**: 41个测试文件涵盖unit/integration/contracts/performance层级

### 🔧 技术债务与对齐差距
- **pure_vocal_pause_detector.py**: 文档中提及但代码中未找到，功能已迁移到增强Silero VAD
- **v2.engine切换**: quick_start.py 硬编码使用VocalPauseDetectorV2，缺少 silero|vocal_prime 引擎选择
- **配置化接口**: vocal_prime_detector.py 使用硬编码参数，未完全接入 get_config() 系统

## 9. 参考规范/文档
- vocal_prime.md（检测技术规范）
- README.md（项目概览与运行）
- development.md（本文件，见“性能与优化”）


## 10. 性能与优化（已合并自 SPEED_OPTIMIZATION.md）

### 10.1 模式与耗时（经验值）
| 模式 | 后端 | 平均耗时 | 适用场景 |
|------|------|----------|----------|
| 快速 ⚡ | hpss_fallback | ~16 秒 | 日常快速分割/低资源 |
| 平衡 ⚖️ | demucs_v4 | ~1–2 分钟 | 质量/速度折中 |
| 精确 🎯 | mdx23 | ~5 分钟+ | 专业质量优先 |

说明：具体耗时随音频时长、GPU/CPU、chunk/segment 参数而变。

### 10.2 切换方式
- 配置切换（推荐）：编辑 `config/default.yaml` 或主配置
  ```yaml
  enhanced_separation:
    backend: "hpss_fallback"   # 快速 ⚡
    # backend: "demucs_v4"     # 平衡 ⚖️
    # backend: "mdx23"         # 精确 🎯
    min_separation_confidence: 0.15
  ```
- 环境变量（可选）：
  ```bash
  export AUDIO_CUT_SPEED_MODE=fast|balanced|accurate
  ```

### 10.3 典型参数模板
- 速度优先（HPSS）
  ```yaml
  enhanced_separation:
    backend: "hpss_fallback"
    min_separation_confidence: 0.10
  vocal_separation:
    hpss_margin: 3.0
    mask_smoothing: 1
  ```
- 质量优先（MDX23）
  ```yaml
  enhanced_separation:
    backend: "mdx23"
    min_separation_confidence: 0.30
    mdx23:
      overlap_large: 0.35
  ```
- 平衡（Demucs v4）
  ```yaml
  enhanced_separation:
    backend: "demucs_v4"
    min_separation_confidence: 0.20
    demucs_v4:
      shifts: 1          # 1=最快，10=最准
      segment: 4         # 平衡内存与速度
  ```

### 10.4 GPU 加速要点
- 启用 CUDA 版 PyTorch 后：
  ```yaml
  enhanced_separation:
    backend: "mdx23"   # 或 "demucs_v4"
    mdx23:
      enable_large_gpu: true
      chunk_size: 1000000
    demucs_v4:
      device: "cuda"
      segment: 8
  ```
- 环境变量建议：
  ```bash
  export PYTORCH_NO_CUDA_MEMORY_CACHING=1
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  ```

### 10.5 故障排除
- 处理太慢：确认 `backend=hpss_fallback`；降低 `min_separation_confidence`；增大 `hpss_margin`。
- 质量不够：切换到 `demucs_v4`/`mdx23`；提高 `min_separation_confidence`；增大 `overlap/segment`。
- 内存不足：降低 `chunk_size/segment`；切回 `hpss_fallback`；在 GPU 上减少 `overlap` 与 `shifts`。


## 11. 无静音平台的谷值切割（Valley-based）

### 11.1 背景与目标
- 问题：纯人声段常见“无绝对静音平台”（尾音/气声/混响），此时仅用“零交叉≈静音”的策略会误切到有声周期内部。
- 目标：当不存在稳定静音平台时，改以“纵轴特征”寻找“可切换气（breath）/摩擦噪声谷”，作为切点；并用“未来静默守卫”避免切在词内摩擦音上。
- 原则：
  - 优先用静音平台（若存在）；
  - 启用“谷值切割”作为无静音平台时的兜底（或手动强制启用）。

### 11.2 触发条件与开关（不破坏兼容）
- 默认行为：保持现有静音平台→零交叉策略；仅当“无静音平台”时自动回退到谷值切割。
- 配置项（建议）：
  - vocal_pause_splitting.enable_valley_mode: false  # 手动强制谷值模式
  - vocal_pause_splitting.auto_valley_fallback: true # 无静音时自动兜底（默认开启）
  - vocal_pause_splitting.local_rms_window_ms: 25
  - vocal_pause_splitting.silence_floor_percentile: 5
  - vocal_pause_splitting.min_valley_width_ms: 120
  - vocal_pause_splitting.lookahead_guard_ms: 120
  - bpm_guard.forbid_ms: 100

### 11.3 核心算法（实现说明）
1) 预处理（纯人声轨 vocal_track）
- 计算短时响度包络 e(t)：RMS（窗 25–50ms，hop 10ms），可叠加 EMA 平滑。
- 动态噪声地板 floor(t)：滚动 5% 分位（或中位数−X dB）近似静默水平。

2) 候选谷检测（时间域为主，频域辅助）
- 谷中心：e(t) 的局部最小值；谷宽度：两侧上升到 e(t)>floor(t)+Δ 所需的时间和。
- 约束：
  - 谷宽 ≥ min_valley_width_ms（防止词内微谷）；
  - 两侧“上坡”必须存在（局部斜率>阈值）。
- 频域/声学特征（在谷中心±W 窗内计算，W≈25–40ms）：
  - depth：floor(t) − e(t)（或 e/floor 比，越低越好）；
  - spectral_flatness（SFM）：越噪声越高，谷期倾向噪声/气声；
  - spectral_centroid：谷期质心偏高更像气声/摩擦；
  - voicing/HNR proxy：自相关主峰强度或简化 F0 置信度（越低越像“无声学”）；
  - ZCR（仅作弱特征，配合其它项使用）。
- 评分函数（线性加权即可）：
  score = +w1·depth_norm + w2·flatness_norm + w3·centroid_norm + w4·(1−voicing_conf) − w5·beat_proximity

3) BPM 禁切区（可选）
- 若启用 bpm_guard：在“强拍±forbid_ms”内降低分数或直接屏蔽，保留乐句连贯性。

4) 未来静默守卫（lookahead guard）
- 在候选切点之后的 lookahead_guard_ms 窗内，要求 e(t) 低于 floor(t)+Δ 的占比 ≥ p（如 70%），
  且不可快速反弹到“元音响度”。不满足则在区间内向右寻找次优谷；仍不满足则放弃该谷。

5) 样本级细化与兜底
- 在最终时间点 ±20ms 内做零交叉吸附，避免点击声；
- 若区间内没有通过守卫的谷：
  - 有静音平台→使用平台中心右偏；
  - 否则→使用 e(t) 全局最小（区间）+ 守卫，仍失败则放弃该切点。

### 11.4 接口与落点（代码对接）
- 主要文件：
  - src/vocal_smart_splitter/core/vocal_pause_detector.py
    - 在 `_calculate_cut_points(...)` 内：
      - 先判定是否存在“稳定静音平台”；
      - 若无且 `auto_valley_fallback` 或 `enable_valley_mode` 为真，调用 `select_valley_cut_point(...)`；
      - 否则沿用现有平台/零交叉策略；
      - 两路径均应应用 lookahead 守卫。
  - src/vocal_smart_splitter/utils/feature_extractor.py（如需新增特征）：
    - 短时 RMS/EMA、rolling percentile floor、SFM、centroid、简化 voicing/HNR、ZCR；
  - src/vocal_smart_splitter/utils/config_manager.py：新增上述配置键，含默认值与范围。

- 复杂度控制：
  - 第一版仅实现“RMS 谷 + floor + 守卫 + 零交叉 + 谷宽/坡度”；
  - 第二版再加入 SFM/centroid/voicing 打分；
  - BPM 禁切区沿用现有 bpm_guard 结构对接。

### 11.5 测试与验收（CI 必须）
- 单元（tests/unit/test_valley_cut.py）：
  - 合成序列：元音(有声) → 气声(无声学/高 SFM) → 元音；
  - 断言：切点落在气声谷 ±20ms；lookahead 守卫通过；零交叉细化存在且不偏离 >20ms。
- 契约（tests/contracts/valley_no_silence.yaml）：
  - 输入：无静音纯人声样本集合；
  - 期望：切点距“高 voicing 区” ≥ 80–120ms；不落在强拍禁切区内（若启用 bpm_guard）。
- 集成（tests/integration/test_pipeline_v2_valley.py）：
  - 在 v2.0 流水线启用 `auto_valley_fallback`；
  - 验证：与仅零交叉方案对比，切点到高-voicing 区中心的距离分布整体右移（更远离人声）。

### 11.6 参数建议与默认值
- local_rms_window_ms: 25（hop 10ms）
- silence_floor_percentile: 5（rolling）
- min_valley_width_ms: 120
- lookahead_guard_ms: 120
- bpm_guard.forbid_ms: 100（如启用）
- enable_valley_mode: false（不破坏既有默认）
- auto_valley_fallback: true（推荐）

### 11.7 兼容性与回退
- “Never break userspace”：默认行为不变；仅在“无静音平台”场景由系统自动兜底至谷值切割；
- 任一环节失败（无合格谷/守卫不通过）→ 回退到平台策略或放弃该切点，不强行切人声。
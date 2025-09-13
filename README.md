# 智能人声分割器（Vocal Smart Splitter）

高质量的人声停顿检测与无缝分割工具，面向歌曲场景优化。支持分离增强（MDX23/Demucs）、BPM自适应、统计学动态裁决与样本级零交叉切割，输出可实现“完美重构”。

## 亮点
- 纯人声域检测：先分离后检测，显著降低误判（换气/摩擦噪声）。
- MDD 动态密度：主/副歌识别与阈值动态调整，减少高潮段过切。
- 终筛改造V2（2025‑09‑12）：
  - 采纳“零交叉吸附 + 安静守卫右推校正”的切点时间；
  - 统一时长治理：自动合并短段、强拆超长段，避免 0s 与 30s+ 片段；
  - v2.1/v2.2 使用纯人声轨进行守卫校正，切割仍在原混音上执行。
- 守卫参数可配置化：`win_ms / guard_db / search_right_ms / floor_percentile` 可在配置中覆盖。
- 样本级无缝拼接：输出WAV/FLAC零处理，拼接差异 0.00e+00。

## 快速开始
1) 将待处理音频放到 `input/`。
2) 运行交互脚本：
```bash
python quick_start.py
```
选择推荐的 `Pure Vocal v2.2 MDD` 模式。

或使用命令行：
```bash
python run_splitter.py input/your_song.mp3 --seamless-vocal
```
分割结果保存在 `output/` 的时间戳目录内。

## 关键配置（摘自 src/vocal_smart_splitter/config.yaml）
- 分离后端：`enhanced_separation.backend: mdx23 | demucs_v4 | auto`
- BPM增强：`vocal_pause_splitting.enable_bpm_adaptation: true`
- 无平台谷值兜底：`vocal_pause_splitting.enable_valley_mode/auto_valley_fallback`
- 守卫参数（可覆盖）：
```yaml
quality_control:
  min_split_gap: 1.0                 # 最小切点间隔（秒）
  segment_min_duration: 1.0          # 片段最小时长（秒）
  segment_max_duration: 20.0         # 片段最长时长（秒）
  enforce_quiet_cut:
    win_ms: 80
    guard_db: 3.0
    search_right_ms: 350
    floor_percentile: 5              # 5 或 0.05 皆可
```
说明：若未显式配置，程序会使用内建默认值。

## 运行模式
- `smart_split`：原混音上进行传统检测（向后兼容）。
- `v2.1`：VocalPrime（RMS/EMA/动态地板/滞回/平台验证）。
- `v2.2_mdd`：在 v2.1 基础上加入 MDD 动态密度增强（推荐）。
- `vocal_separation`：仅做人声/伴奏分离并导出。

## 更新日志
### 2025‑09‑12
- 引入“终筛改造V2”：
  - 采纳守卫校正时间；
  - 合并短段/强拆长段，避免 0s 与 30s+；
  - v2.1/v2.2 路径采用纯人声轨进行守卫校正。
- `enforce_quiet_cut` 支持从配置覆盖 `win_ms / guard_db / search_right_ms / floor_percentile`。
- 修复“样本停顿丰富但出现整段未切”的问题。

## 许可
本项目仅供学习与研究使用。


# 智能人声分割器（Vocal Smart Splitter）

高质量的人声停顿检测与无缝分割工具，面向歌曲场景优化。支持分离增强（MDX23/Demucs）、BPM自适应、VPP一次判定（能量谷）与样本级零交叉切割，输出可实现“完美重构”。

## 亮点
- 纯人声域检测：先分离后检测，显著降低误判（换气/摩擦噪声）。
- MDD 动态密度：主/副歌识别与阈值动态调整，减少高潮段过切。
- 终筛改造V2：
  - 采纳“零交叉吸附 + 安静守卫右推校正”的切点时间；
  - 简化治理：仅“合并短段”，不再二次插点/强拆，充分尊重一次检测的优选切点；
  - v2.2 使用纯人声轨进行守卫校正，切割在原混音上执行。
- 守卫参数可配置化：`win_ms / guard_db / search_right_ms / floor_percentile` 可在配置中覆盖。
- 样本级无缝拼接：输出WAV/FLAC零处理，拼接差异 0.00e+00。

## 快速开始
1) 将待处理音频放到 `input/`。
2) 运行交互脚本：
```bash
python quick_start.py
```
在菜单中可选择：
- `1` 纯人声分离 (仅输出人声/伴奏)
- `2` Pure Vocal v2.2 MDD（默认推荐）

推荐选择 `Pure Vocal v2.2 MDD` 模式完成无缝分割。

或使用命令行：
```bash
python run_splitter.py input/your_song.mp3 --mode v2.2_mdd
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
  segment_min_duration: 5.0          # 片段最小时长（秒）
  segment_max_duration: 18.0         # 片段最长时长（秒）
  pure_music_min_duration: 6.0       # 无人声区间的最小检测窗口
  segment_vocal_threshold_db: -50.0  # 片段人声判定阈值（dB）
  enforce_quiet_cut:
    win_ms: 80
    guard_db: 2.5
    search_right_ms: 150
    floor_percentile: 5
```
说明：若未显式配置，程序会使用内建默认值。

## 调参口诀（VPP + BPM）
- 总体思路：一次检测（VPP 为主，BPM 轻度自适应）、仅合并短段，不做二次插点/强拆。

- 基础阈值（相对能量谷）
  - `pure_vocal_detection.peak_relative_threshold_ratio`: 建议 0.22–0.30（推荐 0.24）
  - `pure_vocal_detection.rms_relative_threshold_ratio`: 建议 0.26–0.34（推荐 0.30）
  - 说明：适度“提高”阈值→更容易形成“更长”的谷，一次拿足优质切点，减少超长段。

- BPM 自适应（收窄幅度，稳健微调）
  - `pure_vocal_detection.relative_threshold_adaptation.clamp_min/max`: 0.85 / 1.15
  - `pure_vocal_detection.pause_stats_adaptation.multipliers`: `{ slow:1.08, medium:1.00, fast:0.92 }`
  - 可选覆盖（启用自适应层时生效）：`bpm_adaptive_core.split_gap_phrases.fast: 2`、`pause_duration_beats.fast: 0.6`、`speech_pad_beats.fast: 0.3`

- 候选质量与压缩
  - `pure_vocal_detection.valley_scoring.w_len`: 0.7（长谷优先）
  - `pure_vocal_detection.valley_scoring.w_quiet`: 0.3
  - `pure_vocal_detection.valley_scoring.merge_close_ms`: 100–150（合并近邻碎谷）
  - `pure_vocal_detection.valley_scoring.max_kept_after_nms`: 150–200（保留足够一次性切点）

- 守卫与最小间隔
  - `quality_control.min_split_gap`: 0.9–1.2（与 `segment_min_duration` 配合）
  - `quality_control.enforce_quiet_cut.guard_db`: 2.0–3.0
  - `quality_control.enforce_quiet_cut.search_right_ms`: 120–200

- 长短段治理（仅合并短段）
  - `quality_control.segment_min_duration`: 5–6（兜底短碎片）
  - `quality_control.segment_max_duration`: 15–18（超长段将自动能量切分）

### 常见现象 → 快速处置
- 仍有超长片段（> seg_max）
  1) `peak_relative_threshold_ratio` / `rms_relative_threshold_ratio` 各+0.02；
  2) `min_split_gap` ↓0.1（如 1.0→0.9）；
  3) `enforce_quiet_cut.search_right_ms` → 150，`guard_db` 保持 2.0–2.5；
  4) `merge_close_ms` ≥ 120，减少碎谷干扰。

- 切点过少 / 分割不足
  1) `peak_relative_threshold_ratio` / `rms_relative_threshold_ratio` 小幅提高；
  2) `min_split_gap` ↓ 到 0.9–1.0；
  3) `max_kept_after_nms` ↑ 到 200。

- 切点过多 / 片段太碎
  1) `merge_close_ms` ↑（120–150）；
  2) `min_split_gap` ↑（1.2–1.5）；
  3) `segment_min_duration` 根据风格在 5–6 范围微调。

- 切点靠边/轻微爆音
  1) `guard_db` ↑（2.5–3.0）；
  2) `search_right_ms` ↑（150→180/200）；
  3) 保持 `enable_zero_crossing_align: true`。

- 快节奏 + 副歌多人和声（间隙短，难完全分离）
  1) 采用推荐阈值：`peak`≈0.24 / `rms`≈0.30；
  2) BPM 自适应收窄（clamp 0.85–1.15），`split_gap_phrases.fast: 2`；
  3) 守卫右推缩短：`search_right_ms: 150`；
  4) 仅“合并短段”，不做二次插点/强拆，保持一次检测自然度。

提示：VPP 一次判定为主；BPM 做轻度自动化辅助。优先用阈值/权重把“最长、最安静”的谷一次挑出来，
终筛仅“合并短段”兜底，避免反复打补丁破坏首次优选切点。

## 运行模式
- `vocal_separation`：仅做人声/伴奏分离并导出。
- `v2.2_mdd`：纯人声检测 + MDD 动态密度增强（默认/推荐）。

## 更新日志
### 2025‑09‑14
- 终筛策略回归 VPP 一次判定：仅“合并短段”，移除二次插点/强拆，依靠 BPM+VPP 一次检测拿到优质切点。
- README 增加“调参口诀（VPP + BPM）”，给出一次检测下的稳健参数区间与常见问题处置方法。
- 守卫参数默认更稳健：更短右推窗口（150ms）与轻度 guard（≈2.0dB）。

## 许可
本项目仅供学习与研究使用。

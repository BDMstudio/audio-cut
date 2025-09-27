<!-- File: docs/tuning.md -->
<!-- AI-SUMMARY: 总结 v3 精简配置的调参方法，涵盖核心旋钮、Profile 与覆盖顺序。 -->

# v3 调参速览

| 核心旋钮 | 作用 | 建议范围 | 备注 |
| --- | --- | --- | --- |
| detection.min_pause_s | 停顿判定下限 | 0.40–0.70 | 慢歌拉大，快歌收紧 |
| detection.min_gap_s | 最终切点最小间隔 | 0.80–1.30 | 影响段落密度 |
| detection.guard.* | 静音守卫窗口 | max_shift_ms: 120–220<br/>guard_db: 2.0–3.0 | 开启 enable 可抑制爆音 |
| detection.threshold.base_ratio | 能量谷阈值基线 | 0.22–0.32 | 同步影响 RMS 阈值 |
| detection.adapt.bpm_strength | BPM 自适应强度 | 0.4–0.9 | 快节奏↑，慢歌↓ |
| detection.adapt.mdd_strength | 动态密度自适应 | 0.3–0.6 | 合唱、副歌段增强 |
| detection.nms.topk | 候选切点上限 | 120–240 | 影响 NMS 稳定性 |
| detection.segment_vocal_ratio | human 判定阈值 | 0.08–0.14 | 过高会漏判弱人声 |

## Profile
- ballad：抒情慢歌，拉高 min_pause_s 与 min_gap_s。
- pop：默认流行曲线。
- edm：快节奏/电子，开启守卫并提高 bpm_strength。
- rap：语速快，降低 segment_vocal_ratio 并缩短停顿。

应用顺序：
1. 选择 profile 建立风格基线；
2. 调整 1~2 个旋钮（如 base_ratio、min_gap_s）；
3. 若仍需细调，使用 set_runtime_config 注入临时覆盖。

## 覆盖层级
1. 包内 schema (audio_cut/config/schema_v3.yaml)
2. VSS_EXTERNAL_CONFIG_PATH
3. 显式 YAML (get_config_manager(path))
4. 环境变量 VSS__...
5. 运行时 set_runtime_config

## 兼容性
- get_config 仍支持旧键路径，例如 quality_control.min_split_gap。
- 派生公式集中在 audio_cut/config/derive.py，外部可按需覆写。

<!-- File: docs/SeamlessSplitter 重构记录.md -->
<!-- AI-SUMMARY: 记录本次 SeamlessSplitter 重构的范围、改动与验证信息。 -->

# SeamlessSplitter 重构记录

日期：2026-01-18

## 目标与背景
- 依据 `docs/SeamlessSplitter 重构方案.md` 对 `seamless_splitter.py` 进行拆分重构，降低重复代码与隐藏状态依赖。
- 保持对外行为不变，避免破坏已有用户流程与输出结构（Never break userspace）。

## 变更范围
- 新增策略：`src/vocal_smart_splitter/core/strategies/mdd_start_strategy.py`（Plan A）。
- 共享工具：`src/vocal_smart_splitter/core/strategies/base.py` 增加工具函数。
- 导出与结果构建：`src/vocal_smart_splitter/core/utils/segment_exporter.py`、`src/vocal_smart_splitter/core/utils/result_builder.py`。
- 编排器改造：`src/vocal_smart_splitter/core/seamless_splitter.py`。
- 分析入口复用：`audio_cut.analysis.BeatAnalyzer` 接入 hybrid_mdd 流程。
- 单元测试：`tests/unit/test_mdd_start_strategy.py`、`tests/unit/test_segment_exporter.py`。

## 核心改动
- **策略化切点生成**：hybrid_mdd 统一走策略分发（Plan A/B/C），Plan A 提取为 `MddStartStrategy`。
- **节拍分析复用**：hybrid_mdd 使用 `BeatAnalyzer`，避免重复 beat/bar/energy 计算。
- **导出逻辑统一**：SegmentExporter 统一 `mix_segments`/`vocal_segments`/全长文件导出，保持命名规则：
  - 常规模式 `segment_{i+1:03d}`（从 1 开始）；
  - hybrid_mdd `segment_{i:03d}`（从 0 开始，含 `_lib` 后缀）。
- **结果结构统一**：ResultBuilder 负责通用字段与分离元数据写入，保持 API 依赖字段不变。
- **隐藏状态移除**：hybrid_mdd 不再依赖 `_last_separation_result` 之类隐式状态。
- **兼容辅助函数**：补回 `_select_hybrid_cut_points` 与 `_apply_layout_refiner_to_cuts` 以保持既有测试与调用路径可用。
- **节拍标记一致性**：BeatOnlyStrategy 允许末尾落在小节边界时保持 `_lib` 标记。

## 兼容性说明
- 输出文件命名、`_lib` 标记规则、返回字典字段保持不变。
- `hybrid_mdd` 在 MDD 失败时仍维持原有回退策略：
  - `snap_to_beat` → 回退 `beat_only`；
  - `beat_only` → 允许无 MDD 切点直接生成；
  - `mdd_start` → 保持失败返回。

## 验证
执行：
- `pytest tests/unit/test_mdd_start_strategy.py tests/unit/test_segment_exporter.py`

结果：通过。

## 未覆盖项与后续建议
- 未跑完整回归与真实素材验证；建议执行：
  - `pytest -m "not slow and not gpu" --cov=src --cov-report=term-missing`
  - `python run_splitter.py input/test.mp3 --mode hybrid_mdd`

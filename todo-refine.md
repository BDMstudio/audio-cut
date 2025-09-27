**性能优化 TODO 文档**（含执行顺序清单与方法简介、接口草案、实施步骤、验收标准与回滚策略）。

## 一、执行顺序清单（里程碑与优先级）

**Milestone 0：基线与护栏（P0）**

* [x] 建立基准数据集与统一基准脚本：`scripts/bench/run_bench.py`（统计总时长、每首处理耗时、峰值内存、切点数分布、>15s 长段比例、<2s 碎片比例、守卫右推平均时长、拼接可逆性误差=0）。
  * CLI：`python scripts/bench/run_bench.py --input-dir input --mode v2.2_mdd --json output/bench.json`
  * 指标覆盖：耗时、片段时长（含长段/短段比例）、守卫位移均已落库；峰值 RSS 通过可选 psutil 采集。
* [x] 打上当前 **v2.2** 的**质量护栏阈值**（作为回归标准）：
  * 护栏模板：`scripts/bench/guardrails/v2_2_guardrails.template.json`，可使用 `--save-guardrails` 覆盖
  * 运行示例：`python scripts/bench/run_bench.py --input-dir input --mode v2.2_mdd --json output/bench.json --save-guardrails scripts/bench/guardrails/v2_2_baseline.json`

  * 速度：总耗时下降 ≥ **30%**（单核 CPU 基准）；
  * 质量：长段比例、碎片比例与当前基线 **±5% 以内**；
  * 拼接可逆性：依旧为 0（样本级复原无差）。

**Milestone 1：核心路径提速（P0）**

1. [x] **抽取“通用切点精炼算法”为独立工具类**（消除跨类隐式依赖）。
  * `src/audio_cut/cutting/refine.py` 提供 `CutContext`/`finalize_cut_points`，统一处理过零吸附、守卫右推、min-gap NMS；`SeamlessSplitter` 已改用该模块。
2. [x] **合并全局/局部特征（BPM/MDD）并引入缓存**（一次计算，多处索引）。
  * `src/audio_cut/analysis/features_cache.py` 构建 `TrackFeatureCache`，集中缓存 BPM、RMS、MDD 序列；`SeamlessSplitter` 在分离后构建并传给纯人声检测与 BPM 增强路径。
  * `PureVocalPauseDetector` 与 `VocalPauseDetectorV2/AdaptiveVADEnhancer` 复用缓存数据，避免重复的 `librosa.beat`/能量特征扫描。
3. [x] **合并重复过滤**（把 Weighted NMS 与最终 min-gap 过滤收敛到单一阶段）。
  * `PureVocalPauseDetector` 仅做候选上限控制，去重/最小间隔过滤统一由 `audio_cut.cutting.refine.finalize_cut_points` 执行，消除重复 min-gap 逻辑。

**Milestone 2：瘦身与简化（P1）**
4. [ ] **清理老模式与冗余配置**（保留兼容层，逐步下线）。
5. [ ] **参数集约化**（去重叠参数、保留少量关键超参+派生规则+预设 Profile）。

**Milestone 3：依赖裁剪与收尾（P1）**
6. [ ] **移除 Silero 依赖**（先 Feature-Flag 关闭，完成 A/B 验证后彻底剔除）。
7. [ ] 补充并行/内存与 I/O 微优化（可选 P2）。

---

## 二、详细 TODO（逐项说明）

### 1) 将“通用切点精炼算法”抽取为独立工具类（P0）

**目标**：把当前 `PureVocalPauseDetector` 内部对 `VocalPauseDetectorV2` 的切点精炼/守卫/过零对齐等逻辑**提炼为共享工具**，杜绝跨类调用与隐式耦合。

**建议目录与接口**

* 新增：`src/audio_cut/cutting/refine.py`
* 数据结构：

  ```python
  # src/audio_cut/cutting/refine.py
  from dataclasses import dataclass
  import numpy as np
  from typing import List, Optional

  @dataclass
  class CutPoint:
      t: float                 # 秒
      score: float             # 置信度/权重
      kind: str = "pause"      # "pause" | "breath" | "uncertain"

  @dataclass
  class CutContext:
      sr: int
      mix_wave: np.ndarray         # 原混音 mono 或 (n, )，外部已做 downmix
      vocal_wave: Optional[np.ndarray] = None  # 纯人声轨（可选）
  ```
* 通用算法函数（**仅此一处实现**）：

  ```python
  def align_to_zero_cross(wave: np.ndarray, sr: int, t: float, win_ms: float=8.0) -> float: ...
  def apply_quiet_guard(wave: np.ndarray, sr: int, t: float,
                        max_shift_ms: float=150.0, floor_db: float=-48.0) -> float: ...
  def nms_min_gap(points: List[CutPoint], min_gap_s: float, topk: Optional[int]=None) -> List[CutPoint]: ...
  def finalize_cut_points(ctx: CutContext, raw_points: List[CutPoint],
                          use_vocal_guard_first: bool=True,
                          min_gap_s: float=1.0) -> List[CutPoint]:
      """
      1) 先在人声轨做过零+守卫 -> 得到 t'
      2) 再在混音轨做过零+守卫 -> 得到 t''
      3) 一次性 NMS(min_gap) 收敛
      """
  ```

**实施步骤**

* [ ] 从 `src/audio_cut/detectors/pure_vocal_v22.py` 与 `src/audio_cut/detectors/vad_v2.py`（示意）中，**剪切**所有与“过零、守卫、min-gap 去重、top-k 限幅”相关的代码到 `refine.py`。
* [ ] 两个检测器仅**调用 `refine.finalize_cut_points`**；不再互相调用或复制逻辑。
* [x] 新增单测：`tests/test_cutting_refiner.py`

  * [x] 过零对齐在纯正弦与锯齿波上**最大偏差 < 1/sample_rate**；
  * [x] 守卫右推**不超过 max_shift_ms**，并能降低局部 RMS；
  * [x] NMS 后**任意相邻切点间隔 ≥ min_gap_s**。

**验收标准**

* 代码层面：核心精炼逻辑仅在 `refine.py` 一处存在。
* 性能：相对基线 CPU 时间下降 ≥ **5%**（减少重复实现带来的额外计算/分支）。
* 质量：切点统计分布与基线差异在护栏范围内。

**风险&回滚**

* 风险：接口迁移遗漏导致某些调用路径未用新逻辑。
* 回滚：保留老实现为隐藏开关 `CUT_REFINER_LEGACY=1`（环境变量）做紧急回退。

---

### 2) 清理老模式与冗余配置（P0）

**要点**：随着 v2.2 成为主路径，建议**保留 1 套主流程**，其他模式仅作为**可选降级**或彻底移除。同步清理配置里早已无效/重复的开关。

**候选清理清单（示意）**

* 模式/代码：

  * [ ] `src/audio_cut/modes/smart_split.py`（若仅用于直切 VAD，转为 demo/示例或 plugin 形式）
  * [ ] `src/audio_cut/modes/pure_vocal_v21.py`（标记 deprecated；一个版本后移除）
  * [ ] 早期“二次插点”“强制拆分”等 dead code 路径
* 配置键（在 `src/audio_cut/config/schema_*.yaml` 中搜索）：

  * [ ] 与“二次插点/强拆”相关的开关与阈值（如 `force_insert_cuts`, `double_pass`, `legacy_guard` 等）
  * [ ] 重复的阈值倍率（见第 4 点参数集约化）
  * [ ] 未被代码引用的旧键（`rg`/`ripgrep` 搜索：`rg -n "config\['(\w+)'\]" -g "src/**.py"` 交叉对照 YAML）

**实施步骤**

* [ ] 在 `src/audio_cut/cli/quick_start.py`：将非主模式**隐藏到“兼容模式”**二级菜单；默认仅展示 v2.2。
* [ ] 新增**配置迁移层**：`src/audio_cut/config/migrate_v2_to_v3.py`

  * [ ] 打印 **DeprecationWarning**，把旧键转换到新键；
  * [ ] 在 1 个稳定版本后移除迁移层与旧键解析。
* [ ] 单测：`tests/test_config_migration.py` 覆盖旧 YAML 的读入与行为一致性。

**验收标准**

* 启动日志中**无**“未使用配置键”与“找不到配置键”警告；
* 二级菜单保留兼容入口但默认流程仅 1 条；
* 基准数据集上功能一致，耗时略降（加载/分支减少）。

**风险&回滚**

* 风险：线上用户旧配置失效。
* 回滚：保留 `--compat-config v2` 开关与 `migrate_v2_to_v3.py` 热修。

---

### 3) 合并重复计算：BPM/MDD 全局与局部一次算好、处处复用（P0）

**目标**：避免在“全曲阈值计算”和“局部停顿增强”中分别重复计算 MDD/BPM；避免在 `AdaptiveVADEnhancer` 与 `PureVocalPauseDetector` 里各自跑一遍节拍检测。

**建议目录与接口**

* 新增：`src/audio_cut/analysis/features_cache.py`

  ```python
  # src/audio_cut/analysis/features_cache.py
  from dataclasses import dataclass
  import numpy as np

  @dataclass
  class TrackFeatureCache:
      sr: int
      hop_s: float
      bpm: float
      bpm_conf: float
      global_mdd: float
      # 统一时间栅格的局部特征序列（全曲长度对齐）
      mdd_series: np.ndarray   # shape: (T,)
      rms_series: np.ndarray   # shape: (T,)
      spec_flatness: np.ndarray  # 可选
      onset_strength: np.ndarray  # 可选

      def idx(self, t: float) -> int: ...
      def get_window(self, t: float, w_s: float) -> dict: ...
  ```
* 入口函数（被检测器与阈值模块共享）：

  ```python
  def build_feature_cache(mix_wave, vocal_wave, sr, hop_s=0.02) -> TrackFeatureCache: ...
  ```

**实施步骤**

* [ ] 在**加载音频后仅一次**调用 `build_feature_cache`，缓存到 `SeamlessSplitter` 上下文（如 `ctx.features`）。
* [ ] 重写使用 BPM 的地方（自适应阈值、pause_stats_adaptation、VAD enhancer）全部改为**索引 `ctx.features`**；
* [ ] MDD 全局值直接用 `cache.global_mdd`；局部值用 `cache.get_window(t, w_s)`。
* [ ] 移除所有**重复的 librosa/自实现**节拍与复杂度计算。

**验收标准**

* 端到端 CPU 时间再降 ≥ **10–20%**（依曲目而异）；
* `features_cache` 单测覆盖**边界对齐**（时间→索引）与**重复调用一致性**（相同输入得到相同特征）。

**风险&回滚**

* 风险：时间栅格对齐误差导致局部指标引用错位。
* 回滚：在 `build_feature_cache` 保留 `legacy_mode`，返回旧 API 的镜像字段。

---

### 4) 参数配置集约化（去重叠，保留少量关键超参 + 派生规则 + 预设 Profile）（P1）

**问题**：阈值比例、倍率、时长门限过多，且有**作用重叠**（如 `relative_threshold_adaptation` 与 `pause_stats_adaptation` 都对阈值生效）。

**方案**

* 新配置文件：`src/audio_cut/config/schema_v3.yaml`

  * **核心可调（5~8 个）**：

    * `min_pause_s`（最小停顿判定时长）
    * `min_gap_s`（最终切点最小间隔）
    * `guard.max_shift_ms`（守卫最大右推）
    * `guard.floor_db`（守卫噪声地板）
    * `threshold.base_ratio`（能量谷相对阈值基线）
    * `adapt.bpm_strength`（BPM 影响强度 0~1）
    * `adapt.mdd_strength`（MDD 影响强度 0~1）
    * `nms.topk`（可选，上限切点数）
  * **派生量**（代码里算，不进配置）：

    * `threshold.effective = base_ratio * f(bpm, mdd)`
    * `min_pause_effective = g(bpm)`（快歌略降，慢歌略升）
* 预设 Profile：`profiles: {ballad, pop, edm, rap}` 仅覆盖 `bpm_strength/mdd_strength/min_pause_s` 三五项。

**实施步骤**

* [ ] 写 `src/audio_cut/config/derive.py`：集中把**所有派生逻辑**实现于此（BPM/MDD→阈值与门限转化）。
* [ ] 清理旧键：把多层倍率/比例**并入**上述核心可调项；迁移脚本见第 2 点。
* [ ] 文档：在 `docs/tuning.md` 用“少即是多”的表格说明**每一项**对切割的单调影响。

**验收标准**

* 配置键数量**腰斩**（例如从 30+ → ≤ 12）；
* 新/旧配置跑同一数据集，结果在护栏内；
* 新使用者仅通过 1–2 个参数+一个 Profile 即可达到接近最佳效果。

**风险&回滚**

* 风险：极端风格下缺少某细粒度开关。
* 回滚：在 `derive.py` 保留 `advanced_overrides: dict` 注入钩子，供专家用户补充细调。

---

### 5) 合并重复过滤：一次 NMS(min-gap) 即终（P1）

**问题**：当前在 **Weighted NMS** 阶段按间隔过滤一次、在最终 `pure_filter_cut_points` 又过滤一次，**重复**。

**方案**：把**间隔与排序**收敛到 `refine.nms_min_gap` / `refine.finalize_cut_points` 一处。

* 排序权重：`score * w(kind) * h(duration)`（可把“真停顿”权重大于“breath”）。
* NMS 规则：

  1. 取当前最高分点；
  2. 抑制与之距离 `< min_gap_s` 的其他点（保留分最高者）；
  3. 可选 `topk`。

**实施步骤**

* [ ] 移除所有额外的“结果后再 min-gap”过滤，仅在 `finalize_cut_points` 中执行。
* [ ] 单测：为合并前后**任意输入候选**，最终**相邻间隔 ≥ min_gap_s** 且**稳定可复现**（排序稳定）。

**验收标准**

* 代码路径更短；
* 端到端再省 **3–5%** CPU；
* 切点数量与分布稳定且不回退。

**风险&回滚**

* 风险：部分输入以前靠第二次过滤修正，现在只做一次可能放过边界点。
* 回滚：保留 `NMS_STRICT=1` 环境开关，启用“二次保险过滤”。

---

### 6) 确定移除对 Silero 的依赖（P1）

**目标**：降低模型加载与推理开销，尤其当主流程是“先分离再做人声停顿检测”时，VAD 模型收益变小。

**策略（两阶段）**

* **A 阶段（默认关闭 Silero）**

  * [ ] 在 `src/audio_cut/plugins/vad_silero.py` 保留封装，但通过配置 `use_silero=false`（默认）禁用；
  * [ ] 若 `Smart Split` 直切模式仍需 VAD，改用**轻量能量门控**（见下）。
* **B 阶段（彻底移除）**

  * [ ] 删除依赖、代码与安装项；`Smart Split` 模式变成示例/实验性。

**轻量能量门控（替代 VAD）**

```python
# src/audio_cut/detectors/energy_gate.py
def energy_gate(wave: np.ndarray, sr: int, frame_s: float=0.02,
                bands=((80,300),(300,3000)),   # 低频+中高频两带
                thr_db=-40.0, hang_ms=120.0):
    """
    返回二值人声活动掩码：能量(RMS)在两带同时低于阈值则判静音；带有hang抑抖。
    结合 features_cache 的局部 MDD 动态提高阈值，避免高潮段误判。
    """
```

**A/B 验证**

* [ ] 基准集上对比 `use_silero=true/false`：

  * 速度增益预期 ≥ **10–15%**；
  * 质量指标保持在护栏内；
  * 重点审查：嘈杂伴奏+弱人声场景的漏检（必要时在 `energy_gate` 增加带通/谱熵特征）。

**验收标准**

* `use_silero=false` 成为默认，质量不过线时才建议用户启用/安装。
* 完成 B 阶段后，依赖树最简、冷启动时间进一步下降。

**风险&回滚**

* 风险：极端素材下 VAD 去除导致“错过短停顿”。
* 回滚：保留 `pip install audio-cut[vad]` 可选 extra；或留独立插件仓库。

---

## 三、工程与性能微优化（可并行推进，P2）

* [ ] **缓存与预分配**：所有帧级特征数组使用 `np.empty` 预分配；尽量 `float32`。
* [ ] **统一重采样**：入口统一到 `sr=44100` 或 `48000`，避免多次 `resample`。
* [ ] **并发**：批处理时用 `concurrent.futures.ProcessPoolExecutor` 按文件并行；模型（如 MDX）允许 GPU 时使用持久会话池。
* [ ] **I/O**：用 `soundfile` 流式读取大文件的单声道 downmix，减少峰值内存。
* [ ] **剖析**：`scripts/bench/profile_cpu.sh`（`py-spy`, `line_profiler`）与 `profile_mem.sh`（`memory_profiler`）定位热点。

---

## 四、示例：关键文件与调用关系（建议）

```
src/
  audio_cut/
    analysis/
      features_cache.py      # 一次计算、全局缓存
    cutting/
      refine.py              # 切点精炼/守卫/NMS 统一入口
    detectors/
      pure_vocal_v22.py      # 调用 features_cache 与 refine
      vad_v2.py              # 如仍保留，亦仅调用 refine
      energy_gate.py         # 轻量替代 VAD
    config/
      schema_v3.yaml         # 精简配置
      derive.py              # 参数派生逻辑
      migrate_v2_to_v3.py    # 配置迁移/兼容
    cli/
      quick_start.py         # 菜单：默认仅 v2.2；其他归入兼容模式
scripts/
  bench/
    run_bench.py             # 基准评测
tests/
  test_cutting_refiner.py
  test_features_cache.py
  test_config_migration.py
```

---

## 五、验收清单（一次性勾完才算过线）

* [ ] **速度**（CPU 单核基准）：端到端平均耗时下降 **≥30%**；
* [ ] **内存峰值**：下降 **≥15%**；
* [ ] **质量护栏**：长段比例、碎片比例在基线 **±5%**；
* [ ] **可逆性**：拼接误差仍为 0；
* [ ] **代码健康度**：

  * 通用精炼逻辑仅在 `refine.py`；
  * 重复 BPM/MDD 计算彻底移除；
  * 配置键数显著减少，文档与迁移脚本齐全；
  * CI 增加基准与回归测试任务（失败即阻断合并）。

---

## 六、直接做的“快刀”动作（建议从这里开工）

1. 在 `src/audio_cut/cutting/refine.py` **落地** `CutPoint/Finalizer`（把过零+守卫+NMS 集中）。
2. 落地 `features_cache.py` 并把 `PureVocalPauseDetector` 改为**只读缓存**。
3. 把 `pure_filter_cut_points` 里的 min-gap 逻辑**删除**，改为只走 `refine.finalize_cut_points`。
4. `schema_v3.yaml` 与 `derive.py` 成形，`migrate_v2_to_v3.py` 跑通旧配置。
5. `use_silero=false` 做一次全量基准，若指标达标，默认关闭并标注为可选插件。

---

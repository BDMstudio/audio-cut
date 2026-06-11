# audio-cut v2.8 意图门面方案（产品面减法）

- 日期: 2026-06-11
- 状态: 提案（待评审冻结）
- 前置: v2.7 引擎侧已合入（统一候选池 / AutoProfile / unified.yaml 瘦身，commits `2d88394`→`380ec5c`）；v2.7 人工验收（I 节）不阻塞本方案
- 关联: `docs/audio_cut_v2_7_unified_cut_proposal.md`、`docs/audio_cut_v2_7_unified_cut_todo.md`
- 执行清单: `docs/audio_cut_v2_8_intent_surface_todo.md`

---

## 0. 核心判断

**✅ 值得做。**

v2.7 把引擎收成了一条统一候选池路径，但产品面还在让用户从五六个算法名词里做选择题。`quick_start.py` 最多连问 6 个问题，词汇表是「MDD 低谷检测」「snap_to_beat」「sidecar」——这是在让用户替我们做架构决策，不是在服务用户。

用户（人或 agent）的真实意图空间只有**二维**：

1. **密度轴**：片段切多碎（含零点 = 不切、只分离）
2. **对齐轴**：切点偏歌词语义，还是偏节拍卡点（一个滑块，不是两个模式）

五个模式只是这个平面上的五个坐标点，却被当成五条用户可见的路径来卖。v2.8 的工作就一句话：**把坐标系交给用户，把坐标点收回来当预设。**

本方案零引擎重构——所有零件 v2.7 已就位（`CandidateSource.BEAT` 入池、`phrase_boundary.weights` 打分、`derive_smart_cut_overrides` 时长派生、QA 指标 `beat_aligned_ratio`/`breath_cut_ratio`）。v2.8 只做接线和门面。

---

## 1. 问题定位：引擎已统一，门面还在卖算法

### 1.1 证据

| # | 现象 | 位置 |
|---|------|------|
| E1 | quick_start 连问最多 6 题：模式 5 选 1 → 风格 5 选 1 → (hybrid) 密度 + 对齐策略 / (vpbd) 歌词来源 5 选 1 + 失败策略 → 输出格式 | `quick_start.py:63-281` |
| E2 | 问题选项是实现名词：「v2.2 MDD 低谷检测」「snap_to_beat」「FireRed sidecar」「VPBD acoustic」 | `quick_start.py:70-201` |
| E3 | 门面自己打架：配置默认 `v2.2_mdd`，quick_start 默认 `vpbd_asr` | `config/unified.yaml:5` vs `quick_start.py:88` |
| E4 | `cut_style` 把两个轴塞进一个枚举：`rhythmic` 是对齐轴的事，`dense` 是密度轴的事；且 `rhythmic` 只把 beat_affinity 从 0.08 挪到 0.12，动态范围名不副实 | `src/audio_cut/config/auto_profile.py:228-237` |
| E5 | 歌词 provider 的回退链（sidecar→cli→声学降级）已存在，但 quick_start 仍要求用户回答「sidecar 还是 CLI」 | `config/expert.yaml` `fire_red.provider_order` vs `quick_start.py:186-245` |
| E6 | 偏节拍 = `hybrid_mdd` 独立路径，偏歌词 = `vpbd_asr` 独立路径，二者互斥——用户感知里"一个滑块的事"被实现成两条不相交的代码路 | `seamless_splitter.py` 模式分发 |

### 1.2 利益相关方

- **人类用户**：交互入口是 quick_start，需要的是"少问、问人话"。
- **agent 工作流（mvagent / hermes / openclaw）**：调用入口是 `audio_cut.api.separate_and_segment()` 与 `SegmentManifest.json`，需要的是**小而稳定、可机读、可复现**的契约——参数语义含混对 agent 是毒药，它没法"试着听一下"。

---

## 2. 数据结构：二维意图空间

> "Bad programmers worry about the code. Good programmers worry about data structures."

用户意图的最小完备表示：

```
intent = {
    target_duration_s: [min_s, max_s]   # 密度轴真值（已存在: smart_cut.target_duration_s）
    alignment: 0.0 .. 1.0               # 对齐轴真值（本方案新增）
    lyrics: auto | off                  # ASR 可用性，auto 自带回退链
}
```

两轴**同构设计**——都是「命名档位 + 数值」双轨，档位只是数值的语法糖，在入口处解糖，内部只有数值一个真值：

| 轴 | 命名档位（人类面） | 解糖为（机器真值） |
|----|------------------|------------------|
| 密度 `segments` | `few` / `medium` / `many` | `target_duration_s` = [10,18] / [5,12] / [3,8] |
| 对齐 `alignment` | `lyric` / `lyric_lean` / `balanced` / `beat_lean` / `beat` | 0.0 / 0.25 / 0.5 / 0.75 / 1.0 |

旧模式 = 平面上的坐标点：

| 旧模式 | 等效意图坐标 | v2.8 处置 |
|--------|-------------|----------|
| `vocal_separation` | 密度零点（不切） | 并入三问第 1 问；mode 别名保留 |
| `vpbd_asr` | alignment 0.5 × lyrics auto | **统一引擎本体**，意图面默认路由 |
| `vpbd_acoustic` | alignment 0.5 × lyrics off | 同上（lyrics 降级落点） |
| `v2.2_mdd` | ≈ alignment 0.5 × lyrics off | 旧路径原样保留为别名 |
| `hybrid_mdd` | ≈ alignment 0.9 × 密度 medium-many | 旧路径原样保留为别名；卡点感 A/B 的对照组 |
| `librosa_onset` | alignment 1.0 且无人声保护 | expert 后门（唯一允许切穿人声的路径），不上产品面 |

消除的特殊情况：

- `vocal_separation` 不再是"模式"，是密度轴的零点——三问第 1 问吃掉它。
- `cut_style` 拆轴回收：`rhythmic` → alignment 0.7，`dense` → segments many，`natural` → 0.5（恒等）。
- 歌词来源不再是问题：`lyrics: auto` 的回退链已实现，人和 agent 永远不需要知道 sidecar 是什么。
- 风格不再是问题：AutoProfile（v2.7 已落地）按 BPM/MDD/能量 CV 自动估计，手动 `--profile` 退为专家逃生口。

---

## 3. 设计

### 3.1 用户面总览

**quick_start 三问**（文件选择之外不再有任何问题；输出格式、歌词 provider、风格全部走配置默认）：

```
1) 要切片，还是只做人声/伴奏分离？     [1] 切片   [2] 只分离
2) 片段密度？                          [1] 少(10-18s)  [2] 中(5-12s)  [3] 多(3-8s)
3) 切点风格？        [1]歌词优先  [2]偏歌词  [3]均衡  [4]偏节拍  [5]强卡点
```

**unified.yaml `smart_cut` 新形态**（用户面唯一需要碰的节点）：

```yaml
smart_cut:
  segments: medium        # few|medium|many，或直接 target_duration_s: [5.0, 12.0]
  alignment: balanced     # lyric|lyric_lean|balanced|beat_lean|beat，或 0.0–1.0
  lyrics: auto            # auto|off；provider 细节在 expert/fire_red，不上用户面
  profile: auto           # 保留；手动 ballad/pop/edm/rap 仍优先（专家用）
  # cut_style: deprecated，读到时拆轴映射并告警（见 §4）
```

**CLI**（`run_splitter.py`）：

```bash
python run_splitter.py input/song.mp3 --segments medium --align beat_lean
python run_splitter.py input/song.mp3 --align 0.8          # agent / 脚本用数值轨
python run_splitter.py input/song.mp3 --mode hybrid_mdd    # 旧别名永久可用
```

给了意图旗标且未显式给 `--mode` 时路由统一引擎（vpbd 路径 + lyrics auto）；显式 `--mode` 永远赢。

**API**（`audio_cut.api.separate_and_segment`）：

```python
manifest = separate_and_segment(
    input_uri="input/song.mp3",
    export_dir="output/job",
    segments="medium",        # 新增，可选：few|medium|many|(min_s, max_s)
    alignment=0.75,           # 新增，可选：档位名或 0.0–1.0
)
# manifest["intent"] 回显解析后的真值，agent 可验证请求被如何理解：
# {"target_duration_s": [5.0, 12.0], "alignment": 0.75, "lyrics": "auto", "profile": "auto"}
```

### 3.2 alignment 滑块语义

接线点：`SeamlessSplitter._apply_smart_cut_runtime`（`seamless_splitter.py:744`，已在 `:345` 于边界检测前调用）。滑块覆盖叠加在 AutoProfile 风格权重**之后**。

机制——**两极表 + 风格基点的两段线性插值**，一个函数，零分支：

```
a = resolve_alignment(smart_cut.alignment)        # 档位解糖 → float，缺省 0.5
style_base = AutoProfile 产出的 phrase_boundary.weights（现状，v2.7 已有）

a <= 0.5:  weights = lerp(LYRIC_POLE, style_base, t=2a)
a >  0.5:  weights = lerp(style_base, BEAT_POLE,  t=2a-1)
```

两极表初值（验收阶段按 A/B 校准，见 §5）：

| 权重键 | LYRIC_POLE (a=0) | 基点 (a=0.5) | BEAT_POLE (a=1) |
|--------|-----------------|--------------|-----------------|
| acoustic_pause | 0.38 | AutoProfile 风格值 | 0.22 |
| asr_gap | 0.26 | 〃 | 0.10 |
| sentence_end | 0.22 | 〃 | 0.08 |
| beat_affinity | 0.02 | 〃 | 0.32 |
| mdd_affinity | 0.06 | 〃 | 0.12 |
| breath | 0.10 | 〃 | 0.10 |
| inside_word_penalty | **0.85** | 〃 | **0.80** |
| singing_penalty | **0.50** | 〃 | **0.50** |

同时由 `a` 派生：

- `vpbd.beat_candidates.base_score`：a<0.3 → 0（节拍网格候选不入池）；a=0.5 → 0.3（现状）；a=1.0 → 0.65
- `global_planner.beat_conflict_weight`：0.0 → 0.15（现状）→ 0.30

三条铁律：

1. **a=0.5 是恒等变换**：`derive_alignment_overrides(0.5, ...)` 返回空 dict，现有 vpbd 用户行为零变化——这是兼容性的基石。
2. **惩罚项不松**：`inside_word_penalty` / `singing_penalty` 两极取值始终高位（0.85→0.80 仅微调，永不松到放行）——滑到最右也不切人声。守卫链（`finalize_cut_points` → 词保护 → layout）对滑块所有位置一视同仁。真想无脑切网格，去走 `librosa_onset` expert 后门。
3. **两轴解耦**：alignment 只动"偏好哪类候选"（权重/入池分），密度只动"要多少切点"（planner 时长目标，`derive_smart_cut_overrides` 已实现）。`beat_candidates.bars_per_cut` 归密度轴管，不随滑块动。

### 3.3 路由与别名

- 意图面（三问 / `--segments`/`--align` / API 新参数）**永远走统一候选池一条路**（vpbd + lyrics auto）。
- `--mode` / `mode=` 六个旧值**原样保留、原路径不动**——v2.8 不删一行旧路径代码（见 §6 非目标）。引擎收路（hybrid_mdd/librosa_onset 归并候选池）是 v2.9 的事，前提是 §5 的 A/B 证明等价。
- `global.default_mode` 与 quick_start 默认统一为意图面默认（修掉 E3 的打架）。

### 3.4 Agent 契约（mvagent / hermes / openclaw）

把 `audio-cut封装为模块.md` 更新为一页契约（不新开文档，文档也做减法），保证四件事：

1. **签名稳定**：`separate_and_segment()` 既有参数零变化，新参数全部可选。
2. **Manifest 字段只增不改**：旧字段一个不动；新增 `intent` 回显节；`segments[].lyrics`（`attach_lyrics_to_segments` 已实现）和 `qa_report`（`breath_cut_ratio`/`beat_aligned_ratio`/`cut_inside_word_rate`）是 agent 的闭环依据。
3. **降级语义明确**：lyrics auto 失败 → 声学候选继续；GPU 失败 → CPU 回退（除非 strict）；任何可选依赖缺失不抛崩，降级路径写进 manifest。
4. **可复现**：同输入 + 同 intent 数值 → 同输出（AutoProfile 估计结果随 manifest 落盘，agent 可固定 profile 复跑）。

---

## 4. 兼容性铁律（Never break userspace）

| # | 承诺 | 验证手段 |
|---|------|---------|
| N1 | `separate_and_segment()` 既有签名与返回结构零变化 | 契约测试 |
| N2 | `--mode` 六值原样可用，输出文件命名不变（`_lib` 后缀仍属 hybrid_mdd 别名） | 旧模式三连跑 diff |
| N3 | `smart_cut.target_duration_s` / `profile` 原语义保留；`alignment` 缺省 = 0.5 = 恒等 | 单测：缺省配置下覆盖集为空 |
| N4 | `cut_style` 继续被读取：natural→0.5（恒等）、rhythmic→0.7、dense→segments many，打 `DeprecationWarning`，v3.0 移除 | 单测 + 告警断言 |
| N5 | Manifest 旧字段不动，只增 `intent` 节 | 契约测试 |
| N6 | `segments` 与 `target_duration_s` 同时给出时，数值轨赢并告警（explicit wins） | 单测 |

---

## 5. 验收标准

| 维度 | 标准 |
|------|------|
| 卡点感 A/B | 同 2–3 首高能量歌：`alignment=1.0 × segments many` vs `hybrid_mdd snap_to_beat` 留存基线；`beat_aligned_ratio` 差 ≤10pp，`cut_inside_word_rate` = 0，主观听感不低于对照组 |
| 恒等性 | 缺省配置（alignment 未设/0.5）下，vpbd_asr 输出与 v2.7 基线逐样本一致 |
| 三问可用性 | quick_start 从启动到出片 ≤4 次输入（文件 + 三问），全程零算法名词 |
| agent 冒烟 | `separate_and_segment(segments='medium', alignment=0.75)` → manifest 含 `intent` 回显 + `segments[].lyrics` + `qa_report` |
| 旧模式回归 | 六个 `--mode` 三连跑输出 diff 为零；快速回归 + 契约 + 拼接精度 ≤1e-12 全绿 |

---

## 6. 非目标

- **不删除**任何旧模式代码路径（hybrid_mdd/librosa_onset 归并候选池是 v2.9，等 A/B 数据说话）。
- 不动分离后端（MDX23/Demucs）与 GPU 流水线。
- 不做连续 alignment 的自动学习/推荐（AutoProfile 管风格，不管用户偏好）。
- 不做 GUI；quick_start 仍是终端问答。
- 不在本轮补 v2.7 的 20 首人工验收（I 节独立推进，playlist 可与本方案 A/B 共用）。

---

## 附录 A：证据与接线锚点

| 锚点 | 位置 |
|------|------|
| 三问接线点 | `src/vocal_smart_splitter/core/seamless_splitter.py:744` `_apply_smart_cut_runtime`（`:345` 调用） |
| 风格权重与 cut_style 现状 | `src/audio_cut/config/auto_profile.py:33-74, 228-237` |
| 时长派生（密度轴已就位） | `src/audio_cut/config/auto_profile.py:146-167` `derive_smart_cut_overrides` |
| 候选源枚举（BEAT 已入池） | `src/audio_cut/cutting/cut_candidate.py:13-23` |
| 节拍网格候选 | `src/audio_cut/cutting/beat_candidates.py`；`config/expert.yaml` `vpbd.beat_candidates` |
| 打分权重节点 | `config/expert.yaml` `phrase_boundary.weights` |
| QA 闭环指标 | `src/audio_cut/qa_report.py:46-47, 62`（`breath_cut_ratio` / `beat_aligned_ratio`） |
| 片段挂歌词 | `src/audio_cut/lyrics/segment_attach.py` `attach_lyrics_to_segments`（`api.py:266-268` 调用） |
| 门面现状（被替换对象） | `quick_start.py:63-281`；`run_splitter.py:60-153`；`config/unified.yaml:4-12` |

现在的两类异常，根因都是**阈值/守卫过严 + 节拍/密度没有进入“快歌模式”**，导致候选点被自己“过滤”光了。再叠加“谷值对齐默认关闭”和合并窗口过大，结果就变成——**只切出两段**、或者**系统提示“中间没有足够候选停顿”**。

---

# 一、问题复盘 → 可能根因

## 1) “MDX23 + 纯人声检测 v2.1（推荐）”只切两段

**可疑点：**

* **未来静默守卫过严**：VocalPrime 描述里有“切点后需≥0.6–1.0s静音”式的未来静默守卫；快歌/说唱段落根本给不出这么长的纯人声静区，候选点全被否了。你在 README 的“谷值切割与拍点禁切”片段里给出的 valley guard 仅 120 ms，但 **VocalPrime 的守卫并不是同一个参数**，容易出现“valley宽松、VocalPrime仍然苛刻”的错配。([GitHub][1])
* **平台平坦度（flatness）判据过严**：快歌里人声平台很少真正“平”，若仍以 ≤6 dB（或 10 dB）波动过滤，快歌的大部分“微停+换气”会被判伪。([GitHub][1])
* **默认没有强启“谷值优先”**：README 片段显示 `enable_valley_mode: false`，且 BPM 守护默认也关闭。于是当 VAD/VocalPrime 过严时，不会自动把切点**对齐到能量谷**，这就是你频谱里看到“落刀在峰”的直接原因。([GitHub][1])
* **合并窗口/最小片段时长**对快歌不友好：如果 `merge_window_ms`、`min_segment_len_s` 没按 BPM 动态缩小，很多近邻候选会被“合并掉”，只剩一短一长两段。

## 2) “MDX23 + MDD 增强 v2.2”提示“歌曲中间没有足够候选停顿”

**可疑点：**

* **MDD错误用在“硬阈值裁剪”**：如果你直接用**全球阈值**（如 MDD > p75 才认“忙”、MDD < p25 才认“闲”）去删选候选点，快歌里 MDD 几乎全程高位，**中段就“没有停顿”**。
* **MDD来源不当**：若你把 **纯人声 stem** 当 MDD 主输入，密度变化被极度压缩（鼓/贝斯/铺底信息缺失），MDD 曲线“平上加平”，更不容易产生日志意义上的“dip”（密度凹陷）。MDD 应以**伴奏 stem**为主，必要时小权重混入原混音，之前我已强调过。
* **没有“拍点网格兜底”**：当**音频学停顿不足**时，系统本该退到**拍点/小节边界的“节拍同步谷值”兜底**；如果没做或参数太严，当然就“候选不足”。
  （README 已写出 BPM 自适应与拍点禁切能力，但默认 `bpm_guard.enable: false`，你没开。) ([GitHub][1])

---

# 二、立刻可做 → 最小侵入式修复（按文件路径给）

> 下面所有路径都包含上一级目录，符合你的习惯要求。

## A. 打开“谷值优先 + 拍点守护”

**文件：** `src/vocal_smart_splitter/config.yaml`
**修改：**

```yaml
vocal_pause_splitting:
  enable_valley_mode: true        # 默认 false → true
  auto_valley_fallback: true      # 保持
  local_rms_window_ms: 25
  silence_floor_percentile: 5
  lookahead_guard_ms: 120         # 仅用于 valley 对齐守卫
  min_valley_width_ms: 120
  bpm_guard:
    enable: true                  # 默认 false → true
    forbid_ms: 100                # 80–120ms 之间可调
```

**效果：** 刀口先对谷、不踩拍；就不会再出“峰上落刀”的尴尬。

## B. 降低快歌的人声守卫强度（关键）

**文件：** `src/vocal_smart_splitter/core/vocal_prime_detector.py`
**策略：** 按 BPM 动态调整**未来静默守卫 + 平台平坦度 + 最小停顿时长**。
**建议参数：**

```python
# 伪代码：在 VocalPrime 聚合阈值处
guard_ms = max(60, 0.25 * 60_000 / max(80, bpm))  # 0.25 拍作为守卫, 快歌≈80–120ms
flatness_db = 12 if bpm >= 120 else 10            # 快歌放宽平台平坦度
min_pause_ms = 60 if bpm >= 120 else 90           # 允许“微停”进入候选
```

**效果：** 快歌不再因“没有 ≥0.6s 静区”而全军覆没。

## C. 别让合并窗口吞掉快歌的微停

**文件：** `src/vocal_smart_splitter/core/seamless_splitter.py`
**修改要点：**

* `merge_close_pauses_ms` 随 BPM 缩小：

  * 慢歌：280–320ms；中速：200–240ms；快歌：120–160ms；极快：90–120ms。
* `min_segment_len_s` 同步下调：

  * 慢歌≥4s；中速≥3s；快歌≥1.8s；极快≥1.2s。

**效果：** 不再把“两三个好候选”合并成一个长段。

## D. 正确使用 MDD：以伴奏为主 + “局部统计”选 dip

**文件：** `src/vocal_smart_splitter/core/mdd_analyzer.py`（若尚未建，建议新增）
**输入：** `accompaniment.wav`（MDX/Demucs 伴奏 stem），质量差时小权重融合 `mix.wav`（0.2–0.35）。
**核心改动：**

1. **局部 z-score dip 选择**（代替全局硬阈值）：

   ```python
   # MDD 序列经节拍对齐后，做滑窗 μ/σ 标准化
   z = (mdd - mu_local) / (sigma_local + 1e-6)
   dips = np.where(z <= -0.8)[0]  # k=0.8~1.2 可调
   ```
2. **拍点网格兜底**：若 `len(dips) < N_min`（比如 < 6），在每 4 拍末或 8 拍边界上**寻找局部能量谷**作为“合成候选”。
3. **副歌保护**（可选）：当 MDD 高密区被识别为副歌段时，切点向最近谷值右推 ≥80ms，避免“唱到一半落刀”。

> 之前 README 的结构与特征模块已给出：`src/vocal_smart_splitter/core/enhanced_vocal_separator.py`、`.../dual_path_detector.py`、`.../bpm_vocal_optimizer.py`等，串接难度低。([GitHub][1])

## E. 默认配置里把 MDD 用对（伴奏主导 + 兜底策略生效）

**文件：** `src/vocal_smart_splitter/config.yaml`
**新增/调整：**

```yaml
mdd:
  input: "accompaniment"   # "accompaniment" | "mix" | "vocals"
  fuse_with_mix: true
  mix_weight: 0.2
  bpm_lock: true
  dip_selector:
    method: "local_sigma"
    sigma_k: 0.9
    min_dip_dur_ms: 80
  beat_fallback:
    enable: true
    every_n_beats: 4
    offset_beats: 0.0       # 在小节边界附近找局部谷值
  chorus_protect:
    enable: true
    push_right_ms: 80
```

---

# 三、你现在就能跑的“三步回归测试”

> 目标：证明“快歌可切、落刀在谷、不再只剩两段”。

1. **启用谷值 & 拍点守护（零代码跑通）**

```bash
# 修改 src/vocal_smart_splitter/config.yaml 如上 A、E
python quick_start.py
```

**预期：** 频谱上刀口向局部能量谷靠拢，不再踩峰；`analysis_report.json` 里候选数量↑，“中间没有足够候选”不再出现。配置项与说明见仓库 README 对应段落。([GitHub][1])

2. **快歌专项**（120–170 BPM 的样本各 5 首）
   把 `vocal_prime_detector.py` 的**未来静默守卫**与**平坦度**改为 BPM 自适应（见 B）。
   **验收指标：**

* 候选数 ≥ 12（3 分钟歌典型）；
* 刀口-谷值偏移 |Δt| ≤ 60–120 ms（统计 P90）；
* 片段长度分布：中位数 4–10 秒（不应再出现“一短一超长”）。

3. **MDD 兜底自检**
   临时把 `vocal_pause_detector` 阈值**刻意调严**（模拟极难歌曲），确认 **beat\_fallback** 会在 4/8 拍边界生成合理候选，且仍然**对齐到局部谷值**而非拍点峰。

---

# 四、为什么这些改动对快歌有效（给你三条以上的理由）

1. **把“静默守卫”从“秒级”降到“拍分数级”**：快歌里 1 秒静默几乎不可用；“0.25 拍”的守卫才是现实。
2. **阈值从“全局固定”改成“局部统计（z-score dips）”**：快歌的 MDD 曲线常态“高密”，只有看**相对下沉**才找得到谷。
3. **以伴奏为 MDD 主输入**：密度/编配的变化发生在鼓贝司和铺底，vocals-only 天生“看不见”这些动态。
4. **拍点网格兜底**：当音频学信号不给候选时，**节拍结构**是唯一可信的“备胎”；用“拍边界附近的能量谷”而不是“正踩拍点峰”，能保证听感自然。
5. **合并窗口/BPM 自适应**：快歌的“微停”密集且短，小窗口+短最小时长才能“保留颗粒度”，不再一刀并万停。

---

# 五、最后两句狠话（但有用）

* 现在的默认配置里，**谷值对齐是关的**、**拍点守护也是关的**，还拿“≥0.6–1.0s 未来静默”当金律，这对快歌就是**自缚手脚**；别怪模型，先把阈值松开。
* **MDD 当“伴奏密度计”用**，不是“人声停顿雷达”；把它从“裁决主轴”挪到“节奏/短时策略增强”，你会立刻看到候选数和切点质量的提升。
# todo-refine.md — GPU + Silero 主路径性能优化计划

> **适用范围**：本项目默认运行在 **NVIDIA GPU**（CUDA）环境，**Silero-VAD 为主路径必需组件**（用于语音/静音裁剪候选窗口，降低后续检测负载）。CPU 兼容与“轻量替代 VAD”仅作为**诊断/应急兜底**的可选插件，不纳入默认路径或性能基线。

---

## 一、执行顺序清单（里程碑与优先级）

### Milestone 0：**GPU 基线与护栏（P0）**

* [x] 基准脚本升级为 GPU 版：`scripts/bench/run_bench.py`

  * 记录指标：**端到端耗时**、**吞吐（sec_audio/sec）**、**GPU 利用率**、**显存峰值**、**H2D/DtoH 传输时间**、切点数分布、>15s 长段比例、<2s 碎片比例、守卫右推平均时长、**可逆性=0**。
  * 打印环境：`cudaDeviceProp.name / driver / runtime / torch, torchaudio 版本`。
  * 示例：

    ```bash
    python scripts/bench/run_bench.py \
      --input-dir input --mode v2.2_mdd \
      --device cuda --gpu-metrics \
      --json output/bench.json \
      --save-guardrails scripts/bench/guardrails/v2_2_gpu_baseline.json
    ```
* [x] **护栏（以 GPU 基线为准）**

  * **速度**：端到端平均耗时 **≥30% 提升**（相对当前 GPU 基线）。
  * **质量**：长段/碎片比例 **±5% 以内**；**可逆性=0**。
  * **资源**：显存峰值 **不高于基线 +10%**；H2D/DtoH 时间 **下降 ≥15%**。

---

### Milestone 1：**计算复用与候选限流（P0）**

* [ ] **一次 STFT（GPU），多处复用**

  * 新增 `src/audio_cut/analysis/features_cache.py`（GPU 版）：使用 `torch.stft/torchaudio` 计算并缓存 |S|、RMS、谱平坦度、onset、MDD 序列与全局 BPM，统一 **hop_s** 时间栅格。
  * 所有频域/节拍/复杂度相关计算 **仅依赖该缓存**，禁止重复 STFT。
* [ ] **Silero → 候选窗口裁剪**

  * Silero 在**人声轨**上推理（`float16`、`inference_mode`），将相邻短间隙（<120ms）合并，得到语音段 `V`。
  * **仅在 `V` 的左右各 ±200ms 边界窗口**内运行 VPP/MDD 的“贵计算”（阈值评估、候选停顿识别），窗口外跳过。
* [ ] **先 NMS，后守卫**（统一到 `refine.finalize_cut_points`）

  * 先执行 `min_gap` + `topK/10s` 的 NMS 限流，再进行守卫/过零。
  * 守卫采用 **O(1) 跳转**：预计算 100 Hz 栅格 `quiet_mask` 与 `next_quiet_right`，候选点直接索引到最近安静帧；过零吸附使用向量化而非逐样本 while 循环。

---

### Milestone 2：**GPU 流水线并行（P0）**

* [ ] **单进程 + 多 CUDA Stream** 的分块流水线（建议块长 10s，overlap 25%）

  * **S1 分离**（MDX/Demucs，FP16，`channels_last`）
  * **S2 VAD**（Silero，对上一块人声）
  * **S3 特征/检测**（features_cache 派生 + VPP/MDD + 精炼）
  * 使用 `cudaEvent.record()/wait_event()` 串联依赖，使相邻块 **分离 / VAD / 检测** 重叠执行。
* [ ] **I/O 与拷贝重叠**

  * 读盘使用 pinned memory；`to(device, non_blocking=True)`；统计 H2D/DtoH 时间以验证重叠效果。
* [ ] **多 GPU**

  * 采用“每卡一进程”模型；通过 `CUDA_VISIBLE_DEVICES` 分配卡位；队列侧的 I/O 与预处理在 CPU 上并行。

---

### Milestone 3：**模型加速与稳定（P1）**

* [ ] **Silero 加速**

  * `torch.inference_mode()` + `float16`；输入长度分桶 + pad 以提升 Tensor Core 利用率；
  * `torch.backends.cudnn.benchmark=True`；形状稳定后尝试 **CUDA Graphs**（保留回退）。
* [ ] **分离模型加速**

  * OLA（overlap-add）分块；热身 1 次；PyTorch 2.x 试 `torch.compile`（遇图捕获失败须自动回退）。
* [ ] **精度护栏**

  * 明确“切点漂移阈值”（守卫右推平均≤150ms，95 分位≤220ms），防止加速牺牲听感。

---

### Milestone 4：**参数集约化与配置整洁（P1）**

* [ ] 新配置 `config/schema_v3.yaml`（保留 6–8 个核心可调）：

  * `min_pause_s`、`min_gap_s`、`guard.max_shift_ms`、`guard.floor_db`、`threshold.base_ratio`、`adapt.{bpm,mdd}_strength`、`nms.topk`（可选）。
* [ ] `config/derive.py`：统一派生逻辑

  * `threshold.effective = base_ratio * f(bpm, mdd)`；`min_pause_effective = g(bpm)`。
* [ ] 预设 Profile：`ballad / pop / edm / rap` 仅覆盖 3–4 个项。
* [ ] 迁移层 `config/migrate_v2_to_v3.py`：打印 `DeprecationWarning`，1 个版本后移除。

---

### Milestone 5：**清理遗留与兼容（P1）**

* [ ] **去除“替代 VAD”的默认表述**；`detectors/energy_gate.py` 标注为**诊断/无 GPU 兜底（默认关闭）**。
* [ ] CLI 菜单默认仅展示 **v2.2 主路径**；旧模式移入“兼容模式”二级菜单。
* [ ] 清理冗余代码与配置键：

  * 老的“二次插点/强制拆分”等 dead code；
  * 未引用/作用重叠的旧键（用 ripgrep 交叉 YAML/源码）；
  * 保留 `--compat-config v2` 回退开关一个版本周期。

---

## 二、详细 TODO（方法与接口要点）

### A. `features_cache.py`（GPU）

* 接口草案：

  ```python
  @dataclass
  class TrackFeatureCache:
      sr: int
      hop_s: float
      bpm: float; bpm_conf: float
      global_mdd: float
      mdd_series: torch.Tensor     # (T,)
      rms_series: torch.Tensor     # (T,)
      spec_flatness: torch.Tensor  # (T,)
      onset_strength: torch.Tensor # (T,)
      def idx(self, t: float) -> int: ...
      def get_window(self, t: float, w_s: float) -> dict: ...
  ```

* 约束：**只做一次 GPU STFT**；所有下游模块从缓存派生；禁止二次 STFT/重复节拍分析。

### B. `cutting/refine.py`（统一精炼）

* 规则：**先 NMS 再守卫**；守卫用 `quiet_mask + next_quiet_right` O(1) 跳转；过零吸附向量化；最终再做一次 `min_gap` 校验确保边界调整后不重叠。

### C. **Silero → 候选窗口**

* 语音段合并策略：小间隙 <120ms 合并；窗口 = 段边界 ±200ms；
* 仅在窗口内跑 VPP/MDD/候选识别与精炼；将候选上限设为 `topK_per_10s`（3–5）。

### D. **流水线并行**

* 分块 10s，overlap 25%；三流并发（分离 / VAD / 特征+检测）；
* I/O 与 H2D 使用 pinned memory + `non_blocking=True`；
* 以 `cudaEvent` 明确依赖，避免隐式同步。

---

## 三、工程与性能微优化（P2）

* [ ] **内存**：优先 `float16`，必要处 `float32`；复用 Tensor，避免频繁分配。
* [ ] **I/O**：`soundfile` 流式 + 单声道 downmix；切片写盘批量化。
* [ ] **日志**：GPU 计时 `cudaEvent`；NVML（`pynvml`）采集利用率/显存；默认关闭大体量调试导出。
* [ ] **剖析**：`nsys`/`nvprof` 端到端；`py-spy`/`line_profiler` 仅用于 CPU 辅助。

---

## 四、示例：关键文件与调用关系

```
src/
  audio_cut/
    analysis/
      features_cache.py      # GPU 一次 STFT，统一特征缓存
    cutting/
      refine.py              # 切点精炼/守卫/NMS 统一入口（先NMS后守卫）
    detectors/
      pure_vocal_v22.py      # 只读 features_cache + 调用 refine
      vad_silero.py          # 主路径 VAD（FP16 + 批处理）
      energy_gate.py         # 可选：诊断/无GPU兜底（默认关闭）
    config/
      schema_v3.yaml         # 精简配置
      derive.py              # 参数派生逻辑
      migrate_v2_to_v3.py    # 配置迁移/兼容（1版本后移除）
    cli/
      quick_start.py         # 默认仅 v2.2；旧模式移入“兼容模式”
scripts/
  bench/
    run_bench.py             # GPU 基线与护栏
tests/
  test_cutting_refiner.py
  test_features_cache.py
  test_config_migration.py
```

---

## 五、验收清单（必须全部通过）

* [ ] **速度（GPU 基线）**：端到端平均耗时 **≥30% 提升**；吞吐提升与 H2D/DtoH 时间下降达标。
* [ ] **资源**：显存峰值 ≤ 基线 +10%；GPU 利用率提升（以均值/95 分位计）。
* [ ] **质量护栏**：长段/碎片比例在 **±5%**；**可逆性=0**；守卫右推平均 ≤150ms（P95 ≤220ms）。
* [ ] **代码健康度**：

  * 精炼逻辑 **仅在 `refine.py`**；
  * BPM/MDD **只算一次**（`features_cache`）；
  * 配置键显著减少，迁移/文档齐全；
  * CI 启用 GPU 基准与回归守护（不达标即阻断合并）。

---

## 六、快刀动作（按序执行）

1. **接入 Silero→候选窗口裁剪**（±200ms）并加 `topK_per_10s`。
2. **落地 GPU 版 `features_cache`**（一次 STFT，所有频域与节拍复用）。
3. **收敛精炼流程到 `refine.finalize_cut_points`**（先 NMS 后守卫，守卫 O(1) 跳转，过零向量化）。
4. **启用多 CUDA Stream 流水线**；bench 记录 GPU 指标作为新基线。
5. **参数集约化 + 迁移层**；移除“替代 VAD/CPU 基线”等陈述并更新文档/CLI。

---

> 备注：如需在极端噪声或超长素材下进一步提速，可引入 **按长度分桶的批推理** 与 **CUDA Graphs**（Silero/分离模型），但务必保留失败回退到 eager 的安全阀。上述条目均已纳入里程碑 3 的“模型加速与稳定”。

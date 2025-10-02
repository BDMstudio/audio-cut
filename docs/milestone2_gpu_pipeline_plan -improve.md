<!-- File: docs/milestone2_gpu_pipeline_plan -improve.md -->
### Milestone 2 — GPU 分块流水线（改进版 · 可执行）

**前提**

* 仅面向 **NVIDIA GPU**；**Silero-VAD 为主路径必需**；CPU 仅作兜底，不纳入性能基线。

#### 0. 启动前置自检（P0 必过）

1. **ORT CUDA 依赖注入 + Provider 固定**

```python
# 在任何 InferenceSession 创建之前执行
import os, sys, pathlib, onnxruntime as ort, logging
if sys.platform == "win32":
    capi = pathlib.Path(ort.__file__).parent / "capi"
    deps = capi / "deps"
    os.add_dll_directory(str(capi))
    if deps.exists(): os.add_dll_directory(str(deps))
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC  # 先保守
providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search":"HEURISTIC"}),
             "CPUExecutionProvider"]  # 禁用 TRT EP
sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
logging.info("ORT EPs=%s", sess.get_providers())
```

* 若日志里 **没有 `CUDAExecutionProvider`**，直接退出并提示 `cuDNN 9 / PATH` 设置（MDX23_SETUP）。

2. **MDX23 输入契约探针**

```python
inp = sess.get_inputs()[0]
# 记录 "[B, 4, F, T]" 的真实维度（动态维以 1 代入），存入 PipelineContext
ctx.mdx23_input = {"name": inp.name, "shape": [d if isinstance(d,int) else 1 for d in inp.shape]}
```

* **禁止猜 shape**；**由会话返回 shape 驱动预处理**。

3. **对齐步长**

* 设 `align_hop=4096`（可配置）；**块内长度 padding 到 hop 的整数倍**。

---

#### 1. 架构与职责（与 TODO 对齐）

| 组件                | 职责                                                  | 关键接口/约束                                                               |
| ----------------- | --------------------------------------------------- | --------------------------------------------------------------------- |
| `gpu_pipeline`    | 设备选择、Streams/Events、分块与背压、Pinned 环形缓冲               | `create_streams()`, `chunk_schedule()`, `record_event()/wait_event()` |
| 分离器（MDX23/Demucs） | **分块→补丁**（按探针 shape）、FP16、`channels_last`、OLA 写回有效区 | `separator.process_chunk(plan, ctx)`                                  |
| Silero VAD        | **块内推理 + 跨块合并**，输出全局时基语音段                           | `silero.infer_chunk(vocal, plan, fp16=True)`                          |
| 特征缓存（GPU）         | **一次 STFT，多处复用**；仅写有效区帧；跨块拼接                        | `features_cache.update_chunk_gpu(vocal, plan)`                        |
| 精炼（refine）        | **先 NMS 后守卫**，守卫 O(1) 跳转，最终 min-gap 校验              | `refine.finalize_cut_points(ctx, cands)`                              |

> 以上职责/接口与旧版计划一致，我仅把“输入契约/对齐”和“断路器”写成硬约束。

---

#### 2. 分块契约（Chunk/Overlap/Halo）

* **chunk/overlap/halo** 使用 **`10s / 2.5s / 0.5s`** 默认值；
* 有效区 = 块去掉 overlap 后再扣 halo；**仅在有效区写回**分离/特征/VAD 结果；分离 OLA 用对称窗归一。

---

#### 3. 运行时序（单进程三流）

```python
plans = chunk_schedule(total_s, 10.0, 2.5, 0.5)
S = create_streams()  # s_sep, s_vad, s_feat

for i, plan in enumerate(plans):
    with torch.cuda.stream(S.s_sep):
        sep_out = separator.process_chunk(plan, ctx)     # GPU tensors, 仅有效区
        E_sep = record_event(S.s_sep)

    with torch.cuda.stream(S.s_vad):
        wait_event(S.s_vad, E_sep)
        vad_out = silero.infer_chunk(sep_out.vocal, plan, fp16=True)
        E_vad = record_event(S.s_vad)

    with torch.cuda.stream(S.s_feat):
        wait_event(S.s_feat, E_vad)
        cache.update_chunk_gpu(sep_out.vocal, plan)      # 一次 STFT
        cand = detect_candidates(cache, vad_out, plan)   # 仅在 VAD 边界 ±200ms
        cuts_i = refine.finalize_cut_points(ctx, cand)   # 先NMS后守卫
torch.cuda.synchronize()
```

* **H2D/DtoH** 使用 pinned 环形缓冲，设置 `inflight_chunks_limit=2` 控制显存。

---

#### 4. 关键实现细节（可直接落地）

**4.1 MDX23：波形→补丁**

* 读取 `ctx.mdx23_input.shape` 得到目标补丁维度 `[B, 4, F, T]`；
* 将 `[1,2,T_wav]` **分块对齐**后，经统一 STFT/特征序列**拼装为 4 通道补丁**（你的模型导出如何映射 4 通道取决于导出脚本——这里按“探针 shape 驱动”避免猜测）；
* **先用 `sess.run(None, {...})` 跑通**，再分阶段恢复 IO Binding（先绑输入→采样输出形状→再绑输出）。

**4.2 Silero：块内 + 跨块合并**

* 块内带 `±halo` 上下文；**短隙 <120 ms 合并**；边界重叠取并集；输出统一为**全局时间**；仅在有效区落地。

**4.3 特征缓存（GPU）**

* **一次 STFT** 统一窗/步长；只写**有效区帧**；
* 提供 `cache.get_window(t, w_s)` 给候选窗口裁剪；与整段 STFT 在同帧索引上 **MAE < 1e-4**。

**4.4 精炼**

* 排序权重建议：`score * w(kind) * h(duration)`（真停顿 > breath）；
* **一次 NMS(min_gap)** 后做守卫/过零；最终再一次 min-gap 校验以防守卫右推造成重叠。

---

#### 5. 异常与回退（断路器）

* **任何 ORT CUDA 异常**（`CUDNN_STATUS_*`/`InvalidPtx`/`illegal memory access`）：

  1. 记录 `gpu_meta.reason` + providers；
  2. **停止本进程的任何后续 CUDA 调用**；
  3. **新开子进程**回退到 CPU（或 Demucs CUDA 但新进程）。
* 提供 `--strict-gpu` 开关：严格模式直接失败并输出诊断。

---

#### 6. 配置示例（新增键）

```yaml
gpu_pipeline:
  enable: true
  chunk_seconds: 10.0
  overlap_seconds: 2.5
  halo_seconds: 0.5
  align_hop: 4096
  use_cuda_streams: true
  prefetch_pinned_buffers: 2
  inflight_chunks_limit: 2
  ort:
    graph_optimization_level: basic      # disable/basic/extended
    cudnn_conv_algo_search: HEURISTIC    # or DEFAULT
    disable_trt: true
```

* 以上键位于统一配置，`SeamlessSplitter`/分离器/Silero/特征共享。

---

#### 7. 基准与成功标准（守门人）

**速度/资源（GPU 基线）**

* 端到端平均耗时 **≥ 30% 提升**；H2D/DtoH 时间 **下降 ≥15%**；显存峰值 **≤ 基线 +10%**。
  **质量/确定性**
* 与整段一次跑相比：切点时间 **均值 ≤10 ms、P95 ≤30 ms**；切点数量差 **≤1%**；守卫右推差 **均值 ≤15 ms**；**可逆性=0**。
  **跑法**：`scripts/bench/run_bench.py --device cuda --gpu-metrics` 输出报告入库。

---

#### 8. 测试矩阵与 CI

* **单元**：STFT 等价（MAE）、VAD 跨块合并（短隙合并/边界并集/全局时基）、refine 最小间隔。
* **契约**：`tests/benchmarks/test_chunk_vs_full_equivalence.py` 产出 JSON/Markdown；CI 未达标阻断合并。

---

#### 9. 风险与回滚

* **形状/图捕获不稳**：保持 eager 回退；
* **隐式同步**：只允许 `cudaEvent`；
* **I/O 抖动**：开启 pinned 环形缓冲 + 背压；
* **回滚**：`gpu_pipeline.enable=false` + 子进程 CPU 流程；保留 `--compat-config v2` 一版周期。

---

#### 10. DoD（完成定义）

* [ ] 三流流水线稳定运行，基线指标达成；
* [ ] STFT/VAD/精炼跨块一致性满足阈值；
* [ ] 断路器与回退在异常注入测试下行为正确；
* [ ] 文档与基准结果入库：更新 `docs\milestone2_gpu_pipeline_todo.md`、``todo-refine.md` 与 `bench` 报表。

---
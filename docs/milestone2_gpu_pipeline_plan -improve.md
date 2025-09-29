<!-- File: docs/milestone2_gpu_pipeline_plan.md -->
<!-- Purpose: Milestone 2 — GPU 分块流水线设计与实施计划（单进程多流 + Silero 保留 + 一次 STFT 复用） -->

# Milestone 2 — GPU 分块流水线设计与实施计划

> **前提**：本项目面向 **NVIDIA GPU** 部署，**Silero-VAD 是主路径必需组件**；CPU 与“轻量门控”仅作诊断/兜底用途，不纳入性能基线与默认路径。本文档用于落地 `todo-refine.md` 的 **Milestone 2：GPU 流水线并行**。:contentReference[oaicite:1]{index=1}

---

## 1. 背景与目标

- **现状**：分离 / VAD / 特征构建仍以“整段同步执行”为主，GPU 空转时间长；局部“焦点窗口”等昂贵计算未形成统一的剪枝与并发模型。:contentReference[oaicite:2]{index=2}  
- **目标**：实现**单进程 + 多 CUDA Stream** 的分块流水线：  
  - **S1 分离**（MDX/Demucs）→ **S2 VAD**（Silero）→ **S3 特征/检测/精炼**（一次 STFT 复用；只在候选窗口内计算）；  
  - 保证**跨块结果一致性**与**最终切割可逆性=0**；  
  - 与 `todo-refine.md` 的基线/护栏一致，作为性能回归守门人。:contentReference[oaicite:3]{index=3}

---

## 2. 成功标准（与基线护栏对齐）

- **速度（GPU 基线）**：端到端平均耗时 **≥ 30% 提升**；吞吐（sec_audio/sec）显著提升。  
- **资源**：显存峰值 **≤ 基线 +10%**；H2D/DtoH 时间 **下降 ≥ 15%**；GPU 利用率均值/95分位上升。  
- **质量**：长段/碎片比例 **±5%** 内；**可逆性=0**；守卫右推**均值 ≤ 150 ms，P95 ≤ 220 ms**。  
- **确定性（契约测试）**：与“整段一次跑”基线相比：  
  - 切点时间误差：**均值 ≤ 10 ms，P95 ≤ 30 ms**；  
  - 切点数量差：**≤ 1%**；守卫右推差：**均值 ≤ 15 ms**。:contentReference[oaicite:4]{index=4}

---

## 3. 分层架构与模块职责

| 组件 | 职责 | 关键接口/实现要点 |
|---|---|---|
| `gpu_pipeline` 工具层 | 设备选择、Pinned Memory、CUDA Stream/Event、分块与背压、环形缓冲 | `select_device()`, `create_streams()`, `record_event()/wait_event()`, `chunk_schedule()` |
| 分离器（MDX/Demucs） | 分块分离（可带 halo），输出 vocal/instrumental；中间有效区写回 | `separator.process_chunk(plan)`（FP16, channels_last, OLA） |
| Silero VAD | **按块推理**并**跨块合并**时间戳，为焦点窗口裁剪提供输入 | `silero_vad.infer_chunk(vocal, plan, fp16=True)`, `_merge_segments()` |
| 特征缓存（GPU） | **一次 STFT** 复用：RMS/谱平坦度/Onset/MDD 序列 + 全局 BPM；跨块拼接 | `features_cache.update_chunk_gpu(vocal, plan)` |
| 精炼（refine） | **先 NMS 后守卫**；守卫 O(1) 跳转 + 过零向量化；终态 min-gap 校验 | `refine.finalize_cut_points(ctx, cands)` |

> 注：以上模块职责与接口名称与现有 `todo-refine.md`/草稿文档一致，仅细化执行细节与契约。

---

## 4. 分块与一致性契约

### 4.1 Chunk / Overlap / Halo

- **Chunk**：默认 `chunk_s = 10.0`。  
- **Overlap**（分离 OLA 用）：`overlap_s = 2.5`，用于跨块**能量无缝拼接**（权重窗归一化）。  
- **Halo**（VAD/特征上下文）：`halo_s = 0.5`，仅用于**推理上下文**，**不写回**有效区结果。  
- **有效区**：`[start + halo_left, end - halo_right]`；只在有效区落地分离/特征/时间戳与切点。:contentReference[oaicite:6]{index=6}

### 4.2 分离 OLA 契约

- 分离输出以 **overlap-add** 拼接：汉宁窗或同类对称窗，**双边归一化权重**保证波形连续，无能量跳变。  
- 若模型内部需要更大上下文，可把 `halo_s` 视为最小保障（建议 **0.5–1.0 s**），有效区内不使用 halo 边缘预测。:contentReference[oaicite:7]{index=7}

### 4.3 Silero VAD 跨块合并

- **上下文帧**：每块推理时包含 `±halo_s` 的上下文，避免边界断句。  
- **短隙合并**：同块内与跨块边界**间隙 < 120 ms** 合并为同一语音段。  
- **重叠规则**：块末尾段与下一块开头段如重叠/相接，取并集。  
- **输出时间轴**：统一用**全局时基**（`t += chunk_start`），并仅输出**有效区**结果。:contentReference[oaicite:8]{index=8}

### 4.4 STFT/特征拼接

- **统一窗口/步长**（`win`, `hop`）在全曲固定；  
- 每块做 halo STF T，但仅把**有效区帧**写入全局缓存；  
- 与整段一次 STFT 在相同帧索引上比较，数值误差 **MAE < 1e-4（float32 基线）**。:contentReference[oaicite:9]{index=9}

---

## 5. 资源预算与默认配置

### 5.1 预算公式（经验）

- **显存占用** ≈ `sep_mem(chunk_s) + vad_mem(chunk_s) + stft_mem(chunk_s, fft, hop) + overlap_buffers + activations`  
- **吞吐/延迟权衡**：`chunk_s ↑ ⇒ 吞吐 ↑, 首块延迟 ↑`；推荐从 `10s/2.5s/0.5s` 起步。:contentReference[oaicite:10]{index=10}

### 5.2 参考配置表（实测为准）

| GPU | chunk / overlap / halo | 期望吞吐（sec_audio/sec） | 显存峰值(估) | 备注 |
|---|---|---:|---:|---|
| 4090 | 10s / 2.5s / 0.5s | 1.6–2.2 | 3.5–4.5 GB | FP16，三流并发 |
| A10  | 8s / 2s / 0.5s | 1.2–1.6 | 3.0–4.0 GB | 减少 in-flight 块 |
| T4   | 6s / 1.5s / 0.5s | 0.8–1.1 | 2.5–3.5 GB | 关闭部分重叠特性 |

> 真实指标以 `scripts/bench/run_bench.py --device cuda --gpu-metrics` 输出为准，并受 `todo-refine.md` 护栏管控。:contentReference[oaicite:11]{index=11}

---

## 6. 实施步骤（按优先级落地）

### 6.1 工具与配置（P0）

- 新增 `audio_cut.utils.gpu_pipeline`：设备选择、Stream/Event、Pinned Memory、分块/背压。  
- 配置键：  
  ```yaml
  gpu_pipeline:
    enable: true
    chunk_seconds: 10.0
    overlap_seconds: 2.5
    halo_seconds: 0.5
    use_cuda_streams: true
    prefetch_pinned_buffers: 2     # 环形预取缓冲数
    inflight_chunks_limit: 2       # 并行在途块上限（背压）
    multi_gpu:
      enabled: false
      preferred_devices: [0]
````

* 为配置与默认值添加单测。

### 6.2 特征缓存 GPU 化（P0）

* `features_cache.update_chunk_gpu(vocal, plan)`：H2D→GPU STF T→派生 RMS/平坦度/Onset/MDD→**有效区帧拼接**。
* **一次 STFT，多处复用**；严禁重复 STFT/重复节拍分析。

### 6.3 Silero VAD chunk 化（P0）

* `silero_vad.infer_chunk(vocal, plan, fp16=True)`：按块推理 + **跨块合并协议**（见 4.3）。
* 输出用于**候选窗口裁剪**：只在语音段**边界 ±200 ms**窗口内运行 VPP/MDD 检测。

### 6.4 分离器流水线（P0）

* `separator.process_chunk(plan)`：FP16、`channels_last`、OLA；仅写回**有效区**。
* 若暂时只有 CLI：先整段跑后切分缓冲提供迭代器，作为过渡实现。

### 6.5 集成与验证（P0）

* 小样本集运行 GPU 管线，对比分离/时间戳/特征与整段一次跑的一致性；
* 明确异常回退路径（OOM/Runtime Error → CPU 路径），并打点 `gpu_pipeline_used`。

---

## 7. 工具层 API 与伪代码

### 7.1 Streams / Events 与分块计划

```python
# audio_cut/utils/gpu_pipeline.py
@dataclass
class Streams:
    s_sep: torch.cuda.Stream
    s_vad: torch.cuda.Stream
    s_feat: torch.cuda.Stream

def create_streams() -> Streams: ...
def record_event(stream: torch.cuda.Stream) -> torch.cuda.Event: ...
def wait_event(stream: torch.cuda.Stream, event: torch.cuda.Event): ...

@dataclass
class ChunkPlan:
    start_s: float
    end_s: float
    halo_left_s: float
    halo_right_s: float

def chunk_schedule(total_s: float, chunk_s=10.0, overlap_s=2.5, halo_s=0.5) -> list[ChunkPlan]: ...
```

* **并发模型**：固定三条流 `S1/S2/S3`；对第 `i` 块记录 `E_sep[i]→E_vad[i]→E_feat[i]`，`S2` 等 `E_sep[i]`，`S3` 等 `E_vad[i]`；块 `i+1` 可与 `i` 重叠执行。

### 7.2 环形缓冲与背压

* **Pinned Memory 预取**：保持 `prefetch_pinned_buffers=2`；
* **在途上限**：`inflight_chunks_limit=2`，超限时 I/O 线程阻塞，防止显存膨胀。

---

## 8. 主循环伪代码（单 GPU、单进程、三流）

```python
streams = create_streams()
plans = chunk_schedule(total_s=duration_s, chunk_s=10.0, overlap_s=2.5, halo_s=0.5)

for i, plan in enumerate(plans):
    # S1: 分离
    with torch.cuda.stream(streams.s_sep):
        sep_out = separator.process_chunk(plan)            # gpu tensors (vocal, inst), OLA+有效区
        E_sep = record_event(streams.s_sep)

    # S2: VAD（等待分离）
    with torch.cuda.stream(streams.s_vad):
        wait_event(streams.s_vad, E_sep)
        vad_out = silero_vad.infer_chunk(sep_out.vocal, plan, fp16=True)
        E_vad = record_event(streams.s_vad)

    # S3: 特征/检测（等待 VAD）
    with torch.cuda.stream(streams.s_feat):
        wait_event(streams.s_feat, E_vad)
        cache.update_chunk_gpu(sep_out.vocal, plan)        # STFT halo + 有效区拼接
        cuts_i = detect_and_refine(cache, vad_out, plan)   # 仅在 VAD 边界 ±200ms 窗口内计算
        E_feat = record_event(streams.s_feat)

torch.cuda.synchronize()  # ensure all done
```

> **检测/精炼**：候选先 **NMS(min_gap + topK/10s)** 限流，再做守卫 O(1) 跳转（`quiet_mask + next_quiet_right`）与过零向量化；最终再做一次最小间隔校验。

---

## 9. 多 GPU 调度

* **模式**：**每卡一进程**，用 `CUDA_VISIBLE_DEVICES` 固定；任务按**文件维度**分片（避免跨进程块级合并）。
* **均衡**：队列 + work-stealing；以显存水位/在途块数做背压。
* **度量**：采集 `gpu_id / util% / mem_peak / H2D / DtoH / S1/S2/S3 耗时`；汇总进 bench 报表。

---

## 10. 测试与验收

### 10.1 单元/合约测试

* **特征拼接**：GPU/CPU STFT 在相同帧索引上 **MAE < 1e-4**；跨块帧索引单调、无 off-by-one。
* **VAD 合并**：短隙合并、跨块并集、有效区裁剪正确；时间戳为全局坐标。
* **精炼一致性**：NMS 限流 + 守卫 O(1) 跳转 + 过零吸附后的相邻间隔 **≥ min_gap_s**。

### 10.2 集成/性能基准

* `scripts/bench/run_bench.py --device cuda --gpu-metrics`：记录吞吐/显存峰值/H2D/DtoH/利用率与切点统计；与 `todo-refine.md` 护栏比对。

---

## 11. 风险与回退

* **OOM / RuntimeError**：立即切换至 CPU 路径，记录 `gpu_pipeline_used=false`；
* **形状/图捕获不稳定**：保持 Eager 回退；**Silero/分离**加 FP16/Graph 时必须保留回退；
* **隐式同步**：仅以 `cudaEvent` 显式同步，避免跨流隐式 barrier；
* **I/O 抖动**：Pinned 环形缓冲 + in-flight 背压。

---

## 12. 完成定义（DoD）清单

* [ ] 单 GPU：三流流水线稳定运行，指标满足**成功标准**；
* [ ] 多 GPU：每卡一进程调度正确，报表呈现分卡指标；
* [ ] 合约测试：chunk vs. 全量一次跑**在容忍度内等价**；
* [ ] 文档：更新 `todo-refine.md` 的 Milestone 2 状态与变更说明；提交 bench 报表与配置快照。

```
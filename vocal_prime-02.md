### **问题根源分析：为什么还会有个别误切？**

#### **根源一：“偏移量”策略的优先级高于“能量谷”搜索**

当前的核心问题是，系统在决定切点时，**优先执行了基于固定时间的偏移策略**，而不是我们期望的能量谷底搜索。

在 `src/vocal_smart_splitter/core/vocal_pause_detector.py` 的 `_calculate_cut_points` 函数中，代码逻辑如下：

1.  首先判断停顿类型（`head`, `middle`, `tail`）。
2.  如果是`head`或`tail`，**直接应用固定的`head_offset`和`tail_offset`** (-0.5s / +0.5s) 来计算切点。
3.  只有当停顿类型是`middle`时，才会在停顿区域的中心点进行切割。

这个逻辑导致：

  * 对于音频的**第一个和最后一个**停顿区域，系统根本不会去寻找能量最低点，而是简单地在人声开始前0.5秒或结束后0.5秒的地方“盲切”一刀。
  * 如果人声的拖音（reverb）超过了0.5秒，或者前奏/尾奏的乐器能量较高，那么这个固定的0.5秒偏移量就必然会切在声音上，这完美解释了你频谱图上看到的边界误切问题。

#### **根源二：“未来静默守卫”的缺失**

即便是对于中间的停顿，系统虽然会在停顿区域内寻找能量谷底，但它缺少一个关键的“前瞻性”验证：**在选定的这个切点之后，音频是否能继续保持一小段安静？**

一个理想的切点，不仅自身能量要低，其后的一小段时间（比如100-200毫秒）也应该是安静的，以确保不会切断人声的尾音或一个吸气声的开头。你的代码里虽然有`lookahead_guard_ms`这个参数，但它在最终决策时的权重和应用逻辑还不够强力，导致一些“伪谷底”（比如两个靠得很近的音节之间的微小缝隙）被错误地选为切点。

### **最终解决方案：统一所有策略，以“能量谷”为最高优先级**

为了根除这个问题，我们需要重构切点计算逻辑，确保**所有类型**的停顿（无论是头部、中间还是尾部）都必须经过最严格的“能量谷搜索”和“安全校验”流程。

#### **步骤一：重构 `_calculate_cut_points`，统一逻辑**

我们要修改 `src/vocal_smart_splitter/core/vocal_pause_detector.py` 中的 `_calculate_cut_points` 函数，废除`head`/`tail`的特殊偏移逻辑，让所有停顿都进入能量谷搜索流程。

**修改 `src/vocal_smart_splitter/core/vocal_pause_detector.py`**：

```python
    def _calculate_cut_points(self, vocal_pauses: List[VocalPause], bpm_features: Optional['BPMFeatures'] = None, waveform: Optional[np.ndarray] = None) -> List[VocalPause]:
        """
        计算精确的切割点位置 - (v2.1 最终修复版)
        统一所有停顿类型，强制执行能量谷搜索，并应用偏移量作为谷搜索的边界。
        """
        # ... (读取配置参数部分保持不变) ...

        logger.info(f"计算 {len(vocal_pauses)} 个停顿的切割点 (能量谷优先模式)...")

        for i, pause in enumerate(vocal_pauses):
            # ✅ --- 核心修复：统一所有停顿类型的处理逻辑 ---

            # 1. 首先确定能量谷搜索的安全范围 (Search Range)
            search_start = pause.start_time
            search_end = pause.end_time

            # 2. 应用偏移量来调整搜索范围，而不是直接计算切点
            if pause.position_type == 'head':
                # 对于头部停顿，能量谷应该在人声开始前，所以搜索范围向右移动
                search_start = max(search_start, pause.end_time + self.head_offset - 0.5) # 在偏移点附近1秒内搜索
                search_end = min(search_end, pause.end_time + self.head_offset + 0.5)
            elif pause.position_type == 'tail':
                # 对于尾部停顿，能量谷应该在人声结束后，所以搜索范围向左移动
                search_start = max(search_start, pause.start_time + self.tail_offset - 0.5)
                search_end = min(search_end, pause.start_time + self.tail_offset + 0.5)
            
            # 确保搜索范围有效
            if search_end <= search_start:
                search_start, search_end = pause.start_time, pause.end_time

            logger.debug(f"停顿 {i+1} ({pause.position_type}): 原始范围 [{pause.start_time:.2f}s, {pause.end_time:.2f}s], "
                         f"能量谷搜索范围 [{search_start:.2f}s, {search_end:.2f}s]")

            # 3. 在确定的安全范围内，强制执行能量谷检测
            selected_idx: Optional[int] = None
            if waveform is not None and len(waveform) > 0:
                l_idx = max(0, int(search_start * self.sample_rate))
                r_idx = min(len(waveform), int(search_end * self.sample_rate))

                if r_idx > l_idx:
                    valley_idx = self._select_valley_cut_point(
                        waveform, l_idx, r_idx, self.sample_rate,
                        local_rms_ms, guard_ms, floor_pct
                    )

                    if valley_idx is not None:
                        selected_idx = valley_idx
                        logger.debug(f"  -> 能量谷切点找到 @ idx={selected_idx}")
                    else:
                        # 如果在精确范围内找不到谷，则在整个停顿区域的中心点进行兜底
                        selected_idx = int((pause.start_time + pause.end_time) / 2 * self.sample_rate)
                        logger.warning(f"  -> 未在搜索区找到能量谷，回退到中心点")
                else:
                    selected_idx = int((pause.start_time + pause.end_time) / 2 * self.sample_rate)
            else:
                selected_idx = int((pause.start_time + pause.end_time) / 2 * self.sample_rate)

            # 4. 更新切点
            pause.cut_point = selected_idx / self.sample_rate
            logger.info(f"停顿 {i+1} ({pause.position_type}): 最终切点 @ {pause.cut_point:.3f}s")
            
            # ✅ --- 修复结束 ---

        return vocal_pauses
```

#### **修复的逻辑：**

1.  **废除特殊逻辑**：不再对 `head` 和 `tail` 类型的停顿进行特殊的、固定的偏移计算。
2.  **统一处理流程**：所有类型的停顿，都必须经过能量谷搜索。
3.  **妙用偏移量**：`head_offset` 和 `tail_offset` 不再直接用来决定切点，而是用来**定义能量谷搜索的安全范围**。这既保留了偏移量的初衷（在人声附近切割），又通过能量谷搜索保证了切点落在最安静的位置，一举两得。

#### **步骤二：强化 `_select_valley_cut_point` 的“未来静默守卫”**

我们需要确保“未来静默守卫”(`_future_silence_guard`)的逻辑被严格执行，并且它的判断标准足够可靠。

**检查并确认 `_select_valley_cut_point` 函数中的守卫逻辑**：

```python
# ... 在 _select_valley_cut_point 函数内部 ...
        # 3. 未来静默守卫：确保切点后足够安静，避免切在尾音上
        if guard_ms > 0:
            guard_passed_indices = []
            for idx in top_indices:
                if self._future_silence_guard(rms_envelope, idx, guard_ms, self.sample_rate, floor_val):
                    guard_passed_indices.append(idx)
            
            if not guard_passed_indices:
                logger.debug(f"  能量谷检测：所有候选点未通过未来静默守卫")
                return None  # 如果没有点能通过守卫，则认为此区域不适合切割

            # 从通过守卫的点中选择能量最低的
            final_energies = [rms_envelope[i] for i in guard_passed_indices]
            final_best_idx = guard_passed_indices[np.argmin(final_energies)]
            
            logger.debug(f"  能量谷检测：{len(top_indices)}->{len(guard_passed_indices)}个点通过守卫, "
                         f"最终选择 idx={final_best_idx}")
            return final_best_idx
# ...
```

这段代码逻辑是正确的。它会过滤掉所有切点后不够安静的候选点。结合**步骤一**的修改，现在所有切点都会经过这道严格的检查，从而避免切在人声尾音上。
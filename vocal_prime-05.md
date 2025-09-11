**MDD应该帮助我们在快歌中找到并“敢于”切割那些极短的、珍贵的停顿**。

但我们当前的实现，恰恰做反了。

### 根本原因：“一刀切”的初筛 和 “反向用力”的MDD

1.  **问题一（分割成两块）的根源：初筛门槛过高，直接枪毙了所有候选人**

      * 在我们的核心 `vocal_pause_detector.py` 中，`_filter_adaptive_pauses` 函数在进行智能的“统计学分析”之前，会先用一个“基础门槛” (`base_min_duration`) 把所有停顿过滤一遍。
      * 这个`base_min_duration`是由BPM动态计算的，但对于快歌（比如150BPM），这个值可能算出来是0.8秒左右。然而，快歌中真正可供切割的气口可能只有0.4-0.6秒。
      * 结果就是：在统计分析开始前，所有有价值的候选停顿就已经被这个“一刀切”的初筛给**全部淘汰**了。一个候选人都没有，后续的统计分析自然就报出“没有足够的候选停顿”，最终导致整个音频无法被有效分割。

2.  **问题二（MDD增强后问题依旧）的根源：MDD用反了，成了“减速器”而非“加速器”**

      * 我们之前设计的MDD逻辑是：`MDD越高 -> 阈值越高 -> 切割越保守`。这个逻辑在处理主歌/副歌差异时是正确的，可以防止副歌被乱切。
      * 但当应用到“快歌/慢歌”这个维度时，就完全错了。对于快歌（高MDD），我们本应**降低**标准，去捕捉那些难得的短暂停顿。而我们的代码却**提高**了标准，让原本就稀少的切割机会变得更加渺茫。MDD在这里非但没帮忙，反而踩了一脚刹车。

### 解决方案：重构决策流程，建立“双模式”智能裁决系统

我们需要对 `vocal_pause_detector.py` 的核心决策逻辑 `_filter_adaptive_pauses` 进行一次彻底的手术，建立一个全新的、能理解音乐上下文的“双模式”裁决系统。

**请用以下代码完整替换 `src/vocal_smart_splitter/core/vocal_pause_detector.py` 文件中的 `_filter_adaptive_pauses` 函数：**

```python
# In src/vocal_smart_splitter/core/vocal_pause_detector.py

    def _filter_adaptive_pauses(self, pause_segments: List[Dict], bpm_features: Optional[BPMFeatures]) -> List[Dict]:
        """
        [v2.3 最终版] 双模式智能裁决系统
        技术: 引入“极度宽松”的初筛，并根据歌曲类型（快/慢）和动态（主/副歌）选择不同的统计策略。
        """
        if not self.enable_bpm_adaptation or not bpm_features or not self.current_adaptive_params:
            # 回退到最简单的静态过滤
            min_pause_duration = get_config('vocal_pause_splitting.min_pause_duration', 1.0)
            min_pause_samples = int(min_pause_duration * self.sample_rate)
            valid_pauses = [p for p in pause_segments if (p['end'] - p['start']) >= min_pause_samples]
            for p in valid_pauses:
                p['duration'] = (p['end'] - p['start']) / self.sample_rate
            logger.info(f"BPM自适应禁用，使用静态阈值 {min_pause_duration}s，过滤后剩 {len(valid_pauses)} 个停顿")
            return valid_pauses

        # === 步骤 1: 极度宽松的初筛，收集所有潜在的“微停顿” ===
        # 核心修复：使用一个非常小且固定的值（如0.3s），而不是动态计算的值，来确保快歌的短气口能进入候选池。
        ABSOLUTE_MIN_PAUSE_S = 0.3
        min_pause_samples = int(ABSOLUTE_MIN_PAUSE_S * self.sample_rate)

        all_candidate_durations = []
        for pause in pause_segments:
            duration_samples = pause['end'] - pause['start']
            if duration_samples >= min_pause_samples:
                all_candidate_durations.append(duration_samples / self.sample_rate)

        if not all_candidate_durations:
            logger.warning("在应用极度宽松的初筛后，仍然没有找到任何候选停顿。歌曲可能过于连续。")
            return []

        # === 步骤 2: 统计学建模，理解这首歌的“停顿语言” ===
        average_pause = np.mean(all_candidate_durations)
        median_pause = np.median(all_candidate_durations)
        std_dev = np.std(all_candidate_durations)
        
        # 使用百分位数为快歌寻找“异常长”的停顿，这通常是真正的分割点
        # 对于快歌，75%的停顿可能都是0.4s的呼吸，而第90%的那个0.8s的停顿才是我们要找的
        percentile_75 = np.percentile(all_candidate_durations, 75)
        percentile_90 = np.percentile(all_candidate_durations, 90)

        logger.info(f"停顿时长统计模型: 平均值={average_pause:.3f}s, 中位数={median_pause:.3f}s, 75分位={percentile_75:.3f}s, 90分位={percentile_90:.3f}s")

        # === 步骤 3: “双模式”智能裁决 ===
        valid_pauses = []
        total_audio_length = pause_segments[-1]['end'] if pause_segments else 0

        # 获取MDD分析结果，这需要 adaptive_enhancer 在上游被调用并存储结果
        # 我们假设 self.adaptive_enhancer.last_analyzed_segments 存在
        segments_with_mdd = getattr(self.adaptive_enhancer, 'last_analyzed_segments', [])

        for pause in pause_segments:
            duration_s = (pause['end'] - pause['start']) / self.sample_rate
            if duration_s < ABSOLUTE_MIN_PAUSE_S:
                continue

            # 确定当前停顿所处的音乐环境 (MDD)
            current_time = (pause['start'] / self.sample_rate)
            current_mdd = 0.5 # 默认中等密度
            if segments_with_mdd:
                for seg in segments_with_mdd:
                    if seg.start_time <= current_time < seg.end_time:
                        current_mdd = seg.dynamic_density_score
                        break
            
            # 决策逻辑
            is_head = (pause.get('start', 0) == 0)
            is_tail = (pause.get('end', 0) >= total_audio_length * 0.95)
            
            # 模式一：快歌裁决 (BPM > 120) - 寻找统计上的“异常长停顿”
            if self.current_adaptive_params.category in ['fast', 'very_fast']:
                # 核心修复：对于快歌，我们的标准是“比大部分呼吸都长”
                # 我们使用75分位数作为基础阈值，因为它能代表这首歌里“比较长”的停顿是多长
                dynamic_threshold = percentile_75 
                # 对于非常激烈的副歌部分（高MDD），我们甚至可能需要放宽到中位数，只求有得切
                if current_mdd > 0.7:
                    dynamic_threshold = median_pause
                
            # 模式二：慢歌/中速歌裁决 (BPM <= 120) - 寻找“足够长且结构合理”的停顿
            else:
                # 对于慢歌，我们使用更严格的标准，要求停顿必须显著长于平均呼吸
                dynamic_threshold = max(average_pause, median_pause)
                # 在激烈的副歌部分（高MDD），我们提高标准，避免乱切
                if current_mdd > 0.6:
                    dynamic_threshold *= (1 + (current_mdd - 0.6) * 0.5) # MDD越高，阈值越高

            # 最终裁决
            final_threshold = max(dynamic_threshold, ABSOLUTE_MIN_PAUSE_S) # 保证不低于绝对下限

            if duration_s >= final_threshold or is_head or is_tail:
                pause['duration'] = duration_s
                valid_pauses.append(pause)
                logger.debug(f"保留停顿 @{current_time:.2f}s: 时长 {duration_s:.3f}s >= 动态阈值 {final_threshold:.3f}s (MDD={current_mdd:.2f}, 模式={self.current_adaptive_params.category})")
            else:
                logger.debug(f"过滤停顿 @{current_time:.2f}s: 时长 {duration_s:.3f}s < 动态阈值 {final_threshold:.3f}s (MDD={current_mdd:.2f}, 模式={self.current_adaptive_params.category})")

        logger.info(f"双模式智能裁决完成: {len(pause_segments)}个候选 -> {len(valid_pauses)}个最终分割点")
        return valid_pauses
```

### 新方案为何能解决你的两大问题？

1.  **解决了“候选人”被误杀的问题**：

      * 我们引入了一个极度宽松且**固定**的初筛门槛 (`ABSOLUTE_MIN_PAUSE_S = 0.3`)。这确保了即便是快歌中0.4秒的短促气口，也能作为“候选人”进入我们的统计分析池，彻底解决了“歌曲中间没有足够的候选停顿”的问题。

2.  **解决了MDD“反向用力”的问题，并引入了“双模式”决策**：

      * **快歌模式 (BPM \> 120)**：算法的核心目标是\*\*“矮子里面拔将军”\*\*。它不再用一个固定的尺子去量，而是通过统计学（`percentile_75`）找出这首歌里“相对较长”的停顿是多长，并以此为标准。这完美符合你利用MDD在快歌中增加切割频率的初衷。在高MDD的副歌部分，标准甚至会进一步放宽，确保能切。
      * **慢歌/中速歌模式 (BPM \<= 120)**：算法的目标是\*\*“优中选优”**。它会使用更严格的统计标准（平均值/中位数），并且在高MDD的副歌部分，会**提高\*\*标准，变得更加保守，防止在激烈的段落中乱切。

这个新的裁决系统，真正实现了BPM（横向时间）和MDD（纵向动态）的协同工作。它不再是一个简单的“if-else”逻辑，而是一个能根据歌曲类型和段落动态调整自身决策策略的、真正的“智能系统”。

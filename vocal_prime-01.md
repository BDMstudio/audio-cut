
## vocal_prime-01.md (已实现)

### VAD检测瓶颈的根源：静态的"标尺"无法丈量动态的"音乐"

问题的根源在于我们核心的停顿检测模块 `vocal_pause_detector.py` 中的 `_filter_adaptive_pauses` 函数。

你可能会觉得奇怪，我们不是已经有了BPM自适应系统吗？为什么还会出问题？

**答案是：我们只做到了“自适应地检测”，却没有做到“自适应地筛选”。**

当前的逻辑是这样的：

1.  BPM系统根据歌曲快慢，计算出一个动态的`min_pause_duration`（例如，快歌0.84秒，慢歌1.8秒）。
2.  然后，`_filter_adaptive_pauses` 函数就像一个无情的保安，拿着这把“0.84秒”的标尺，把所有VAD找到的、时长大于0.84秒的停顿**全部**当成了有效的分割点。

在慢歌里，这没问题，因为歌手两次呼吸的间隔通常都小于1.8秒，只有乐句结束才有长停顿。但在快歌里，这就成了灾难：

  * **密集的“微停顿”**：快歌中，歌手为了抢拍，两次呼吸或单词间的停顿可能恰好就在0.8秒左右，这在音乐上根本不是一句的结束。
  * **“一视同仁”的屠杀**：当前的“保安”逻辑无法区分“乐句间的长停顿”和“单词间的短呼吸”。只要时长达标，它就一刀切下去，结果就是你看到的，一句话被切成了好几段。

**所以，瓶颈就在于：我们用一个静态、固定的时长阈值去判断音乐中动态、复杂的停顿，这是典型的“刻舟求剑”。**

-----

### 解决方案：引入“统计学裁判”，让数据自己说话

我们必须抛弃那把静态的“标尺”，转而引入一个更智能的“统计学裁判”。这个裁判不再依赖一个固定的阈值，而是先观察这首歌里**所有**可能的停顿，找出它们的分布规律，然后动态地决定“什么才算是一个真正的、值得分割的长停顿”。

我们将对 `vocal_pause_detector.py` 里的 `_filter_adaptive_pauses` 函数进行一次“大脑升级手术”。

**请将 `src/vocal_smart_splitter/core/vocal_pause_detector.py` 文件中的 `_filter_adaptive_pauses` 函数替换为以下版本：**

```python
    def _filter_adaptive_pauses(self, pause_segments: List[Dict],
                              bpm_features: Optional['BPMFeatures']) -> List[Dict]:
        """
        [大脑升级版] 基于停顿分布的统计学动态筛选
        技术：两遍扫描法，第一遍收集并分析数据，第二遍根据动态阈值进行裁决。
        """
        # ✅ 关键改动：如果BPM系统未启用或失败，则退回旧的、简单的过滤方法
        if not self.enable_bpm_adaptation or not bpm_features:
            return self._filter_valid_pauses(pause_segments) # _filter_valid_pauses 是你旧代码里的静态过滤方法

        # === 第一遍扫描：数据收集与统计分析 ===
        
        # 1. 动态计算一个非常宽松的“基础门槛”
        # 这个门槛只用来过滤掉明显的噪音，保留所有可能是呼吸或停顿的候选者
        base_min_duration = self.current_adaptive_params.min_pause_duration * 0.7 # 使用动态计算值的70%作为基础门槛
        min_pause_samples = int(base_min_duration * self.sample_rate)

        middle_pause_durations = []
        total_audio_length = pause_segments[-1]['end'] if pause_segments else 0

        for i, pause in enumerate(pause_segments):
            duration_samples = pause['end'] - pause['start']
            # 只统计满足基础门槛的、且不是开头和结尾的“中间停顿”
            is_head = (i == 0 and pause['start'] == 0)
            is_tail = (i == len(pause_segments) - 1 and pause['end'] >= total_audio_length * 0.95)

            if duration_samples >= min_pause_samples and not is_head and not is_tail:
                middle_pause_durations.append(duration_samples / self.sample_rate)

        if not middle_pause_durations:
            logger.warning("在歌曲中间没有找到足够的候选停顿，可能导致分割过少。将放宽标准处理。")
            # 回退策略：如果中间没有停顿，就用所有停顿来做统计
            middle_pause_durations = [p['duration'] for p in pause_segments if p['duration'] >= base_min_duration]
            if not middle_pause_durations:
                # 极端情况：整首歌都没有像样的停顿，直接返回所有满足基础门槛的
                return [p for p in pause_segments if p['duration'] >= base_min_duration]


        # 2. 统计学分析：计算所有中间停顿的平均值和中位数
        average_pause_duration = np.mean(middle_pause_durations)
        median_pause_duration = np.median(middle_pause_durations)
        std_dev = np.std(middle_pause_durations)

        logger.info(f"停顿时长统计分析: 平均值={average_pause_duration:.3f}s, 中位数={median_pause_duration:.3f}s, 标准差={std_dev:.3f}s")
        
        # 3. 动态生成“裁决阈值”
        # 优先使用中位数，因为它对极端短或极端长的异常值不敏感，更能代表“普遍”的长停顿
        # 如果标准差很大，说明停顿长短不一，此时用平均值更合适
        if std_dev > average_pause_duration * 0.5:
             duration_threshold = average_pause_duration
             logger.info(f"停顿分布离散，使用平均值作为动态阈值: {duration_threshold:.3f}s")
        else:
             duration_threshold = max(average_pause_duration, median_pause_duration)
             logger.info(f"停顿分布集中，使用平均值/中位数较大者作为动态阈值: {duration_threshold:.3f}s")

        # 设置一个绝对下限，防止阈值过低导致乱切
        duration_threshold = max(duration_threshold, self.current_adaptive_params.min_pause_duration)
        logger.info(f"最终裁决阈值 (应用绝对下限后): {duration_threshold:.3f}s")


        # === 第二遍扫描：执行裁决 ===
        valid_pauses = []
        for pause in pause_segments:
            duration_seconds = pause['duration'] # 这里的duration是在上游计算好的
            
            # 规则：
            # 1. 开头和结尾的停顿，只要满足BPM系统计算的基础时长要求，就保留。
            # 2. 中间的停顿，必须满足我们动态计算出的“裁决阈值”，才被认为是真正的分割点。
            is_head = (pause.get('start', 0) == 0)
            is_tail = (pause.get('end', 0) >= total_audio_length * 0.95)
            
            passes_base_threshold = duration_seconds >= self.current_adaptive_params.min_pause_duration
            passes_dynamic_threshold = duration_seconds >= duration_threshold

            if passes_base_threshold and (is_head or is_tail):
                valid_pauses.append(pause)
                logger.debug(f"保留边界停顿: {duration_seconds:.3f}s")
            elif passes_dynamic_threshold and not (is_head or is_tail):
                valid_pauses.append(pause)
                logger.debug(f"保留中间停顿 (满足动态阈值): {duration_seconds:.3f}s")
            else:
                logger.debug(f"过滤中间停顿 (不满足动态阈值): {duration_seconds:.3f}s < {duration_threshold:.3f}s")
        
        logger.info(f"统计学动态裁决完成: {len(pause_segments)}个候选 -> {len(valid_pauses)}个最终分割点")
        return valid_pauses

```

### 这次升级为什么能解决快歌的问题？

1.  **个性化标准，而非“一刀切”**：新算法为**每一首歌**量身定制了一个分割标准 (`duration_threshold`)。对于快歌，它能从一大堆短促的呼吸中，识别出那个相对“最长”的、代表乐句结束的停顿，从而避免了在句子中间下刀。
2.  **数据驱动决策**：决策不再依赖于我们在配置文件里猜的一个静态数字，而是基于对歌曲自身停顿模式的**统计分析**。这使得系统对不同风格、不同节奏的歌曲具有了更强的鲁棒性。
3.  **保留边界，尊重整体**：算法特殊处理了歌曲的开头和结尾，确保了引子和尾奏的完整性，同时只对歌曲主体部分进行严格的统计筛选，决策更加合理。
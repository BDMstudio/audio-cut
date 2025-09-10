## vocal_prime-03.md (增强方案)

**只有横向的BPM（时间节奏）是不够的，必须引入纵向的动态（响度、能量、情绪）指标，才能让切割真正拥有"音乐感"**。

### 当前实现状态

本文档描述的"音乐动态密度(MDD)"指标已部分实现：
- ✅ ArrangementComplexitySegment数据结构已存在于adaptive_vad_enhancer.py
- ✅ BPM自适应系统已完整实现并稳定运行
- ⚠️ MDD纵向指标计算尚未完全集成到生产代码
- 📝 建议作为v2.2版本的增强特性

### 设计纵向判断指标：“音乐动态密度 (Musical Dynamic Density)”

为了解决切不准切的生硬这个问题，我们需要设计一套能够量化“音乐激烈程度”的纵向指标。我将其命名为\*\*“音乐动态密度 (Musical Dynamic Density, MDD)”\*\*。MDD是一个综合评分，分数越高，代表音乐越激烈、越不应该频繁切割。

这个指标将通过分析音频块的三个维度来计算：

1.  **能量维度 (Loudness & Power)**: 副歌部分的能量通常远高于主歌。
2.  **频谱维度 (Spectral Fullness)**: 副歌部分的频谱更“满”，从低频到高频都有声音。
3.  **节奏维度 (Rhythmic Intensity)**: 副歌部分的节奏更密集、更强烈。

#### 技术实现方案

我们将对核心的 `AdaptiveVADEnhancer` 模块 (`src/vocal_smart_splitter/core/adaptive_vad_enhancer.py`) 进行升级，让它在分析时不仅输出BPM，还要输出每个时间段的MDD评分。

**1. 升级 `ArrangementComplexitySegment` 数据结构**

在 `adaptive_vad_enhancer.py` 文件的开头，我们需要给这个数据结构增加新的字段来存储我们的纵向指标：

```python
# In src/vocal_smart_splitter/core/adaptive_vad_enhancer.py

@dataclass
class ArrangementComplexitySegment:
    # ... (原有字段)
    # 新增纵向指标
    rms_energy: float                 # 能量维度：均方根能量
    spectral_flatness: float          # 频谱维度：频谱平坦度，越接近1越像噪音/满频谱
    onset_rate: float                 # 节奏维度：音符起始率，越高节奏越密集
    dynamic_density_score: float      # 最终的“音乐动态密度”综合评分
```

**2. 实现MDD指标的计算**

在 `AdaptiveVADEnhancer` 类中，我们需要一个新函数来计算这些指标。

```python
# In src/vocal_smart_splitter/core/adaptive_vad_enhancer.py -> class AdaptiveVADEnhancer

    def _calculate_dynamic_density_metrics(self, audio_segment: np.ndarray) -> Dict[str, float]:
        """计算音乐动态密度（MDD）相关指标"""
        metrics = {}
        sr = self.sample_rate

        # 1. 能量维度: RMS Energy
        rms = librosa.feature.rms(y=audio_segment)[0]
        metrics['rms_energy'] = np.mean(rms)

        # 2. 频谱维度: Spectral Flatness
        # 频谱平坦度衡量声音的“类噪音”程度。副歌部分频谱饱满，平坦度会更高。
        flatness = librosa.feature.spectral_flatness(y=audio_segment)
        metrics['spectral_flatness'] = np.mean(flatness)

        # 3. 节奏维度: Onset Rate
        # 计算每秒的音符起始数量，反映节奏的密集程度
        onset_env = librosa.onset.onset_strength(y=audio_segment, sr=sr)
        onset_rate = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / (len(audio_segment) / sr)
        metrics['onset_rate'] = onset_rate
        
        return metrics

    def _calculate_overall_dynamic_density(self, metrics: Dict[str, float], all_segments_metrics: List[Dict]) -> float:
        """根据全局分布计算当前片段的MDD综合评分 (0-1)"""
        
        # 提取所有片段的指标用于归一化
        all_rms = [m['rms_energy'] for m in all_segments_metrics]
        all_flatness = [m['spectral_flatness'] for m in all_segments_metrics]
        all_onset_rate = [m['onset_rate'] for m in all_segments_metrics]

        # 计算归一化得分 (将每个指标映射到0-1范围)
        rms_score = (metrics['rms_energy'] - np.min(all_rms)) / (np.max(all_rms) - np.min(all_rms) + 1e-6)
        flatness_score = (metrics['spectral_flatness'] - np.min(all_flatness)) / (np.max(all_flatness) - np.min(all_flatness) + 1e-6)
        onset_score = (metrics['onset_rate'] - np.min(all_onset_rate)) / (np.max(all_onset_rate) - np.min(all_onset_rate) + 1e-6)
        
        # 加权平均得到最终MDD评分 (能量权重最高)
        weights = {'rms': 0.5, 'flatness': 0.3, 'onset': 0.2}
        mdd_score = rms_score * weights['rms'] + flatness_score * weights['flatness'] + onset_score * weights['onset']
        
        return np.clip(mdd_score, 0, 1)
```

**3. 在主分析流程中集成MDD计算**

我们需要修改 `analyze_arrangement_complexity` 函数，让它在分析每个片段时都计算MDD，并进行全局归一化。

```python
# In src/vocal_smart_splitter/core/adaptive_vad_enhancer.py -> class AdaptiveVADEnhancer

    def analyze_arrangement_complexity(self, audio: np.ndarray) -> Tuple[List[ArrangementComplexitySegment], BPMFeatures]:
        # ... (前面的BPM分析等代码不变) ...

        # 两遍扫描法：第一遍收集所有片段的原始指标
        all_metrics = []
        raw_segments_info = []
        for i in range(0, len(audio) - window_samples, hop_samples):
             # ... (获取 segment_audio) ...
            raw_metrics = self._calculate_dynamic_density_metrics(segment_audio)
            all_metrics.append(raw_metrics)
            raw_segments_info.append({'start_time': start_time, 'end_time': end_time, 'raw_metrics': raw_metrics})

        # 第二遍：计算每个片段的最终MDD评分并构建结果
        final_segments = []
        for info in raw_segments_info:
            mdd_score = self._calculate_overall_dynamic_density(info['raw_metrics'], all_metrics)
            
            # ... (计算其他的复杂度指标和自适应参数) ...

            segment = ArrangementComplexitySegment(
                # ... (原有字段)
                # 填充新的MDD相关字段
                rms_energy=info['raw_metrics']['rms_energy'],
                spectral_flatness=info['raw_metrics']['spectral_flatness'],
                onset_rate=info['raw_metrics']['onset_rate'],
                dynamic_density_score=mdd_score
            )
            final_segments.append(segment)
            
        return final_segments, bpm_features
```

**4. 应用MDD指标：动态调整切割策略**

最后，也是最关键的一步，我们在 `vocal_pause_detector.py` 中利用这个MDD评分来调整我们的切割“狠度”。

```python
# In src/vocal_smart_splitter/core/vocal_pause_detector.py -> class VocalPauseDetectorV2

    def _filter_adaptive_pauses(self, pause_segments: List[Dict], bpm_features: Optional[BPMFeatures]) -> List[Dict]:
        # ... (上一轮我们做的统计学动态阈值逻辑) ...

        # 新增：应用MDD评分调整最终的裁决阈值
        
        final_valid_pauses = []
        for pause in valid_pauses: # valid_pauses是上一轮统计学筛选后的结果
            
            # 找到这个停顿点对应的MDD评分
            current_time = (pause['start'] + pause['end']) / 2.0 / self.sample_rate
            current_mdd = 0.5 # 默认值
            if self.adaptive_enhancer and hasattr(self.adaptive_enhancer, 'last_analyzed_segments'):
                for seg in self.adaptive_enhancer.last_analyzed_segments:
                    if seg.start_time <= current_time < seg.end_time:
                        current_mdd = seg.dynamic_density_score
                        break
            
            # 核心策略：MDD越高，对停顿时长的要求就越高（越不倾向于切割）
            # MDD为0时，使用原阈值；MDD为1时，阈值提高50%
            mdd_multiplier = 1.0 + (current_mdd * 0.5) 
            final_duration_threshold = duration_threshold * mdd_multiplier

            if pause['duration'] >= final_duration_threshold:
                final_valid_pauses.append(pause)
                logger.debug(f"保留停顿: duration {pause['duration']:.2f}s >= MDD调整后阈值 {final_duration_threshold:.2f}s (MDD={current_mdd:.2f})")
            else:
                logger.debug(f"过滤停顿: duration {pause['duration']:.2f}s < MDD调整后阈值 {final_duration_threshold:.2f}s (MDD={current_mdd:.2f})")

        return final_valid_pauses
```

### 总结：从“一维”到“二维”的决策进化

通过引入“音乐动态密度 (MDD)”这个纵向指标，我们的决策系统从原来只关心\*\*“时间上是否够长”（一维）**，进化到了同时关心**“时间上是否够长” AND “音乐上是否激烈”（二维）\*\*的全新层面。

  - 在**主歌部分** (MDD低)，系统会采取\*\*“切的柔”\*\*的策略，标准会放宽，允许在更多自然的呼吸点进行分割。
  - 在**副歌部分** (MDD高)，系统会自动切换到\*\*“切的狠”\*\*的策略，标准会变得极其严格，只有那些最长、最明显的停顿才会被考虑，从而保证了高潮部分的音乐完整性和连贯性。

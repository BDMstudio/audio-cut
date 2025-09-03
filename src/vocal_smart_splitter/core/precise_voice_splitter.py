#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/precise_voice_splitter.py
# AI-SUMMARY: 基于精确人声检测的分割器，只在真正的人声停顿处分割

"""
精确人声分割器

基于用户反馈重新设计的分割器：
1. 使用先进的VAD算法精确检测人声活动
2. 只在真正的人声停顿处进行分割
3. 不考虑片段长度，优先保证分割精准度
4. 如果没有人声的部分可以保持为长片段
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

from .advanced_vad import AdvancedVAD
from ..utils.config_manager import get_config


class PreciseVoiceSplitter:
    """精确人声分割器"""

    def __init__(self, sample_rate: int = 44100):
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate

        # 初始化先进的VAD
        self.vad = AdvancedVAD(sample_rate=16000)  # VAD通常使用16kHz

        # 配置参数
        self.min_silence_duration = get_config('precise_voice_splitting.min_silence_duration', 0.5)  # 最小静音时长
        self.min_voice_duration = get_config('precise_voice_splitting.min_voice_duration', 1.0)     # 最小人声时长
        self.silence_threshold = get_config('precise_voice_splitting.silence_threshold', 0.3)       # 静音阈值

        self.logger.info("精确人声分割器初始化完成")

    def split_by_voice_activity(self, audio: np.ndarray, sample_rate: int, breath_pauses=None) -> Dict:
        """
        基于人声活动进行精确分割 - 支持三种切点策略：
        1) 静音中心切（内部静音）
        2) 人声边界±偏移（首/尾静音）
        3) 呼吸停顿备选（向后兼容，可禁用）
        """
        self.logger.info("开始基于人声活动的精确分割...")

        total_duration = len(audio) / sample_rate
        self.logger.info(f"音频总时长: {total_duration:.2f}秒")

        placement_strategy = get_config('precise_voice_splitting.placement_strategy', 'center_internal_offset_edges')
        use_breath = get_config('precise_voice_splitting.use_breath_pauses', True)

        # 当采用按人声边界+偏移策略时，强制走 VAD 路径
        if placement_strategy == 'center_internal_offset_edges':
            use_breath = False
        # 更保守的静音阈值（减少碎片），允许通过配置覆盖
        self.min_silence_duration = get_config('precise_voice_splitting.min_silence_duration', 1.2)

        if breath_pauses is not None and use_breath:
            # 使用breath_detector检测到的真实人声停顿点（向后兼容）
            self.logger.info("使用breath_detector提供的人声停顿点...")
            qualified_pauses = self._analyze_breath_pauses(breath_pauses)
            split_points = self._select_split_points_from_pauses(qualified_pauses, total_duration)
        else:
            # 使用VAD方法（推荐）
            self.logger.info("使用 VAD 检测人声片段与静音间隔...")

            # 1. 检测人声活动区间
            voice_segments = self.vad.detect_voice_activity(audio, sample_rate)
            self.logger.info(f"检测到 {len(voice_segments)} 个人声片段")
            # 合并短静音、过滤过短人声，稳定边界
            voice_segments = self._refine_voice_segments(voice_segments)
            self.logger.info(f"边界稳定后的人声片段数: {len(voice_segments)}")

            if placement_strategy == 'center_internal_offset_edges':
                # 基于人声边界±0.5s与内部静音中心点的策略
                split_points = self._select_points_offset_strategy(voice_segments, total_duration)
            else:
                # 2. 找到静音间隔
                silence_gaps = self.vad.find_silence_gaps(voice_segments, total_duration)
                self.logger.info(f"检测到 {len(silence_gaps)} 个静音间隔")

                # 3. 分析静音间隔质量
                qualified_silences = self._analyze_silence_quality(silence_gaps)

                # 4. 选择最佳分割点
                split_points = self._select_split_points(qualified_silences, voice_segments, total_duration)

        # 4.5 切点吸附到能量谷底，确保落在真实停顿处
        split_points_times = [p['split_time'] for p in split_points]
        snapped_times = self._snap_splits_to_energy_valleys(audio, sample_rate, split_points_times)
        for i, t in enumerate(snapped_times):
            split_points[i]['split_time'] = t

        # 5. 生成分割方案
        split_plan = self._generate_split_plan(split_points, total_duration, sample_rate)

        self.logger.info(f"精确人声分割完成，共 {len(split_points)} 个分割点")

        return split_plan

    def _snap_splits_to_energy_valleys(self, audio: np.ndarray, sample_rate: int,
                                        split_times: List[float]) -> List[float]:
        """在每个切点附近寻找能量谷底以对齐到更自然的停顿
        在 ±window 秒范围内计算短时能量，选择最小能量位置。
        """
        if not split_times:
            return []
        snapped = []
        # 短时能量窗口 50ms，步长 10ms
        win = int(0.05 * sample_rate)
        hop = int(0.01 * sample_rate)
        search = 0.25  # 搜索范围 ±0.25s
        n = len(audio)
        for t in split_times:
            center = int(t * sample_rate)
            start = max(0, center - int(search * sample_rate))
            end = min(n, center + int(search * sample_rate))
            if end - start <= win:
                snapped.append(t)
                continue
            # 计算窗口能量
            best_e = None
            best_i = center
            for i in range(start, end - win, hop):
                seg = audio[i:i+win]
                e = float(np.mean(seg * seg)) if len(seg) > 0 else 0.0
                if best_e is None or e < best_e:
                    best_e = e
                    best_i = i + win // 2
            snapped_t = best_i / sample_rate
            snapped.append(snapped_t)
        return snapped


    def _select_points_offset_strategy(self, voice_segments: List[Dict], total_duration: float) -> List[float]:
        """根据人声边界与静音规则选择分割点
        规则：
        - 内部静音（两段人声之间，静音≥min_silence_duration）：在静音中心点切
        - 头部静音（起点至首段人声，静音≥min_silence_duration）：在 人声开始−offset 处切
        - 尾部静音（末段人声至终点，静音≥min_silence_duration）：在 人声结束+offset 处切
        不做任何时长约束，以保证拼接还原。
        """
        min_sil = get_config('precise_voice_splitting.min_silence_duration', 1.0)
        off_before = get_config('precise_voice_splitting.cut_offset_before_vocal_start', 0.5)
        off_after = get_config('precise_voice_splitting.cut_offset_after_vocal_end', 0.5)
        min_gap = get_config('precise_voice_splitting.min_split_gap', 0.3)
        boundary_buffer = max(0.0, min(get_config('precise_voice_splitting.boundary_buffer', 2.0), 1.0))

        cuts: List[float] = []

        # 无人声：不生成任何切点，保持整段
        if not voice_segments:
            return []

        # 头部静音
        first_start = voice_segments[0]['start']
        head_sil = first_start - 0.0
        if head_sil >= min_sil:
            cut = max(0.0, first_start - off_before)
            cuts.append(cut)

        # 内部静音：中心切
        for i in range(len(voice_segments) - 1):
            prev_end = voice_segments[i]['end']
            next_start = voice_segments[i + 1]['start']
            gap = next_start - prev_end
            if gap >= min_sil:
                # 在静音区间内部选择：边界±offset的中点，确保落在静音内部
                left = prev_end + off_after
                right = next_start - off_before
                if right > left:
                    split = (left + right) / 2.0
                else:
                    # 极端情况下退回静音中心
                    split = (prev_end + next_start) / 2.0
                cuts.append(split)

        # 尾部静音
        last_end = voice_segments[-1]['end']
        tail_sil = total_duration - last_end
        if tail_sil >= min_sil:
            cut = min(total_duration, last_end + off_after)
            cuts.append(cut)

        # 排序并按最小间距去重
        cuts.sort()
        deduped: List[float] = []
        for c in cuts:
            if not deduped or abs(c - deduped[-1]) >= min_gap:
                deduped.append(c)

        self.logger.info(f"按边界与偏移策略选择了 {len(deduped)} 个分割点")
        return deduped


    def _refine_voice_segments(self, voice_segments: List[Dict]) -> List[Dict]:
        """合并短静音并过滤过短的人声片段，稳定边界
        - 合并相邻人声片段：若间隔 < merge_silence_below 则合并
        - 过滤过短片段：若片段时长 < min_voice_duration 则丢弃
        """
        merge_gap = get_config('precise_voice_splitting.merge_silence_below', 0.8)
        min_voice = get_config('precise_voice_splitting.min_voice_duration', 0.8)
        if not voice_segments:
            return []
        segs = sorted(voice_segments, key=lambda s: s['start'])
        # 先过滤明显过短的人声片段
        segs = [s for s in segs if (s['end'] - s['start']) >= min_voice]
        if not segs:
            return []
        merged: List[Dict] = []
        cur = segs[0].copy()
        for s in segs[1:]:
            gap = s['start'] - cur['end']
            if gap < merge_gap:
                # 合并
                cur['end'] = max(cur['end'], s['end'])
            else:
                merged.append(cur)
                cur = s.copy()
        merged.append(cur)
        return merged

    def _analyze_breath_pauses(self, breath_pauses: List[Tuple[float, float, float]]) -> List[Dict]:
        """分析breath_detector提供的人声停顿点质量

        Args:
            breath_pauses: [(start, end, confidence), ...] 格式的停顿点列表

        Returns:
            分析后的停顿点列表
        """
        self.logger.info(f"分析 {len(breath_pauses)} 个人声停顿点...")

        qualified_pauses = []

        # 获取多级停顿阈值
        short_pause = get_config('precise_voice_splitting.short_pause_duration', 0.1)
        medium_pause = get_config('precise_voice_splitting.medium_pause_duration', 0.3)
        long_pause = get_config('precise_voice_splitting.long_pause_duration', 0.8)

        for start, end, confidence in breath_pauses:
            duration = end - start

            # 只处理符合最小时长要求的停顿
            if duration >= self.min_silence_duration:
                # 多级停顿分类
                pause_type = "unknown"
                base_score = 0.0

                if duration >= long_pause:
                    pause_type = "long_pause"
                    base_score = 0.95
                elif duration >= medium_pause:
                    pause_type = "medium_pause"
                    base_score = 0.85
                elif duration >= short_pause:
                    pause_type = "short_pause"
                    base_score = 0.75
                else:
                    pause_type = "breath"
                    base_score = 0.65

                # 综合质量评分：结合停顿时长和breath_detector的置信度
                quality_score = base_score * 0.7 + confidence * 0.3

                # 只选择高质量的停顿点
                if quality_score >= self.silence_threshold:
                    pause_info = {
                        'start': start,
                        'end': end,
                        'duration': duration,
                        'pause_type': pause_type,
                        'quality_score': quality_score,
                        'confidence': confidence,
                        'type': 'breath_pause'
                    }
                    qualified_pauses.append(pause_info)

                    self.logger.debug(f"合格停顿 [{pause_type}]: {start:.2f}-{end:.2f}s, "
                                    f"时长: {duration:.3f}s, 评分: {quality_score:.3f}")

        # 按质量和停顿类型排序
        def sort_key(pause):
            type_priority = {
                'long_pause': 4,
                'medium_pause': 3,
                'short_pause': 2,
                'breath': 1
            }
            return (type_priority.get(pause['pause_type'], 0), pause['quality_score'])

        qualified_pauses.sort(key=sort_key, reverse=True)

        self.logger.info(f"找到 {len(qualified_pauses)} 个合格的人声停顿点")
        for pause in qualified_pauses[:5]:  # 记录前5个最佳停顿
            self.logger.info(f"  优质停顿 [{pause['pause_type']}]: {pause['start']:.2f}s, "
                           f"时长: {pause['duration']:.3f}s, 评分: {pause['quality_score']:.3f}")

        return qualified_pauses

    def _select_split_points_from_pauses(self, qualified_pauses: List[Dict], total_duration: float) -> List[float]:
        """从人声停顿点中选择最佳分割点

        Args:
            qualified_pauses: 合格的停顿点列表
            total_duration: 音频总时长

        Returns:
            分割点时间列表
        """
        self.logger.info("从人声停顿点选择最佳分割点...")

        split_points = []

        # 选择所有高质量的自然停顿点
        for pause in qualified_pauses:
            # 在停顿间隔的中心点分割
            split_time = pause['start'] + pause['duration'] / 2

            # 边界检查
            boundary_buffer = get_config('precise_voice_splitting.boundary_buffer', 2.0)
            adjusted_buffer = min(boundary_buffer, 1.0)  # 最大1秒边界缓冲

            if adjusted_buffer < split_time < total_duration - adjusted_buffer:
                # 检查与其他分割点的距离，避免过密
                min_gap = 1.5  # 分割点之间最小间距1.5秒
                too_close = False

                for existing_split in split_points:
                    if abs(split_time - existing_split) < min_gap:
                        too_close = True
                        break

                if not too_close:
                    split_points.append(split_time)
                    self.logger.info(f"选择分割点 [{pause['pause_type']}]: {split_time:.2f}s "
                                   f"(停顿时长: {pause['duration']:.3f}s, 评分: {pause['quality_score']:.3f})")
                else:
                    self.logger.debug(f"跳过分割点 {split_time:.2f}s，与现有分割点过近")

        # 如果分割点太少，降低质量阈值重新尝试
        if len(split_points) < 5 and len(qualified_pauses) > len(split_points):
            self.logger.info("分割点数量较少，尝试降低质量要求...")

            # 临时降低质量阈值，选择更多分割点
            for pause in qualified_pauses:
                if pause['quality_score'] >= 0.3:  # 降低阈值到0.3
                    split_time = pause['start'] + pause['duration'] / 2

                    if 1.0 < split_time < total_duration - 1.0:  # 进一步减少边界缓冲
                        # 检查距离
                        too_close = any(abs(split_time - existing) < 1.2 for existing in split_points)  # 减少最小间距

                        if not too_close:
                            split_points.append(split_time)
                            self.logger.info(f"补充分割点 [{pause['pause_type']}]: {split_time:.2f}s")

                            if len(split_points) >= 15:  # 限制最大分割点数量
                                break

        # 排序分割点
        split_points.sort()

        self.logger.info(f"最终选择了 {len(split_points)} 个分割点")
        for i, point in enumerate(split_points, 1):
            self.logger.info(f"  分割点 {i}: {point:.2f}s")

        return split_points

    def _analyze_silence_quality(self, silence_gaps: List[Dict]) -> List[Dict]:
        """分析静音间隔的质量 - 改进版本，更好识别自然停顿"""
        self.logger.info("分析静音间隔质量...")

        # 获取多级停顿阈值
        short_pause = get_config('precise_voice_splitting.short_pause_duration', 0.1)
        medium_pause = get_config('precise_voice_splitting.medium_pause_duration', 0.3)
        long_pause = get_config('precise_voice_splitting.long_pause_duration', 0.8)

        qualified_silences = []

        for gap in silence_gaps:
            duration = gap['duration']

            # 改进的质量评分算法 - 优先识别自然停顿
            quality_score = 0.0
            pause_type = "unknown"

            # 多级停顿分类和评分
            if duration >= long_pause:
                duration_score = 0.95  # 长停顿，优秀的分割点
                pause_type = "long_pause"
            elif duration >= medium_pause:
                duration_score = 0.85  # 中等停顿，很好的分割点
                pause_type = "medium_pause"
            elif duration >= short_pause:
                duration_score = 0.7   # 短停顿，良好的分割点
                pause_type = "short_pause"
            elif duration >= 0.06:
                duration_score = 0.5   # 极短停顿，可能是换气
                pause_type = "breath"
            else:
                duration_score = 0.2   # 太短，可能是噪音
                pause_type = "noise"

            quality_score += duration_score * 0.6  # 时长权重60%

            # 位置评分 - 更宽松的边界限制
            position_score = 1.0
            if gap['start'] < 2.0:  # 开头2秒内
                position_score = 0.7
            elif gap['start'] > gap['start'] + duration - 2.0:  # 结尾2秒内
                position_score = 0.7

            quality_score += position_score * 0.2  # 位置权重20%

            # 新增：停顿连续性评分
            # 检查前后是否有其他停顿，孤立的停顿更可能是自然分割点
            isolation_score = 1.0  # 暂时设为1.0，后续可优化
            quality_score += isolation_score * 0.2  # 连续性权重20%

            # 更宽松的筛选条件 - 接受更多自然停顿
            if duration >= self.min_silence_duration and quality_score >= self.silence_threshold:
                gap['quality_score'] = quality_score
                gap['pause_type'] = pause_type
                qualified_silences.append(gap)

                self.logger.debug(f"合格静音 [{pause_type}]: {gap['start']:.2f}-{gap['end']:.2f}s, "
                                f"时长: {duration:.3f}s, 评分: {quality_score:.3f}")

        # 优化排序策略：先按类型排序（优先自然停顿），再按质量评分
        def sort_key(gap):
            type_priority = {
                'long_pause': 4,
                'medium_pause': 3,
                'short_pause': 2,
                'breath': 1,
                'noise': 0
            }
            return (type_priority.get(gap['pause_type'], 0), gap['quality_score'])

        qualified_silences.sort(key=sort_key, reverse=True)

        self.logger.info(f"找到 {len(qualified_silences)} 个合格的静音间隔")
        for gap in qualified_silences[:5]:  # 记录前5个最佳停顿
            self.logger.info(f"  优质停顿 [{gap['pause_type']}]: {gap['start']:.2f}s, "
                           f"时长: {gap['duration']:.3f}s, 评分: {gap['quality_score']:.3f}")

        return qualified_silences

    def _select_split_points(self, qualified_silences: List[Dict],
                           voice_segments: List[Dict],
                           total_duration: float) -> List[float]:
        """选择最佳分割点 - 改进版本，选择更多自然停顿点"""
        self.logger.info("选择最佳分割点...")

        split_points = []

        # 新策略：选择所有高质量的自然停顿点，而不仅仅是最长的
        for silence in qualified_silences:
            # 在静音间隔的中心点分割
            split_time = silence['start'] + silence['duration'] / 2

            # 更宽松的边界检查 - 减少边界缓冲从2秒到1秒
            boundary_buffer = get_config('precise_voice_splitting.boundary_buffer', 2.0)
            adjusted_buffer = min(boundary_buffer, 1.0)  # 最大1秒边界缓冲

            if adjusted_buffer < split_time < total_duration - adjusted_buffer:
                # 检查与其他分割点的距离，避免过密
                min_gap = 2.0  # 分割点之间最小间距2秒
                too_close = False

                for existing_split in split_points:
                    if abs(split_time - existing_split) < min_gap:
                        too_close = True
                        break

                if not too_close:
                    split_points.append(split_time)
                    self.logger.info(f"选择分割点 [{silence['pause_type']}]: {split_time:.2f}s "
                                   f"(静音时长: {silence['duration']:.3f}s, 评分: {silence['quality_score']:.3f})")
                else:
                    self.logger.debug(f"跳过分割点 {split_time:.2f}s，与现有分割点过近")

        # 如果分割点太少，降低质量阈值重新尝试
        if len(split_points) < 3 and len(qualified_silences) > len(split_points):
            self.logger.info("分割点数量较少，尝试降低质量要求...")

            # 临时降低质量阈值，选择更多分割点
            for silence in qualified_silences:
                if silence['quality_score'] >= 0.3:  # 降低阈值到0.3
                    split_time = silence['start'] + silence['duration'] / 2

                    if 1.0 < split_time < total_duration - 1.0:  # 进一步减少边界缓冲
                        # 检查距离
                        too_close = any(abs(split_time - existing) < 1.5 for existing in split_points)  # 减少最小间距

                        if not too_close:
                            split_points.append(split_time)
                            self.logger.info(f"补充分割点 [{silence['pause_type']}]: {split_time:.2f}s")

                            if len(split_points) >= 10:  # 限制最大分割点数量
                                break

        # 如果仍然没有找到合适的分割点，使用备用策略
        if not split_points:
            self.logger.warning("未找到理想的分割点，使用备用策略")
            split_points = self._fallback_split_strategy(voice_segments, total_duration)

        # 排序分割点
        split_points.sort()

        self.logger.info(f"最终选择了 {len(split_points)} 个分割点")
        for i, point in enumerate(split_points, 1):
            self.logger.info(f"  分割点 {i}: {point:.2f}s")

        return split_points

    def _fallback_split_strategy(self, voice_segments: List[Dict], total_duration: float) -> List[float]:
        """备用分割策略：在人声片段之间的间隔分割 - 改进版本"""
        self.logger.info("执行备用分割策略...")

        split_points = []

        if len(voice_segments) >= 2:
            # 在人声片段之间的间隔中分割 - 降低最小间隔要求
            for i in range(len(voice_segments) - 1):
                gap_start = voice_segments[i]['end']
                gap_end = voice_segments[i + 1]['start']
                gap_duration = gap_end - gap_start

                # 大幅降低间隔要求，从200ms降至100ms
                if gap_duration >= 0.1:  # 至少100ms的间隔
                    split_time = gap_start + gap_duration / 2
                    if 0.5 < split_time < total_duration - 0.5:  # 减少边界限制
                        split_points.append(split_time)
                        self.logger.info(f"备用分割点: {split_time:.2f}s (间隔: {gap_duration:.3f}s)")

        # 如果还是分割点太少，尝试更激进的策略
        if len(split_points) < 5:
            self.logger.info("分割点仍然较少，尝试基于人声片段长度的分割...")

            # 策略：如果人声片段超过10秒，尝试在中间分割
            for segment in voice_segments:
                duration = segment['end'] - segment['start']
                if duration > 10.0:  # 长片段
                    # 在片段的3/4处尝试分割（通常是句子或段落的自然停顿点）
                    split_time = segment['start'] + duration * 0.75
                    if 1.0 < split_time < total_duration - 1.0:
                        # 检查是否与现有分割点太近
                        too_close = any(abs(split_time - existing) < 2.0 for existing in split_points)
                        if not too_close:
                            split_points.append(split_time)
                            self.logger.info(f"长片段分割点: {split_time:.2f}s (片段长度: {duration:.1f}s)")

        # 如果还是没有分割点，保持整个音频为一个片段
        if not split_points:
            self.logger.info("无法找到合适的分割点，保持为单个片段")
        else:
            self.logger.info(f"备用策略生成了 {len(split_points)} 个分割点")

        return sorted(split_points)

    def _generate_split_plan(self, split_points: List[float],
                           total_duration: float,
                           sample_rate: int) -> Dict:
        """生成最终的分割方案"""

        # 添加开始和结束时间
        all_points = [0.0] + split_points + [total_duration]

        # 生成片段信息
        segments = []
        formatted_split_points = []

        for i in range(len(all_points) - 1):
            start_time = all_points[i]
            end_time = all_points[i + 1]
            duration = end_time - start_time

            segments.append({
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'start_sample': int(start_time * sample_rate),
                'end_sample': int(end_time * sample_rate)
            })

        # 格式化分割点（排除开始和结束点）
        for split_time in split_points:
            formatted_split_points.append({
                'split_time': split_time,
                'quality_score': 0.9,  # 高质量分割点
                'pause_duration': 0.5,  # 估计的停顿时长
                'confidence': 0.9,     # 高置信度
                'method': 'precise_vad'
            })

        return {
            'segments': segments,
            'split_points': formatted_split_points,
            'total_segments': len(segments),
            'method': 'precise_voice_activity',
            'vad_method': self.vad.vad_method
        }

    def get_split_summary(self, split_plan: Dict) -> str:
        """获取分割结果摘要"""
        segments = split_plan['segments']
        total_segments = len(segments)

        if total_segments == 1:
            return f"保持为单个片段 (时长: {segments[0]['duration']:.1f}秒)"

        durations = [seg['duration'] for seg in segments]
        avg_duration = np.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        summary = f"分割为 {total_segments} 个片段:\n"
        summary += f"  - 平均时长: {avg_duration:.1f}秒\n"
        summary += f"  - 时长范围: {min_duration:.1f}-{max_duration:.1f}秒\n"
        summary += f"  - VAD方法: {split_plan.get('vad_method', 'unknown')}"

        return summary

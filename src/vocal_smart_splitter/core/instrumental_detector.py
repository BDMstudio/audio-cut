#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/instrumental_detector.py
# AI-SUMMARY: 纯音乐段检测器 - 检测音频中的纯音乐部分并根据阈值决定是否单独分割

import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..utils.config_manager import get_config

logger = logging.getLogger(__name__)

@dataclass
class InstrumentalSegment:
    """纯音乐段数据结构"""
    start_time: float
    end_time: float
    duration: float
    segment_type: str  # 'instrumental' or 'vocal'
    should_split: bool = False  # 是否应该单独分割

class InstrumentalDetector:
    """纯音乐段检测器

    使用VAD反向逻辑检测纯音乐段，并根据配置的阈值决定是否单独分割
    """

    def __init__(self, sample_rate: int = 44100):
        """初始化纯音乐检测器

        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        self.enabled = get_config('instrumental_detection.enable', True)
        self.music_threshold = get_config('instrumental_detection.music_threshold', 5.0)

        # 借用现有的segment长度约束
        self.min_duration = get_config('quality_control.segment_min_duration', 5.0)
        self.max_duration = get_config('quality_control.segment_max_duration', 18.0)

        logger.info(f"InstrumentalDetector initialized (SR: {sample_rate}, threshold: {self.music_threshold}s)")

    def detect_instrumental_segments(self, vocal_activity_mask: np.ndarray,
                                    audio_duration: float) -> List[InstrumentalSegment]:
        """检测所有纯音乐段

        Args:
            vocal_activity_mask: 人声活动掩码（1=有人声，0=无人声）
            audio_duration: 音频总时长（秒）

        Returns:
            纯音乐段列表
        """
        if not self.enabled:
            logger.info("Instrumental detection is disabled")
            return []

        logger.info("Detecting instrumental segments from vocal activity mask...")

        # 反转掩码获取纯音乐区域
        instrumental_mask = 1 - vocal_activity_mask

        # 找到所有连续的纯音乐段
        segments = self._find_continuous_segments(instrumental_mask, audio_duration)

        # 判定每个段是否应该单独分割
        for segment in segments:
            segment.should_split = self._should_split_segment(segment.duration)
            if segment.should_split:
                logger.info(f"Instrumental segment [{segment.start_time:.2f}-{segment.end_time:.2f}]s "
                          f"({segment.duration:.2f}s) will be split separately")

        logger.info(f"Found {len(segments)} instrumental segments, "
                   f"{sum(1 for s in segments if s.should_split)} will be split separately")

        return segments

    def _find_continuous_segments(self, mask: np.ndarray, total_duration: float) -> List[InstrumentalSegment]:
        """找到掩码中的所有连续段

        Args:
            mask: 二值掩码
            total_duration: 总时长

        Returns:
            连续段列表
        """
        segments = []

        # 找到所有从0到1和从1到0的转换点
        padded_mask = np.concatenate(([0], mask, [0]))
        diff = np.diff(padded_mask)

        # 上升沿（开始）和下降沿（结束）
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # 转换为时间
        time_per_frame = total_duration / len(mask)

        for start_idx, end_idx in zip(starts, ends):
            start_time = start_idx * time_per_frame
            end_time = min(end_idx * time_per_frame, total_duration)
            duration = end_time - start_time

            # 过滤掉极短的段（噪音）
            # 提高最小段长度，避免碎片化
            if duration >= 1.0:  # 至少1秒
                segments.append(InstrumentalSegment(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    segment_type='instrumental',
                    should_split=False
                ))

        return segments

    def _should_split_segment(self, duration: float) -> bool:
        """判断是否应该单独分割此段

        Args:
            duration: 段时长（秒）

        Returns:
            是否应该分割
        """
        return duration >= self.music_threshold

    def create_complete_segment_list(self, instrumental_segments: List[InstrumentalSegment],
                                    audio_duration: float) -> List[InstrumentalSegment]:
        """创建完整的段列表（包括人声段和纯音乐段）

        Args:
            instrumental_segments: 纯音乐段列表
            audio_duration: 音频总时长

        Returns:
            完整的段列表，按时间顺序排列
        """
        if not instrumental_segments:
            # 整个音频都是人声
            return [InstrumentalSegment(
                start_time=0,
                end_time=audio_duration,
                duration=audio_duration,
                segment_type='vocal',
                should_split=False
            )]

        all_segments = []
        current_time = 0

        # 按时间顺序处理
        instrumental_segments.sort(key=lambda x: x.start_time)

        for inst_segment in instrumental_segments:
            # 添加纯音乐段之前的人声段
            if inst_segment.start_time > current_time:
                vocal_duration = inst_segment.start_time - current_time
                all_segments.append(InstrumentalSegment(
                    start_time=current_time,
                    end_time=inst_segment.start_time,
                    duration=vocal_duration,
                    segment_type='vocal',
                    should_split=False
                ))

            # 添加纯音乐段
            all_segments.append(inst_segment)
            current_time = inst_segment.end_time

        # 添加最后的人声段
        if current_time < audio_duration:
            all_segments.append(InstrumentalSegment(
                start_time=current_time,
                end_time=audio_duration,
                duration=audio_duration - current_time,
                segment_type='vocal',
                should_split=False
            ))

        # 验证完整性
        total_segments_duration = sum(s.duration for s in all_segments)
        if abs(total_segments_duration - audio_duration) > 0.01:  # 允许10ms误差
            logger.warning(f"Segment duration mismatch: {total_segments_duration:.3f}s vs {audio_duration:.3f}s")

        return all_segments

    def merge_short_instrumental_segments(self, all_segments: List[InstrumentalSegment]) -> List[Tuple[float, float, str]]:
        """合并不满足阈值的纯音乐段到相邻段

        Args:
            all_segments: 完整的段列表

        Returns:
            合并后的段边界列表 [(start, end, type), ...]
        """
        if not all_segments:
            return []

        merged_segments = []
        i = 0

        while i < len(all_segments):
            current = all_segments[i]

            if current.segment_type == 'instrumental' and not current.should_split:
                # 这个纯音乐段需要合并
                # 找到前后的段进行合并
                if i > 0 and merged_segments:
                    # 与前一段合并
                    prev_start, prev_end, prev_type = merged_segments[-1]
                    merged_segments[-1] = (prev_start, current.end_time, f"{prev_type}_with_music")
                elif i < len(all_segments) - 1:
                    # 与后一段合并
                    next_segment = all_segments[i + 1]
                    merged_segments.append((current.start_time, next_segment.end_time, f"{next_segment.segment_type}_with_music"))
                    i += 1  # 跳过下一段，因为已经合并了
                else:
                    # 只有这一段，保留
                    merged_segments.append((current.start_time, current.end_time, current.segment_type))
            else:
                # 人声段或需要单独分割的纯音乐段
                merged_segments.append((current.start_time, current.end_time, current.segment_type))

            i += 1

        return merged_segments
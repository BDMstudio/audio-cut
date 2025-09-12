#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/core/seamless_splitter.py

import os
import numpy as np
import librosa
import logging
from typing import List, Dict, Optional
from pathlib import Path
import time
import tempfile
import soundfile as sf

from ..utils.config_manager import get_config
from ..utils.audio_processor import AudioProcessor
from .vocal_pause_detector import VocalPauseDetectorV2
from .pure_vocal_pause_detector import PureVocalPauseDetector
from .quality_controller import QualityController
from .enhanced_vocal_separator import EnhancedVocalSeparator

logger = logging.getLogger(__name__)

class SeamlessSplitter:
    """
    [v2.3 统一指挥中心]
    无缝分割器 - 负责编排所有分割模式的唯一引擎。
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate)
        self.pause_detector = VocalPauseDetectorV2(sample_rate)  # 用于smart_split模式
        self.pure_vocal_detector = PureVocalPauseDetector(sample_rate)  # 用于v2.1/v2.2模式
        self.quality_controller = QualityController(sample_rate)
        self.separator = EnhancedVocalSeparator(sample_rate)
        logger.info(f"无缝分割器统一指挥中心初始化完成 (SR: {self.sample_rate}) - 已加载双检测器")

    def split_audio_seamlessly(self, input_path: str, output_dir: str, mode: str = 'v2.2_mdd') -> Dict:
        """
        执行无缝分割的主入口。
        
        Args:
            input_path: 输入文件路径
            output_dir: 输出目录
            mode: 分割模式 ('v2.1', 'v2.2_mdd', 'smart_split', 'vocal_separation')
            
        Returns:
            分割结果信息
        """
        logger.info(f"开始无缝分割: {input_path} (模式: {mode})")
        
        try:
            if mode == 'vocal_separation':
                return self._process_vocal_separation_only(input_path, output_dir)
            elif mode in ['v2.1', 'v2.2_mdd']:
                return self._process_pure_vocal_split(input_path, output_dir, mode)
            elif mode == 'smart_split':
                return self._process_smart_split(input_path, output_dir)
            else:
                logger.warning(f"未知模式 {mode}，使用默认v2.2 MDD模式")
                return self._process_pure_vocal_split(input_path, output_dir, 'v2.2_mdd')
                
        except Exception as e:
            logger.error(f"无缝分割失败: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'input_file': input_path}

    def _load_and_resample_if_needed(self, input_path: str):
        """加载音频并根据需要重采样"""
        original_audio, sr = self.audio_processor.load_audio(input_path, normalize=False)
        if sr != self.sample_rate:
            logger.info(f"音频采样率 {sr}Hz 与目标 {self.sample_rate}Hz 不符，将进行重采样。")
            original_audio = librosa.resample(original_audio, orig_sr=sr, target_sr=self.sample_rate)
        return original_audio

# In src/vocal_smart_splitter/core/seamless_splitter.py

    def _process_pure_vocal_split(self, input_path: str, output_dir: str, mode: str) -> Dict:
        """
        [v2.9 终极修正版] 处理v2.1和v2.2 MDD模式的核心逻辑
        关键修复: 调用新的双音频输入接口，确保音乐分析在 original_audio 上，而VAD检测在 vocal_track 上。
        """
        logger.info(f"[{mode.upper()}] 执行纯人声分割流程...")
        overall_start_time = time.time()

        # 1. 加载音频
        original_audio = self._load_and_resample_if_needed(input_path)
        
        # 2. 高质量人声分离
        logger.info(f"[{mode.upper()}-STEP1] 执行高质量人声分离...")
        separation_start = time.time()
        separation_result = self.separator.separate_for_detection(original_audio)
        separation_time = time.time() - separation_start

        if separation_result.vocal_track is None:
            return {'success': False, 'error': '人声分离失败', 'input_file': input_path}
        
        vocal_track = separation_result.vocal_track
        logger.info(f"[{mode.upper()}-STEP1] 人声分离完成 - 后端: {separation_result.backend_used}, 质量: {separation_result.separation_confidence:.3f}, 耗时: {separation_time:.1f}s")
        
        # 3. 关键修复：使用正确的PureVocalPauseDetector进行停顿检测
        logger.info(f"[{mode.upper()}-STEP2] 使用PureVocalPauseDetector在[纯人声轨道]上进行多维特征检测...")
        
        # 检查是否为v2.2 MDD模式
        enable_mdd = (mode == 'v2.2_mdd')
        vocal_pauses = self.pure_vocal_detector.detect_pure_vocal_pauses(
            vocal_track, 
            enable_mdd_enhancement=enable_mdd,
            original_audio=original_audio  # 提供原始音频用于MDD分析
        )

        if not vocal_pauses:
            return self._create_single_segment_result(original_audio, input_path, output_dir, "未在纯人声中找到停顿")

        # 4. 生成、过滤并分割 (在 vocal_track 上进行最终验证)
        # 转换PureVocalPause到样本点
        cut_points_samples = []
        for p in vocal_pauses:
            if hasattr(p, 'cut_point'):
                cut_points_samples.append(int(p.cut_point * self.sample_rate))
            else:
                # 兜底：使用停顿中心
                center_time = (p.start_time + p.end_time) / 2
                cut_points_samples.append(int(center_time * self.sample_rate))
                
        logger.info(f"[{mode.upper()}-STEP3] 生成{len(cut_points_samples)}个候选分割点")
        final_cut_points = self._finalize_and_filter_cuts(cut_points_samples, vocal_track)
        
        # 使用最终确定的分割点来切割原始音频
        segments = self._split_at_sample_level(original_audio, final_cut_points)
        saved_files = self._save_segments(segments, output_dir)
        
        # 5. 保存完整的分离文件
        input_name = Path(input_path).stem
        full_vocal_file = Path(output_dir) / f"{input_name}_{mode}_vocal_full.wav"
        sf.write(full_vocal_file, vocal_track, self.sample_rate, subtype='PCM_24')
        saved_files.append(str(full_vocal_file))

        if separation_result.instrumental_track is not None:
            full_instrumental_file = Path(output_dir) / f"{input_name}_{mode}_instrumental.wav"
            sf.write(full_instrumental_file, separation_result.instrumental_track, self.sample_rate, subtype='PCM_24')
            saved_files.append(str(full_instrumental_file))

        total_time = time.time() - overall_start_time
        
        return {
            'success': True, 'method': f'pure_vocal_split_{mode}', 'num_segments': len(segments),
            'saved_files': saved_files, 'backend_used': separation_result.backend_used,
            'separation_confidence': separation_result.separation_confidence, 'processing_time': total_time,
            'input_file': input_path, 'output_dir': output_dir
        }

    def _process_smart_split(self, input_path: str, output_dir: str) -> Dict:
        """处理传统智能分割模式"""
        logger.info("[SMART_SPLIT] 执行传统智能分割...")
        original_audio = self._load_and_resample_if_needed(input_path)
        vocal_pauses = self.pause_detector.detect_vocal_pauses(original_audio)
        if not vocal_pauses:
            return self._create_single_segment_result(original_audio, input_path, output_dir, "未找到符合条件的停顿")

        cut_points_samples = [int(p.cut_point * self.sample_rate) for p in vocal_pauses]
        final_cut_points = self._finalize_and_filter_cuts(cut_points_samples, original_audio)
        segments = self._split_at_sample_level(original_audio, final_cut_points)
        saved_files = self._save_segments(segments, output_dir)

        return {'success': True, 'method': 'smart_split', 'num_segments': len(segments), 'saved_files': saved_files, 'input_file': input_path, 'output_dir': output_dir}

    def _process_vocal_separation_only(self, input_path: str, output_dir: str) -> Dict:
        """处理纯人声分离模式"""
        logger.info("[VOCAL_SEPARATION] 执行纯人声分离...")
        start_time = time.time()
        original_audio = self._load_and_resample_if_needed(input_path)
        separation_result = self.separator.separate_for_detection(original_audio)
        
        if separation_result.vocal_track is None:
            return {'success': False, 'error': '人声分离失败', 'input_file': input_path}

        input_name = Path(input_path).stem
        saved_files = []
        vocal_file = Path(output_dir) / f"{input_name}_vocal.wav"
        sf.write(vocal_file, separation_result.vocal_track, self.sample_rate, subtype='PCM_24')
        saved_files.append(str(vocal_file))

        if separation_result.instrumental_track is not None:
            instrumental_file = Path(output_dir) / f"{input_name}_instrumental.wav"
            sf.write(instrumental_file, separation_result.instrumental_track, self.sample_rate, subtype='PCM_24')
            saved_files.append(str(instrumental_file))
            
        processing_time = time.time() - start_time

        return {
            'success': True, 'method': 'vocal_separation_only', 'num_segments': 0, 'saved_files': saved_files,
            'backend_used': separation_result.backend_used, 'separation_confidence': separation_result.separation_confidence,
            'processing_time': processing_time, 'input_file': input_path, 'output_dir': output_dir
        }

    def _finalize_and_filter_cuts(self, cut_points_samples: List[int], audio: np.ndarray) -> List[int]:
        """对切割点进行最终的排序、去重和安全校验"""
        audio_duration_s = len(audio) / self.sample_rate
        cut_times = sorted(list(set([p / self.sample_rate for p in cut_points_samples])))
        validated_times = [t for t in cut_times if self.quality_controller.enforce_quiet_cut(audio, self.sample_rate, t) >= 0]
        # 修复：降低最小间隔限制，允许更密集的切割
        min_interval = get_config('quality_control.min_split_gap', 1.0)  # 从2.0降到1.0秒
        final_times = self.quality_controller.pure_filter_cut_points(
            validated_times, audio_duration_s, min_interval=min_interval
        )
        final_samples = [0] + [int(t * self.sample_rate) for t in final_times] + [len(audio)]
        return sorted(list(set(final_samples)))

    def _split_at_sample_level(self, audio: np.ndarray, final_cut_points: List[int]) -> List[np.ndarray]:
        """执行样本级分割"""
        segments = []
        for i in range(len(final_cut_points) - 1):
            start = final_cut_points[i]
            end = final_cut_points[i+1]
            segments.append(audio[start:end])
        return segments
    
    def _save_segments(self, segments: List[np.ndarray], output_dir: str) -> List[str]:
        """保存分割后的片段"""
        saved_files = []
        for i, segment_audio in enumerate(segments):
            output_path = Path(output_dir) / f"segment_{i+1:03d}.wav"
            sf.write(output_path, segment_audio, self.sample_rate, subtype='PCM_24')
            saved_files.append(str(output_path))
        return saved_files

    def _create_single_segment_result(self, audio: np.ndarray, input_path: str, output_dir: str, reason: str) -> Dict:
        """当无法分割时，创建单个片段的结果"""
        logger.warning(f"{reason}，将输出为单个文件。")
        saved_files = self._save_segments([audio], output_dir)
        return {
            'success': True, 'num_segments': 1, 'saved_files': saved_files,
            'note': reason, 'input_file': input_path, 'output_dir': output_dir
        }
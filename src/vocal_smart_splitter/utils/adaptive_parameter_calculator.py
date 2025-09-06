#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/vocal_smart_splitter/utils/adaptive_parameter_calculator.py
# AI-SUMMARY: BPM驱动的自适应参数计算器，将静态配置转换为动态节拍相关参数

import logging
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from .config_manager import get_config

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveParameters:
    """自适应参数结果"""
    # 核心检测参数
    min_pause_duration: float         # 最小停顿时长(秒)
    speech_pad_ms: float             # 语音边界保护(毫秒)
    vad_threshold: float             # VAD阈值
    min_split_gap: float             # 最小分割间隙(秒)
    consecutive_silence_frames: int   # 连续静音帧数要求
    
    # 节拍相关参数
    beat_interval: float             # 节拍间隔(秒)
    category: str                    # 速度类别
    compensation_factor: float       # 复杂度补偿系数
    
    # 上下文信息
    bpm_value: float                # 原始BPM值
    complexity_score: float         # 复杂度评分
    instrument_count: int           # 乐器数量
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'min_pause_duration': self.min_pause_duration,
            'speech_pad_ms': self.speech_pad_ms,
            'vad_threshold': self.vad_threshold,
            'min_split_gap': self.min_split_gap,
            'consecutive_silence_frames': self.consecutive_silence_frames,
            'beat_interval': self.beat_interval,
            'category': self.category,
            'compensation_factor': self.compensation_factor,
            'bpm_value': self.bpm_value,
            'complexity_score': self.complexity_score,
            'instrument_count': self.instrument_count
        }

class AdaptiveParameterCalculator:
    """BPM驱动的自适应参数计算器
    
    根据音乐的BPM、复杂度和乐器数量，动态计算所有与时间相关的参数，
    替换config.yaml中的静态固定值，解决"个别分割不准确"问题。
    
    核心理念：
    - 所有时间参数都基于音乐节拍计算
    - 考虑编曲复杂度对检测难度的影响
    - 乐器数量越多，需要越长的停顿确认时间
    """
    
    def __init__(self):
        """初始化自适应参数计算器"""
        self.logger = logging.getLogger(__name__)
        
        # 从配置加载BPM分类阈值
        self._load_bpm_categories()
        
        # 加载基础节拍参数
        self._load_base_parameters()
        
        # 加载复杂度补偿参数
        self._load_complexity_compensation()
        
        self.logger.info("BPM驱动自适应参数计算器初始化完成")
    
    
    def _load_bpm_categories(self):
        """加载BPM分类阈值 - 修正方法名"""
        # 从todo-phase.md中的设计规范加载
        self.tempo_categories = {
            'slow': {'min': 0, 'max': 70, 'label': '巴拉德/民谣'},
            'medium': {'min': 70, 'max': 100, 'label': '流行/摇滚'},
            'fast': {'min': 100, 'max': 140, 'label': '舞曲/电音'},
            'very_fast': {'min': 140, 'max': 999, 'label': 'Drum&Bass'}
        }
        
        # 也可以从配置文件覆盖
        try:
            config_categories = get_config('bpm_adaptive_core.tempo_categories')
            if config_categories:
                self.tempo_categories.update(config_categories)
        except:
            pass  # 使用默认值
    
    def _load_base_parameters(self):
        """加载基础节拍参数"""
        # 换气停顿(以拍为单位)
        self.pause_duration_beats = {
            'slow': get_config('bpm_adaptive_core.pause_duration_beats.slow', 1.5),
            'medium': get_config('bpm_adaptive_core.pause_duration_beats.medium', 1.0),
            'fast': get_config('bpm_adaptive_core.pause_duration_beats.fast', 0.75),
            'very_fast': get_config('bpm_adaptive_core.pause_duration_beats.very_fast', 0.5)
        }
        
        # 边界保护(以拍为单位)
        self.speech_pad_beats = {
            'slow': get_config('bpm_adaptive_core.speech_pad_beats.slow', 0.8),
            'medium': get_config('bpm_adaptive_core.speech_pad_beats.medium', 0.5),
            'fast': get_config('bpm_adaptive_core.speech_pad_beats.fast', 0.3),
            'very_fast': get_config('bpm_adaptive_core.speech_pad_beats.very_fast', 0.2)
        }
        
        # 分割间隙(以乐句为单位 - 拍数)
        self.split_gap_phrases = {
            'slow': get_config('bpm_adaptive_core.split_gap_phrases.slow', 4),
            'medium': get_config('bpm_adaptive_core.split_gap_phrases.medium', 4),
            'fast': get_config('bpm_adaptive_core.split_gap_phrases.fast', 8),
            'very_fast': get_config('bpm_adaptive_core.split_gap_phrases.very_fast', 8)
        }
    
    def _load_complexity_compensation(self):
        """加载复杂度补偿参数"""
        self.complexity_compensation = {
            'base_factor': get_config('bpm_adaptive_core.complexity_compensation.base_factor', 1.0),
            'complexity_boost': get_config('bpm_adaptive_core.complexity_compensation.complexity_boost', 0.5),
            'instrument_boost': get_config('bpm_adaptive_core.complexity_compensation.instrument_boost', 0.15),
            'min_instruments': get_config('bpm_adaptive_core.complexity_compensation.min_instruments', 2)
        }
    
    def calculate_all_parameters(self, bpm: float, complexity: float, 
                               instrument_count: int) -> AdaptiveParameters:
        """根据BPM和复杂度计算所有参数
        
        Args:
            bpm: 检测到的BPM值
            complexity: 编曲复杂度 (0-1)
            instrument_count: 乐器数量
            
        Returns:
            计算得出的自适应参数对象
        """
        self.logger.info(f"计算自适应参数: BPM={bpm:.1f}, 复杂度={complexity:.3f}, 乐器={instrument_count}")
        
        # 1. 确定节拍类别
        category = self._categorize_tempo(bpm)
        beat_interval = 60.0 / bpm
        
        self.logger.debug(f"音乐分类: {category}, 节拍间隔: {beat_interval:.3f}s")
        
        # 2. 获取基础参数
        base_pause_beats = self.pause_duration_beats[category]
        base_pad_beats = self.speech_pad_beats[category]
        base_gap_phrases = self.split_gap_phrases[category]
        
        # 3. 复杂度补偿计算
        complexity_factor = 1.0 + (complexity * self.complexity_compensation['complexity_boost'])
        
        instrument_factor = 1.0
        if instrument_count > self.complexity_compensation['min_instruments']:
            extra_instruments = instrument_count - self.complexity_compensation['min_instruments']
            instrument_factor = 1.0 + (extra_instruments * self.complexity_compensation['instrument_boost'])
        
        total_compensation = complexity_factor * instrument_factor
        
        self.logger.debug(f"补偿系数: 复杂度×{complexity_factor:.2f}, 乐器×{instrument_factor:.2f}, 总计×{total_compensation:.2f}")
        
        # 4. 最终参数计算
        final_pause_duration = base_pause_beats * beat_interval * total_compensation
        final_speech_pad_ms = base_pad_beats * beat_interval * 1000  # 转换为毫秒
        final_split_gap = base_gap_phrases * beat_interval  # 乐句间隙
        
        # 5. VAD阈值动态调整
        base_vad_threshold = get_config('vocal_pause_splitting.voice_threshold', 0.5)
        complexity_threshold_boost = complexity * 0.3  # 复杂度高时提高阈值
        final_vad_threshold = min(0.8, base_vad_threshold + complexity_threshold_boost)
        
        # 6. 连续静音帧数 (基于30ms帧长)
        frame_duration = 0.03  # 30ms
        consecutive_frames = int(final_pause_duration / frame_duration)
        
        # 7. 构建结果对象
        adaptive_params = AdaptiveParameters(
            min_pause_duration=round(final_pause_duration, 3),
            speech_pad_ms=round(final_speech_pad_ms, 1),
            vad_threshold=round(final_vad_threshold, 3),
            min_split_gap=round(final_split_gap, 2),
            consecutive_silence_frames=consecutive_frames,
            beat_interval=round(beat_interval, 3),
            category=category,
            compensation_factor=round(total_compensation, 3),
            bpm_value=round(bpm, 1),
            complexity_score=round(complexity, 3),
            instrument_count=instrument_count
        )
        
        # 8. 记录计算结果
        self._log_calculation_results(adaptive_params, base_pause_beats, base_pad_beats, base_gap_phrases)
        
        return adaptive_params
    
    def _categorize_tempo(self, bpm: float) -> str:
        """BPM分类
        
        Args:
            bpm: 节拍速度
            
        Returns:
            节拍类别字符串
        """
        for category, thresholds in self.tempo_categories.items():
            if thresholds['min'] <= bpm < thresholds['max']:
                return category
        
        # 处理边界情况
        if bpm < 50:
            return 'slow'  # 极慢当作慢歌处理
        else:
            return 'very_fast'  # 超快当作极快处理
    
    def _log_calculation_results(self, params: AdaptiveParameters, 
                               base_pause_beats: float, base_pad_beats: float, 
                               base_gap_phrases: float):
        """记录计算结果日志"""
        self.logger.info("=== BPM自适应参数计算结果 ===")
        self.logger.info(f"音乐特征: {params.bpm_value} BPM ({params.category})")
        self.logger.info(f"节拍间隔: {params.beat_interval}s")
        self.logger.info(f"复杂度补偿: ×{params.compensation_factor}")
        self.logger.info("")
        self.logger.info("核心参数对比:")
        self.logger.info(f"  停顿时长: {base_pause_beats}拍 → {params.min_pause_duration}s")
        self.logger.info(f"  边界保护: {base_pad_beats}拍 → {params.speech_pad_ms}ms")
        self.logger.info(f"  分割间隙: {base_gap_phrases}拍 → {params.min_split_gap}s")
        self.logger.info(f"  VAD阈值: → {params.vad_threshold}")
        self.logger.info(f"  静音帧数: → {params.consecutive_silence_frames}")
    
    def get_static_override_parameters(self, params: AdaptiveParameters) -> Dict[str, any]:
        """获取用于覆盖静态配置的参数字典
        
        Args:
            params: 自适应参数对象
            
        Returns:
            可直接用于覆盖config.yaml静态值的参数字典
        """
        return {
            # advanced_vad 参数覆盖
            'advanced_vad.silero_min_silence_ms': int(params.min_pause_duration * 1000),
            'advanced_vad.silero_speech_pad_ms': int(params.speech_pad_ms),
            'advanced_vad.silero_min_consecutive_silence_frames': params.consecutive_silence_frames,
            
            # vocal_pause_splitting 参数覆盖
            'vocal_pause_splitting.min_pause_duration': params.min_pause_duration,
            'vocal_pause_splitting.voice_threshold': params.vad_threshold,
            
            # quality_control 参数覆盖
            'quality_control.min_pause_at_split': params.min_pause_duration,
            'quality_control.min_split_gap': params.min_split_gap,
        }
    
    def apply_dynamic_parameters(self, params: AdaptiveParameters, 
                               config_overrides: Optional[Dict] = None) -> Dict:
        """应用动态参数到配置系统
        
        Args:
            params: 自适应参数对象
            config_overrides: 额外的配置覆盖
            
        Returns:
            完整的动态配置字典
        """
        # 获取基础覆盖参数
        override_params = self.get_static_override_parameters(params)
        
        # 合并额外覆盖
        if config_overrides:
            override_params.update(config_overrides)
        
        # 应用到配置管理器
        from .config_manager import set_runtime_config
        try:
            set_runtime_config(override_params)
            self.logger.info("动态参数已应用到配置系统")
        except AttributeError:
            self.logger.warning("配置管理器不支持运行时配置设置，需要扩展config_manager.py")
        
        return override_params
    
    def validate_parameters(self, params: AdaptiveParameters) -> Dict[str, any]:
        """验证参数合理性
        
        Args:
            params: 自适应参数对象
            
        Returns:
            验证结果字典
        """
        issues = []
        warnings = []
        
        # 1. 停顿时长验证
        if params.min_pause_duration < 0.5:
            issues.append(f"停顿时长过短: {params.min_pause_duration}s < 0.5s")
        elif params.min_pause_duration > 3.0:
            warnings.append(f"停顿时长较长: {params.min_pause_duration}s > 3.0s")
        
        # 2. VAD阈值验证
        if params.vad_threshold < 0.2 or params.vad_threshold > 0.8:
            issues.append(f"VAD阈值超出合理范围: {params.vad_threshold}")
        
        # 3. 节拍间隔验证
        if params.beat_interval < 0.3 or params.beat_interval > 2.0:
            warnings.append(f"节拍间隔异常: {params.beat_interval}s (BPM: {params.bpm_value})")
        
        # 4. 补偿系数验证
        if params.compensation_factor > 3.0:
            warnings.append(f"复杂度补偿过高: ×{params.compensation_factor}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'validation_passed': len(issues) == 0
        }

def create_adaptive_calculator() -> AdaptiveParameterCalculator:
    """工厂函数：创建自适应参数计算器实例
    
    Returns:
        配置好的AdaptiveParameterCalculator实例
    """
    return AdaptiveParameterCalculator()

# 便于测试的示例函数
def test_adaptive_calculation():
    """测试自适应参数计算"""
    calculator = create_adaptive_calculator()
    
    # 测试不同风格音乐
    test_cases = [
        {'bpm': 65, 'complexity': 0.3, 'instruments': 2, 'name': '慢歌民谣'},
        {'bpm': 85, 'complexity': 0.5, 'instruments': 4, 'name': '流行歌曲'},
        {'bpm': 128, 'complexity': 0.7, 'instruments': 6, 'name': '舞曲'},
        {'bpm': 175, 'complexity': 0.8, 'instruments': 8, 'name': '电子乐'}
    ]
    
    for case in test_cases:
        print(f"\n=== {case['name']} 测试 ===")
        params = calculator.calculate_all_parameters(
            case['bpm'], case['complexity'], case['instruments']
        )
        
        # 验证参数
        validation = calculator.validate_parameters(params)
        print(f"验证结果: {'通过' if validation['valid'] else '失败'}")
        
        if validation['warnings']:
            print("警告:", validation['warnings'])
        if validation['issues']:
            print("问题:", validation['issues'])

if __name__ == "__main__":
    test_adaptive_calculation()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/__init__.py
# AI-SUMMARY: 包初始化 - 暴露无缝分割核心组件
"""
Vocal Smart Splitter
====================

精简后的智能人声分割工具，聚焦两条主线：
- 纯人声分离 (vocal_separation)
- v2.2 MDD 增强的纯人声停顿分割

典型用法::

    from vocal_smart_splitter.core import SeamlessSplitter

    splitter = SeamlessSplitter(sample_rate=44100)
    splitter.split_audio_seamlessly('input.wav', 'output_dir', mode='v2.2_mdd')
"""

__version__ = "2.3.0"
__author__ = "AI Assistant"
__description__ = "智能人声分割器 - 纯人声分离与 v2.2 MDD 无缝分割"

from .core.seamless_splitter import SeamlessSplitter
from .core.enhanced_vocal_separator import EnhancedVocalSeparator
from .core.pure_vocal_pause_detector import PureVocalPauseDetector
from .core.quality_controller import QualityController

from .utils.config_manager import ConfigManager, get_config_manager
from .utils.audio_processor import AudioProcessor
from .utils.feature_extractor import FeatureExtractor

__all__ = [
    'SeamlessSplitter',
    'EnhancedVocalSeparator',
    'PureVocalPauseDetector',
    'QualityController',
    'ConfigManager',
    'get_config_manager',
    'AudioProcessor',
    'FeatureExtractor',
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/__init__.py
# AI-SUMMARY: 智能人声分割器包初始化文件

"""
智能人声分割器 (Vocal Smart Splitter)

一个基于人声内容和换气停顿的智能音频分割工具，专门针对歌曲进行优化。

主要特性:
- 高质量人声分离
- 精确的换气和停顿检测
- 基于内容的智能分割决策
- 严格的质量控制
- 模块化架构设计

使用示例:
    from vocal_smart_splitter import VocalSmartSplitter
    
    splitter = VocalSmartSplitter()
    result = splitter.split_audio('input.mp3', 'output_dir')
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "智能人声分割器 - 基于人声内容和换气停顿的智能音频分割工具"

# 导入主要类
from .main import VocalSmartSplitter

# 导入核心模块
from .core.vocal_separator import VocalSeparator
from .core.breath_detector import BreathDetector
from .core.content_analyzer import ContentAnalyzer
from .core.smart_splitter import SmartSplitter
from .core.quality_controller import QualityController

# 导入工具模块
from .utils.config_manager import ConfigManager, get_config_manager
from .utils.audio_processor import AudioProcessor
from .utils.feature_extractor import FeatureExtractor

__all__ = [
    'VocalSmartSplitter',
    'VocalSeparator',
    'BreathDetector', 
    'ContentAnalyzer',
    'SmartSplitter',
    'QualityController',
    'ConfigManager',
    'get_config_manager',
    'AudioProcessor',
    'FeatureExtractor'
]

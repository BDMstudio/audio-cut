#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/__init__.py
# AI-SUMMARY: 包装模块元信息，并通过惰性加载暴露核心组件。

"""Top-level package for the vocal smart splitter toolkit."""

__version__ = "2.4.0"
__author__ = "BDM Team"
__description__ = "智能人声分割器 - 纯人声分离与 Librosa onset 分割模式"

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

_LAZY_IMPORTS = {
    'SeamlessSplitter': 'vocal_smart_splitter.core.seamless_splitter.SeamlessSplitter',
    'EnhancedVocalSeparator': 'vocal_smart_splitter.core.enhanced_vocal_separator.EnhancedVocalSeparator',
    'PureVocalPauseDetector': 'vocal_smart_splitter.core.pure_vocal_pause_detector.PureVocalPauseDetector',
    'QualityController': 'vocal_smart_splitter.core.quality_controller.QualityController',
    'ConfigManager': 'vocal_smart_splitter.utils.config_manager.ConfigManager',
    'get_config_manager': 'vocal_smart_splitter.utils.config_manager.get_config_manager',
    'AudioProcessor': 'vocal_smart_splitter.utils.audio_processor.AudioProcessor',
    'FeatureExtractor': 'vocal_smart_splitter.utils.feature_extractor.FeatureExtractor',
}


def __getattr__(name: str):
    """Lazily import heavy modules on first access."""

    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'vocal_smart_splitter' has no attribute {name!r}")
    module_name, attr_name = target.rsplit('.', 1)
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(__all__ + list(globals().keys())))


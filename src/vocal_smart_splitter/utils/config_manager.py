#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/utils/config_manager.py
# AI-SUMMARY: 配置管理器，负责加载和管理所有配置参数

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """智能人声分割器配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 默认配置文件路径
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / 'config.yaml'
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
        logger.info(f"配置管理器初始化完成，配置文件: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info("配置文件加载成功")
            return config
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        required_sections = [
            'audio', 'pure_vocal_detection', 'musical_dynamic_density',
            'quality_control', 'vocal_separation', 'output', 'logging'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必需的节: {section}")
        
        # 验证关键参数
        self._validate_audio_config()
        self._validate_quality_config()
        
        logger.info("配置参数验证通过")
    
    def _validate_audio_config(self):
        """验证音频配置"""
        audio_config = self.config['audio']
        
        if audio_config['sample_rate'] not in [16000, 22050, 44100, 48000]:
            raise ValueError(f"不支持的采样率: {audio_config['sample_rate']}")
        
        if audio_config['channels'] not in [1, 2]:
            raise ValueError(f"不支持的声道数: {audio_config['channels']}")
    
    def _validate_quality_config(self):
        """验证质量控制配置"""
        quality_config = self.config['quality_control']
        
        if not (0 < quality_config['min_vocal_content_ratio'] < 1):
            raise ValueError("人声内容比例必须在0-1之间")
        
        if not (0 < quality_config['max_silence_ratio'] < 1):
            raise ValueError("静音比例必须在0-1之间")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key_path: 配置键路径，如 'audio.sample_rate'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"配置键不存在: {key_path}")
    
    def set(self, key_path: str, value: Any):
        """设置配置值
        
        Args:
            key_path: 配置键路径
            value: 配置值
        """
        keys = key_path.split('.')
        config = self.config
        
        # 导航到父级
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置值
        config[keys[-1]] = value
        logger.debug(f"配置更新: {key_path} = {value}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置节
        
        Args:
            section: 节名称
            
        Returns:
            配置节字典
        """
        if section not in self.config:
            raise KeyError(f"配置节不存在: {section}")
        
        return self.config[section].copy()
    
    def save_config(self, output_path: Optional[str] = None):
        """保存配置到文件
        
        Args:
            output_path: 输出路径，如果为None则覆盖原文件
        """
        if output_path is None:
            output_path = self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            logger.info(f"配置文件已保存: {output_path}")
            
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")
            raise
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        log_config = self.get_section('logging')
        
        # 转换日志级别
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        log_config['level'] = level_map.get(log_config['level'], logging.INFO)
        return log_config
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return f"ConfigManager(config_path={self.config_path})"
    
    def __repr__(self) -> str:
        return self.__str__()

# 全局配置管理器实例
_config_manager = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """获取全局配置管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigManager实例
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager

def get_config(key_path: str, default: Any = None) -> Any:
    """快捷方式：获取配置值
    
    Args:
        key_path: 配置键路径
        default: 默认值
        
    Returns:
        配置值
    """
    return get_config_manager().get(key_path, default)

def set_runtime_config(config_overrides: Dict[str, Any]):
    """设置运行时配置覆盖（用于动态参数）
    
    Args:
        config_overrides: 配置覆盖字典，键为配置路径，值为新值
    """
    config_manager = get_config_manager()
    
    for key_path, value in config_overrides.items():
        config_manager.set(key_path, value)
    
    logger.info(f"已设置 {len(config_overrides)} 个运行时配置覆盖")

def reset_runtime_config():
    """重置运行时配置（重新加载原始配置文件）"""
    global _config_manager
    if _config_manager:
        original_path = _config_manager.config_path
        _config_manager = ConfigManager(str(original_path))
        logger.info("运行时配置已重置")

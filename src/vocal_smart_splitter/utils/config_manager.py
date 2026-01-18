#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/utils/config_manager.py
# AI-SUMMARY: 配置管理器，负责加载和管理所有配置参数（支持 unified.yaml 统一配置）

import os
import yaml
import logging
from typing import Dict, Any, Optional

from audio_cut.config.derive import (
    build_legacy_overrides,
    load_default_schema,
    merge_schema,
    schema_from_mapping,
    is_v3_schema,
)
from pathlib import Path

logger = logging.getLogger(__name__)

# 项目根目录
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

_UNSET = object()


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是字典结构: {path}")
    return data


def _deep_merge_dict(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = dict(base)
    if not override:
        return result
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _merge_schema_v3(base: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Apply schema v3 overrides on top of legacy configuration."""

    default_schema = load_default_schema()
    try:
        if mapping.get("version") == 3:
            schema = merge_schema(default_schema, mapping)
        else:
            partial = mapping.get("overrides", mapping)
            schema = merge_schema(default_schema, partial)
    except Exception:
        schema = schema_from_mapping(mapping)

    overrides = build_legacy_overrides(schema)
    # Ensure meta fields co-exist even when legacy base misses them.
    meta = overrides.get("meta", {})
    meta.setdefault("schema_source", mapping.get("name", schema.name))
    overrides["meta"] = meta

    return _deep_merge_dict(base, overrides)


def _parse_env_value(raw: str) -> Any:
    value = raw.strip()
    lower = value.lower()
    if lower in {'true', 'false'}:
        return lower == 'true'
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        return raw


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(config)
    prefix = 'VSS__'
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = [part.lower() for part in key[len(prefix):].split('__') if part]
        if not parts:
            continue
        value = _parse_env_value(raw)
        cursor = result
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return result

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
        """加载配置文件

        加载优先级（从低到高）：
        1. config/unified.yaml (统一配置入口，必须存在)
        2. VSS_EXTERNAL_CONFIG_PATH 环境变量指定的配置
        3. 显式指定的 config_path
        4. VSS__* 环境变量覆盖
        """
        try:
            # 1. 加载统一配置文件 (config/unified.yaml) 作为基础配置
            unified_path = _PROJECT_ROOT / 'config' / 'unified.yaml'
            if not unified_path.exists():
                raise FileNotFoundError(f"统一配置文件不存在: {unified_path}")
            config = self._load_unified_config(unified_path)
            logger.info(f"[ConfigManager] 加载统一配置: {unified_path}")

            # 3. 加载外部配置 (VSS_EXTERNAL_CONFIG_PATH)
            external_path = os.environ.get('VSS_EXTERNAL_CONFIG_PATH')
            if external_path:
                external_raw = _load_yaml_file(Path(external_path))
                if is_v3_schema(external_raw) or (
                    isinstance(external_raw, dict)
                    and is_v3_schema(external_raw.get('overrides', {}))
                ):
                    config = _merge_schema_v3(config, external_raw)
                else:
                    config = _deep_merge_dict(config, external_raw)
                logger.info(f"[ConfigManager] 加载外部配置: {external_path}")

            # 3. 加载显式指定的配置文件
            explicit_path = Path(self.config_path)
            if explicit_path.resolve() != unified_path.resolve():
                if explicit_path.exists():
                    explicit_raw = _load_yaml_file(explicit_path)
                    if is_v3_schema(explicit_raw) or (
                        isinstance(explicit_raw, dict)
                        and is_v3_schema(explicit_raw.get('overrides', {}))
                    ):
                        config = _merge_schema_v3(config, explicit_raw)
                    else:
                        config = _deep_merge_dict(config, explicit_raw)
                    logger.info(f"[ConfigManager] 加载显式配置: {explicit_path}")

            # 4. 应用环境变量覆盖
            config = _apply_env_overrides(config)

            logger.info("配置文件加载成功")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise

    def _load_unified_config(self, path: Path) -> Dict[str, Any]:
        """加载统一配置文件并扁平化 v2_mdd 嵌套结构

        Args:
            path: unified.yaml 文件路径

        Returns:
            扁平化后的配置字典
        """
        unified_raw = _load_yaml_file(path)
        return self._merge_unified_config({}, unified_raw)

    def _merge_unified_config(self, base: Dict[str, Any], unified: Dict[str, Any]) -> Dict[str, Any]:
        """合并统一配置文件，处理特殊的嵌套结构

        unified.yaml 结构：
        - global: 全局设置
        - audio/gpu_pipeline/output/logging: 直接合并
        - v2_mdd: 映射到 pure_vocal_detection, quality_control 等
        - librosa_onset: 存储为独立节
        """
        result = dict(base)

        # 直接合并的顶级键
        direct_merge_keys = [
            'global', 'audio', 'gpu_pipeline', 'enhanced_separation',
            'output', 'logging', 'analysis', 'vocal_separation',
            'vocal_pause_splitting', 'bpm_adaptive_core', 'hybrid_mdd'
        ]

        for key in direct_merge_keys:
            if key in unified:
                if key in result and isinstance(result[key], dict) and isinstance(unified[key], dict):
                    result[key] = _deep_merge_dict(result[key], unified[key])
                else:
                    result[key] = unified[key]

        # v2_mdd 节的特殊处理：映射到对应的顶级键
        if 'v2_mdd' in unified:
            v2_mdd = unified['v2_mdd']

            # 映射 v2_mdd.pure_vocal_detection -> pure_vocal_detection
            if 'pure_vocal_detection' in v2_mdd:
                if 'pure_vocal_detection' in result:
                    result['pure_vocal_detection'] = _deep_merge_dict(
                        result['pure_vocal_detection'], v2_mdd['pure_vocal_detection']
                    )
                else:
                    result['pure_vocal_detection'] = v2_mdd['pure_vocal_detection']

            # 映射 v2_mdd.musical_dynamic_density -> musical_dynamic_density
            if 'musical_dynamic_density' in v2_mdd:
                if 'musical_dynamic_density' in result:
                    result['musical_dynamic_density'] = _deep_merge_dict(
                        result['musical_dynamic_density'], v2_mdd['musical_dynamic_density']
                    )
                else:
                    result['musical_dynamic_density'] = v2_mdd['musical_dynamic_density']

            # 映射 v2_mdd.advanced_vad -> advanced_vad
            if 'advanced_vad' in v2_mdd:
                if 'advanced_vad' in result:
                    result['advanced_vad'] = _deep_merge_dict(
                        result['advanced_vad'], v2_mdd['advanced_vad']
                    )
                else:
                    result['advanced_vad'] = v2_mdd['advanced_vad']

            # 映射 v2_mdd.quality_control -> quality_control
            if 'quality_control' in v2_mdd:
                if 'quality_control' in result:
                    result['quality_control'] = _deep_merge_dict(
                        result['quality_control'], v2_mdd['quality_control']
                    )
                else:
                    result['quality_control'] = v2_mdd['quality_control']

            # 映射 v2_mdd.segment_layout -> segment_layout
            if 'segment_layout' in v2_mdd:
                if 'segment_layout' in result:
                    result['segment_layout'] = _deep_merge_dict(
                        result['segment_layout'], v2_mdd['segment_layout']
                    )
                else:
                    result['segment_layout'] = v2_mdd['segment_layout']

        # librosa_onset 节保持独立存储
        if 'librosa_onset' in unified:
            result['librosa_onset'] = unified['librosa_onset']

        return result

    
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
    
    def get(self, key_path: str, default: Any = _UNSET) -> Any:
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
            if default is not _UNSET:
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

def get_config(key_path: str, default: Any = _UNSET) -> Any:
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


def get_librosa_onset_config() -> Dict[str, Any]:
    """获取 librosa_onset 模式的配置

    优先级：环境变量 > unified.yaml > 默认值

    Returns:
        librosa_onset 配置字典
    """
    config_manager = get_config_manager()

    # 从配置文件获取基础配置
    base_config = config_manager.get('librosa_onset', {})

    # 构建最终配置，应用环境变量覆盖
    result = {
        'use_vocal_separation': _get_with_env_override(
            base_config.get('use_vocal_separation', True),
            'AUDIOCUT_LIBROSA_USE_VOCAL',
            lambda x: x.lower() == 'true'
        ),
        'silence': {
            'threshold_db': _get_with_env_override(
                base_config.get('silence', {}).get('threshold_db', -40),
                'AUDIOCUT_SILENCE_THRESHOLD_DB',
                float
            ),
            'min_duration': _get_with_env_override(
                base_config.get('silence', {}).get('min_duration', 0.3),
                'AUDIOCUT_SILENCE_MIN_DURATION',
                float
            ),
        },
        'density': _get_with_env_override(
            base_config.get('density', 'medium'),
            'AUDIOCUT_DENSITY',
            str
        ),
        'density_custom': base_config.get('density_custom', {
            'enable': False,
            'verse_bars': 4,
            'chorus_bars': 2,
        }),
        'energy_analysis': base_config.get('energy_analysis', {
            'hop_length': 512,
            'chorus_percentile': 60,
            'chorus_peak_percentile': 80,
        }),
        'beat': base_config.get('beat', {
            'time_signature': 4,
        }),
    }

    return result


def _get_with_env_override(default_value: Any, env_key: str, converter=None) -> Any:
    """获取配置值，优先使用环境变量覆盖

    Args:
        default_value: 默认值
        env_key: 环境变量名
        converter: 类型转换函数

    Returns:
        最终配置值
    """
    env_value = os.environ.get(env_key)
    if env_value is not None:
        if converter:
            try:
                return converter(env_value)
            except (ValueError, TypeError):
                logger.warning(f"环境变量 {env_key}={env_value} 转换失败，使用默认值")
                return default_value
        return env_value
    return default_value


def get_hybrid_mdd_config(density_override: Optional[str] = None) -> Dict[str, Any]:
    """获取 hybrid_mdd 模式的配置

    优先级：density_override > 环境变量 > unified.yaml > 默认值

    Args:
        density_override: 用户交互时选择的密度 (low/medium/high)

    Returns:
        hybrid_mdd 配置字典
    """
    config_manager = get_config_manager()

    # 从配置文件获取基础配置
    base_config = config_manager.get('hybrid_mdd', {})

    # 获取密度设置
    density = density_override or _get_with_env_override(
        base_config.get('beat_cut_density', 'medium'),
        'AUDIOCUT_HYBRID_DENSITY',
        str
    )

    # 获取密度预设
    density_presets = base_config.get('density_presets', {})
    preset = density_presets.get(density, density_presets.get('medium', {}))

    # 构建最终配置
    result = {
        'density': density,
        'enable_beat_cuts': preset.get('enable_beat_cuts', True),
        'energy_percentile': preset.get('energy_percentile', 70),
        'bars_per_cut': preset.get('bars_per_cut', 2),
        'lib_alignment': _get_with_env_override(
            base_config.get('lib_alignment', 'mdd_start'),
            'AUDIOCUT_HYBRID_LIB_ALIGNMENT',
            str
        ),
        # Plan C (snap_to_beat) specific options
        'snap_tolerance_ms': _get_with_env_override(
            base_config.get('snap_tolerance_ms', 300),
            'AUDIOCUT_SNAP_TOLERANCE_MS',
            int
        ),
        'vad_protection': _get_with_env_override(
            base_config.get('vad_protection', True),
            'AUDIOCUT_VAD_PROTECTION',
            lambda x: str(x).lower() in ('true', '1', 'yes')
        ),
        'beat_detection': {
            'hop_length': base_config.get('beat_detection', {}).get('hop_length', 512),
            'time_signature': base_config.get('beat_detection', {}).get('time_signature', 4),
            'snap_to_pause_ms': base_config.get('beat_detection', {}).get('snap_to_pause_ms', 300),
        },
        'labeling': {
            'lib_suffix': base_config.get('labeling', {}).get('lib_suffix', '_lib'),
        },
    }

    return result

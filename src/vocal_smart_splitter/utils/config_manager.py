#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/utils/config_manager.py
# AI-SUMMARY: 统一管理配置加载、分层合并与派生，支持 v3 精简 schema 与旧版兼容输出。

from __future__ import annotations

import copy
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from audio_cut.config import (
    compose_app_config,
    load_schema_config,
    merge_schema_override,
    resolve_legacy_config,
)

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _is_schema_payload(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get('version') == 3:
        return True
    if 'defaults' in payload and isinstance(payload['defaults'], dict) and 'detection' in payload['defaults']:
        return True
    if 'detection' in payload and 'min_pause_s' in payload['detection']:
        return True
    return False


def _deep_merge_dicts(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    target = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict):
            node = target.get(key)
            if isinstance(node, dict):
                target[key] = _deep_merge_dicts(node, value)
            else:
                target[key] = copy.deepcopy(value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _set_by_path(config: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split('.')
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _parse_env_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {'true', 'false'}:
        return lowered == 'true'
    try:
        if '.' in lowered:
            return float(lowered)
        return int(lowered)
    except ValueError:
        return raw


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    prefix = 'VSS__'
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().replace('__', '.')
        parsed = _parse_env_value(value)
        _set_by_path(config, path, parsed)


class ConfigManager:
    """智能人声分割器配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        current_dir = Path(__file__).parent.parent
        self.base_config_path = current_dir / 'config.yaml'
        self.explicit_config_path = Path(config_path) if config_path else None
        self._schema_data: Optional[Dict[str, Any]] = None
        self._app_config = None
        self.config = self._compose_effective_config()
        self._validate_config()
        logger.info(
            "配置管理器初始化完成 (base=%s, explicit=%s)",
            self.base_config_path,
            self.explicit_config_path or 'None',
        )

    def _compose_effective_config(self) -> Dict[str, Any]:
        base_payload = _load_yaml(self.base_config_path)
        schema_data: Optional[Dict[str, Any]] = None
        effective_config: Dict[str, Any]

        if isinstance(base_payload, dict) and base_payload.get('defaults_from'):
            schema_file = (self.base_config_path.parent / base_payload['defaults_from']).resolve()
            schema_data = load_schema_config(schema_file)
            overrides = copy.deepcopy(base_payload)
            overrides.pop('defaults_from', None)
            overrides.pop('version', None)
            if overrides:
                merge_schema_override(schema_data, overrides)
        elif _is_schema_payload(base_payload):
            schema_data = load_schema_config(self.base_config_path)
        else:
            effective_config = copy.deepcopy(base_payload)

        # 收集覆盖层
        schema_overrides: list[Dict[str, Any]] = []
        legacy_overrides: list[Dict[str, Any]] = []

        external_path = os.environ.get('VSS_EXTERNAL_CONFIG_PATH')
        if external_path:
            ext_payload = _load_yaml(Path(external_path))
            (schema_overrides if _is_schema_payload(ext_payload) else legacy_overrides).append(ext_payload)

        if self.explicit_config_path:
            explicit_payload = _load_yaml(self.explicit_config_path)
            (schema_overrides if _is_schema_payload(explicit_payload) else legacy_overrides).append(explicit_payload)

        if schema_data is not None:
            for override in schema_overrides:
                merge_schema_override(schema_data, override)
            app_config = compose_app_config(schema_data)
            legacy = resolve_legacy_config(app_config)
            self._schema_data = schema_data
            self._app_config = app_config
            effective_config = legacy
        else:
            self._schema_data = None
            self._app_config = None

        for override in legacy_overrides:
            effective_config = _deep_merge_dicts(effective_config, override)

        _apply_env_overrides(effective_config)
        return effective_config

    def _validate_config(self) -> None:
        required_sections = [
            'audio', 'pure_vocal_detection', 'musical_dynamic_density',
            'quality_control', 'vocal_separation', 'output', 'logging'
        ]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必需的节: {section}")
        self._validate_audio_config()
        self._validate_quality_config()
        logger.info("配置参数验证通过")

    def _validate_audio_config(self) -> None:
        audio_config = self.config['audio']
        if audio_config['sample_rate'] not in [16000, 22050, 44100, 48000]:
            raise ValueError(f"不支持的采样率: {audio_config['sample_rate']}")
        if audio_config['channels'] not in [1, 2]:
            raise ValueError(f"不支持的声道数: {audio_config['channels']}")

    def _validate_quality_config(self) -> None:
        quality_config = self.config['quality_control']
        if not (0 < quality_config['min_vocal_content_ratio'] < 1):
            raise ValueError("人声内容比例必须在0-1之间")
        if not (0 < quality_config['max_silence_ratio'] < 1):
            raise ValueError("静音比例必须在0-1之间")

    def get(self, key_path: str, default: Any = None) -> Any:
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

    def set(self, key_path: str, value: Any) -> None:
        _set_by_path(self.config, key_path, value)
        logger.debug("配置更新: %s = %s", key_path, value)

    def get_section(self, section: str) -> Dict[str, Any]:
        if section not in self.config:
            raise KeyError(f"配置节不存在: {section}")
        return copy.deepcopy(self.config[section])

    def save_config(self, output_path: Optional[str] = None) -> None:
        target = Path(output_path) if output_path else (self.explicit_config_path or self.base_config_path)
        with open(target, 'w', encoding='utf-8') as handle:
            yaml.dump(self.config, handle, default_flow_style=False, allow_unicode=True, indent=2)
        logger.info("配置文件已保存: %s", target)

    def get_logging_config(self) -> Dict[str, Any]:
        log_config = self.get_section('logging')
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }
        log_config['level'] = level_map.get(str(log_config.get('level', 'INFO')).upper(), logging.INFO)
        return log_config


_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    global _config_manager
    if _config_manager is None or (config_path and _config_manager.explicit_config_path != Path(config_path)):
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config(key_path: str, default: Any = None) -> Any:
    return get_config_manager().get(key_path, default)


def set_runtime_config(config_overrides: Dict[str, Any]) -> None:
    cfg = get_config_manager()
    for key_path, value in config_overrides.items():
        cfg.set(key_path, value)
    logger.info("已设置 %d 个运行时配置覆盖", len(config_overrides))


def reset_runtime_config() -> None:
    global _config_manager
    explicit = None
    if _config_manager and _config_manager.explicit_config_path:
        explicit = str(_config_manager.explicit_config_path)
    _config_manager = ConfigManager(explicit)
    logger.info("运行时配置已重置")

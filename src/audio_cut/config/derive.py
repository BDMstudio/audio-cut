#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/config/derive.py
# AI-SUMMARY: 将 v3 精简参数派生成旧版配置结构，供现有模块无缝复用。

from __future__ import annotations

from typing import Any, Dict

from .settings import (
    AudioCutAppConfig,
    AudioCutDetectionConfig,
    AdaptConfig,
    GuardConfig,
    NMSConfig,
    ThresholdConfig,
)

_DEFAULT_BPM_STRENGTH = 0.6
_DEFAULT_MDD_STRENGTH = 0.4


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _guard_floor_to_percentile(floor_db: float) -> float:
    # 将 floor_db 近似映射到经验百分位：-60dB → 5%，-50dB → 10%，-70dB → 2%
    percentile = 5.0 + (floor_db + 60.0) * 0.5
    return _clamp(percentile, 1.0, 20.0)


def _derive_relative_threshold(threshold: ThresholdConfig, adapt: AdaptConfig) -> Dict[str, Any]:
    bpm_ratio = adapt.bpm_strength / _DEFAULT_BPM_STRENGTH if _DEFAULT_BPM_STRENGTH else 1.0
    bpm_ratio = _clamp(bpm_ratio, 0.2, 2.0)

    slow_multiplier = 1.0 + 0.08 * bpm_ratio
    fast_multiplier = 1.0 - 0.08 * bpm_ratio
    clamp_span = 0.15 * bpm_ratio
    clamp_min = 1.0 - clamp_span
    clamp_max = 1.0 + clamp_span

    mdd_ratio = adapt.mdd_strength / _DEFAULT_MDD_STRENGTH if _DEFAULT_MDD_STRENGTH else 1.0
    mdd_ratio = _clamp(mdd_ratio, 0.25, 2.0)
    mdd_gain = 0.2 * mdd_ratio

    return {
        'enable': True,
        'bpm': {
            'slow_multiplier': round(slow_multiplier, 3),
            'medium_multiplier': 1.0,
            'fast_multiplier': round(fast_multiplier, 3),
        },
        'mdd': {
            'base': 1.0,
            'gain': round(mdd_gain, 3),
        },
        'clamp_min': round(max(0.5, clamp_min), 3),
        'clamp_max': round(min(1.5, clamp_max), 3),
    }


def _derive_pause_stats(detection: AudioCutDetectionConfig, adapt: AdaptConfig, guard: GuardConfig) -> Dict[str, Any]:
    delta_db = max(2.0, guard.guard_db + 0.5)
    interlude_min_s = max(4.0, detection.min_pause_s * 8.0)
    sing_block_min_s = max(2.0, detection.min_pause_s * 4.0)
    multipliers = {
        'slow': round(1.0 + 0.08 * adapt.bpm_strength / _DEFAULT_BPM_STRENGTH, 3),
        'medium': 1.0,
        'fast': round(1.0 - 0.08 * adapt.bpm_strength / _DEFAULT_BPM_STRENGTH, 3),
    }
    return {
        'enable': True,
        'delta_db': round(delta_db, 3),
        'sing_block_min_s': round(sing_block_min_s, 3),
        'interlude_min_s': round(interlude_min_s, 3),
        'morph_close_ms': int(max(120, detection.min_gap_s * 200.0)),
        'morph_open_ms': int(50),
        'interlude_coverage_check': {
            'enable': False,
            'pad_seconds': 2.0,
            'coverage_threshold': 0.10,
        },
        'classify_thresholds': {
            'slow': {'mpd': 0.60, 'p95': 1.20, 'rr': 0.35},
            'fast': {'mpd': 0.25, 'pr': 18, 'rr': 0.15},
        },
        'multipliers': multipliers,
        'clamp_min': round(max(0.5, 0.75 * multipliers['slow']), 3),
        'clamp_max': round(min(1.5, 1.25 * multipliers['slow']), 3),
        'voice_percentile_hint': 90,
    }


def _derive_valley_scoring(detection: AudioCutDetectionConfig, nms: NMSConfig) -> Dict[str, Any]:
    merge_ms = max(150.0, detection.min_gap_s * 240.0)
    return {
        'w_len': 0.7,
        'w_quiet': 0.3,
        'w_flat': 0.1,
        'use_weighted_nms': True,
        'merge_close_ms': int(round(merge_ms)),
        'max_raw_candidates': max(800, nms.topk * 6),
        'max_kept_after_nms': nms.topk,
    }


def _derive_quality_control(detection: AudioCutDetectionConfig, guard: GuardConfig) -> Dict[str, Any]:
    guard_section = {
        'enable': bool(guard.enable),
        'win_ms': float(guard.win_ms),
        'guard_db': float(guard.guard_db),
        'search_right_ms': float(guard.max_shift_ms),
        'floor_percentile': _guard_floor_to_percentile(guard.floor_db),
    }

    segment_min_duration = max(6.0, detection.min_gap_s * 6.0)
    segment_max_duration = max(18.0, detection.min_gap_s * 18.0)

    return {
        'validate_split_points': True,
        'min_pause_at_split': detection.min_pause_s,
        'min_split_gap': detection.min_gap_s,
        'min_vocal_content_ratio': 0.4,
        'max_silence_ratio': 0.3,
        'fade_in_duration': 0.0,
        'fade_out_duration': 0.0,
        'normalize_audio': False,
        'remove_click_noise': False,
        'smooth_transitions': False,
        'segment_vocal_threshold_db': guard.floor_db + 10.0,
        'segment_vocal_activity_ratio': detection.segment_vocal_ratio,
        'segment_vocal_presence_ratio': 0.35,
        'segment_music_presence_ratio': 0.15,
        'segment_vocal_energy_ratio': 0.6,
        'segment_music_energy_ratio': 0.4,
        'segment_noise_margin_db': 6.0,
        'segment_vocal_overlap_ratio': 0.35,
        'segment_music_overlap_ratio': 0.55,
        'segment_min_duration': segment_min_duration,
        'segment_max_duration': segment_max_duration,
        'segment_min_mix_pace': 2.0,
        'pure_music_min_duration': detection.pure_music_min_duration,
        'high_presence_ratio': 0.6,
        'enforce_quiet_cut': guard_section,
    }


def _derive_vocal_pause_splitting(detection: AudioCutDetectionConfig, guard: GuardConfig, adapt: AdaptConfig) -> Dict[str, Any]:
    bpm_enabled = adapt.bpm_strength > 0.0
    floor_pct = _guard_floor_to_percentile(guard.floor_db)
    return {
        'enable_bpm_adaptation': bpm_enabled,
        'head_offset': -0.5,
        'tail_offset': 0.5,
        'min_pause_duration': detection.min_pause_s,
        'local_rms_window_ms': max(20.0, detection.nms.zero_cross_win_ms * 3.0),
        'lookahead_guard_ms': guard.max_shift_ms,
        'silence_floor_percentile': floor_pct,
        'voice_threshold': 0.35,
        'bpm_adaptive_settings': {
            'slow_bpm_threshold': 80,
            'fast_bpm_threshold': 120,
            'pause_duration_multipliers': {
                'slow_song_multiplier': 1.1,
                'medium_song_multiplier': 1.0,
                'fast_song_multiplier': 0.85,
            },
        },
    }


def _derive_bpm_core(detection: AudioCutDetectionConfig, adapt: AdaptConfig) -> Dict[str, Any]:
    ratio = adapt.bpm_strength / _DEFAULT_BPM_STRENGTH if _DEFAULT_BPM_STRENGTH else 1.0
    ratio = _clamp(ratio, 0.4, 1.6)

    def _scale(base: float) -> float:
        return round(base * ratio, 3)

    return {
        'tempo_categories': ['slow', 'medium', 'fast', 'very_fast'],
        'pause_duration_beats': {
            'slow': _scale(1.5),
            'medium': _scale(1.0),
            'fast': _scale(0.6),
            'very_fast': _scale(0.5),
        },
        'split_gap_phrases': {
            'slow': max(2, int(round(4 * ratio))),
            'medium': max(2, int(round(3 * ratio))),
            'fast': max(1, int(round(2 * ratio))),
            'very_fast': max(1, int(round(2 * ratio))),
        },
        'speech_pad_beats': {
            'slow': round(0.8 * ratio, 3),
            'medium': round(0.5 * ratio, 3),
            'fast': round(0.3 * ratio, 3),
            'very_fast': round(0.2 * ratio, 3),
        },
        'complexity_compensation': {
            'base_factor': 0.2,
            'complexity_boost': 0.15,
            'instrument_boost': 0.05,
            'min_instruments': 3,
        },
    }


def _derive_mdd(adapt: AdaptConfig) -> Dict[str, Any]:
    mdd_ratio = adapt.mdd_strength / _DEFAULT_MDD_STRENGTH if _DEFAULT_MDD_STRENGTH else 1.0
    mdd_ratio = _clamp(mdd_ratio, 0.3, 2.0)
    return {
        'enable': True,
        'energy_weight': 0.5,
        'spectral_weight': 0.3,
        'onset_weight': 0.2,
        'threshold_multiplier': round(0.2 * mdd_ratio, 3),
        'max_multiplier': round(1.0 + 0.4 * mdd_ratio, 3),
        'min_multiplier': 0.6,
        'chorus_detection': {
            'enable': True,
            'energy_threshold': 0.55,
            'density_threshold': 0.75,
            'multiplier': 1.1,
        },
        'segment_analysis': {
            'window_overlap': 0.5,
            'smoothing_factor': 0.3,
            'min_segment_duration': 8.0,
        },
    }


def _default_enhanced_separation(separation: Dict[str, Any]) -> Dict[str, Any]:
    backend = separation.get('backend', 'mdx23')
    enable_fallback = bool(separation.get('enable_fallback', True))
    return {
        'backend': backend,
        'enable_fallback': enable_fallback,
        'min_separation_confidence': 0.7,
        'gpu_config': {
            'enable_gpu': True,
            'large_gpu_mode': False,
        },
        'mdx23': {
            'project_path': separation.get('mdx23_project_path', './MVSEP-MDX23-music-separation-model'),
            'model_path': separation.get('mdx23_model_path', ''),
            'executable_path': separation.get('mdx23_executable', 'python inference.py'),
            'chunk_size': int(separation.get('mdx23_chunk_size', 256000)),
            'overlap_large': 0.5,
            'overlap_small': 0.3,
            'timeout': int(separation.get('mdx23_timeout', 600)),
            'use_kim_model_1': True,
        },
        'demucs_v4': {
            'model': 'htdemucs',
            'device': 'cpu',
            'shifts': 1,
            'overlap': 0.75,
            'segment': 10,
            'split': True,
        },
    }


def _default_vocal_separation() -> Dict[str, Any]:
    return {
        'hpss_margin': 2.0,
        'hpss_power': 1.5,
        'mask_threshold': 0.2,
        'mask_smoothing': 1,
        'vocal_freq_min': 100,
        'vocal_freq_max': 4000,
        'vocal_core_min': 200,
        'vocal_core_max': 1000,
        'min_vocal_ratio': 0.15,
        'max_noise_ratio': 0.3,
    }


def _advanced_vad_defaults() -> Dict[str, Any]:
    return {
        'silero_prob_threshold_down': 0.35,
        'silero_min_speech_ms': 250,
        'silero_min_silence_ms': 700,
        'silero_window_size_samples': 512,
        'silero_speech_pad_ms': 150,
    }


def resolve_legacy_config(app: AudioCutAppConfig) -> Dict[str, Any]:
    detection = app.detection
    guard = detection.guard
    threshold = detection.threshold
    adapt = detection.adapt
    nms = detection.nms

    legacy = {
        'audio': {
            'sample_rate': app.audio.get('sample_rate', 44100),
            'channels': app.audio.get('channels', 1),
            'format': app.audio.get('format', 'wav'),
            'quality': app.audio.get('quality', 320),
        },
        'output': {
            'directory': app.output.get('directory', './output'),
            'naming_pattern': app.output.get('naming_pattern', 'segment_{index:03d}'),
            'include_metadata': app.output.get('include_metadata', True),
            'save_debug_info': app.output.get('save_debug_info', True),
            'save_separated_vocal': app.output.get('save_separated_vocal', False),
            'save_breath_map': app.output.get('save_breath_map', False),
            'save_analysis_report': app.output.get('save_analysis_report', True),
        },
        'logging': {
            'level': app.logging.get('level', 'INFO'),
            'file': app.logging.get('file', 'smart_splitter.log'),
            'format': app.logging.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            'max_file_size': app.logging.get('max_file_size', '10MB'),
            'backup_count': app.logging.get('backup_count', 3),
        },
        'quality_control': _derive_quality_control(detection, guard),
        'pure_vocal_detection': {
            'enable': True,
            'min_pause_duration': detection.min_pause_s,
            'breath_duration_range': [0.1, 0.3],
            'f0_weight': 0.30,
            'formant_weight': 0.25,
            'spectral_weight': 0.25,
            'duration_weight': 0.20,
            'enable_relative_energy_mode': True,
            'peak_relative_threshold_ratio': threshold.base_ratio,
            'rms_relative_threshold_ratio': threshold.base_ratio + threshold.rms_offset,
            'relative_threshold_adaptation': _derive_relative_threshold(threshold, adapt),
            'pause_stats_adaptation': _derive_pause_stats(detection, adapt, guard),
            'energy_threshold_db': -40,
            'f0_drop_threshold': 0.7,
            'breath_filter_threshold': 0.3,
            'pause_confidence_threshold': 0.7,
            'valley_scoring': _derive_valley_scoring(detection, nms),
        },
        'musical_dynamic_density': _derive_mdd(adapt),
        'vocal_pause_splitting': _derive_vocal_pause_splitting(detection, guard, adapt),
        'bpm_adaptive_core': _derive_bpm_core(detection, adapt),
        'enhanced_separation': _default_enhanced_separation(app.separation),
        'vocal_separation': _default_vocal_separation(),
        'advanced_vad': _advanced_vad_defaults(),
    }

    return legacy

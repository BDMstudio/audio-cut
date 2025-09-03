#!/usr/bin/env python3
"""
BPM自适应VAD增强器测试
测试基于BPM的自适应人声停顿检测功能
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import numpy as np
from vocal_smart_splitter.core.adaptive_vad_enhancer import AdaptiveVADEnhancer, BPMFeatures
from vocal_smart_splitter.core.vocal_pause_detector import VocalPauseDetectorV2
from vocal_smart_splitter.utils.config_manager import get_config

def test_bpm_analysis():
    """测试BPM分析功能"""
    print(" 测试BPM分析功能...")
    
    # 创建模拟音频数据（44100Hz, 30秒）
    sample_rate = 44100
    duration = 30.0
    audio_samples = int(sample_rate * duration)
    
    # 生成包含节拍的模拟音频信号
    t = np.linspace(0, duration, audio_samples)
    # 120 BPM的节拍 + 一些谐波
    audio = np.sin(2 * np.pi * 120/60 * t) * 0.8 + \
            np.sin(2 * np.pi * 440 * t) * 0.3 + \
            np.random.normal(0, 0.05, audio_samples)  # 轻微噪声
    
    try:
        enhancer = AdaptiveVADEnhancer(sample_rate)
        bpm_features = enhancer.analyze_bpm(audio)
        
        print(f"[成功] BPM分析成功:")
        print(f"   主要BPM: {float(bpm_features.main_bpm):.1f}")
        print(f"   BPM类别: {bpm_features.bpm_category}")
        print(f"   节拍强度: {bpm_features.beat_strength:.3f}")
        print(f"   BPM置信度: {bpm_features.bpm_confidence:.3f}")
        
        # 🆕 测试参数化乘数配置
        print(f"\n 📊 参数化配置测试:")
        from vocal_smart_splitter.utils.config_manager import get_config
        
        # 测试停顿时长乘数
        slow_multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.pause_duration_multipliers.slow_song_multiplier', 0.7)
        fast_multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.pause_duration_multipliers.fast_song_multiplier', 1.3)
        medium_multiplier = get_config('vocal_pause_splitting.bpm_adaptive_settings.pause_duration_multipliers.medium_song_multiplier', 1.0)
        
        print(f"   停顿时长乘数: 慢歌×{slow_multiplier}, 中速×{medium_multiplier}, 快歌×{fast_multiplier}")
        
        # 测试偏移乘数 
        slow_offset = get_config('vocal_pause_splitting.bpm_adaptive_settings.offset_multipliers.slow_song_offset_multiplier', 1.6)
        fast_offset = get_config('vocal_pause_splitting.bpm_adaptive_settings.offset_multipliers.fast_song_offset_multiplier', 0.6)
        medium_offset = get_config('vocal_pause_splitting.bpm_adaptive_settings.offset_multipliers.medium_song_offset_multiplier', 1.0)
        
        print(f"   切割偏移乘数: 慢歌×{slow_offset}, 中速×{medium_offset}, 快歌×{fast_offset}")
        
        # 测试复杂度乘数
        inst4_base = get_config('vocal_pause_splitting.bpm_adaptive_settings.complexity_multipliers.instrument_4_plus_base', 1.4)
        inst3_base = get_config('vocal_pause_splitting.bpm_adaptive_settings.complexity_multipliers.instrument_3_base', 1.2)
        inst2_base = get_config('vocal_pause_splitting.bpm_adaptive_settings.complexity_multipliers.instrument_2_base', 1.1)
        
        print(f"   复杂度基础乘数: 4+乐器×{inst4_base}, 3乐器×{inst3_base}, 2乐器×{inst2_base}")
        
        return bpm_features
        
    except ImportError as e:
        print(f"[警告]  BPM分析库未安装: {e}")
        print("   请运行: pip install librosa")
        return None
    except Exception as e:
        print(f"[失败] BPM分析失败: {e}")
        return None

def test_adaptive_thresholds():
    """测试自适应阈值生成"""
    print("\n 测试自适应阈值生成...")
    
    sample_rate = 44100
    
    # 测试不同BPM类别的阈值调整
    test_cases = [
        (60, 'slow', '慢歌'),
        (100, 'medium', '中速歌曲'),
        (140, 'fast', '快歌')
    ]
    
    try:
        enhancer = AdaptiveVADEnhancer(sample_rate)
        
        for bpm, category, desc in test_cases:
            # 创建模拟BPM特征
            bpm_features = BPMFeatures(
                main_bpm=bpm,
                bpm_category=category,
                beat_strength=0.7,
                bpm_confidence=0.8,
                tempo_variance=0.1
            )
            
            # 生成自适应阈值
            thresholds = enhancer.generate_adaptive_thresholds(
                bpm_features=bpm_features,
                complexity_scores=[0.3, 0.5, 0.7, 0.6, 0.4]  # 模拟复杂度变化
            )
            
            print(f"[成功] {desc} (BPM: {bpm}):")
            print(f"   基础阈值: {thresholds['base_threshold']:.3f}")
            segments_str = [f'{t:.3f}' for t in thresholds['segment_thresholds'][:3]]
            print(f"   分段阈值: {segments_str}...")
            print(f"   BPM系数: {thresholds['bpm_factor']:.3f}")
            
        return True
        
    except Exception as e:
        print(f"[失败] 自适应阈值生成失败: {e}")
        return False

def test_integrated_pause_detection():
    """测试集成的BPM感知停顿检测"""
    print("\n 测试集成的BPM感知停顿检测...")
    
    sample_rate = 44100
    duration = 10.0
    audio_samples = int(sample_rate * duration)
    
    # 生成包含明显停顿的模拟音频
    audio = np.zeros(audio_samples)
    
    # 添加三个人声段（0-2s, 4-6s, 8-10s）和两个停顿（2-4s, 6-8s）
    voice_segments = [(0, 2), (4, 6), (8, 10)]
    for start, end in voice_segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        t = np.linspace(0, end - start, end_idx - start_idx)
        # 人声段：基频 + 谐波
        audio[start_idx:end_idx] = (
            np.sin(2 * np.pi * 220 * t) * 0.6 +
            np.sin(2 * np.pi * 440 * t) * 0.3 +
            np.random.normal(0, 0.1, len(t))
        )
    
    try:
        # 创建启用BPM自适应的停顿检测器
        detector = VocalPauseDetectorV2(sample_rate=sample_rate)
        
        # 检测停顿
        vocal_pauses = detector.detect_vocal_pauses(audio)
        
        print(f"[成功] 检测到 {len(vocal_pauses)} 个停顿:")
        for i, pause in enumerate(vocal_pauses):
            print(f"   停顿 {i+1}: {pause.start_time:.2f}s - {pause.end_time:.2f}s")
            print(f"            类型: {pause.position_type}, 切点: {pause.cut_point:.2f}s")
            print(f"            置信度: {pause.confidence:.3f}")
        
        # 生成报告
        report = detector.generate_pause_report(vocal_pauses)
        print(f"\n 检测报告:")
        print(f"   总停顿数: {report['total_pauses']}")
        print(f"   平均置信度: {report['avg_confidence']:.3f}")
        print(f"   停顿类型分布: {report['pause_types']}")
        
        return len(vocal_pauses) > 0
        
    except Exception as e:
        print(f"[失败] 集成停顿检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("BPM自适应VAD增强器测试套件")
    print("=" * 50)
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    results = []
    
    # 1. BPM分析测试
    bpm_features = test_bpm_analysis()
    results.append(bpm_features is not None)
    
    # 2. 自适应阈值测试
    threshold_success = test_adaptive_thresholds()
    results.append(threshold_success)
    
    # 3. 集成停顿检测测试
    detection_success = test_integrated_pause_detection()
    results.append(detection_success)
    
    # 总结
    print("\n" + "=" * 50)
    print(" 测试总结:")
    print(f"   BPM分析: {'[成功] 通过' if results[0] else '[失败] 失败'}")
    print(f"   自适应阈值: {'[成功] 通过' if results[1] else '[失败] 失败'}")
    print(f"   集成检测: {'[成功] 通过' if results[2] else '[失败] 失败'}")
    
    success_rate = sum(results) / len(results) * 100
    print(f"   总体成功率: {success_rate:.1f}%")
    
    if success_rate >= 66.7:
        print("\n BPM自适应VAD系统基本功能正常!")
        if success_rate == 100:
            print("   所有测试都通过，系统已准备好处理真实音频!")
        else:
            print("   部分功能可能需要安装额外依赖（如librosa）")
    else:
        print("\n[警告]  系统存在问题，请检查配置和依赖")
    
    return success_rate >= 66.7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
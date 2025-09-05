#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: test_dual_path_system.py
# AI-SUMMARY: 双路检测系统综合测试脚本，验证MDX23集成、性能表现和质量提升效果

import sys
import os
import time
import numpy as np
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_separator():
    """测试增强型人声分离器"""
    print("\n" + "="*60)
    print("测试1：增强型人声分离器")
    print("="*60)
    
    try:
        from vocal_smart_splitter.core.enhanced_vocal_separator import EnhancedVocalSeparator
        
        # 初始化分离器
        separator = EnhancedVocalSeparator(sample_rate=44100)
        
        print(f"后端状态: {separator}")
        backend_info = separator.get_backend_info()
        
        print(f"当前后端: {backend_info['current_backend']}")
        print("后端可用性:")
        for backend, status in backend_info['backend_status'].items():
            status_icon = "[OK]" if status['available'] else "[ERROR]"
            error_msg = f" ({status['error']})" if status['error'] else ""
            print(f"  {status_icon} {backend}{error_msg}")
        
        # 测试基本功能
        print(f"\n高质量后端可用: {'[OK]' if separator.is_high_quality_backend_available() else '[ERROR]'}")
        
        return True
        
    except Exception as e:
        print(f"增强分离器测试失败: {e}")
        return False

def test_dual_path_detector():
    """测试双路检测器"""
    print("\n" + "="*60)
    print("测试2：双路检测器")
    print("="*60)
    
    try:
        from vocal_smart_splitter.core.dual_path_detector import DualPathVocalDetector
        
        # 初始化双路检测器
        detector = DualPathVocalDetector(sample_rate=44100)
        
        print(f"检测器状态: {detector}")
        
        # 生成测试音频（模拟带有停顿的音频信号）
        print("生成测试音频信号...")
        test_audio = generate_test_audio_with_pauses()
        
        # 执行双路检测
        print("执行双路检测...")
        start_time = time.time()
        dual_result = detector.detect_with_dual_validation(test_audio)
        detection_time = time.time() - start_time
        
        # 显示结果
        print(f"检测耗时: {detection_time:.2f}秒")
        print(f"检测到停顿数: {len(dual_result.validated_pauses)}")
        print(f"整体质量: {dual_result.quality_report['overall_quality']:.3f}")
        
        # 显示每个停顿的详情
        for i, pause in enumerate(dual_result.validated_pauses[:5], 1):  # 只显示前5个
            print(f"  停顿{i}: {pause.start_time:.2f}s-{pause.end_time:.2f}s, "
                  f"置信度: {pause.confidence:.3f}, 方法: {pause.validation_method}")
        
        # 显示统计信息
        stats = detector.get_performance_stats()
        print(f"\n📊 性能统计:")
        print(f"  双路使用率: {stats['dual_path_usage_rate']*100:.1f}%")
        print(f"  高质量分离率: {stats['high_quality_rate']*100:.1f}%")
        print(f"  平均处理时间: {stats['avg_processing_time']:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 双路检测器测试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_seamless_splitter_integration():
    """测试SeamlessSplitter集成"""
    print("\n" + "="*60)
    print("🔗 测试3：SeamlessSplitter集成")
    print("="*60)
    
    try:
        from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
        
        # 检查是否有测试音频
        test_audio_path = "input/01.mp3"
        if not os.path.exists(test_audio_path):
            print(f"⚠️  测试音频文件不存在: {test_audio_path}")
            print("    请将音频文件放置到 input/01.mp3 以进行完整测试")
            return False
        
        # 初始化分割器
        splitter = SeamlessSplitter(sample_rate=44100)
        print("✅ SeamlessSplitter初始化成功")
        
        # 创建测试输出目录
        output_dir = f"output/dual_path_test_{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 执行分割
        print("🔄 执行音频分割...")
        start_time = time.time()
        result = splitter.split_audio_seamlessly(test_audio_path, output_dir)
        split_time = time.time() - start_time
        
        # 显示结果
        print(f"⏱️  分割耗时: {split_time:.2f}秒")
        print(f"🎯 生成片段数: {result['num_segments']}")
        print(f"📈 分离质量: {result['separation_quality']['overall_quality']:.3f}")
        print(f"🔄 重构验证: {'✅完美' if result['seamless_validation']['perfect_reconstruction'] else '❌有差异'}")
        
        # 显示文件信息
        print(f"📁 输出文件:")
        for file_path in result['saved_files'][:3]:  # 显示前3个文件
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  {Path(file_path).name} ({file_size:.1f}MB)")
        
        if len(result['saved_files']) > 3:
            print(f"  ... 共{len(result['saved_files'])}个文件")
        
        return True
        
    except Exception as e:
        print(f"❌ SeamlessSplitter集成测试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_performance_comparison():
    """性能对比测试"""
    print("\n" + "="*60)
    print("⚡ 测试4：性能对比（双路 vs 单路）")
    print("="*60)
    
    try:
        from vocal_smart_splitter.core.dual_path_detector import DualPathVocalDetector
        from vocal_smart_splitter.core.vocal_pause_detector import VocalPauseDetectorV2
        
        # 生成测试音频
        test_audio = generate_test_audio_with_pauses(duration=30)  # 30秒音频
        
        # 单路检测测试
        print("🔄 单路检测基准测试...")
        single_detector = VocalPauseDetectorV2(sample_rate=44100)
        
        start_time = time.time()
        single_pauses = single_detector.detect_vocal_pauses(test_audio)
        single_time = time.time() - start_time
        
        # 双路检测测试
        print("🔄 双路检测性能测试...")
        dual_detector = DualPathVocalDetector(sample_rate=44100)
        
        start_time = time.time()
        dual_result = dual_detector.detect_with_dual_validation(test_audio)
        dual_time = time.time() - start_time
        
        # 结果对比
        print(f"\n📊 性能对比结果:")
        print(f"  单路检测: {len(single_pauses)} 个停顿, 耗时 {single_time:.2f}秒")
        print(f"  双路检测: {len(dual_result.validated_pauses)} 个停顿, 耗时 {dual_time:.2f}秒")
        print(f"  性能开销: +{((dual_time/single_time - 1)*100):.1f}%")
        print(f"  质量提升: {dual_result.quality_report['overall_quality']:.3f}")
        
        # 详细统计
        processing_stats = dual_result.processing_stats
        print(f"\n📈 双路检测详情:")
        print(f"  使用双路: {'✅' if processing_stats['dual_path_used'] else '❌'}")
        print(f"  分离质量: {processing_stats['separation_confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能对比测试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def generate_test_audio_with_pauses(duration=10, sample_rate=44100):
    """生成带有停顿的测试音频信号"""
    # 生成基础信号
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 创建复合信号（人声 + 背景音乐）
    vocal_signal = np.sin(2 * np.pi * 440 * t)  # 440Hz人声
    background = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)  # 背景音乐
    
    # 添加停顿（在特定时间段将人声设为0）
    pause_segments = [(2, 3), (5, 6), (8, 8.5)]  # 停顿时间段
    
    for start, end in pause_segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        vocal_signal[start_idx:end_idx] = 0
    
    # 合成最终音频
    mixed_audio = vocal_signal + background
    
    # 添加轻微噪声
    noise = 0.01 * np.random.normal(0, 1, len(mixed_audio))
    mixed_audio += noise
    
    # 标准化到[-1, 1]范围
    mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
    
    return mixed_audio.astype(np.float32)

def test_configuration():
    """测试配置系统"""
    print("\n" + "="*60)
    print("⚙️ 测试5：配置系统验证")
    print("="*60)
    
    try:
        from vocal_smart_splitter.utils.config_manager import get_config
        
        # 测试增强分离配置
        backend = get_config('enhanced_separation.backend', 'unknown')
        enable_fallback = get_config('enhanced_separation.enable_fallback', False)
        min_confidence = get_config('enhanced_separation.min_separation_confidence', 0.0)
        
        print(f"✅ 分离后端: {backend}")
        print(f"✅ 启用降级: {enable_fallback}")
        print(f"✅ 最小置信度: {min_confidence}")
        
        # 测试双路检测配置
        enable_cross_validation = get_config('enhanced_separation.dual_detection.enable_cross_validation', False)
        pause_tolerance = get_config('enhanced_separation.dual_detection.pause_matching_tolerance', 0.0)
        
        print(f"✅ 启用交叉验证: {enable_cross_validation}")
        print(f"✅ 停顿匹配容差: {pause_tolerance}")
        
        # 测试MDX23配置
        mdx23_project_path = get_config('enhanced_separation.mdx23.project_path', '')
        mdx23_model_name = get_config('enhanced_separation.mdx23.model_name', '')
        
        print(f"✅ MDX23项目路径: {mdx23_project_path}")
        print(f"✅ MDX23模型名: {mdx23_model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("双路检测系统综合测试")
    print("=" * 60)
    print("测试目标：验证MDX23集成、双路检测和性能表现")
    print("=" * 60)
    
    tests = [
        ("配置系统", test_configuration),
        ("增强分离器", test_enhanced_separator),
        ("双路检测器", test_dual_path_detector),
        ("SeamlessSplitter集成", test_seamless_splitter_integration),
        ("性能对比", test_performance_comparison)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n正在运行: {test_name}")
            if test_func():
                print(f"通过: {test_name}")
                passed += 1
            else:
                print(f"失败: {test_name}")
        except KeyboardInterrupt:
            print(f"\n测试被用户中断")
            break
        except Exception as e:
            print(f"异常: {test_name} - {e}")
    
    print("\n" + "="*60)
    print(f"测试总结: {passed}/{total} 通过")
    
    if passed == total:
        print("所有测试通过！双路检测系统运行正常")
        print("\n系统已准备好用于生产环境测试")
        print("   建议：使用真实音频文件进行更全面的验证")
    else:
        print("部分测试失败，请检查配置和依赖")
        print("   1. 确保已安装所需的依赖包")
        print("   2. 检查MDX23或Demucs是否正确配置")
        print("   3. 验证config.yaml配置文件")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 简化版双路检测系统测试

import sys
import os
import time
import numpy as np
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """基础功能测试"""
    print("\n" + "="*50)
    print("基础功能测试")
    print("="*50)
    
    try:
        # 测试配置系统
        print("测试配置系统...")
        from vocal_smart_splitter.utils.config_manager import get_config
        backend = get_config('enhanced_separation.backend', 'mdx23')
        print(f"配置后端: {backend}")
        
        # 测试增强分离器
        print("测试增强分离器...")
        from vocal_smart_splitter.core.enhanced_vocal_separator import EnhancedVocalSeparator
        separator = EnhancedVocalSeparator(sample_rate=44100)
        print(f"分离器状态: {separator}")
        
        # 测试双路检测器
        print("测试双路检测器...")
        from vocal_smart_splitter.core.dual_path_detector import DualPathVocalDetector
        detector = DualPathVocalDetector(sample_rate=44100)
        print(f"检测器状态: {detector}")
        
        # 生成测试音频
        print("生成测试音频...")
        test_audio = generate_simple_test_audio()
        
        # 执行检测
        print("执行双路检测...")
        start_time = time.time()
        result = detector.detect_with_dual_validation(test_audio)
        detection_time = time.time() - start_time
        
        print(f"检测完成 - 耗时: {detection_time:.2f}秒")
        print(f"检测到停顿: {len(result.validated_pauses)} 个")
        print(f"质量评分: {result.quality_report['overall_quality']:.3f}")
        
        # 显示后端状态
        backend_info = separator.get_backend_info()
        print("\n后端可用性:")
        for name, status in backend_info['backend_status'].items():
            available = "可用" if status['available'] else "不可用"
            print(f"  {name}: {available}")
        
        print("\n基础功能测试: 通过")
        return True
        
    except Exception as e:
        print(f"基础功能测试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def generate_simple_test_audio(duration=5, sample_rate=44100):
    """生成简单的测试音频"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 基础信号
    signal = np.sin(2 * np.pi * 440 * t)
    
    # 添加停顿
    pause_start = int(2 * sample_rate)
    pause_end = int(3 * sample_rate)
    signal[pause_start:pause_end] *= 0.1  # 降低音量模拟停顿
    
    # 标准化
    signal = signal / np.max(np.abs(signal))
    return signal.astype(np.float32)

def test_with_real_audio():
    """使用真实音频文件测试"""
    print("\n" + "="*50)
    print("真实音频测试")
    print("="*50)
    
    audio_file = "input/01.mp3"
    if not os.path.exists(audio_file):
        print(f"跳过真实音频测试: {audio_file} 不存在")
        return True
    
    try:
        from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
        
        splitter = SeamlessSplitter(sample_rate=44100)
        print("SeamlessSplitter 初始化成功")
        
        output_dir = f"output/test_dual_{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        
        print("执行音频分割...")
        start_time = time.time()
        result = splitter.split_audio_seamlessly(audio_file, output_dir)
        split_time = time.time() - start_time
        
        print(f"分割完成 - 耗时: {split_time:.2f}秒")
        print(f"生成片段: {result['num_segments']} 个")
        print(f"输出目录: {output_dir}")
        
        print("真实音频测试: 通过")
        return True
        
    except Exception as e:
        print(f"真实音频测试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    print("双路检测系统测试")
    print("="*50)
    
    tests = [
        ("基础功能", test_basic_functionality),
        ("真实音频", test_with_real_audio)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        if test_func():
            print(f"测试通过: {test_name}")
            passed += 1
        else:
            print(f"测试失败: {test_name}")
    
    print(f"\n测试总结: {passed}/{len(tests)} 通过")
    
    if passed == len(tests):
        print("所有测试通过! 系统运行正常")
    else:
        print("部分测试失败，请检查配置")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
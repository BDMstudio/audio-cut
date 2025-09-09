#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# quick_test.py - 快速测试MDX23是否工作

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def quick_test():
    """快速测试不同后端"""
    from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
    
    # 设置环境变量强制使用MDX23
    os.environ['FORCE_SEPARATION_BACKEND'] = 'mdx23'
    
    print("测试MDX23后端...")
    
    try:
        splitter = SeamlessSplitter(sample_rate=44100)
        
        # 使用一个小音频文件测试
        input_file = "input/15.MP3"
        output_dir = "output/quick_test_mdx23"
        
        if not Path(input_file).exists():
            print(f"音频文件 {input_file} 不存在")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("开始分割...")
        result = splitter.split_audio_seamlessly(input_file, output_dir)
        
        if 'processing_stats' in result:
            stats = result['processing_stats']
            print(f"实际使用后端: {stats.get('backend_used', 'unknown')}")
            print(f"双路检测: {'是' if stats.get('dual_path_used', False) else '否'}")
            if 'separation_confidence' in stats:
                print(f"分离置信度: {stats['separation_confidence']:.3f}")
        
        print(f"分割成功: {result.get('success', False)}")
        print(f"生成片段: {result.get('num_segments', 0)}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
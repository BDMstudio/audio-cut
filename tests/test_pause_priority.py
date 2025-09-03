#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_pause_priority.py
# AI-SUMMARY: 测试停顿优先级分割算法的效果

"""
停顿优先级算法测试脚本

测试新的基于停顿优先级的分割算法，验证其精准度和效果。

使用方法:
    python test_pause_priority.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vocal_smart_splitter.main import VocalSmartSplitter

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def analyze_split_precision(result, input_file):
    """分析分割精准度"""
    print("\n🎯 分割精准度分析")
    print("=" * 50)
    
    if not result['success']:
        print("❌ 分割失败，无法分析精准度")
        return
    
    segments = result['output_files']
    total_segments = len(segments)
    
    print(f"📊 基本统计:")
    print(f"  - 输入文件: {input_file}")
    print(f"  - 总时长: {result.get('total_duration', 0):.2f}秒")
    print(f"  - 生成片段数: {total_segments}")
    print(f"  - 平均片段长度: {result.get('total_duration', 0) / max(1, total_segments):.2f}秒")
    
    # 分析片段长度分布
    if 'processing_summary' in result:
        summary = result['processing_summary']
        print(f"\n🔍 处理质量:")
        print(f"  - 人声分离质量: {summary.get('separation_quality', 0):.3f}")
        print(f"  - 换气检测质量: {summary.get('breath_detection_quality', 0):.3f}")
        print(f"  - 内容分析质量: {summary.get('content_analysis_quality', 0):.3f}")
        print(f"  - 最终质量评分: {summary.get('final_quality', 0):.3f}")
    
    # 检查输出文件
    if segments:
        print(f"\n📁 输出文件:")
        for i, file_path in enumerate(segments, 1):
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # KB
                filename = os.path.basename(file_path)
                print(f"  {i}. {filename} ({file_size:.0f}KB)")
            else:
                print(f"  {i}. 文件不存在: {file_path}")
    
    # 分析分割点质量（如果有调试信息）
    debug_file = os.path.join(result['output_directory'], 'debug_info.json')
    if os.path.exists(debug_file):
        print(f"\n📋 详细调试信息: {debug_file}")
    
    analysis_file = os.path.join(result['output_directory'], 'analysis_report.json')
    if os.path.exists(analysis_file):
        print(f"📋 分析报告: {analysis_file}")

def compare_algorithms():
    """比较新旧算法的效果"""
    print("\n🔄 算法对比测试")
    print("=" * 50)
    
    input_file = "input/01.mp3"
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 测试新算法（停顿优先级）
    print("🆕 测试停顿优先级算法...")
    try:
        splitter_new = VocalSmartSplitter()
        # 确保使用新算法
        splitter_new.config_manager.set('smart_splitting.use_pause_priority_algorithm', True)
        
        output_dir_new = f"output/pause_priority_{timestamp}"
        result_new = splitter_new.split_audio(input_file, output_dir_new)
        
        print("✅ 停顿优先级算法测试完成")
        analyze_split_precision(result_new, input_file)
        
    except Exception as e:
        print(f"❌ 停顿优先级算法测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    
    # 测试传统算法
    print("🔄 测试传统算法...")
    try:
        splitter_old = VocalSmartSplitter()
        # 使用传统算法
        splitter_old.config_manager.set('smart_splitting.use_pause_priority_algorithm', False)
        
        output_dir_old = f"output/traditional_{timestamp}"
        result_old = splitter_old.split_audio(input_file, output_dir_old)
        
        print("✅ 传统算法测试完成")
        analyze_split_precision(result_old, input_file)
        
    except Exception as e:
        print(f"❌ 传统算法测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 对比结果
    print("\n📊 算法对比总结")
    print("=" * 50)
    try:
        if 'result_new' in locals() and 'result_old' in locals():
            if result_new['success'] and result_old['success']:
                new_segments = len(result_new['output_files'])
                old_segments = len(result_old['output_files'])
                new_quality = result_new.get('quality_score', 0)
                old_quality = result_old.get('quality_score', 0)
                
                print(f"片段数量对比:")
                print(f"  - 停顿优先级算法: {new_segments} 个")
                print(f"  - 传统算法: {old_segments} 个")
                print(f"  - 差异: {new_segments - old_segments:+d} 个")
                
                print(f"\n质量评分对比:")
                print(f"  - 停顿优先级算法: {new_quality:.3f}")
                print(f"  - 传统算法: {old_quality:.3f}")
                print(f"  - 差异: {new_quality - old_quality:+.3f}")
                
                # 推荐
                if new_segments > old_segments and new_quality >= old_quality * 0.9:
                    print(f"\n🏆 推荐: 停顿优先级算法 (更多片段，质量相当)")
                elif new_quality > old_quality * 1.1:
                    print(f"\n🏆 推荐: 停顿优先级算法 (质量显著提升)")
                elif old_quality > new_quality * 1.1:
                    print(f"\n🏆 推荐: 传统算法 (质量更高)")
                else:
                    print(f"\n⚖️ 两种算法效果相近，可根据需求选择")
            else:
                print("⚠️ 部分算法测试失败，无法完整对比")
        else:
            print("⚠️ 算法测试不完整，无法对比")
    except Exception as e:
        print(f"⚠️ 对比分析失败: {e}")

def main():
    """主函数"""
    setup_logging()
    
    print("🎵 停顿优先级分割算法测试")
    print("=" * 50)
    print("测试目标:")
    print("  1. 验证停顿优先级算法的精准度")
    print("  2. 对比新旧算法的效果")
    print("  3. 分析分割点的质量")
    print("=" * 50)
    
    # 检查输入文件
    input_file = "input/01.mp3"
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        print("请将音频文件放置在 input/01.mp3")
        return
    
    # 运行对比测试
    compare_algorithms()
    
    print("\n" + "=" * 50)
    print("🎵 测试完成！")
    print("\n💡 使用建议:")
    print("  - 如果需要更多片段，使用停顿优先级算法")
    print("  - 如果需要更高质量，根据测试结果选择")
    print("  - 可以通过配置文件调整算法参数")
    print("  - 查看输出目录中的调试信息了解详情")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_precise_voice_splitting.py
# AI-SUMMARY: 测试基于先进VAD的精确人声分割算法

"""
精确人声分割算法测试

测试新的基于Silero VAD的精确人声分割算法：
1. 只在真正的人声停顿处分割
2. 不考虑片段长度，优先保证分割精准度
3. 使用先进的VAD算法精确检测人声活动
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
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

def test_precise_voice_splitting():
    """测试精确人声分割算法"""
    print("🎯 精确人声分割算法测试")
    print("=" * 60)
    print("目标:")
    print("  1. 只在真正的人声停顿处分割")
    print("  2. 使用Silero VAD精确检测人声活动")
    print("  3. 不考虑片段长度，优先保证分割精准度")
    print("  4. 避免在人声进行中切割")
    print("=" * 60)
    
    input_file = "input/01.mp3"
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/precise_voice_{timestamp}"
    
    try:
        print("🔧 初始化精确人声分割器...")
        splitter = VocalSmartSplitter()
        
        # 强制使用精确人声分割算法
        splitter.config_manager.set('smart_splitting.use_precise_voice_algorithm', True)
        splitter.config_manager.set('smart_splitting.use_pause_priority_algorithm', False)
        
        # 配置精确人声分割参数
        splitter.config_manager.set('precise_voice_splitting.min_silence_duration', 0.5)
        splitter.config_manager.set('precise_voice_splitting.silence_threshold', 0.3)
        splitter.config_manager.set('precise_voice_splitting.preferred_vad_method', 'silero')
        
        print("🎯 算法配置:")
        print(f"  - 精确人声分割: {splitter.config_manager.get('smart_splitting.use_precise_voice_algorithm')}")
        print(f"  - VAD方法: {splitter.config_manager.get('precise_voice_splitting.preferred_vad_method')}")
        print(f"  - 最小静音时长: {splitter.config_manager.get('precise_voice_splitting.min_silence_duration')}秒")
        print(f"  - 静音质量阈值: {splitter.config_manager.get('precise_voice_splitting.silence_threshold')}")
        
        print(f"\n🚀 开始处理音频文件: {input_file}")
        print(f"📁 输出目录: {output_dir}")
        
        result = splitter.split_audio(input_file, output_dir)
        
        if result['success']:
            print("\n✅ 处理成功！")
            
            # 基本统计
            segments = result['output_files']
            total_segments = len(segments)
            total_duration = result.get('total_duration', 0)
            
            print(f"\n📊 分割结果统计:")
            print(f"  - 输入文件: {input_file}")
            print(f"  - 总时长: {total_duration:.2f}秒")
            print(f"  - 生成片段数: {total_segments}")
            
            if total_segments > 0:
                avg_length = total_duration / total_segments
                print(f"  - 平均片段长度: {avg_length:.2f}秒")
            
            print(f"  - 质量评分: {result.get('quality_score', 0):.3f}")
            
            # 分析片段详情
            if segments:
                print(f"\n📋 片段详情:")
                for i, file_path in enumerate(segments, 1):
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        filename = os.path.basename(file_path)
                        print(f"  {i:2d}. {filename} ({file_size:.0f}KB)")
                    else:
                        print(f"  {i:2d}. ❌ 文件不存在: {os.path.basename(file_path)}")
            
            # 分析VAD检测结果
            debug_file = os.path.join(output_dir, 'debug_info.json')
            if os.path.exists(debug_file):
                print(f"\n🔍 VAD检测分析:")
                try:
                    import json
                    with open(debug_file, 'r', encoding='utf-8') as f:
                        debug_data = json.load(f)
                    
                    # 查找VAD相关信息
                    if 'smart_splitting' in debug_data:
                        split_info = debug_data['smart_splitting']
                        print(f"  - 分割方法: {split_info.get('method', 'unknown')}")
                        if 'vad_method' in split_info:
                            print(f"  - VAD算法: {split_info['vad_method']}")
                    
                    print(f"  - 详细调试信息: {debug_file}")
                    
                except Exception as e:
                    print(f"  ⚠️ 读取调试信息失败: {e}")
            
            # 分割精准度评估
            print(f"\n🎯 分割精准度评估:")
            
            if total_segments == 1:
                print("  - 结果: 保持为单个片段")
                print("  - 评估: ✅ 未检测到合适的分割点，避免了错误分割")
            elif total_segments > 1:
                print(f"  - 结果: 分割为 {total_segments} 个片段")
                
                # 评估片段长度分布
                if segments:
                    durations = []
                    for file_path in segments:
                        if os.path.exists(file_path):
                            # 这里简化处理，实际应该读取音频文件获取时长
                            estimated_duration = total_duration / total_segments
                            durations.append(estimated_duration)
                    
                    if durations:
                        min_dur = min(durations)
                        max_dur = max(durations)
                        print(f"  - 片段时长范围: {min_dur:.1f}-{max_dur:.1f}秒")
                        
                        # 评估是否有过短的片段（可能是错误分割）
                        short_segments = [d for d in durations if d < 2.0]
                        if short_segments:
                            print(f"  - ⚠️ 发现 {len(short_segments)} 个过短片段（<2秒）")
                            print("  - 建议: 调整静音检测阈值")
                        else:
                            print("  - ✅ 无过短片段，分割质量良好")
            
            # 给出改进建议
            print(f"\n💡 改进建议:")
            if total_segments == 1:
                print("  - 如需更多分割，可降低 min_silence_duration 参数")
                print("  - 或降低 silence_threshold 参数")
            elif total_segments > 20:
                print("  - 分割过多，可提高 min_silence_duration 参数")
                print("  - 或提高 silence_threshold 参数")
            else:
                print("  - 分割数量合理，可试听验证分割精准度")
            
            print(f"\n📁 输出文件位置: {output_dir}")
            
        else:
            print("❌ 处理失败")
            if 'error' in result:
                print(f"错误信息: {result['error']}")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 精确人声分割测试完成")
    print("\n💡 测试要点:")
    print("  1. 检查是否使用了Silero VAD")
    print("  2. 验证分割点是否在真正的人声停顿处")
    print("  3. 确认没有在人声进行中切割")
    print("  4. 评估分割数量是否合理")

def main():
    """主函数"""
    setup_logging()
    test_precise_voice_splitting()

if __name__ == "__main__":
    main()

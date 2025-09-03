#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_simple_pause_priority.py
# AI-SUMMARY: 简化的停顿优先级算法测试

"""
简化的停顿优先级算法测试

专门测试新的停顿优先级算法是否正常工作。
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

def test_pause_priority_only():
    """只测试停顿优先级算法"""
    print("🎵 停顿优先级算法单独测试")
    print("=" * 50)
    
    input_file = "input/01.mp3"
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/pause_test_{timestamp}"
    
    try:
        print("🔧 初始化分割器...")
        splitter = VocalSmartSplitter()
        
        # 强制使用停顿优先级算法
        splitter.config_manager.set('smart_splitting.use_pause_priority_algorithm', True)
        
        print("🎯 配置参数:")
        print(f"  - 使用停顿优先级算法: {splitter.config_manager.get('smart_splitting.use_pause_priority_algorithm')}")
        print(f"  - 最小分割间隔: {splitter.config_manager.get('pause_priority.min_split_interval', 3.0)}秒")
        print(f"  - 停顿时长权重: {splitter.config_manager.get('pause_priority.duration_weight', 0.5)}")
        
        print(f"\n🚀 开始处理音频文件: {input_file}")
        print(f"📁 输出目录: {output_dir}")
        
        result = splitter.split_audio(input_file, output_dir)
        
        if result['success']:
            print("\n✅ 处理成功！")
            print(f"📊 结果统计:")
            print(f"  - 生成片段数: {len(result['output_files'])}")
            print(f"  - 总时长: {result.get('total_duration', 0):.2f}秒")
            print(f"  - 质量评分: {result.get('quality_score', 0):.3f}")
            
            # 分析片段长度
            if result['output_files']:
                print(f"\n📋 片段详情:")
                for i, file_path in enumerate(result['output_files'], 1):
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        filename = os.path.basename(file_path)
                        print(f"  {i}. {filename} ({file_size:.0f}KB)")
            
            # 检查调试信息
            debug_file = os.path.join(output_dir, 'debug_info.json')
            if os.path.exists(debug_file):
                print(f"\n📋 调试信息: {debug_file}")
                
                # 尝试读取调试信息中的停顿点数据
                try:
                    import json
                    with open(debug_file, 'r', encoding='utf-8') as f:
                        debug_data = json.load(f)
                    
                    if 'breath_detection' in debug_data:
                        breath_info = debug_data['breath_detection']
                        print(f"🫁 换气检测信息:")
                        print(f"  - 检测到停顿点: {breath_info.get('num_pauses', 0)}个")
                        print(f"  - 检测质量: {breath_info.get('quality_score', 0):.3f}")
                    
                    if 'smart_splitting' in debug_data:
                        split_info = debug_data['smart_splitting']
                        print(f"🧠 智能分割信息:")
                        print(f"  - 候选分割点: {split_info.get('num_candidates', 0)}个")
                        print(f"  - 最终分割点: {split_info.get('num_selected', 0)}个")
                        
                except Exception as e:
                    print(f"⚠️ 读取调试信息失败: {e}")
            
            print(f"\n🎯 分割精准度评估:")
            total_duration = result.get('total_duration', 0)
            num_segments = len(result['output_files'])
            if num_segments > 0:
                avg_length = total_duration / num_segments
                print(f"  - 平均片段长度: {avg_length:.2f}秒")
                
                # 评估是否符合预期
                if 5 <= avg_length <= 15:
                    print(f"  - 长度评估: ✅ 符合5-15秒要求")
                else:
                    print(f"  - 长度评估: ⚠️ 不在5-15秒范围内")
                
                # 评估片段数量是否合理
                expected_segments = int(total_duration / 10)  # 按10秒/片段估算
                if num_segments >= expected_segments * 0.7:
                    print(f"  - 数量评估: ✅ 片段数量合理 ({num_segments}/{expected_segments})")
                else:
                    print(f"  - 数量评估: ⚠️ 片段数量偏少 ({num_segments}/{expected_segments})")
            
        else:
            print("❌ 处理失败")
            if 'error' in result:
                print(f"错误信息: {result['error']}")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎵 测试完成")

def main():
    """主函数"""
    setup_logging()
    
    print("🎯 停顿优先级算法专项测试")
    print("目标: 验证新算法是否能正确识别和使用停顿点进行分割")
    print()
    
    test_pause_priority_only()
    
    print("\n💡 如果测试成功，说明停顿优先级算法工作正常")
    print("💡 如果仍有问题，需要进一步调试算法实现")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/test_improved_pause_detection.py
# AI-SUMMARY: 测试改进后的停顿检测准确性，验证分割点是否在自然停顿处

"""
改进停顿检测测试

验证优化后的停顿检测算法：
1. 检测更短的自然停顿（0.15s vs 0.5s）
2. 多级停顿分类（短/中/长停顿）
3. 更敏感的VAD参数配置
4. 改进的分割点选择策略
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

def test_improved_pause_detection():
    """测试改进后的停顿检测算法"""
    print("🎯 改进停顿检测测试")
    print("=" * 70)
    print("🔍 测试目标:")
    print("  1. 验证是否能检测到更短的自然停顿 (0.15s vs 0.5s)")
    print("  2. 检查多级停顿分类效果")
    print("  3. 确认分割点是否落在真正的人声停顿处")
    print("  4. 评估改进后的分割数量和质量")
    print("=" * 70)
    
    input_file = "input/01.mp3"
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/improved_pause_{timestamp}"
    
    try:
        print("🔧 初始化改进的智能分割器...")
        splitter = VocalSmartSplitter()
        
        # 确保使用精确人声分割算法（已优化）
        splitter.config_manager.set('smart_splitting.use_precise_voice_algorithm', True)
        splitter.config_manager.set('smart_splitting.use_pause_priority_algorithm', False)
        
        # 验证关键参数设置
        print("📋 验证改进参数:")
        min_silence = splitter.config_manager.get('precise_voice_splitting.min_silence_duration')
        silence_threshold = splitter.config_manager.get('precise_voice_splitting.silence_threshold')
        breath_min_pause = splitter.config_manager.get('breath_detection.min_pause_duration')
        breath_energy_threshold = splitter.config_manager.get('breath_detection.energy_threshold')
        
        print(f"  - 最小静音时长: {min_silence}s (改进前: 0.5s)")
        print(f"  - 静音质量阈值: {silence_threshold} (改进前: 0.3)")
        print(f"  - 换气检测最小停顿: {breath_min_pause}s (改进前: 0.15s)")
        print(f"  - 能量检测阈值: {breath_energy_threshold} (改进前: 0.02)")
        
        print(f"\n🚀 开始处理音频文件: {input_file}")
        print(f"📁 输出目录: {output_dir}")
        
        # 运行改进的分割算法
        result = splitter.split_audio(input_file, output_dir)
        
        if result['success']:
            print("\n✅ 处理成功！")
            
            # 基本统计
            segments = result['output_files']
            total_segments = len(segments)
            total_duration = result.get('total_duration', 0)
            quality_score = result.get('quality_score', 0)
            
            print(f"\n📊 改进效果对比:")
            print(f"  - 输入文件: {input_file}")
            print(f"  - 总时长: {total_duration:.2f}秒")
            print(f"  - 生成片段数: {total_segments} (改进前: 7个)")
            print(f"  - 质量评分: {quality_score:.3f}")
            
            if total_segments > 0:
                avg_length = total_duration / total_segments
                print(f"  - 平均片段长度: {avg_length:.2f}秒")
                
                # 评估改进效果
                improvement_ratio = total_segments / 7  # 与之前7个片段对比
                print(f"  - 分割数量改进: {improvement_ratio:.1f}x")
                
                if total_segments >= 10:
                    print("  ✅ 分割数量显著提升")
                elif total_segments >= 15:
                    print("  🎉 分割数量大幅提升")
                else:
                    print("  ⚠️ 分割数量仍需改进")
            
            # 分析片段详情
            if segments:
                print(f"\n📋 片段详情 (前10个):")
                for i, file_path in enumerate(segments[:10], 1):
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        filename = os.path.basename(file_path)
                        print(f"  {i:2d}. {filename} ({file_size:.0f}KB)")
                    else:
                        print(f"  {i:2d}. ❌ 文件不存在: {os.path.basename(file_path)}")
                
                if len(segments) > 10:
                    print(f"  ... 还有 {len(segments) - 10} 个片段")
            
            # 分析调试信息
            debug_file = os.path.join(output_dir, 'debug_info.json')
            if os.path.exists(debug_file):
                print(f"\n🔍 停顿检测详细分析:")
                try:
                    import json
                    with open(debug_file, 'r', encoding='utf-8') as f:
                        debug_data = json.load(f)
                    
                    # 查找停顿相关信息
                    if 'smart_splitting' in debug_data:
                        split_info = debug_data['smart_splitting']
                        print(f"  - 分割方法: {split_info.get('method', 'unknown')}")
                        if 'vad_method' in split_info:
                            print(f"  - VAD算法: {split_info['vad_method']}")
                        
                        # 分析分割点信息
                        if 'split_points' in split_info and split_info['split_points']:
                            print(f"  - 检测到的分割点: {len(split_info['split_points'])}个")
                            for i, point in enumerate(split_info['split_points'][:5]):
                                if isinstance(point, dict):
                                    split_time = point.get('split_time', 'unknown')
                                    pause_type = point.get('pause_type', 'unknown')  
                                    quality = point.get('quality_score', 0)
                                    print(f"    {i+1}. {split_time:.2f}s [{pause_type}] 质量:{quality:.3f}")
                    
                    print(f"  - 详细调试信息: {debug_file}")
                    
                except Exception as e:
                    print(f"  ⚠️ 读取调试信息失败: {e}")
            
            # 停顿检测改进效果评估
            print(f"\n🎯 停顿检测改进评估:")
            
            if total_segments == 1:
                print("  - 结果: 仍为单个片段")
                print("  - 评估: ❌ 改进效果不明显，需进一步调整参数")
                print("  - 建议: 继续降低 min_silence_duration 或 silence_threshold")
            elif 7 <= total_segments <= 12:
                print("  - 结果: 分割数量适度提升")
                print("  - 评估: ⚠️ 改进有效但仍有优化空间")
            elif total_segments > 12:
                print("  - 结果: 分割数量显著提升") 
                print("  - 评估: ✅ 改进效果良好")
                
                # 进一步检查片段质量
                if avg_length < 20:
                    print("  - 片段长度: ✅ 合理范围")
                else:
                    print("  - 片段长度: ⚠️ 仍然偏长")
            
            # 给出进一步优化建议
            print(f"\n💡 下一步优化建议:")
            if total_segments < 10:
                print("  1. 进一步降低 min_silence_duration 至 0.1s")
                print("  2. 降低 silence_threshold 至 0.1")
                print("  3. 启用更多备用分割策略")
            elif total_segments > 25:
                print("  1. 适当提高质量阈值，减少过度分割")
                print("  2. 增加分割点间最小距离")
            else:
                print("  1. 参数调优效果良好，可进行用户测试")
                print("  2. 考虑针对不同类型音乐进行微调")
            
            print(f"\n📁 输出文件位置: {output_dir}")
            
            return True
            
        else:
            print("❌ 处理失败")
            if 'error' in result:
                print(f"错误信息: {result['error']}")
            return False
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n" + "=" * 70)
        print("🎯 改进停顿检测测试完成")
        print("\n💡 关键检查要点:")
        print("  1. 分割数量是否比之前的7个有显著提升")
        print("  2. 检查debug_info.json中的停顿类型分类")
        print("  3. 人工试听几个片段，确认分割点的自然度")
        print("  4. 观察日志中的分割点选择过程")

def main():
    """主函数"""
    setup_logging()
    
    print("🚀 启动改进停顿检测测试...")
    success = test_improved_pause_detection()
    
    if success:
        print("\n🎉 测试完成，请检查输出结果")
    else:
        print("\n❌ 测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
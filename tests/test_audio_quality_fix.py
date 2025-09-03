#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_audio_quality_fix.py
# AI-SUMMARY: 测试音质修复效果的脚本

"""
音质修复测试脚本

测试修复后的音频分割器是否解决了音质损失和破音问题。

使用方法:
    python test_audio_quality_fix.py
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

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

def analyze_audio_quality(file_path):
    """分析音频质量"""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        
        # 基本统计
        duration = len(audio) / sr
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
        
        # 检测削波
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
        
        # 频谱分析
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        avg_spectral_centroid = np.mean(spectral_centroid)
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'peak': peak,
            'rms': rms,
            'dynamic_range': dynamic_range,
            'clipping_ratio': clipping_ratio,
            'spectral_centroid': avg_spectral_centroid,
            'file_size': os.path.getsize(file_path)
        }
    except Exception as e:
        print(f"分析音频失败: {e}")
        return None

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🎵 音质修复测试开始")
    print("=" * 50)
    
    # 检查输入文件
    input_file = "input/01.mp3"
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 分析原始音频质量
    print("📊 分析原始音频质量...")
    original_quality = analyze_audio_quality(input_file)
    if original_quality:
        print(f"原始音频:")
        print(f"  - 时长: {original_quality['duration']:.2f}秒")
        print(f"  - 采样率: {original_quality['sample_rate']}Hz")
        print(f"  - 峰值: {original_quality['peak']:.3f}")
        print(f"  - RMS: {original_quality['rms']:.3f}")
        print(f"  - 动态范围: {original_quality['dynamic_range']:.1f}dB")
        print(f"  - 削波比例: {original_quality['clipping_ratio']*100:.2f}%")
        print(f"  - 频谱重心: {original_quality['spectral_centroid']:.0f}Hz")
        print(f"  - 文件大小: {original_quality['file_size']/1024/1024:.1f}MB")
    
    print("\n🔧 运行修复后的分割器...")
    
    try:
        # 初始化分割器
        splitter = VocalSmartSplitter()
        
        # 创建测试输出目录
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/quality_test_{timestamp}"
        
        # 运行分割
        result = splitter.split_audio(input_file, output_dir)
        
        if result['success']:
            print(f"✅ 分割成功！")
            print(f"📁 输出目录: {output_dir}")
            print(f"🎵 生成片段数: {len(result['output_files'])}")
            print(f"⭐ 总体质量评分: {result['quality_score']:.3f}")
            
            # 分析每个输出片段的质量
            print("\n📊 分析输出片段质量...")
            print("-" * 50)
            
            total_clipping = 0
            total_dynamic_range = 0
            valid_segments = 0
            
            for i, file_path in enumerate(result['output_files'], 1):
                if os.path.exists(file_path):
                    quality = analyze_audio_quality(file_path)
                    if quality:
                        valid_segments += 1
                        total_clipping += quality['clipping_ratio']
                        total_dynamic_range += quality['dynamic_range']
                        
                        filename = os.path.basename(file_path)
                        print(f"片段 {i}: {filename}")
                        print(f"  - 时长: {quality['duration']:.2f}秒")
                        print(f"  - 采样率: {quality['sample_rate']}Hz")
                        print(f"  - 峰值: {quality['peak']:.3f}")
                        print(f"  - RMS: {quality['rms']:.3f}")
                        print(f"  - 动态范围: {quality['dynamic_range']:.1f}dB")
                        print(f"  - 削波比例: {quality['clipping_ratio']*100:.2f}%")
                        print(f"  - 文件大小: {quality['file_size']/1024:.0f}KB")
                        
                        # 质量评估
                        quality_issues = []
                        if quality['clipping_ratio'] > 0.01:  # 超过1%削波
                            quality_issues.append("⚠️ 削波严重")
                        if quality['dynamic_range'] < 10:  # 动态范围小于10dB
                            quality_issues.append("⚠️ 动态范围过小")
                        if quality['peak'] > 0.99:  # 峰值过高
                            quality_issues.append("⚠️ 峰值过高")
                        
                        if quality_issues:
                            print(f"  - 问题: {', '.join(quality_issues)}")
                        else:
                            print(f"  - 状态: ✅ 质量良好")
                        print()
            
            # 总体质量评估
            if valid_segments > 0:
                avg_clipping = total_clipping / valid_segments
                avg_dynamic_range = total_dynamic_range / valid_segments
                
                print("📈 总体质量评估:")
                print(f"  - 平均削波比例: {avg_clipping*100:.2f}%")
                print(f"  - 平均动态范围: {avg_dynamic_range:.1f}dB")
                
                # 质量判断
                if avg_clipping < 0.001 and avg_dynamic_range > 15:
                    print("  - 总体评价: ✅ 优秀")
                elif avg_clipping < 0.01 and avg_dynamic_range > 10:
                    print("  - 总体评价: ✅ 良好")
                elif avg_clipping < 0.05 and avg_dynamic_range > 5:
                    print("  - 总体评价: ⚠️ 一般")
                else:
                    print("  - 总体评价: ❌ 需要改进")
            
        else:
            print(f"❌ 分割失败: {result.get('error', '未知错误')}")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎵 音质修复测试完成")

if __name__ == "__main__":
    main()

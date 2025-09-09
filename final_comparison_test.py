#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# final_comparison_test.py - 最终对比测试：修复后的不同后端效果

import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def compare_all_backends():
    """对比所有后端的最终分割结果"""
    from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
    
    input_file = "input/15.MP3"
    if not Path(input_file).exists():
        print(f"测试文件 {input_file} 不存在")
        return
    
    print("=" * 60)
    print("最终对比测试：修复后的后端差异验证")
    print("=" * 60)
    
    backends = ['hpss_fallback', 'mdx23', 'demucs_v4']
    results = {}
    
    for backend in backends:
        print(f"\n[测试] {backend} 后端:")
        
        # 设置环境变量
        os.environ['FORCE_SEPARATION_BACKEND'] = backend
        
        try:
            # 创建分割器
            splitter = SeamlessSplitter(sample_rate=44100)
            
            # 创建输出目录
            timestamp = datetime.now().strftime("%H%M%S")
            output_dir = f"output/final_test_{backend}_{timestamp}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # 执行分割
            result = splitter.split_audio_seamlessly(input_file, output_dir)
            
            if result.get('success'):
                num_segments = result.get('num_segments', 0)
                stats = result.get('processing_stats', {})
                backend_used = stats.get('backend_used', 'unknown')
                dual_path_used = stats.get('dual_path_used', False)
                confidence = stats.get('separation_confidence', 0.0)
                
                print(f"  [成功] 分割完成")
                print(f"    实际后端: {backend_used}")
                print(f"    双路检测: {'是' if dual_path_used else '否'}")
                print(f"    分离置信度: {confidence:.4f}")
                print(f"    生成片段: {num_segments}个")
                
                # 记录片段文件大小（用于验证差异）
                saved_files = result.get('saved_files', [])
                file_sizes = []
                for file_path in saved_files:
                    if Path(file_path).exists():
                        size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                        file_sizes.append(size_mb)
                
                total_size = sum(file_sizes)
                print(f"    总文件大小: {total_size:.2f}MB")
                print(f"    输出目录: {output_dir}")
                
                results[backend] = {
                    'success': True,
                    'backend_used': backend_used,
                    'dual_path_used': dual_path_used,
                    'confidence': float(confidence) if confidence else 0.0,
                    'num_segments': num_segments,
                    'total_size_mb': total_size,
                    'file_sizes': file_sizes,
                    'output_dir': output_dir
                }
            else:
                print(f"  [失败] 分割失败: {result.get('error', '未知错误')}")
                results[backend] = {'success': False, 'error': result.get('error')}
                
        except Exception as e:
            print(f"  [异常] {e}")
            results[backend] = {'success': False, 'exception': str(e)}
    
    # 清理环境变量
    if 'FORCE_SEPARATION_BACKEND' in os.environ:
        del os.environ['FORCE_SEPARATION_BACKEND']
    
    print(f"\n" + "="*60)
    print("对比结果分析")
    print("="*60)
    
    successful_results = {k: v for k, v in results.items() if v.get('success')}
    
    if len(successful_results) >= 2:
        print(f"\n[片段数量对比]")
        segment_counts = {}
        for backend, result in successful_results.items():
            backend_used = result['backend_used']
            num_segments = result['num_segments']
            dual_path = "双路" if result['dual_path_used'] else "单路"
            confidence = result['confidence']
            
            print(f"  {backend:15} ({backend_used:12}) -> {num_segments:2d}个片段 [{dual_path}] 置信度:{confidence:.3f}")
            segment_counts[backend] = num_segments
        
        print(f"\n[文件大小对比]")
        for backend, result in successful_results.items():
            total_size = result['total_size_mb']
            print(f"  {backend:15} -> 总大小: {total_size:7.2f}MB")
        
        # 检查是否存在差异
        unique_segments = set(segment_counts.values())
        unique_sizes = set(round(r['total_size_mb'], 1) for r in successful_results.values())
        
        print(f"\n[差异分析]")
        print(f"  不同片段数量: {len(unique_segments)} 种 ({list(unique_segments)})")
        print(f"  不同文件大小: {len(unique_sizes)} 种")
        
        if len(unique_segments) == 1 and len(unique_sizes) == 1:
            print(f"  [WARNING] 所有后端仍然产生相同结果！可能需要进一步调试。")
        else:
            print(f"  [SUCCESS] 不同后端产生了不同结果！修复成功。")
            
            # 显示具体差异
            for backend, result in successful_results.items():
                if result['dual_path_used'] and result['backend_used'] in ['mdx23', 'demucs_v4']:
                    print(f"    {backend}: 使用了分离增强检测，片段数={result['num_segments']}")
    
    return results

if __name__ == "__main__":
    compare_all_backends()
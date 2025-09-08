#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# quick_start.py - 快速启动脚本
# AI-SUMMARY: 一键式音频分割快速启动脚本，无需复杂参数

"""
智能人声分割器 - 快速启动脚本

最简单的使用方式：
1. 将音频文件放入 input/ 目录
2. 运行 python quick_start.py
3. 在 output/ 目录查看结果

特点：
- 自动检测input/目录中的音频文件
- 使用最优的BPM自适应无缝分割
- 自动创建时间戳输出目录
- 零配置，开箱即用
"""

# PyTorch 2.8.0兼容性修复 - 必须在导入torch相关模块之前执行
try:
    import pytorch_compatibility_fix
    print("[COMPAT] PyTorch 2.8.0兼容性修复已加载")
except Exception as e:
    print(f"[WARN] 兼容性修复加载失败: {e}")

import os
import sys
from pathlib import Path
from datetime import datetime
import torch  # 添加torch导入用于系统检查

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def find_audio_files():
    """查找输入目录中的音频文件"""
    input_dir = project_root / "input"
    if not input_dir.exists():
        input_dir.mkdir()
        print(f"已创建输入目录: {input_dir}")
        print("请将音频文件放入该目录后重新运行")
        return []
    
    # 支持的音频格式
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
        audio_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    return sorted(audio_files)

def check_system_status():
    """检查系统状态"""
    print("\n" + "=" * 60)
    print("系统状态检查")
    print("=" * 60)
    
    # 检查PyTorch和CUDA
    try:
        import torch
        print(f"[OK] PyTorch版本: {torch.__version__}")
        print(f"[OK] CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA版本: {torch.version.cuda}")
            print(f"[OK] GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"[OK] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("[!] GPU不可用，将使用CPU模式")
    except ImportError:
        print("[ERR] PyTorch未安装")
        return False
    
    # 检查MDX23
    mdx23_available = False
    mdx23_path = Path(project_root) / "MVSEP-MDX23-music-separation-model"
    if mdx23_path.exists():
        models_path = mdx23_path / "models"
        if models_path.exists():
            onnx_files = list(models_path.glob("*.onnx"))
            if onnx_files:
                print(f"[OK] MDX23已安装: {mdx23_path}")
                print(f"    找到{len(onnx_files)}个ONNX模型文件")
                for f in onnx_files[:3]:  # 显示前3个
                    print(f"    - {f.name} ({f.stat().st_size / 1024**2:.1f} MB)")
                mdx23_available = True
            else:
                print(f"[!] MDX23模型文件缺失，请运行: python download_mdx23.py")
        else:
            print(f"[!] MDX23 models目录不存在")
    else:
        print(f"[!] MDX23未安装，建议克隆: https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model")
    
    # 检查配置的backend
    try:
        from src.vocal_smart_splitter.utils.config_manager import get_config
        backend = get_config('enhanced_separation.backend', 'auto')
        enable_gpu = get_config('enhanced_separation.gpu_config.enable_gpu', True)
        print(f"\n[配置信息]")
        print(f"  分离后端: {backend}")
        print(f"  GPU加速: {'启用' if enable_gpu else '禁用'}")
        
        # 确定最终backend
        if backend == 'auto':
            if mdx23_available and torch.cuda.is_available():
                final_backend = 'mdx23'
            else:
                final_backend = 'hpss_fallback'
                print(f"  [!] 将使用HPSS后备方案（MDX23不可用）")
        else:
            final_backend = backend
        
        print(f"  最终后端: {final_backend}")
        
        # 如果是增强模式但MDX23不可用，给出警告
        if final_backend == 'hpss_fallback' and not mdx23_available:
            print("\n" + "=" * 60)
            print("[警告] MDX23未安装，无法使用高精度分离")
            print("建议安装MDX23以获得更好的分割效果：")
            print("1. git clone https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model")
            print("2. python download_mdx23.py")
            print("当前将使用HPSS基础方案")
            print("=" * 60)
            
    except Exception as e:
        print(f"[!] 配置读取失败: {e}")
    
    print("=" * 60)
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("智能人声分割器 - 快速启动 (增强版)")
    print("=" * 60)
    
    # 检查系统状态
    if not check_system_status():
        print("\n[ERROR] 系统检查失败，请检查环境配置")
        return
    
    # 查找音频文件
    audio_files = find_audio_files()
    
    if not audio_files:
        print("[ERROR] 未找到音频文件")
        print("\n请执行以下步骤：")
        print("1. 将音频文件复制到 input/ 目录")
        print("2. 支持格式: MP3, WAV, FLAC, M4A")
        print("3. 重新运行此脚本")
        return
    
    print(f"[INFO] 发现 {len(audio_files)} 个音频文件:")
    for i, file_path in enumerate(audio_files, 1):
        print(f"  {i}. {file_path.name}")
    
    # 选择文件
    if len(audio_files) == 1:
        selected_file = audio_files[0]
        print(f"\n[AUTO] 自动选择: {selected_file.name}")
    else:
        print(f"\n请选择要分割的文件 (1-{len(audio_files)}):")
        try:
            choice = int(input("输入序号: ").strip())
            if 1 <= choice <= len(audio_files):
                selected_file = audio_files[choice - 1]
            else:
                print("[ERROR] 序号无效")
                return
        except ValueError:
            print("[ERROR] 输入无效")
            return
    
    print(f"[SELECT] 选择文件: {selected_file.name}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "output" / f"quick_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[OUTPUT] 输出目录: {output_dir.name}")
    print("\n[START] 开始分割...")
    
    try:
        # 导入分割器
        from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
        from src.vocal_smart_splitter.utils.config_manager import get_config
        
        # 获取配置
        sample_rate = get_config('audio.sample_rate', 44100)
        backend = get_config('enhanced_separation.backend', 'auto')
        
        print(f"\n[CONFIG] 使用配置：")
        print(f"  采样率: {sample_rate} Hz")
        print(f"  分离后端: {backend}")
        print(f"  双路检测: 启用")
        print(f"  BPM自适应: 启用")
        
        # 创建分割器
        print("\n[INIT] 初始化分割器...")
        splitter = SeamlessSplitter(sample_rate=sample_rate)
        
        # 执行分割
        print("[PROCESS] 开始分割处理...")
        result = splitter.split_audio_seamlessly(str(selected_file), str(output_dir))
        
        # 显示处理统计
        if 'processing_stats' in result:
            stats = result['processing_stats']
            print(f"\n[STATS] 处理统计：")
            if 'backend_used' in stats:
                print(f"  实际使用后端: {stats['backend_used']}")
            if 'dual_path_used' in stats:
                print(f"  双路检测执行: {'是' if stats['dual_path_used'] else '否'}")
            if 'processing_time' in stats:
                print(f"  处理时间: {stats['processing_time']:.1f}秒")
        
        if result.get('success', False):
            print("\n" + "=" * 50)
            print("[SUCCESS] 分割成功完成!")
            print("=" * 50)
            
            # 显示结果
            num_segments = result.get('num_segments', 0)
            print(f"[INFO] 生成片段数量: {num_segments}")
            
            # 显示分割文件
            saved_files = result.get('saved_files', [])
            if saved_files:
                print("\n[FILES] 生成的文件:")
                for i, file_path in enumerate(saved_files, 1):
                    file_name = Path(file_path).name
                    file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
                    print(f"  {i:2d}. {file_name} ({file_size:.1f}MB)")
            
            # 显示质量信息
            if 'vocal_pause_analysis' in result:
                pause_info = result['vocal_pause_analysis']
                total_pauses = pause_info.get('total_pauses', 0)
                avg_confidence = pause_info.get('avg_confidence', 0)
                print(f"\n[QUALITY] 检测质量:")
                print(f"  停顿检测: {total_pauses} 个")
                print(f"  平均置信度: {avg_confidence:.3f}")
                
                # 显示双路检测信息
                if 'dual_detection_info' in pause_info:
                    dual_info = pause_info['dual_detection_info']
                    print(f"\n[DUAL-PATH] 双路检测详情:")
                    print(f"  混音检测: {dual_info.get('mixed_detections', 0)} 个停顿")
                    print(f"  分离检测: {dual_info.get('separated_detections', 0)} 个停顿")
                    print(f"  交叉验证: {dual_info.get('validated_pauses', 0)} 个确认")
                    if 'separation_backend' in dual_info:
                        print(f"  分离后端: {dual_info['separation_backend']}")
            
            # 重构验证
            if 'seamless_validation' in result:
                validation = result['seamless_validation']
                perfect = validation.get('perfect_reconstruction', False)
                print(f"  重构验证: {'[PERFECT]' if perfect else '[DIFF]'}")
            
            print(f"\n[OUTPUT] 输出目录: {output_dir}")
            print("[SUCCESS] 可以直接使用这些音频片段!")
            
        else:
            print("[ERROR] 分割失败")
            if 'error' in result:
                print(f"错误: {result['error']}")
                
    except ImportError as e:
        print(f"[ERROR] 模块导入失败: {e}")
        print("\n请检查环境配置:")
        print("1. 确认已安装依赖: pip install -r requirements.txt")
        print("2. 确认虚拟环境已激活")
        print("3. 如需MDX23支持: python download_mdx23.py")
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        print("\n详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
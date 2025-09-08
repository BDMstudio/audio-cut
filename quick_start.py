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

def check_backend_availability():
    """检查各分离后端的可用性，返回可用后端列表"""
    available_backends = {}
    
    # 检查PyTorch和CUDA
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_info = f"{gpu_name} ({gpu_memory:.1f}GB)"
        else:
            gpu_info = "不可用"
    except ImportError:
        return {}
    
    # 检查MDX23
    mdx23_path = Path(project_root) / "MVSEP-MDX23-music-separation-model"
    if mdx23_path.exists() and (mdx23_path / "models").exists():
        onnx_files = list((mdx23_path / "models").glob("*.onnx"))
        if onnx_files:
            available_backends['mdx23'] = {
                'name': 'MDX23 (最高质量)',
                'description': f'ONNX神经网络分离，找到{len(onnx_files)}个模型',
                'gpu_required': True,
                'gpu_available': gpu_available
            }
    
    # 检查Demucs
    try:
        import demucs.pretrained
        available_backends['demucs_v4'] = {
            'name': 'Demucs v4 (高质量)',
            'description': 'Facebook开源神经网络分离',
            'gpu_required': False,  # CPU也能工作，但GPU更快
            'gpu_available': gpu_available
        }
    except ImportError:
        pass
    
    # HPSS总是可用作为后备
    available_backends['hpss_fallback'] = {
        'name': 'HPSS (传统方法)',
        'description': '谐波-冲击分离，速度快但质量一般',
        'gpu_required': False,
        'gpu_available': True  # 总是可用
    }
    
    return available_backends, gpu_info

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
    
    # 检查各个后端
    available_backends, gpu_info = check_backend_availability()
    print(f"\n[INFO] 可用的分离后端:")
    for backend_id, info in available_backends.items():
        status = "✓" if (not info['gpu_required'] or info['gpu_available']) else "!"
        print(f"  [{status}] {info['name']}: {info['description']}")
    
    return True

def select_backend():
    """让用户选择分离后端"""
    available_backends, gpu_info = check_backend_availability()
    
    print("\n" + "=" * 60)
    print("选择分离技术")
    print("=" * 60)
    print("请选择要使用的人声分离技术：")
    print()
    
    backend_options = []
    option_num = 1
    
    for backend_id, info in available_backends.items():
        if not info['gpu_required'] or info['gpu_available']:
            print(f"  {option_num}. {info['name']}")
            print(f"     {info['description']}")
            if info['gpu_required'] and info['gpu_available']:
                print(f"     [GPU加速] {gpu_info}")
            elif info['gpu_required'] and not info['gpu_available']:
                print(f"     [需要GPU] GPU不可用，此选项将无法使用")
                continue
            else:
                print(f"     [CPU/GPU兼容]")
            print()
            backend_options.append(backend_id)
            option_num += 1
    
    # 添加自动选择选项
    print(f"  {option_num}. 自动选择 (推荐)")
    print(f"     系统自动选择最佳可用后端")
    print()
    backend_options.append('auto')
    
    try:
        choice = int(input(f"请选择 (1-{len(backend_options)}): ").strip())
        if 1 <= choice <= len(backend_options):
            selected_backend = backend_options[choice - 1]
            backend_name = available_backends.get(selected_backend, {}).get('name', '自动选择')
            print(f"[SELECT] 已选择: {backend_name}")
            return selected_backend
        else:
            print("[ERROR] 选择无效，使用自动模式")
            return 'auto'
    except ValueError:
        print("[ERROR] 输入无效，使用自动模式")
        return 'auto'

def apply_backend_config(selected_backend):
    """应用用户选择的后端配置，强制使用指定后端"""
    if selected_backend == 'auto':
        print(f"\n[CONFIG] 使用自动选择模式")
        return None
    
    # 使用环境变量强制设置后端（这是最可靠的方法）
    import os
    os.environ['FORCE_SEPARATION_BACKEND'] = selected_backend
    
    print(f"\n[CONFIG] ✓ 强制设置分离后端: {selected_backend}")
    print(f"[CONFIG] ✓ 环境变量已设置: FORCE_SEPARATION_BACKEND={selected_backend}")
    
    return selected_backend

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
    
    # 选择分离后端
    selected_backend = select_backend()
    forced_backend = apply_backend_config(selected_backend)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if forced_backend:
        output_dir = project_root / "output" / f"quick_{forced_backend}_{timestamp}"
    else:
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
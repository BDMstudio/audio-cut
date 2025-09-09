#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_separator_only.py - 增强型人声分离脚本，支持交互式选择
# AI-SUMMARY: 独立的人声分离工具，支持多种分离后端，交互式文件和技术选择，只分离人声和伴奏，不做分割

"""
智能人声分离器 - 增强版

特点：
- 交互式文件选择：自动扫描input/目录，支持多种音频格式
- 智能后端选择：MDX23/Demucs v4/HPSS，根据系统自动推荐
- 系统状态检查：检测GPU、CUDA、模型文件可用性
- 多种使用模式：
  1. 交互模式：python vocal_separator_only.py
  2. 快速批量：python vocal_separator_only.py --quick
  3. 命令行模式：python vocal_separator_only.py input.mp3 -b mdx23

支持格式：MP3, WAV, FLAC, M4A
输出格式：WAV (无损)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# PyTorch兼容性修复
try:
    import pytorch_compatibility_fix
    print("[COMPAT] PyTorch兼容性修复已加载")
except Exception as e:
    print(f"[WARN] 兼容性修复加载失败: {e}")

from src.vocal_smart_splitter.core.enhanced_vocal_separator import EnhancedVocalSeparator
from src.vocal_smart_splitter.utils.audio_processor import AudioProcessor
import numpy as np
import librosa
import soundfile as sf
import torch

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_audio_files():
    """查找输入目录中的音频文件"""
    project_root = Path(__file__).parent
    input_dir = project_root / "input"
    if not input_dir.exists():
        input_dir.mkdir()
        print(f"已创建输入目录: {input_dir}")
        print("请将音频文件放入该目录后重新运行")
        return []
    
    # 支持的音频格式
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a']
    audio_files = set()  # 使用集合避免重复
    
    for ext in audio_extensions:
        audio_files.update(input_dir.glob(f"*{ext}"))
        audio_files.update(input_dir.glob(f"*{ext.upper()}"))
    
    return sorted(list(audio_files))

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
    project_root = Path(__file__).parent
    mdx23_path = project_root / "MVSEP-MDX23-music-separation-model"
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
        status = "OK" if (not info['gpu_required'] or info['gpu_available']) else "NO"
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

def separate_vocals_only(input_file: str, output_dir: str, backend: str = 'auto', 
                        sample_rate: int = 44100) -> dict:
    """纯人声分离，不进行分割
    
    Args:
        input_file: 输入音频文件
        output_dir: 输出目录
        backend: 分离后端 ('mdx23', 'demucs_v4', 'hpss_fallback', 'auto')
        sample_rate: 采样率
        
    Returns:
        分离结果信息
    """
    logger.info(f"开始人声分离: {input_file}")
    logger.info(f"使用后端: {backend}")
    
    try:
        # 设置后端环境变量
        if backend != 'auto':
            os.environ['FORCE_SEPARATION_BACKEND'] = backend
            logger.info(f"强制使用后端: {backend}")
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. 加载音频
        logger.info("加载音频文件...")
        audio_processor = AudioProcessor(sample_rate)
        audio, sr = librosa.load(input_file, sr=sample_rate, mono=True)
        
        logger.info(f"音频信息: 时长 {len(audio)/sr:.2f}秒, 采样率 {sr}Hz")
        
        # 2. 初始化分离器
        logger.info("初始化人声分离器...")
        separator = EnhancedVocalSeparator(sample_rate)
        
        # 3. 执行分离
        logger.info("开始人声分离...")
        start_time = datetime.now()
        
        separation_result = separator.separate_for_detection(audio)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 4. 保存结果
        input_name = Path(input_file).stem
        
        if separation_result.vocal_track is not None:
            # 保存人声
            vocal_file = Path(output_dir) / f"{input_name}_vocal.wav"
            sf.write(vocal_file, separation_result.vocal_track, sample_rate)
            logger.info(f"人声已保存: {vocal_file}")
            
            # 保存伴奏（如果有）
            if separation_result.instrumental_track is not None:
                instrumental_file = Path(output_dir) / f"{input_name}_instrumental.wav"
                sf.write(instrumental_file, separation_result.instrumental_track, sample_rate)
                logger.info(f"伴奏已保存: {instrumental_file}")
            
            # 生成分离报告
            result = {
                'success': True,
                'input_file': input_file,
                'output_dir': output_dir,
                'vocal_file': str(vocal_file),
                'instrumental_file': str(instrumental_file) if separation_result.instrumental_track is not None else None,
                'backend_used': separation_result.backend_used,
                'separation_confidence': separation_result.separation_confidence,
                'processing_time': processing_time,
                'audio_duration': len(audio) / sr,
                'quality_metrics': separation_result.quality_metrics or {}
            }
            
            logger.info(f"分离完成!")
            logger.info(f"  使用后端: {separation_result.backend_used}")
            logger.info(f"  分离质量: {separation_result.separation_confidence:.3f}")
            logger.info(f"  处理时间: {processing_time:.1f}秒")
            
            return result
            
        else:
            logger.error("人声分离失败")
            return {
                'success': False,
                'error': '人声分离返回空结果',
                'input_file': input_file
            }
            
    except Exception as e:
        logger.error(f"人声分离失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'input_file': input_file
        }
    finally:
        # 清理环境变量
        if 'FORCE_SEPARATION_BACKEND' in os.environ:
            del os.environ['FORCE_SEPARATION_BACKEND']

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='纯人声分离工具')
    parser.add_argument('input', nargs='?', help='输入音频文件（可选，不提供时进入交互模式）')
    parser.add_argument('-o', '--output', help='输出目录', 
                       default='output/vocal_separation')
    parser.add_argument('-b', '--backend', 
                       choices=['mdx23', 'demucs_v4', 'hpss_fallback', 'auto'],
                       default='auto', help='分离后端')
    parser.add_argument('-sr', '--sample-rate', type=int, default=44100,
                       help='采样率')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='详细输出')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='强制进入交互模式')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 如果没有提供输入文件或强制交互模式，进入交互模式
    if not args.input or args.interactive:
        return interactive_mode()
    
    # 检查输入文件
    if not Path(args.input).exists():
        logger.error(f"输入文件不存在: {args.input}")
        return 1
    
    # 添加时间戳到输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_{timestamp}"
    
    # 执行分离
    result = separate_vocals_only(
        args.input, 
        output_dir, 
        args.backend, 
        args.sample_rate
    )
    
    if result['success']:
        print(f"\n分离成功!")
        print(f"输出目录: {result['output_dir']}")
        print(f"人声文件: {Path(result['vocal_file']).name}")
        if result['instrumental_file']:
            print(f"伴奏文件: {Path(result['instrumental_file']).name}")
        print(f"使用后端: {result['backend_used']}")
        print(f"分离质量: {result['separation_confidence']:.1%}")
        print(f"处理时间: {result['processing_time']:.1f}秒")
        return 0
    else:
        print(f"\n分离失败: {result.get('error', '未知错误')}")
        return 1

def interactive_mode():
    """交互模式主函数"""
    print("=" * 60)
    print("智能人声分离器 - 交互模式")
    print("=" * 60)
    
    # 检查系统状态
    if not check_system_status():
        print("\n[ERROR] 系统检查失败，请检查环境配置")
        return 1
    
    # 查找音频文件
    audio_files = find_audio_files()
    
    if not audio_files:
        print("[ERROR] 未找到音频文件")
        print("\n请执行以下步骤：")
        print("1. 将音频文件复制到 input/ 目录")
        print("2. 支持格式: MP3, WAV, FLAC, M4A")
        print("3. 重新运行此脚本")
        return 1
    
    print(f"[INFO] 发现 {len(audio_files)} 个音频文件:")
    for i, file_path in enumerate(audio_files, 1):
        print(f"  {i}. {file_path.name}")
    
    # 选择文件
    if len(audio_files) == 1:
        selected_file = audio_files[0]
        print(f"\n[AUTO] 自动选择: {selected_file.name}")
    else:
        print(f"\n请选择要分离的文件 (1-{len(audio_files)}):")
        try:
            choice = int(input("输入序号: ").strip())
            if 1 <= choice <= len(audio_files):
                selected_file = audio_files[choice - 1]
            else:
                print("[ERROR] 序号无效")
                return 1
        except ValueError:
            print("[ERROR] 输入无效")
            return 1
    
    print(f"[SELECT] 选择文件: {selected_file.name}")
    
    # 选择分离后端
    selected_backend = select_backend()
    
    # 应用后端配置
    if selected_backend != 'auto':
        os.environ['FORCE_SEPARATION_BACKEND'] = selected_backend
        print(f"\n[CONFIG] 强制设置分离后端: {selected_backend}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if selected_backend != 'auto':
        output_dir = f"output/vocal_{selected_backend}_{timestamp}"
    else:
        output_dir = f"output/vocal_auto_{timestamp}"
    
    print(f"[OUTPUT] 输出目录: {Path(output_dir).name}")
    print("\n[START] 开始分离...")
    
    try:
        # 执行分离
        result = separate_vocals_only(
            str(selected_file), 
            output_dir, 
            selected_backend, 
            44100
        )
        
        if result['success']:
            print("\n" + "=" * 50)
            print("[SUCCESS] 分离成功完成!")
            print("=" * 50)
            
            print(f"[INFO] 输出目录: {result['output_dir']}")
            print(f"[INFO] 人声文件: {Path(result['vocal_file']).name}")
            if result['instrumental_file']:
                print(f"[INFO] 伴奏文件: {Path(result['instrumental_file']).name}")
            
            # 显示文件大小
            vocal_size = Path(result['vocal_file']).stat().st_size / (1024 * 1024)  # MB
            print(f"[INFO] 人声文件大小: {vocal_size:.1f}MB")
            
            if result['instrumental_file']:
                inst_size = Path(result['instrumental_file']).stat().st_size / (1024 * 1024)  # MB
                print(f"[INFO] 伴奏文件大小: {inst_size:.1f}MB")
            
            print(f"\n[QUALITY] 分离质量: {result['separation_confidence']:.1%}")
            print(f"[BACKEND] 使用后端: {result['backend_used']}")
            print(f"[TIME] 处理时间: {result['processing_time']:.1f}秒")
            
            # 显示质量指标
            if result.get('quality_metrics'):
                metrics = result['quality_metrics']
                print(f"\n[METRICS] 质量指标:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
            
            print(f"\n[SUCCESS] 分离完成! 可以直接使用这些音频文件!")
            return 0
            
        else:
            print("[ERROR] 分离失败")
            if 'error' in result:
                print(f"错误: {result['error']}")
            return 1
            
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        print("\n详细错误信息:")
        print(traceback.format_exc())
        return 1
    finally:
        # 清理环境变量
        if 'FORCE_SEPARATION_BACKEND' in os.environ:
            del os.environ['FORCE_SEPARATION_BACKEND']

def quick_separate():
    """快速分离函数，用于直接调用"""
    input_files = [
        "input/15.MP3",
        "input/16.MP3", 
        "input/17.MP3"
    ]
    
    output_base = "output/vocal_only"
    
    for input_file in input_files:
        if not Path(input_file).exists():
            print(f"跳过不存在的文件: {input_file}")
            continue
            
        print(f"\n处理文件: {input_file}")
        
        # 为每个文件创建独立目录
        file_stem = Path(input_file).stem
        output_dir = f"{output_base}/{file_stem}"
        
        result = separate_vocals_only(input_file, output_dir, 'auto')
        
        if result['success']:
            print(f"{file_stem} 分离完成")
        else:
            print(f"{file_stem} 分离失败")

if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) == 1:
        # 没有参数时，进入交互模式
        sys.exit(interactive_mode())
    elif len(sys.argv) == 2 and sys.argv[1] == "--quick":
        # 快速批量分离模式
        print("快速分离模式 - 处理input/目录中的音频文件")
        quick_separate()
    else:
        # 有其他参数时，使用命令行模式
        sys.exit(main())
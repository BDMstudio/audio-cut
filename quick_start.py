#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# quick_start.py - 快速启动脚本 (重构版)
# AI-SUMMARY: 精简的传令兵模式快速启动脚本，统一调用SeamlessSplitter

"""
智能人声分割器 - 快速启动脚本 (重构版)

重构优势：
- 移除了200多行的"影子大脑"逻辑
- 统一调用SeamlessSplitter作为唯一指挥中心  
- 清晰的职责划分：快速启动脚本只负责用户交互
- 所有模式的核心逻辑集中在seamless_splitter.py

使用方式：
1. 将音频文件放入 input/ 目录
2. 运行 python quick_start.py
3. 选择处理模式和后端
4. 在 output/ 目录查看结果
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
    
    # 支持的音频格式（统一用小写匹配，避免Windows大小写重复）
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a'}
    seen = set()
    audio_files = []

    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in audio_extensions:
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                audio_files.append(p)

    # 名称不区分大小写排序
    return sorted(audio_files, key=lambda x: x.name.lower())

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
        return {}, "PyTorch未安装"
    
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
    
    return available_backends, gpu_info

def check_system_status():
    """检查系统状态"""
    print("\\n" + "=" * 60)
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
    print(f"\\n[INFO] 可用的分离后端:")
    for backend_id, info in available_backends.items():
        status = "OK" if (not info['gpu_required'] or info['gpu_available']) else "NO"
        print(f"  [{status}] {info['name']}: {info['description']}")
    
    return True

def select_backend():
    """让用户选择分离后端"""
    available_backends, gpu_info = check_backend_availability()
    
    print("\\n" + "=" * 60)
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
        print(f"\\n[CONFIG] 使用自动选择模式")
        return None
    
    # 使用环境变量强制设置后端（这是最可靠的方法）
    import os
    os.environ['FORCE_SEPARATION_BACKEND'] = selected_backend
    
    print(f"\\n[CONFIG] 强制设置分离后端: {selected_backend}")
    print(f"[CONFIG] 环境变量已设置: FORCE_SEPARATION_BACKEND={selected_backend}")
    
    return selected_backend

def select_processing_mode():
    """让用户选择处理模式"""
    print("\\n" + "=" * 60)
    print("选择处理模式")
    print("=" * 60)
    print("请选择要执行的处理类型：")
    print()
    
    print("  1. 智能分割")
    print("     根据人声停顿点自动分割音频为多个片段")
    print("     适合：语音训练、音频片段制作")
    print()
    
    print("  2. 纯人声分离")
    print("     只分离人声和伴奏，不进行分割")
    print("     适合：音乐制作、卡拉OK制作")
    print()
    
    print("  3. [NEW] 纯人声检测v2.1 (推荐)")
    print("     统计学动态裁决+BPM自适应优化+边界保护")
    print("     适合：高质量语音训练、解决快歌切割瓶颈问题")
    print()
    
    print("  4. [LATEST] MDD增强纯人声检测v2.2 (最新)")
    print("     集成音乐动态密度(MDD)主副歌智能识别")
    print("     适合：专业级音频处理、自动主副歌差异化切割")
    print()
    
    try:
        choice = int(input("请选择 (1-4): ").strip())
        if choice == 1:
            print("[SELECT] 已选择: 智能分割")
            return 'smart_split'
        elif choice == 2:
            print("[SELECT] 已选择: 纯人声分离")
            return 'vocal_separation'
        elif choice == 3:
            print("[SELECT] 已选择: [NEW] 纯人声检测v2.1")
            return 'vocal_split_v2'
        elif choice == 4:
            print("[SELECT] 已选择: [LATEST] MDD增强纯人声检测v2.2")
            return 'vocal_split_mdd'
        else:
            print("[ERROR] 选择无效，使用默认MDD v2.2模式")
            return 'vocal_split_mdd'
    except ValueError:
        print("[ERROR] 输入无效，使用默认MDD v2.2模式")
        return 'vocal_split_mdd'

def main():
    """主函数 - 重构为纯传令兵模式"""
    print("=" * 60)
    print("智能人声分割器 - 快速启动 (重构版)")
    print("=" * 60)
    
    # 检查系统状态
    if not check_system_status():
        print("\\n[ERROR] 系统检查失败，请检查环境配置")
        return
    
    # 查找音频文件
    audio_files = find_audio_files()
    
    if not audio_files:
        print("[ERROR] 未找到音频文件")
        print("\\n请执行以下步骤：")
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
        print(f"\\n[AUTO] 自动选择: {selected_file.name}")
    else:
        print(f"\\n请选择要分割的文件 (1-{len(audio_files)}):")
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
    
    # 选择处理模式
    processing_mode = select_processing_mode()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if forced_backend and forced_backend != 'auto':
        output_dir = project_root / "output" / f"unified_{forced_backend}_{timestamp}"
    else:
        output_dir = project_root / "output" / f"unified_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] 输出目录: {output_dir.name}")
    
    try:
        # === 核心改造：统一调用指挥中心 ===
        
        # 1. 实例化官方大脑
        from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
        from src.vocal_smart_splitter.utils.config_manager import get_config
        
        # 从配置文件获取初始采样率，但它会在加载时动态调整
        sample_rate = get_config('audio.sample_rate', 44100)
        splitter = SeamlessSplitter(sample_rate=sample_rate)
        
        # 2. 模式映射：将用户界面的模式映射到SeamlessSplitter的模式
        mode_mapping = {
            'vocal_separation': 'vocal_separation',
            'vocal_split_v2': 'v2.1', 
            'vocal_split_mdd': 'v2.2_mdd',
            'smart_split': 'smart_split'
        }
        
        seamless_mode = mode_mapping.get(processing_mode, 'v2.2_mdd')  # 默认使用v2.2 MDD
        
        # 3. 将用户选择的模式作为参数，下达指令
        print(f"\\n[START] 正在启动统一分割引擎，模式: {seamless_mode}...")
        print(f"[CONFIG] 使用分离后端: {selected_backend}")
        
        result = splitter.split_audio_seamlessly(
            str(selected_file), 
            str(output_dir), 
            mode=seamless_mode
        )
        
        # 4. 显示结果
        if result.get('success'):
            print("\\n" + "=" * 60)
            print("[SUCCESS] 智能分割成功完成!")
            print("=" * 60)
            print(f"  生成片段数量: {result.get('num_segments', 0)}")
            print(f"  文件保存在: {output_dir}")
            print(f"  处理方法: {result.get('method', seamless_mode)}")
            if 'backend_used' in result:
                print(f"  使用后端: {result['backend_used']}")
            if 'processing_time' in result:
                print(f"  处理时间: {result['processing_time']:.1f}秒")
        else:
            print("\\n[ERROR] 处理失败:")
            print(f"错误: {result.get('error', '未知错误')}")

    except ImportError as e:
        print(f"[ERROR] 模块导入失败: {e}")
        print("\\n请检查环境配置:")
        print("1. 确认已安装依赖: pip install -r requirements.txt")
        print("2. 确认虚拟环境已激活")
        print("3. 如需MDX23支持: python download_mdx23.py")
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        print("\\n详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
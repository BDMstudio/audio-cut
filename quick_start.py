#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# quick_start.py - 快速启动脚本 (v2.3 统一指挥中心版)
# AI-SUMMARY: 精简的传令兵模式快速启动脚本，统一调用SeamlessSplitter

import sys
import logging
from pathlib import Path
from datetime import datetime
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from src.vocal_smart_splitter.utils.config_manager import get_config

# --- (保留 find_audio_files, check_system_status, select_backend 等所有用户交互函数，无需改动) ---

def find_audio_files():
    """查找输入目录中的音频文件"""
    input_dir = project_root / "input"
    if not input_dir.exists():
        input_dir.mkdir()
        print(f"已创建输入目录: {input_dir}")
        print("请将音频文件放入该目录后重新运行")
        return []
    
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a'}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in audio_extensions], key=lambda x: x.name.lower())

def check_system_status():
    """检查系统状态"""
    print("\n" + "=" * 60)
    print("系统状态检查")
    print("=" * 60)
    try:
        import torch
        print(f"[OK] PyTorch版本: {torch.__version__}")
        gpu_available = torch.cuda.is_available()
        print(f"[OK] CUDA可用: {gpu_available}")
        if gpu_available:
            print(f"[OK] GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[ERR] PyTorch未安装")
        return False
    return True

def select_processing_mode():
    """让用户选择处理模式"""
    print("\n" + "=" * 60)
    print("选择处理模式")
    print("=" * 60)
    print("  1. 智能分割 (Smart Split)")
    print("     - 在原始混音上识别人声停顿并分割。")
    print("  2. 纯人声分离 (Vocal Separation Only)")
    print("     - 仅分离人声和伴奏，不分割。")
    print("  3. [推荐] 纯人声检测v2.1 (Pure Vocal v2.1)")
    print("     - 先分离再检测，使用统计学动态裁决，适合快歌。")
    print("  4. [最新] MDD增强纯人声检测v2.2 (Pure Vocal v2.2 MDD)")
    print("     - 在v2.1基础上，集成音乐动态密度(MDD)识别主副歌。")
    print()
    
    try:
        choice = int(input("请选择 (1-4): ").strip())
        modes = {1: 'smart_split', 2: 'vocal_separation', 3: 'v2.1', 4: 'v2.2_mdd'}
        mode = modes.get(choice, 'v2.2_mdd')
        print(f"[SELECT] 已选择模式: {mode}")
        return mode
    except ValueError:
        print("[ERROR] 输入无效，使用默认MDD v2.2模式")
        return 'v2.2_mdd'

def main():
    """主函数 - 重构为纯传令兵模式"""
    # 轻量日志配置：让核心模块的INFO日志在控制台可见
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    print("=" * 60)
    print("智能人声分割器 - 快速启动 (v2.3 统一指挥中心版)")
    print("=" * 60)
    
    if not check_system_status(): return
    audio_files = find_audio_files()
    if not audio_files: return

    print(f"[INFO] 发现 {len(audio_files)} 个音频文件:")
    for i, file_path in enumerate(audio_files, 1): print(f"  {i}. {file_path.name}")
    
    try:
        choice = 1 if len(audio_files) == 1 else int(input(f"\n请选择要分割的文件 (1-{len(audio_files)}): ").strip())
        selected_file = audio_files[choice - 1]
    except (ValueError, IndexError):
        print("[ERROR] 选择无效")
        return

    print(f"[SELECT] 选择文件: {selected_file.name}")
    processing_mode = select_processing_mode()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "output" / f"quick_{processing_mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] 输出目录: {output_dir.name}")

    try:
        # 轻量环境诊断日志，便于定位虚拟环境与后端问题
        import sys as _sys, os as _os
        print(f"[DIAG] Python: {_sys.executable}")
        print(f"[DIAG] VIRTUAL_ENV: {_os.environ.get('VIRTUAL_ENV', '')}")
        print(f"[DIAG] FORCE_SEPARATION_BACKEND: {_os.environ.get('FORCE_SEPARATION_BACKEND', '')}")

        # === 核心改造：统一调用指挥中心 ===
        sample_rate = get_config('audio.sample_rate', 44100)
        splitter = SeamlessSplitter(sample_rate=sample_rate)
        
        print(f"\n[START] 正在启动统一分割引擎，模式: {processing_mode}...")
        result = splitter.split_audio_seamlessly(
            str(selected_file), 
            str(output_dir), 
            mode=processing_mode
        )
        
        if result.get('success'):
            print("\n" + "=" * 50)
            print("[SUCCESS] 处理成功完成!")
            print("=" * 50)
            print(f"  处理方法: {result.get('method', 'N/A')}")
            print(f"  生成片段数量: {result.get('num_segments', 0)}")
            print(f"  文件保存在: {output_dir}")
            if 'backend_used' in result: print(f"  使用后端: {result['backend_used']}")
            if 'processing_time' in result: print(f"  总耗时: {result['processing_time']:.1f}秒")
        else:
            print(f"\n[ERROR] 处理失败: {result.get('error', '未知错误')}")
            # 输出最小诊断信息
            print(f"[DIAG] Python: {_sys.executable}")
            print(f"[DIAG] VIRTUAL_ENV: {_os.environ.get('VIRTUAL_ENV', '')}")
            print(f"[DIAG] FORCE_SEPARATION_BACKEND: {_os.environ.get('FORCE_SEPARATION_BACKEND', '')}")

    except Exception as e:
        print(f"[FATAL] 脚本顶层出现未捕获异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

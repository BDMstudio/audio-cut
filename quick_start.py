#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# quick_start.py - 快速启动脚本 (v2.3 统一指挥中心版)
# AI-SUMMARY: 精简的传令兵模式快速启动脚本，统一调用SeamlessSplitter

import sys
import logging
from pathlib import Path
from datetime import datetime
import torch

def _fmt_float(value):
    if value is None:
        return 'N/A'
    try:
        return f'{float(value):.3f}'
    except (TypeError, ValueError):
        return str(value)


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from src.vocal_smart_splitter.utils.config_manager import get_config
from src.vocal_smart_splitter.utils.audio_export import (
    get_supported_formats,
    ensure_supported_format,
)

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
    print("  1. 纯人声分离 (Vocal Separation Only)")
    print("     - 仅分离人声和伴奏，不执行切割")
    print("  2. [最新] MDD增强纯人声检测v2.2 (Pure Vocal v2.2 MDD)")
    print("     - 先分离再检测，集成音乐动态密度识别主副歌")
    print()

    try:
        choice = int(input("请选择 (1-2): ").strip())
        modes = {1: 'vocal_separation', 2: 'v2.2_mdd'}
        mode = modes.get(choice, 'v2.2_mdd')
        print(f"[SELECT] 已选择模式: {mode}")
        return mode
    except ValueError:
        print("[ERROR] 输入无效，使用默认MDD v2.2模式")
        return 'v2.2_mdd'


def select_output_format(default_format: str) -> str:
    """选择输出格式（模块化扩展接口）"""
    formats = get_supported_formats()
    default_key = ensure_supported_format(default_format)
    format_map = {fmt.name.lower(): fmt for fmt in formats}
    index_map = {idx: fmt for idx, fmt in enumerate(formats, 1)}

    print("\n" + "=" * 60)
    print("选择输出格式")
    print("=" * 60)
    for idx, fmt in index_map.items():
        is_default = "(默认)" if fmt.name.lower() == default_key else ""
        print(f"  {idx}. {fmt.name.upper():<6} -> .{fmt.extension} {fmt.description} {is_default}")

    prompt = f"请选择输出格式 (输入序号或名称，直接回车沿用 {default_key.upper()}): "
    selection = input(prompt).strip().lower()
    if not selection:
        return default_key

    chosen_format = None
    if selection.isdigit():
        chosen_format = index_map.get(int(selection))
    else:
        try:
            chosen_key = ensure_supported_format(selection)
            chosen_format = format_map.get(chosen_key)
        except ValueError:
            chosen_format = None

    if chosen_format is None:
        print("[WARN] 输入无效，沿用默认格式。")
        return default_key

    return chosen_format.name.lower()
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
    try:
        default_format = ensure_supported_format(get_config('output.format', 'wav'))
    except ValueError:
        default_format = 'wav'
    export_format = select_output_format(default_format)
    print(f"[SELECT] 输出格式: {export_format}")
    
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
            mode=processing_mode,
            export_format=export_format
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
            debug = result.get('segment_classification_debug', [])
            for idx, info in enumerate(debug, 1):
                print(
                    f"Segment {idx:02d}: label={result['segment_labels'][idx-1]} "
                    f"energy_ratio={_fmt_float(info.get('energy_ratio'))} "
                    f"presence_ratio={_fmt_float(info.get('presence_ratio'))} "
                    f"presence_baseline={_fmt_float(info.get('presence_baseline_db'))} "
                    f"marker_vote={info.get('marker_vote')} "
                    f"energy_vote={info.get('energy_vote')} "
                    f"presence_vote={info.get('presence_vote')} "
                    f"reason={info.get('decision_reason') or 'N/A'}"
                )
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

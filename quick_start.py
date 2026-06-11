#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: quick_start.py
# AI-SUMMARY: v2.8 intent-surface quick start with file selection plus three user-facing questions.

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from src.vocal_smart_splitter.utils.audio_export import ensure_supported_format
from src.vocal_smart_splitter.utils.config_manager import get_config, set_runtime_config

_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a'}


def find_audio_files() -> List[Path]:
    """Find supported audio files under the local input directory."""

    input_dir = project_root / 'input'
    if not input_dir.exists():
        input_dir.mkdir()
        print(f'已创建输入目录: {input_dir}')
        print('请将音频文件放入该目录后重新运行')
        return []
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in _AUDIO_EXTENSIONS],
        key=lambda item: item.name.lower(),
    )


def check_system_status() -> bool:
    """Check minimal runtime status before processing."""

    print('\n' + '=' * 60)
    print('系统状态检查')
    print('=' * 60)
    try:
        import torch
    except ImportError:
        print('[ERR] PyTorch未安装')
        return False
    print(f'[OK] PyTorch版本: {torch.__version__}')
    gpu_available = torch.cuda.is_available()
    print(f'[OK] CUDA可用: {gpu_available}')
    if gpu_available:
        print(f'[OK] GPU设备: {torch.cuda.get_device_name(0)}')
    return True


def select_processing_mode() -> str:
    """Ask whether to split the track or only export separated stems."""

    print('\n' + '=' * 60)
    print('要切片，还是只分离？')
    print('=' * 60)
    print('  1. 切片')
    print('     输出适合剪辑和 agent 工作流的片段。')
    print('  2. 只分离')
    print('     只导出人声和伴奏，不生成片段。')
    try:
        choice = int(input('请选择 (1-2，直接回车=1): ').strip() or '1')
    except ValueError:
        choice = 1
    mode = 'vocal_separation' if choice == 2 else 'vpbd_asr'
    print(f"[SELECT] {'只分离' if mode == 'vocal_separation' else '切片'}")
    return mode


def select_segment_density() -> str:
    """Ask for the target segment density."""

    print('\n' + '=' * 60)
    print('片段密度')
    print('=' * 60)
    print('  1. 少：约 10-18 秒')
    print('  2. 中：约 5-12 秒')
    print('  3. 多：约 3-8 秒')
    try:
        choice = int(input('请选择 (1-3，直接回车=2): ').strip() or '2')
    except ValueError:
        choice = 2
    value = {1: 'few', 2: 'medium', 3: 'many'}.get(choice, 'medium')
    print(f'[SELECT] 片段密度: {value}')
    return value


def select_alignment_style() -> str:
    """Ask for the lyric-to-beat alignment preference."""

    print('\n' + '=' * 60)
    print('切点风格')
    print('=' * 60)
    print('  1. 歌词优先')
    print('  2. 偏歌词')
    print('  3. 均衡')
    print('  4. 偏节拍')
    print('  5. 强卡点')
    try:
        choice = int(input('请选择 (1-5，直接回车=3): ').strip() or '3')
    except ValueError:
        choice = 3
    value = {
        1: 'lyric',
        2: 'lyric_lean',
        3: 'balanced',
        4: 'beat_lean',
        5: 'beat',
    }.get(choice, 'balanced')
    print(f'[SELECT] 切点风格: {value}')
    return value


def build_intent_runtime_overrides(*, segments: str, alignment: str) -> Dict[str, Any]:
    """Build runtime overrides for the v2.8 intent surface."""

    return {
        'smart_cut.segments': segments,
        'smart_cut.alignment': alignment,
        'lyrics_alignment.enabled': True,
        'lyrics_alignment.provider': 'auto',
        'lyrics_alignment.strict': False,
    }


def _select_target_files(audio_files: List[Path]) -> List[Path]:
    print(f'[INFO] 发现 {len(audio_files)} 个音频文件')
    for i, file_path in enumerate(audio_files, 1):
        print(f'  {i}. {file_path.name}')

    print('\n' + '=' * 60)
    print('选择处理范围')
    print('=' * 60)
    print('  1. 选择单个文件')
    print('  2. 处理 input 目录下全部音频')
    try:
        scope_choice = int(input('请选择处理范围 (1-2，直接回车=1): ').strip() or '1')
    except ValueError:
        scope_choice = 1
    if scope_choice == 2:
        print(f'[SELECT] 批量处理 {len(audio_files)} 个文件')
        return audio_files
    if len(audio_files) == 1:
        return [audio_files[0]]
    try:
        index = int(input(f'请选择文件 (1-{len(audio_files)}，直接回车=1): ').strip() or '1')
        return [audio_files[index - 1]]
    except (ValueError, IndexError):
        print('[ERROR] 选择无效')
        return []


def _output_dir_for(file_path: Path) -> Path:
    now = datetime.now()
    dirname = f"{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}_{file_path.stem}"
    output_dir = project_root / 'output' / dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def process_audio_file(
    splitter: SeamlessSplitter,
    file_path: Path,
    processing_mode: str,
    export_format: str,
) -> None:
    """Process one selected audio file."""

    print('\n' + '=' * 60)
    print(f'[SELECT] 正在处理: {file_path.name}')
    print('=' * 60)
    output_dir = _output_dir_for(file_path)
    print(f'[OUTPUT] 输出目录: {output_dir.name}')

    try:
        print('\n[START] 正在处理音频...')
        result = splitter.split_audio_seamlessly(
            str(file_path),
            str(output_dir),
            mode=processing_mode,
            export_format=export_format,
        )
    except Exception as exc:
        print(f'[FATAL] 处理 {file_path.name} 时出现未捕获异常: {exc}')
        import traceback
        traceback.print_exc()
        return

    if not result.get('success'):
        print(f"\n[ERROR] {file_path.name} 处理失败: {result.get('error', '未知错误')}")
        return

    print('\n' + '=' * 50)
    print(f'[SUCCESS] {file_path.name} 处理完成')
    print('=' * 50)
    print(f"  生成片段数量: {result.get('num_segments', 0)}")
    print(f'  文件保存至: {output_dir}')
    if 'backend_used' in result:
        print(f"  使用后端: {result['backend_used']}")
    if 'processing_time' in result:
        print(f"  总耗时: {result['processing_time']:.1f}s")



def main() -> None:
    """Run the quick-start flow."""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print('=' * 60)
    print('智能人声分割器 - 快速启动')
    print('=' * 60)

    if not check_system_status():
        return
    audio_files = find_audio_files()
    if not audio_files:
        return
    target_files = _select_target_files(audio_files)
    if not target_files:
        return

    processing_mode = select_processing_mode()
    if processing_mode != 'vocal_separation':
        density = select_segment_density()
        alignment = select_alignment_style()
        set_runtime_config(build_intent_runtime_overrides(segments=density, alignment=alignment))

    try:
        export_format = ensure_supported_format(get_config('output.format', 'wav'))
    except ValueError:
        export_format = 'wav'
    sample_rate = int(get_config('audio.sample_rate', 44100))
    splitter = SeamlessSplitter(sample_rate=sample_rate)

    for file_path in target_files:
        process_audio_file(splitter, file_path, processing_mode, export_format)


if __name__ == '__main__':
    main()

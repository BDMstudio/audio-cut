#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# quick_start.py - 快速启动脚本 (v2.7 主流程选择)
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
from src.vocal_smart_splitter.utils.config_manager import get_config, set_runtime_config
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
    """让用户按使用目标选择主流程，而不是按内部版本标签选择。"""
    print("\n" + "=" * 60)
    print("选择主流程")
    print("=" * 60)
    print("  1. 只做人声/伴奏分离")
    print("     用途: 准备素材，不切片。")
    print("  2. 稳定声学切分")
    print("     用途: 不依赖歌词 ASR，复用 v2.2 MDD 低谷检测。")
    print("  3. 歌词辅助自然切分")
    print("     用途: v2.7 主路径；声学低谷为主，歌词边界加分；mvagent 默认。")
    print("  4. 音乐卡点切分")
    print("     用途: MV/短视频卡点，Hybrid MDD + 节拍吸附。")
    print("  5. 节拍网格基线")
    print("     用途: librosa onset 调试或节奏基线对比。")
    print()

    modes = {
        1: 'vocal_separation',
        2: 'v2.2_mdd',
        3: 'vpbd_asr',
        4: 'hybrid_mdd',
        5: 'librosa_onset',
    }
    try:
        choice = int(input("请选择主流程 (1-5，直接回车=3): ").strip() or "3")
    except ValueError:
        choice = 3
    mode = modes.get(choice, 'vpbd_asr')
    print(f"[SELECT] 已选择主流程: {mode}")
    return mode


def select_smart_profile():
    """让用户选择 smart_cut profile；默认 auto。"""
    print("\n" + "=" * 60)
    print("选择风格策略")
    print("=" * 60)
    print("  1. Auto - 自动识别风格，适合批量处理和 mvagent")
    print("  2. Ballad - 抒情慢歌，切点更保守")
    print("  3. Pop - 流行歌曲，均衡默认")
    print("  4. EDM - 强节奏，提升节拍亲和")
    print("  5. Rap - 密集人声，偏向短句和气口")
    print()
    try:
        choice = int(input("请选择风格策略 (1-5，直接回车=1): ").strip() or "1")
    except ValueError:
        choice = 1
    profiles = {1: 'auto', 2: 'ballad', 3: 'pop', 4: 'edm', 5: 'rap'}
    profile = profiles.get(choice, 'auto')
    print(f"[SELECT] 已选择风格策略: {profile}")
    return profile


def select_hybrid_density():
    """让用户选择 hybrid_mdd 模式的卡点密度"""
    print("\n" + "=" * 60)
    print("选择卡点密度")
    print("=" * 60)
    print("  1. 少 - 较少或没有节拍卡点")
    print("  2. 中 - 默认卡点数量")
    print("  3. 多 - 更多卡点，更有灵动感 (可能碎片化)")
    print()

    try:
        choice = int(input("请选择 (1-3，默认2): ").strip() or "2")
        densities = {1: 'low', 2: 'medium', 3: 'high'}
        density = densities.get(choice, 'medium')
        print(f"[SELECT] 已选择卡点密度: {density}")
        return density
    except ValueError:
        print("[INFO] 使用默认密度 medium")
        return 'medium'


def select_lib_alignment():
    """让用户选择 hybrid_mdd 模式的节拍对齐策略"""
    print("\n" + "=" * 60)
    print("选择节拍对齐策略")
    print("=" * 60)
    print("  1. 强制节拍分割 (beat_only) - 副歌每小节切割")
    print("     - _lib片段: 副歌段纯节拍切割，适合强节奏卡点")
    print("  2. MDD智能吸附到节拍 (snap_to_beat) - 平衡方案")
    print("     - _lib片段: MDD切点吸附最近节拍，副歌额外添加节拍切点")
    print()

    try:
        choice = int(input("请选择 (1-2，默认2): ").strip() or "2")
        strategies = {1: 'beat_only', 2: 'snap_to_beat'}
        strategy = strategies.get(choice, 'snap_to_beat')
        print(f"[SELECT] 已选择对齐策略: {strategy}")
        return strategy
    except ValueError:
        print("[INFO] 使用默认策略 snap_to_beat")
        return 'snap_to_beat'


def build_vpbd_asr_runtime_overrides(
    provider: str,
    endpoint: str | None = None,
    cli_executable: str | None = None,
    fixture_path: str | None = None,
    strict: bool = False,
):
    """Build runtime config overrides for VPBD ASR quick-start choices."""

    normalized = str(provider or 'auto').strip().lower()
    if normalized == 'null':
        normalized = 'disabled'
    overrides = {
        'lyrics_alignment.enabled': normalized not in {'disabled'},
        'lyrics_alignment.provider': normalized,
        'lyrics_alignment.strict': bool(strict),
    }
    if endpoint:
        overrides['fire_red.endpoint'] = endpoint.strip()
    if cli_executable:
        overrides['fire_red.cli.executable'] = cli_executable.strip()
    if fixture_path:
        overrides['lyrics_alignment.fixture_path'] = fixture_path.strip()
    return overrides


def select_vpbd_asr_runtime_overrides():
    """让用户选择 VPBD ASR 歌词来源策略并返回运行时配置覆盖。"""

    print("\n" + "=" * 60)
    print("选择歌词来源策略")
    print("=" * 60)
    print("  1. 自动选择")
    print("     常规用户和 mvagent 默认: sidecar -> CLI -> 声学降级；无需知道部署方式。")
    print("  2. FireRed CLI worker")
    print("     已有本地 worker 脚本时使用；适合一次性批处理。")
    print("  3. FireRed sidecar")
    print("     已启动 HTTP 常驻服务时使用；适合高频批量。")
    print("  4. Fake fixture")
    print("     只用于回归测试，使用固定歌词时间轴。")
    print("  5. 关闭歌词")
    print("     退回 VPBD acoustic，只用声学低谷、气口和弱节拍候选。")
    print()

    try:
        choice = int(input("请选择歌词来源 (1-5，直接回车=1): ").strip() or "1")
    except ValueError:
        choice = 1

    provider_map = {1: 'auto', 2: 'cli', 3: 'sidecar', 4: 'fake', 5: 'disabled'}
    provider = provider_map.get(choice, 'auto')
    endpoint = None
    cli_executable = None
    fixture_path = None
    strict = False

    if provider == 'sidecar':
        default_endpoint = get_config('fire_red.endpoint', None) or 'http://127.0.0.1:8765'
        endpoint = input(f"FireRed sidecar endpoint (直接回车={default_endpoint}): ").strip() or default_endpoint
    elif provider == 'cli':
        default_cli = get_config('fire_red.cli.executable', None) or 'scripts/fireredasr2s_worker.py'
        cli_executable = input(f"FireRed CLI worker 路径或命令 (直接回车={default_cli}): ").strip() or default_cli
    elif provider == 'fake':
        default_fixture = 'tests/fixtures/lyrics/simple_song_timeline.json'
        fixture_path = input(f"lyrics fixture JSON (直接回车={default_fixture}): ").strip() or default_fixture

    if provider in {'sidecar', 'cli', 'fake'}:
        print("\nASR 失败处理")
        print("  1. 降级继续 - ASR 失败时退回声学 VPBD，适合批量处理")
        print("  2. 严格失败 - ASR 输出异常时中止，适合调试 provider")
        try:
            strict_choice = int(input("请选择 (1-2，直接回车=1): ").strip() or "1")
        except ValueError:
            strict_choice = 1
        strict = strict_choice == 2

    overrides = build_vpbd_asr_runtime_overrides(
        provider=provider,
        endpoint=endpoint,
        cli_executable=cli_executable,
        fixture_path=fixture_path,
        strict=strict,
    )
    strict_label = '严格失败' if strict else '降级继续'
    print(f"[SELECT] 歌词来源: {provider}；ASR失败处理: {strict_label}")
    return overrides


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

def process_audio_file(
    splitter: SeamlessSplitter,
    file_path: Path,
    processing_mode: str,
    export_format: str,
) -> None:
    """针对单个音频文件执行完整处理流程。"""

    print("\n" + "=" * 60)
    print(f"[SELECT] 正在处理: {file_path.name}")
    print("=" * 60)

    now = datetime.now()
    date_part = now.strftime("%Y%m%d")
    time_part = now.strftime("%H%M%S")
    output_dir_name = f"{date_part}_{time_part}_{file_path.stem}"
    output_dir = project_root / "output" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] 输出目录: {output_dir.name}")

    try:
        import sys as _sys, os as _os
        print(f"[DIAG] Python: {_sys.executable}")
        print(f"[DIAG] VIRTUAL_ENV: {_os.environ.get('VIRTUAL_ENV', '')}")
        print(f"[DIAG] FORCE_SEPARATION_BACKEND: {_os.environ.get('FORCE_SEPARATION_BACKEND', '')}")

        print(f"\n[START] 正在启动统一分割引擎，模式: {processing_mode}...")
        result = splitter.split_audio_seamlessly(
            str(file_path),
            str(output_dir),
            mode=processing_mode,
            export_format=export_format
        )

        if result.get('success'):
            print("\n" + "=" * 50)
            print(f"[SUCCESS] {file_path.name} 处理完成!")
            print("=" * 50)
            print(f"  处理方法: {result.get('method', 'N/A')}")
            print(f"  生成片段数量: {result.get('num_segments', 0)}")
            print(f"  文件保存至: {output_dir}")
            if 'backend_used' in result:
                print(f"  使用后端: {result['backend_used']}")
            if 'processing_time' in result:
                print(f"  总耗时: {result['processing_time']:.1f}s")
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
            print(f"\n[ERROR] {file_path.name} 处理失败: {result.get('error', '未知错误')}")
            print(f"[DIAG] Python: {_sys.executable}")
            print(f"[DIAG] VIRTUAL_ENV: {_os.environ.get('VIRTUAL_ENV', '')}")
            print(f"[DIAG] FORCE_SEPARATION_BACKEND: {_os.environ.get('FORCE_SEPARATION_BACKEND', '')}")

    except Exception as e:
        print(f"[FATAL] 处理 {file_path.name} 时出现未捕获异常: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数 - 重构为纯传令兵模式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    print("=" * 60)
    print("智能人声分割器 - 快速启动 (v2.7)")
    print("=" * 60)

    if not check_system_status():
        return
    audio_files = find_audio_files()
    if not audio_files:
        return

    print(f"[INFO] 发现 {len(audio_files)} 个音频文件")
    for i, file_path in enumerate(audio_files, 1):
        print(f"  {i}. {file_path.name}")

    print("\n" + "=" * 60)
    print("选择处理方式")
    print("=" * 60)
    print("  1. 选择单个文件处理")
    print("  2. 批量处理 input 目录下全部音频")

    scope_choice = 1
    try:
        scope_text = input("请选择处理方式 (1-2，默认1): ").strip()
        if scope_text:
            scope_choice = int(scope_text)
    except ValueError:
        scope_choice = 1

    if scope_choice == 2:
        target_files = audio_files
        print(f"[SELECT] 批量处理 {len(target_files)} 个文件。")
    else:
        if len(audio_files) == 1:
            target_files = [audio_files[0]]
        else:
            try:
                index_text = input(f"\n请选择要分割的文件 (1-{len(audio_files)}，默认1): ").strip()
                index = int(index_text) if index_text else 1
                selected_file = audio_files[index - 1]
            except (ValueError, IndexError):
                print("[ERROR] 选择无效")
                return
            target_files = [selected_file]

    processing_mode = select_processing_mode()
    if processing_mode != 'vocal_separation':
        smart_profile = select_smart_profile()
        set_runtime_config({'smart_cut.profile': smart_profile})

    # 如果选择了 hybrid_mdd 模式，询问卡点密度和对齐策略
    hybrid_density = None
    lib_alignment = None
    if processing_mode == 'hybrid_mdd':
        hybrid_density = select_hybrid_density()
        lib_alignment = select_lib_alignment()
        # 通过运行时配置覆盖密度和策略设置
        set_runtime_config({
            'hybrid_mdd.beat_cut_density': hybrid_density,
            'hybrid_mdd.lib_alignment': lib_alignment,
        })
    elif processing_mode == 'vpbd_asr':
        set_runtime_config(select_vpbd_asr_runtime_overrides())

    try:
        default_format = ensure_supported_format(get_config('output.format', 'wav'))
    except ValueError:
        default_format = 'wav'
    export_format = select_output_format(default_format)
    print(f"[SELECT] 输出格式: {export_format}")

    try:
        sample_rate = get_config('audio.sample_rate', 44100)
        splitter = SeamlessSplitter(sample_rate=sample_rate)

        for file_path in target_files:
            process_audio_file(
                splitter,
                file_path,
                processing_mode,
                export_format,
            )
    except Exception as e:
        print(f"[FATAL] 脚本顶层出现未捕获异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

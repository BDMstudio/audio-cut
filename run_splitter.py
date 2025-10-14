#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_splitter.py
# AI-SUMMARY: 简化的运行脚本，仅支持纯人声分离与 v2.2 MDD 无缝分割

"""
智能人声分割器运行脚本（精简版）

- 支持两种模式：
  1. `vocal_separation` —— 仅输出人声/伴奏轨
  2. `v2.2_mdd` —— 启用纯人声检测 + MDD 增强的无缝分割
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from src.vocal_smart_splitter.utils.config_manager import get_config, set_runtime_config
from audio_cut.config.derive import (
    apply_profile_overrides,
    build_runtime_override_map,
    schema_from_mapping,
)
from audio_cut.config.migrate_v2_to_v3 import migrate_to_schema_v3


def setup_logging(verbose: bool = False) -> None:
    """配置日志输出"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_output_dir_for(input_path: Path) -> str:
    """按照 <原音频名>_<日期>_<时间> 的规则创建输出目录"""
    now = datetime.now()
    date_part = now.strftime("%Y%m%d")
    time_part = now.strftime("%H%M%S")
    dirname = f"{date_part}_{time_part}_{input_path.stem}"
    output_dir = project_root / "output" / dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="智能人声分割器 - 纯人声分离 / v2.2 MDD 无缝分割",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_splitter.py input/song.mp3 --mode vocal_separation
  python run_splitter.py input/song.mp3 --mode v2.2_mdd --validate-reconstruction
        """,
    )

    parser.add_argument('input_file', help='输入音频文件路径')
    parser.add_argument(
        '--mode',
        choices=['vocal_separation', 'v2.2_mdd'],
        default='v2.2_mdd',
        help='运行模式 (默认: v2.2_mdd)',
    )
    parser.add_argument(
        '--validate-reconstruction',
        action='store_true',
        help='验证拼接完整性 (仅在 v2.2_mdd 模式下有效)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='输出调试日志',
    )
    parser.add_argument(
        '--gpu-device',
        default=None,
        help='指定 GPU 设备，例如 cuda:0、cuda:1 或 cpu（默认读取配置）',
    )
    parser.add_argument(
        '--strict-gpu',
        action='store_true',
        help='启用 strict GPU 模式，GPU 失败时直接报错、不回退 CPU',
    )
    parser.add_argument(
        '--profile',
        choices=['ballad', 'pop', 'edm', 'rap'],
        help='应用预设配置 Profile（ballad/pop/edm/rap）',
    )
    parser.add_argument(
        '--compat-config',
        choices=['v2'],
        help='迁移旧版配置（当前仅支持 v2）并覆盖运行时参数',
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)

    output_dir = create_output_dir_for(input_path)
    logger.info(f"输出目录: {output_dir}")

    runtime_overrides = {}
    active_profile = None
    compat_flag = None
    if args.profile:
        try:
            _, profile_overrides = apply_profile_overrides(args.profile)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            sys.exit(1)
        runtime_overrides.update(profile_overrides)
        active_profile = args.profile
        logger.info("应用配置 Profile: %s", args.profile)
    if args.compat_config == 'v2':
        legacy_path = project_root / 'config' / 'default.yaml'
        if not legacy_path.exists():
            logger.error("未找到 legacy 配置: %s", legacy_path)
            sys.exit(1)
        schema_mapping = migrate_to_schema_v3(legacy_path)
        schema = schema_from_mapping(schema_mapping)
        compat_overrides = build_runtime_override_map(schema)
        compat_overrides.setdefault('meta.compat_config', 'v2')
        runtime_overrides.update(compat_overrides)
        compat_flag = 'v2'
        logger.info("应用兼容配置: %s", legacy_path.name)
    if args.gpu_device:
        runtime_overrides['gpu_pipeline.prefer_device'] = args.gpu_device
        logger.info("[GPU Pipeline] prefer_device=%s", args.gpu_device)
    if args.strict_gpu:
        runtime_overrides['gpu_pipeline.strict_gpu'] = True
        logger.info("[GPU Pipeline] strict_gpu 模式已开启")
    if runtime_overrides:
        set_runtime_config(runtime_overrides)

    sample_rate = get_config('audio.sample_rate', 44100)
    logger.info(f"使用采样率: {sample_rate}Hz")

    splitter = SeamlessSplitter(sample_rate=sample_rate)

    logger.info(f"运行模式: {args.mode}")
    result = splitter.split_audio_seamlessly(str(input_path), output_dir, mode=args.mode)
    if active_profile:
        meta = result.setdefault('meta', {})
        meta['profile'] = active_profile
    if compat_flag:
        meta = result.setdefault('meta', {})
        meta['compat_config'] = compat_flag

    if not result.get('success'):
        logger.error(f"处理失败: {result.get('error', '未知错误')}")
        sys.exit(1)

    logger.info("=" * 50)
    if args.mode == 'vocal_separation':
        logger.info("纯人声分离完成")
    else:
        logger.info("无缝分割完成")
        logger.info(f"生成片段数: {result.get('num_segments', 0)}")
        logger.info(f"使用后端: {result.get('backend_used', 'unknown')}")
        if 'processing_time' in result:
            logger.info(f"处理耗时: {result['processing_time']:.1f}s")

        if args.validate_reconstruction:
            logger.info("运行拼接完整性验证...")
            try:
                from tests.test_seamless_reconstruction import SeamlessReconstructionTester

                tester = SeamlessReconstructionTester(sample_rate)
                passed = tester.test_perfect_reconstruction(str(input_path), output_dir)
                logger.info(f"拼接验证: {'PASS' if passed else 'FAIL'}")
            except ImportError:
                logger.warning("未找到拼接验证模块，跳过")

    saved_files = result.get('saved_files', [])
    if saved_files:
        logger.info("生成的文件:")
        for idx, file_path in enumerate(saved_files, 1):
            logger.info(f"  {idx}. {Path(file_path).name}")

    logger.info("=" * 50)


if __name__ == "__main__":
    main()

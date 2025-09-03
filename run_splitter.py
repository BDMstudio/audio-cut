#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_splitter.py
# AI-SUMMARY: 智能人声分割器运行脚本，自动创建时间戳输出目录

"""
智能人声分割器运行脚本

自动创建按时间命名的输出目录，运行人声分割任务。

使用方法:
    python run_splitter.py [input_file] [options]
    
示例:
    python run_splitter.py input/01.mp3
    python run_splitter.py input/01.mp3 --min-length 8 --max-length 12
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vocal_smart_splitter.main import VocalSmartSplitter
from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from src.vocal_smart_splitter.utils.config_manager import get_config

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_timestamped_output_dir():
    """创建时间戳命名的输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "output" / f"test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="智能人声分割器 - 自动在换气/停顿处分割音频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_splitter.py input/01.mp3
  python run_splitter.py input/01.mp3 --min-length 8 --max-length 12
  python run_splitter.py input/01.mp3 --target-length 10 --verbose
        """
    )
    
    parser.add_argument(
        'input_file',
        help='输入音频文件路径'
    )
    
    parser.add_argument(
        '--min-length',
        type=float,
        default=5,
        help='最小片段长度（秒），默认: 5'
    )
    
    parser.add_argument(
        '--max-length', 
        type=float,
        default=15,
        help='最大片段长度（秒），默认: 15'
    )
    
    parser.add_argument(
        '--target-length',
        type=float,
        default=10,
        help='目标片段长度（秒），默认: 10'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志'
    )
    
    parser.add_argument(
        '--seamless-vocal',
        action='store_true',
        help='使用无缝人声停顿分割模式（基于人声停顿的精确分割，确保完美拼接）'
    )
    
    parser.add_argument(
        '--validate-reconstruction',
        action='store_true',
        help='验证拼接完整性（仅在--seamless-vocal模式下有效）'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 检查输入文件
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)
    
    # 创建时间戳输出目录
    output_dir = create_timestamped_output_dir()
    logger.info(f"创建输出目录: {output_dir}")
    
    try:
        if args.seamless_vocal:
            # 无缝人声停顿分割模式
            logger.info("使用无缝人声停顿分割模式...")
            
            # 从配置文件读取采样率
            sample_rate = get_config('audio.sample_rate', 44100)
            logger.info(f"使用采样率: {sample_rate}Hz")
            splitter = SeamlessSplitter(sample_rate=sample_rate)
            
            logger.info(f"开始无缝分割: {input_path}")
            logger.info(f"输出目录: {output_dir}")
            logger.info("分割策略: 基于人声停顿的精确分割")
            logger.info("输出格式: 无损WAV/FLAC，零音频处理")
            
            result = splitter.split_audio_seamlessly(str(input_path), output_dir)
            
        else:
            # 传统智能分割模式
            logger.info("初始化传统智能人声分割器...")
            splitter = VocalSmartSplitter()
            
            # 更新配置
            if args.verbose:
                splitter.config_manager.set('logging.level', 'DEBUG')
            
            splitter.config_manager.set('smart_splitting.min_segment_length', args.min_length)
            splitter.config_manager.set('smart_splitting.max_segment_length', args.max_length)
            splitter.config_manager.set('smart_splitting.target_segment_length', args.target_length)
            
            # 运行分割
            logger.info(f"开始处理音频文件: {input_path}")
            logger.info(f"输出目录: {output_dir}")
            logger.info(f"片段长度范围: {args.min_length}-{args.max_length}秒，目标: {args.target_length}秒")
            
            result = splitter.split_audio(str(input_path), output_dir)
        
        # 显示结果
        logger.info("=" * 50)
        logger.info("分割完成！")
        logger.info(f"输出目录: {output_dir}")
        
        if args.seamless_vocal:
            # 无缝分割结果显示
            logger.info(f"生成片段数: {result['num_segments']}")
            logger.info(f"处理模式: {result.get('processing_type', 'seamless_vocal_pause_splitting')}")
            
            # 显示人声停顿分析
            pause_analysis = result.get('vocal_pause_analysis', {})
            if pause_analysis:
                logger.info(f"检测到人声停顿: {pause_analysis.get('total_pauses', 0)} 个")
                logger.info(f"平均置信度: {pause_analysis.get('avg_confidence', 0):.3f}")
            
            # 显示拼接验证结果
            validation = result.get('seamless_validation', {})
            if validation:
                logger.info(f"拼接验证: {'✅ 完美重构' if validation.get('perfect_reconstruction') else '❌ 存在差异'}")
                if 'max_difference' in validation:
                    logger.info(f"最大差异: {validation['max_difference']:.2e}")
                    
            # 运行额外验证（如果请求）
            if args.validate_reconstruction and result['success']:
                logger.info("运行拼接完整性验证...")
                try:
                    from tests.test_seamless_reconstruction import SeamlessReconstructionTester
                    tester = SeamlessReconstructionTester(44100)
                    validation_passed = tester.test_perfect_reconstruction(str(input_path), output_dir)
                    logger.info(f"详细验证结果: {'PASS' if validation_passed else 'FAIL'}")
                except ImportError:
                    logger.warning("验证模块未找到，跳过详细验证")
            
        else:
            # 传统分割结果显示
            output_files_key = 'saved_files' if 'saved_files' in result else 'output_files'
            logger.info(f"生成片段数: {len(result[output_files_key])}")
            if 'quality_report' in result:
                logger.info(f"总体质量评分: {result['quality_report']['overall_quality']:.3f}")
        
        # 显示生成的文件
        saved_files = result.get('saved_files', result.get('output_files', []))
        if saved_files:
            logger.info("生成的片段文件:")
            for i, file_path in enumerate(saved_files, 1):
                file_name = Path(file_path).name
                logger.info(f"  {i}. {file_name}")
        
        # 显示分析报告路径
        report_path = Path(output_dir) / "analysis_report.json"
        if report_path.exists():
            logger.info(f"详细分析报告: {report_path}")
            
        if args.seamless_vocal:
            # 无缝分割专用报告
            validation_report_path = Path(output_dir) / "seamless_validation_report.json"
            if validation_report_path.exists():
                logger.info(f"拼接验证报告: {validation_report_path}")
        
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

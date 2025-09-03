#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vocal_smart_splitter/main.py
# AI-SUMMARY: 智能人声分割器主程序，整合所有核心模块实现完整的分割流程

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vocal_smart_splitter.utils.config_manager import ConfigManager, get_config_manager, get_config
from vocal_smart_splitter.utils.audio_processor import AudioProcessor
from vocal_smart_splitter.core.vocal_separator import VocalSeparator
from vocal_smart_splitter.core.breath_detector import BreathDetector
from vocal_smart_splitter.core.content_analyzer import ContentAnalyzer
from vocal_smart_splitter.core.smart_splitter import SmartSplitter
from vocal_smart_splitter.core.quality_controller import QualityController

class VocalSmartSplitter:
    """智能人声分割器主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化智能人声分割器
        
        Args:
            config_path: 配置文件路径
        """
        # 初始化配置管理器
        self.config_manager = get_config_manager(config_path)
        
        # 设置日志
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("智能人声分割器初始化开始...")
        
        # 获取音频配置
        self.sample_rate = get_config('audio.sample_rate', 22050)
        
        # 初始化核心模块
        self.audio_processor = AudioProcessor(self.sample_rate)
        self.vocal_separator = VocalSeparator(self.sample_rate)
        self.breath_detector = BreathDetector(self.sample_rate)
        self.content_analyzer = ContentAnalyzer(self.sample_rate)
        self.smart_splitter = SmartSplitter(self.sample_rate)
        self.quality_controller = QualityController(self.sample_rate)
        
        self.logger.info("智能人声分割器初始化完成")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = self.config_manager.get_logging_config()
        
        # 清除现有的处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 设置新的日志配置
        logging.basicConfig(
            level=log_config['level'],
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_config['file'], encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def split_audio(self, input_path: str, output_dir: Optional[str] = None) -> Dict:
        """分割音频文件
        
        Args:
            input_path: 输入音频文件路径
            output_dir: 输出目录路径
            
        Returns:
            分割结果信息
        """
        self.logger.info(f"开始处理音频文件: {input_path}")
        
        try:
            # 1. 加载和预处理音频
            self.logger.info("步骤 1/7: 加载和预处理音频...")
            audio, sample_rate = self.audio_processor.load_audio(input_path)
            audio, sample_rate = self.audio_processor.preprocess_audio(
                audio, sample_rate, target_sr=self.sample_rate
            )
            
            total_duration = len(audio) / sample_rate
            self.logger.info(f"音频加载完成，时长: {total_duration:.2f}秒")
            
            # 2. 人声分离
            self.logger.info("步骤 2/7: 进行高质量人声分离...")
            vocal_track, accompaniment_track, separation_quality = self.vocal_separator.separate_vocals(audio)
            self.logger.info(f"人声分离完成，质量评分: {separation_quality['overall_score']:.3f}")
            
            # 3. 换气和停顿检测
            self.logger.info("步骤 3/7: 检测换气和停顿点...")
            breath_results = self.breath_detector.detect_breath_points(vocal_track, audio)
            self.logger.info(f"检测到 {len(breath_results['pauses'])} 个停顿点，质量评分: {breath_results['quality_score']:.3f}")
            
            # 4. 内容分析
            self.logger.info("步骤 4/7: 分析人声内容...")
            content_results = self.content_analyzer.analyze_vocal_content(vocal_track, breath_results['pauses'])
            self.logger.info(f"识别到 {len(content_results['vocal_segments'])} 个人声片段，质量评分: {content_results['quality_score']:.3f}")
            
            # 5. 智能分割决策
            self.logger.info("步骤 5/7: 创建智能分割方案...")
            split_points = self.smart_splitter.create_smart_splits(
                total_duration, breath_results, content_results,
                vocal_track, self.sample_rate
            )
            self.logger.info(f"创建了 {len(split_points)} 个分割点")
            
            # 6. 质量控制和音频处理
            self.logger.info("步骤 6/7: 质量控制和音频处理...")
            quality_results = self.quality_controller.validate_and_process_segments(
                audio, vocal_track, split_points
            )
            processed_segments = quality_results['segments']
            quality_report = quality_results['quality_report']
            
            self.logger.info(f"质量控制完成，{len(processed_segments)} 个片段通过验证")
            self.logger.info(f"整体质量评分: {quality_report['overall_quality']:.3f}")
            
            # 7. 保存分割结果
            self.logger.info("步骤 7/7: 保存分割结果...")
            if output_dir is None:
                output_dir = get_config('output.directory', '../output')
            
            saved_files = self._save_segments(processed_segments, output_dir)
            
            # 保存调试信息（如果启用）
            if get_config('output.save_debug_info', False):
                self._save_debug_info(output_dir, {
                    'separation_quality': separation_quality,
                    'breath_results': breath_results,
                    'content_results': content_results,
                    'split_points': split_points,
                    'quality_report': quality_report
                })
            
            # 保存分析报告
            if get_config('output.save_analysis_report', True):
                self._save_analysis_report(output_dir, {
                    'input_file': input_path,
                    'total_duration': total_duration,
                    'separation_quality': separation_quality,
                    'breath_detection': {
                        'num_pauses': len(breath_results['pauses']),
                        'quality_score': breath_results['quality_score']
                    },
                    'content_analysis': {
                        'num_segments': len(content_results['vocal_segments']),
                        'num_groups': len(content_results['content_groups']),
                        'quality_score': content_results['quality_score']
                    },
                    'splitting_results': {
                        'num_split_points': len(split_points),
                        'num_final_segments': len(processed_segments)
                    },
                    'quality_report': quality_report
                })
            
            # 可选：保存分离的人声轨道
            if get_config('output.save_separated_vocal', False):
                # 按配置格式保存分离人声
                audio_format = get_config('audio.format', 'wav').lower()
                ext = 'flac' if audio_format == 'flac' else 'wav'
                vocal_path = os.path.join(output_dir, f'separated_vocal.{ext}')
                self.audio_processor.save_audio(vocal_track, sample_rate, vocal_path)
                self.logger.info(f"已保存分离的人声轨道: {vocal_path}")
            
            result_info = {
                'success': True,
                'input_file': input_path,
                'output_directory': output_dir,
                'total_duration': total_duration,
                'num_segments': len(processed_segments),
                'output_files': saved_files,  # 修复键名
                'saved_files': saved_files,   # 保持兼容性
                'quality_report': quality_report,  # 添加完整的质量报告
                'quality_score': quality_report['overall_quality'],
                'processing_summary': {
                    'separation_quality': separation_quality['overall_score'],
                    'breath_detection_quality': breath_results['quality_score'],
                    'content_analysis_quality': content_results['quality_score'],
                    'final_quality': quality_report['overall_quality']
                }
            }
            
            self.logger.info("音频分割处理完成！")
            self.logger.info(f"输出目录: {output_dir}")
            self.logger.info(f"生成片段: {len(processed_segments)} 个")
            self.logger.info(f"整体质量: {quality_report['overall_quality']:.3f}")
            
            return result_info
            
        except Exception as e:
            self.logger.error(f"音频分割处理失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_file': input_path
            }
    
    def _save_segments(self, segments: List[Dict], output_dir: str) -> List[str]:
        """保存音频片段
        
        Args:
            segments: 处理后的片段列表
            output_dir: 输出目录
            
        Returns:
            保存的文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        naming_pattern = get_config('output.naming_pattern', 'vocal_segment_{index:02d}')
        audio_quality = get_config('audio.quality', 192)
        
        for segment in segments:
            # 生成文件名
            # 依据配置 audio.format 决定扩展名（建议使用无损 WAV/FLAC）
            audio_format = get_config('audio.format', 'wav').lower()
            ext = '.flac' if audio_format == 'flac' else '.wav'
            filename = f"{naming_pattern.format(index=segment['index'] + 1)}{ext}"
            output_path = os.path.join(output_dir, filename)
            
            # 保存音频
            success = self.audio_processor.save_audio(
                segment['processed_audio'],
                self.sample_rate,
                output_path,
                quality=audio_quality
            )
            
            if success:
                saved_files.append(output_path)
                duration = segment['duration']
                quality = segment['quality_metrics']['overall_quality']
                
                self.logger.info(f"已保存片段 {segment['index'] + 1}: {filename}")
                self.logger.info(f"  时长: {duration:.2f}秒, 质量: {quality:.3f}")
            else:
                self.logger.error(f"片段保存失败: {filename}")
        
        return saved_files
    
    def _save_debug_info(self, output_dir: str, debug_data: Dict):
        """保存调试信息
        
        Args:
            output_dir: 输出目录
            debug_data: 调试数据
        """
        try:
            debug_file = os.path.join(output_dir, 'debug_info.json')
            
            # 转换numpy数组为列表以便JSON序列化
            serializable_data = self._make_json_serializable(debug_data)
            
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"调试信息已保存: {debug_file}")
            
        except Exception as e:
            self.logger.warning(f"调试信息保存失败: {e}")
    
    def _save_analysis_report(self, output_dir: str, report_data: Dict):
        """保存分析报告
        
        Args:
            output_dir: 输出目录
            report_data: 报告数据
        """
        try:
            report_file = os.path.join(output_dir, 'analysis_report.json')
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"分析报告已保存: {report_file}")
            
        except Exception as e:
            self.logger.warning(f"分析报告保存失败: {e}")
    
    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        else:
            return obj

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='智能人声分割器 - 基于人声内容和换气停顿的智能音频分割工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py input.mp3                          # 使用默认设置
  python main.py input.mp3 -o output_dir            # 指定输出目录
  python main.py input.mp3 --min-length 4 --max-length 12  # 自定义时长范围
  python main.py input.mp3 -v --save-vocal          # 详细输出并保存人声轨道
        """
    )
    
    parser.add_argument('input_path', help='输入音频文件路径')
    parser.add_argument('-o', '--output', help='输出目录路径', default=None)
    parser.add_argument('-c', '--config', help='配置文件路径', default=None)
    
    # 分割参数
    parser.add_argument('--min-length', type=int, help='最小片段长度(秒)', default=None)
    parser.add_argument('--max-length', type=int, help='最大片段长度(秒)', default=None)
    parser.add_argument('--target-length', type=int, help='目标片段长度(秒)', default=None)
    
    # 音频参数
    parser.add_argument('--sample-rate', type=int, help='采样率', default=None)
    parser.add_argument('--quality', type=int, help='输出音频质量(kbps)', default=None)
    
    # 输出选项
    parser.add_argument('--save-vocal', action='store_true', help='保存分离的人声轨道')
    parser.add_argument('--save-debug', action='store_true', help='保存调试信息')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    try:
        # 检查输入文件
        if not os.path.exists(args.input_path):
            print(f"错误: 输入文件不存在: {args.input_path}")
            sys.exit(1)
        
        # 初始化分割器
        splitter = VocalSmartSplitter(args.config)
        
        # 更新配置
        config_updates = {}
        if args.min_length is not None:
            config_updates['min_length'] = args.min_length
        if args.max_length is not None:
            config_updates['max_length'] = args.max_length
        if args.target_length is not None:
            config_updates['target_length'] = args.target_length
        if args.sample_rate is not None:
            config_updates['sample_rate'] = args.sample_rate
        if args.quality is not None:
            config_updates['quality'] = args.quality
        if args.verbose:
            config_updates['verbose'] = True
        if args.save_vocal:
            splitter.config_manager.set('output.save_separated_vocal', True)
        if args.save_debug:
            splitter.config_manager.set('output.save_debug_info', True)
        
        if config_updates:
            splitter.config_manager.update_from_args(config_updates)
        
        # 执行分割
        result = splitter.split_audio(args.input_path, args.output)
        
        if result['success']:
            print("\n🎉 音频分割完成！")
            print(f"📁 输出目录: {result['output_directory']}")
            print(f"🎵 生成片段: {result['num_segments']} 个")
            print(f"⏱️  总时长: {result['total_duration']:.2f} 秒")
            print(f"⭐ 质量评分: {result['quality_score']:.3f}")
            
            print("\n📋 处理摘要:")
            summary = result['processing_summary']
            print(f"  人声分离质量: {summary['separation_quality']:.3f}")
            print(f"  换气检测质量: {summary['breath_detection_quality']:.3f}")
            print(f"  内容分析质量: {summary['content_analysis_quality']:.3f}")
            print(f"  最终质量评分: {summary['final_quality']:.3f}")
            
            print("\n📄 生成的文件:")
            for i, file_path in enumerate(result['saved_files'], 1):
                filename = os.path.basename(file_path)
                print(f"  {i}. {filename}")
        else:
            print(f"\n❌ 处理失败: {result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断处理")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

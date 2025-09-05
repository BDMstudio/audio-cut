#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/test_seamless_reconstruction.py
# AI-SUMMARY: 无缝拼接验证测试，确保分割片段可以完美重构原音频

import os
import sys
import numpy as np
import librosa
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from vocal_smart_splitter.utils.audio_processor import AudioProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SeamlessReconstructionTester:
    """无缝拼接验证测试器"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.splitter = SeamlessSplitter(sample_rate)
        self.audio_processor = AudioProcessor(sample_rate)
    
    def test_perfect_reconstruction(self, input_file: str, output_dir: str) -> bool:
        """测试完美重构
        
        Args:
            input_file: 输入音频文件
            output_dir: 输出目录
            
        Returns:
            是否完美重构
        """
        logger.info(f"开始测试无缝拼接: {input_file}")
        
        try:
            # 1. 执行无缝分割
            result = self.splitter.split_audio_seamlessly(input_file, output_dir)
            
            if not result['success']:
                logger.error(f"分割失败: {result.get('error', 'Unknown error')}")
                return False
            
            # 2. 验证拼接结果
            validation = result['seamless_validation']
            
            logger.info("=== 无缝拼接验证结果 ===")
            logger.info(f"长度匹配: {validation['length_match']}")
            logger.info(f"完美重构: {validation['perfect_reconstruction']}")
            
            if 'max_difference' in validation:
                logger.info(f"最大差异: {validation['max_difference']:.2e}")
                logger.info(f"RMS差异: {validation['rms_difference']:.2e}")
            
            # 3. 手动验证（双重检查）
            manual_validation = self._manual_validation(result['saved_files'], input_file)
            
            logger.info("=== 手动验证结果 ===")
            logger.info(f"手动长度匹配: {manual_validation['length_match']}")
            logger.info(f"手动完美重构: {manual_validation['perfect_reconstruction']}")
            logger.info(f"手动最大差异: {manual_validation['max_difference']:.2e}")
            
            # 4. 生成验证报告
            self._generate_validation_report(result, validation, manual_validation, output_dir)
            
            return validation['perfect_reconstruction'] and manual_validation['perfect_reconstruction']
            
        except Exception as e:
            logger.error(f"测试失败: {e}")
            return False
    
    def _manual_validation(self, segment_files: list, original_file: str) -> dict:
        """手动验证拼接质量
        
        Args:
            segment_files: 分割片段文件列表
            original_file: 原始音频文件
            
        Returns:
            验证结果
        """
        try:
            # 加载原始音频
            original_audio, original_sr = librosa.load(original_file, sr=self.sample_rate, mono=True)
            
            # 加载并拼接所有片段
            reconstructed_segments = []
            for file_path in sorted(segment_files):
                segment_audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
                reconstructed_segments.append(segment_audio)
            
            # 拼接
            reconstructed_audio = np.concatenate(reconstructed_segments)
            
            # 长度检查
            length_match = len(original_audio) == len(reconstructed_audio)
            
            # 精确度检查
            if length_match:
                diff = original_audio - reconstructed_audio
                max_diff = np.max(np.abs(diff))
                rms_diff = np.sqrt(np.mean(diff**2))
                perfect_reconstruction = max_diff < 1e-10
            else:
                max_diff = float('inf')
                rms_diff = float('inf')
                perfect_reconstruction = False
            
            return {
                'length_match': length_match,
                'perfect_reconstruction': perfect_reconstruction,
                'max_difference': max_diff,
                'rms_difference': rms_diff,
                'original_length': len(original_audio),
                'reconstructed_length': len(reconstructed_audio)
            }
            
        except Exception as e:
            logger.error(f"手动验证失败: {e}")
            return {
                'length_match': False,
                'perfect_reconstruction': False,
                'error': str(e)
            }
    
    def _generate_validation_report(self, split_result: dict, auto_validation: dict,
                                  manual_validation: dict, output_dir: str):
        """生成验证报告
        
        Args:
            split_result: 分割结果
            auto_validation: 自动验证结果
            manual_validation: 手动验证结果
            output_dir: 输出目录
        """
        report = {
            'test_info': {
                'input_file': split_result['input_file'],
                'num_segments': split_result['num_segments'],
                'processing_type': split_result['processing_type']
            },
            'auto_validation': auto_validation,
            'manual_validation': manual_validation,
            'validation_passed': (
                auto_validation['perfect_reconstruction'] and 
                manual_validation['perfect_reconstruction']
            ),
            'segments_info': split_result['segments']
        }
        
        # 保存报告
        import json
        report_file = os.path.join(output_dir, 'seamless_validation_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"验证报告已保存: {report_file}")
    
    def test_vocal_pause_accuracy(self, input_file: str) -> dict:
        """测试人声停顿检测精度
        
        Args:
            input_file: 输入音频文件
            
        Returns:
            检测精度报告
        """
        logger.info(f"测试人声停顿检测精度: {input_file}")
        
        try:
            # 加载音频
            original_audio, _ = librosa.load(input_file, sr=self.sample_rate, mono=True)
            
            # 人声分离
            vocal_track, _, separation_quality = self.splitter.vocal_separator.separate_vocals(original_audio)
            
            # 人声停顿检测
            vocal_pauses = self.splitter.vocal_pause_detector.detect_vocal_pauses(vocal_track, original_audio)
            
            # 分析检测结果
            pause_report = self.splitter.vocal_pause_detector.generate_pause_report(vocal_pauses)
            
            # 质量评估
            total_audio_duration = len(original_audio) / self.sample_rate
            pause_coverage = pause_report['total_pause_duration'] / total_audio_duration
            
            accuracy_report = {
                'separation_quality': separation_quality['overall_score'],
                'num_pauses_detected': pause_report['total_pauses'],
                'pause_types': pause_report['pause_types'],
                'avg_confidence': pause_report['avg_confidence'],
                'pause_coverage_ratio': pause_coverage,
                'total_audio_duration': total_audio_duration,
                'pause_details': []
            }
            
            # 详细停顿信息
            for pause in vocal_pauses:
                accuracy_report['pause_details'].append({
                    'start_time': pause.start_time,
                    'end_time': pause.end_time,
                    'duration': pause.duration,
                    'position_type': pause.position_type,
                    'confidence': pause.confidence,
                    'cut_point': pause.cut_point
                })
            
            logger.info(f"人声停顿检测完成: {pause_report['total_pauses']} 个停顿，"
                       f"平均置信度 {pause_report['avg_confidence']:.3f}")
            
            return accuracy_report
            
        except Exception as e:
            logger.error(f"人声停顿检测测试失败: {e}")
            return {'error': str(e)}

def main():
    """主测试函数"""
    # 配置
    input_file = "input/01.mp3"
    output_dir = "output/seamless_test"
    
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 创建测试器
    tester = SeamlessReconstructionTester(sample_rate=44100)
    
    logger.info("=== 开始无缝拼接完整性测试 ===")
    
    # 1. 完美重构测试
    perfect_reconstruction = tester.test_perfect_reconstruction(input_file, output_dir)
    
    logger.info(f"完美重构测试结果: {'PASS' if perfect_reconstruction else 'FAIL'}")
    
    # 2. 人声停顿精度测试
    logger.info("\n=== 人声停顿检测精度测试 ===")
    pause_accuracy = tester.test_vocal_pause_accuracy(input_file)
    
    if 'error' not in pause_accuracy:
        logger.info(f"检测到 {pause_accuracy['num_pauses_detected']} 个人声停顿")
        logger.info(f"平均置信度: {pause_accuracy['avg_confidence']:.3f}")
        logger.info(f"人声分离质量: {pause_accuracy['separation_quality']:.3f}")
        
        # 保存精度报告
        import json
        accuracy_file = os.path.join(output_dir, 'pause_accuracy_report.json')
        os.makedirs(output_dir, exist_ok=True)
        with open(accuracy_file, 'w', encoding='utf-8') as f:
            json.dump(pause_accuracy, f, indent=2, ensure_ascii=False)
        logger.info(f"精度报告已保存: {accuracy_file}")
    else:
        logger.error(f"人声停顿测试失败: {pause_accuracy['error']}")
    
    # 3. 综合评估
    logger.info("\n=== 综合测试结果 ===")
    overall_pass = perfect_reconstruction and ('error' not in pause_accuracy)
    logger.info(f"综合测试结果: {'PASS' if overall_pass else 'FAIL'}")
    
    if overall_pass:
        logger.info("[SUCCESS] 无缝分割系统运行正常，可以投入使用")
    else:
        logger.warning("[WARNING] 系统存在问题，需要进一步调试")
    
    return overall_pass

if __name__ == "__main__":
    main()
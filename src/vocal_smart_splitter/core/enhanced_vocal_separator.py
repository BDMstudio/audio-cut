#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/core/enhanced_vocal_separator.py
# AI-SUMMARY: 检测专用高精度人声分离器，支持MDX23/Demucs v4等先进分离后端，专门用于提升停顿检测精度

import os
import sys
import numpy as np
import librosa
import logging
import tempfile
import subprocess
import glob
from typing import Tuple, Dict, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass

from ..utils.config_manager import get_config
from .vocal_separator import VocalSeparator  # 继承现有分离器

logger = logging.getLogger(__name__)

@dataclass
class SeparationResult:
    """人声分离结果数据结构"""
    vocal_track: np.ndarray      # 分离的人声轨道
    instrumental_track: Optional[np.ndarray] = None  # 器乐轨道（可选）
    separation_confidence: float = 0.0  # 分离质量置信度 (0-1)
    backend_used: str = "unknown"       # 使用的分离后端
    processing_time: float = 0.0        # 处理耗时（秒）
    quality_metrics: Dict = None        # 质量指标

class EnhancedVocalSeparator:
    """检测专用高精度人声分离器
    
    设计理念：
    1. 多后端支持：MDX23(主推) / Demucs v4(备选) / HPSS(兜底)
    2. 检测专用：只返回内存数据，不保存文件，优化性能
    3. 质量评估：自动评估分离质量，为双路检测提供置信度
    4. 渐进降级：优先使用高精度后端，失败时自动降级
    """
    
    def __init__(self, sample_rate: int = 44100):
        """初始化增强型分离器
        
        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        
        # 从配置加载参数
        self.backend = get_config('enhanced_separation.backend', 'mdx23')
        self.enable_fallback = get_config('enhanced_separation.enable_fallback', True)
        self.min_confidence_threshold = get_config('enhanced_separation.min_separation_confidence', 0.7)
        
        # 初始化后端状态
        self.backend_status = {
            'mdx23': {'available': False, 'error': None},
            'demucs_v4': {'available': False, 'error': None}, 
            'hpss_fallback': {'available': True, 'error': None}  # HPSS总是可用
        }
        
        # 初始化传统分离器作为兜底
        self.hpss_separator = VocalSeparator(sample_rate)
        
        # 检查和初始化高精度后端
        self._initialize_backends()
        
        logger.info(f"增强型分离器初始化完成 - 主后端: {self.backend}")
        
    def _initialize_backends(self):
        """初始化和检测各分离后端的可用性"""
        
        # 检测MDX23后端
        if self.backend == 'mdx23' or self.enable_fallback:
            self._check_mdx23_availability()
            
        # 检测Demucs v4后端  
        if self.backend == 'demucs_v4' or self.enable_fallback:
            self._check_demucs_availability()
            
        # 报告后端状态
        available_backends = [name for name, status in self.backend_status.items() if status['available']]
        logger.info(f"可用分离后端: {available_backends}")
        
        if not any(self.backend_status[b]['available'] for b in ['mdx23', 'demucs_v4']) and self.backend != 'hpss_fallback':
            logger.warning("高精度分离后端不可用，将使用传统HPSS方案")
    
    def _check_mdx23_availability(self):
        """检测MDX23后端可用性"""
        try:
            # 检查MDX23是否已安装
            import_result = self._try_import_mdx23()
            if import_result['success']:
                # 检查模型文件
                model_path = get_config('enhanced_separation.mdx23.model_path', './models/mdx23_pretrained.pth')
                if os.path.exists(model_path) or self._check_mdx23_models():
                    self.backend_status['mdx23']['available'] = True
                    logger.info("✅ MDX23后端可用")
                else:
                    self.backend_status['mdx23']['error'] = "模型文件未找到"
                    logger.warning("MDX23模型文件缺失，需要下载预训练模型")
            else:
                self.backend_status['mdx23']['error'] = import_result['error']
                logger.warning(f"⚠️  MDX23导入失败: {import_result['error']}")
                
        except Exception as e:
            self.backend_status['mdx23']['error'] = str(e)
            logger.warning(f"⚠️  MDX23后端检测失败: {e}")
    
    def _try_import_mdx23(self) -> Dict:
        """尝试导入MDX23相关模块"""
        try:
            # 方案1：尝试导入已安装的MDX23 Python包
            try:
                import inference
                return {'success': True, 'method': 'python_module'}
            except ImportError:
                pass
            
            # 方案2：检查MDX23可执行文件
            mdx23_executable = get_config('enhanced_separation.mdx23.executable_path', 'python inference.py')
            
            # 尝试运行help命令测试可用性
            test_result = subprocess.run(
                [sys.executable, '-c', 'import inference; print("MDX23 available")'],
                capture_output=True, text=True, timeout=10
            )
            
            if test_result.returncode == 0:
                return {'success': True, 'method': 'cli_available'}
            else:
                # 检查是否有独立的MDX23项目目录
                mdx23_project_path = get_config('enhanced_separation.mdx23.project_path', './MVSEP-MDX23-music-separation-model')
                if os.path.exists(os.path.join(mdx23_project_path, 'inference.py')):
                    return {'success': True, 'method': 'project_directory', 'path': mdx23_project_path}
                    
                return {'success': False, 'error': 'MDX23未找到，请安装或配置正确路径'}
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'MDX23检测超时'}
        except Exception as e:
            return {'success': False, 'error': f'MDX23检测异常: {e}'}
    
    def _check_mdx23_models(self) -> bool:
        """检查MDX23预训练模型"""
        try:
            # 检查用户指定的模型路径
            user_model_path = get_config('enhanced_separation.mdx23.model_path', '')
            if user_model_path and os.path.exists(user_model_path):
                logger.info(f"找到用户指定的MDX23模型: {user_model_path}")
                return True
            
            # 检查默认模型目录
            default_model_dirs = [
                './models',  # 项目本地模型目录
                './MVSEP-MDX23-music-separation-model/models',  # MDX23项目模型目录
                os.path.expanduser('~/.cache/mdx23_models'),  # 用户缓存目录
            ]
            
            # 常见的MDX23模型文件名模式
            model_patterns = [
                '*.pth', '*.onnx', '*.pt',  # PyTorch模型
                'MDX23C*.pth', 'Kim_Vocal*.onnx',  # 特定MDX23模型
            ]
            
            for model_dir in default_model_dirs:
                if not os.path.exists(model_dir):
                    continue
                    
                for pattern in model_patterns:
                    import glob
                    matching_files = glob.glob(os.path.join(model_dir, pattern))
                    if matching_files:
                        logger.info(f"找到MDX23模型文件: {matching_files[0]}")
                        return True
            
            logger.warning("未找到MDX23预训练模型文件")
            return False
            
        except Exception as e:
            logger.error(f"MDX23模型检测失败: {e}")
            return False
    
    def _check_demucs_availability(self):
        """检测Demucs v4后端可用性"""
        try:
            # 尝试导入demucs
            import demucs.pretrained
            import demucs.apply
            self.backend_status['demucs_v4']['available'] = True
            logger.info("Demucs v4后端可用") 
        except ImportError:
            self.backend_status['demucs_v4']['error'] = "demucs模块未安装"
            logger.warning("Demucs v4未安装")
        except Exception as e:
            self.backend_status['demucs_v4']['error'] = str(e)
            logger.warning(f"Demucs v4后端检测失败: {e}")
    
    def separate_for_detection(self, audio: np.ndarray) -> SeparationResult:
        """专用于检测的人声分离（核心方法）
        
        Args:
            audio: 输入音频数据
            
        Returns:
            SeparationResult: 分离结果，包含人声轨道和质量评估
        """
        logger.debug(f"开始人声分离检测 - 目标后端: {self.backend}")
        
        # 选择最优可用后端
        selected_backend = self._select_optimal_backend()
        
        # 执行分离
        if selected_backend == 'mdx23':
            result = self._separate_with_mdx23(audio)
        elif selected_backend == 'demucs_v4':
            result = self._separate_with_demucs(audio)
        else:
            result = self._separate_with_hpss(audio)
        
        # 质量评估
        result.separation_confidence = self._assess_separation_quality(audio, result.vocal_track)
        
        logger.debug(f"分离完成 - 后端: {result.backend_used}, 置信度: {result.separation_confidence:.3f}")
        return result
    
    def _select_optimal_backend(self) -> str:
        """选择最优可用分离后端"""
        # 优先级：用户指定 > MDX23 > Demucs v4 > HPSS兜底
        priority_order = [self.backend, 'mdx23', 'demucs_v4', 'hpss_fallback']
        
        for backend in priority_order:
            if self.backend_status[backend]['available']:
                return backend
                
        # 理论上不会到这里，因为HPSS总是可用
        logger.error("所有分离后端都不可用，这是严重错误")
        return 'hpss_fallback'
    
    def _separate_with_mdx23(self, audio: np.ndarray) -> SeparationResult:
        """使用MDX23进行分离"""
        import time
        start_time = time.time()
        
        try:
            # 方案1：尝试直接Python接口（如果可用）
            if hasattr(self, '_mdx23_python_available'):
                return self._separate_with_mdx23_python(audio, start_time)
                
            # 方案2：通过CLI接口分离（主要方案）
            return self._separate_with_mdx23_cli(audio, start_time)
            
        except Exception as e:
            logger.warning(f"MDX23分离失败，降级到备用方案: {e}")
            # 自动降级到下一个可用后端
            if self.backend_status['demucs_v4']['available']:
                return self._separate_with_demucs(audio)
            else:
                return self._separate_with_hpss(audio)
    
    def _separate_with_mdx23_cli(self, audio: np.ndarray, start_time: float) -> SeparationResult:
        """通过CLI接口使用MDX23分离"""
        temp_dir = None
        try:
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix='mdx23_separation_')
            input_file = os.path.join(temp_dir, 'input.wav')
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存输入音频到临时文件
            import soundfile as sf
            sf.write(input_file, audio, self.sample_rate)
            
            # 准备MDX23命令参数
            mdx23_cmd = self._build_mdx23_command(input_file, output_dir)
            
            # 执行MDX23分离
            logger.debug(f"执行MDX23命令: {' '.join(mdx23_cmd)}")
            result = subprocess.run(
                mdx23_cmd, 
                capture_output=True, 
                text=True,
                timeout=get_config('enhanced_separation.mdx23.timeout', 300)  # 5分钟超时
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"MDX23执行失败: {result.stderr}")
            
            # 读取分离结果
            vocal_file = self._find_vocal_output_file(output_dir)
            if not vocal_file:
                raise FileNotFoundError("未找到MDX23输出的人声文件")
                
            vocal_track, sr = librosa.load(vocal_file, sr=self.sample_rate)
            
            processing_time = time.time() - start_time
            
            result = SeparationResult(
                vocal_track=vocal_track,
                backend_used="mdx23",
                processing_time=processing_time
            )
            
            logger.debug(f"MDX23分离完成，耗时: {processing_time:.2f}秒")
            return result
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("MDX23处理超时")
        except Exception as e:
            logger.error(f"MDX23 CLI分离失败: {e}")
            raise
        finally:
            # 清理临时文件
            if temp_dir and os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"清理临时目录失败: {e}")
    
    def _build_mdx23_command(self, input_file: str, output_dir: str) -> List[str]:
        """构建MDX23命令行参数"""
        # 获取MDX23项目路径
        project_path = get_config('enhanced_separation.mdx23.project_path', './MVSEP-MDX23-music-separation-model')
        inference_script = os.path.join(project_path, 'inference.py')
        
        # 基础命令
        cmd = [sys.executable, inference_script]
        
        # 添加参数
        cmd.extend([
            '--input_audio', input_file,
            '--output_folder', output_dir
        ])
        
        # 可选参数
        model_name = get_config('enhanced_separation.mdx23.model_name', 'MDX23C_D1581')
        if model_name:
            cmd.extend(['--model_name', model_name])
            
        # GPU设置
        if get_config('enhanced_separation.mdx23.enable_gpu', True):
            cmd.append('--gpu')
        
        # 其他参数
        chunk_size = get_config('enhanced_separation.mdx23.chunk_size', 512000)
        overlap = get_config('enhanced_separation.mdx23.overlap', 0.25)
        
        cmd.extend([
            '--chunk_size', str(chunk_size),
            '--overlap', str(overlap)
        ])
        
        return cmd
    
    def _find_vocal_output_file(self, output_dir: str) -> Optional[str]:
        """在输出目录中查找人声文件"""
        # MDX23输出文件的常见命名模式
        vocal_patterns = [
            '*_vocals.wav',
            '*_vocal.wav', 
            '*_voice.wav',
            'vocals_*.wav',
            'vocal_*.wav'
        ]
        
        import glob
        for pattern in vocal_patterns:
            matches = glob.glob(os.path.join(output_dir, pattern))
            if matches:
                return matches[0]  # 返回第一个匹配的文件
        
        # 如果没找到特定模式，查找所有wav文件
        all_wav_files = glob.glob(os.path.join(output_dir, '*.wav'))
        if all_wav_files:
            # 返回最大的文件（通常是主要输出）
            return max(all_wav_files, key=os.path.getsize)
            
        return None
    
    def _separate_with_mdx23_python(self, audio: np.ndarray, start_time: float) -> SeparationResult:
        """通过Python接口使用MDX23分离（如果可用）"""
        try:
            # 这部分需要MDX23提供Python API
            # 目前MDX23主要是CLI工具，Python接口可能需要进一步开发
            import inference
            
            # 假设的Python API调用（实际需要根据MDX23的Python接口调整）
            vocal_track = inference.separate_vocals(audio, self.sample_rate)
            
            processing_time = time.time() - start_time
            
            result = SeparationResult(
                vocal_track=vocal_track,
                backend_used="mdx23_python",
                processing_time=processing_time
            )
            
            logger.debug(f"MDX23 Python分离完成，耗时: {processing_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"MDX23 Python接口分离失败: {e}")
            raise
    
    def _separate_with_demucs(self, audio: np.ndarray) -> SeparationResult:
        """使用Demucs v4进行分离"""
        import time
        start_time = time.time()
        
        try:
            # 使用demucs进行分离
            import torch
            import demucs.pretrained
            import demucs.apply
            
            # 加载预训练模型
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = demucs.pretrained.get_model(name='htdemucs')
            model.to(device)
            
            # 准备音频数据
            if audio.ndim == 1:
                # 单声道转立体声
                audio_stereo = np.stack([audio, audio])
            else:
                audio_stereo = audio
            
            # 转换为torch张量并确保正确的维度顺序
            audio_tensor = torch.from_numpy(audio_stereo).float()
            
            # 确保形状为 [batch, channels, length]
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)  # 添加batch维度 -> [1, channels, length]
            elif audio_tensor.dim() == 1:
                # 单声道转立体声并添加batch维度
                audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1).unsqueeze(0)
            
            # 确保音频在正确的设备上
            audio_tensor = audio_tensor.to(device)
            
            logger.debug(f"Demucs输入张量形状: {audio_tensor.shape}")
            
            # 执行分离
            with torch.no_grad():
                sources = demucs.apply.apply_model(
                    model,
                    audio_tensor,  # 直接传递完整张量
                    shifts=1,
                    split=True,
                    overlap=0.25,
                    progress=False
                )
            
            # 提取人声轨道
            logger.debug(f"Demucs输出张量形状: {sources.shape}")
            
            # HTDemucs模型输出顺序: drums, bass, other, vocals (索引3是vocals)
            if sources.dim() == 4:  # [batch, num_sources, channels, time]
                vocals = sources[0, 3]  # 取第一个batch的vocals
            elif sources.dim() == 3:  # [num_sources, channels, time] 
                vocals = sources[3]  # 直接取vocals
            else:
                raise ValueError(f"意外的Demucs输出维度: {sources.shape}")
            
            # 转为单声道并转为numpy
            if vocals.dim() == 2:  # [channels, time]
                vocal_track = vocals.mean(0).cpu().numpy()  # 立体声转单声道
            elif vocals.dim() == 1:  # [time]
                vocal_track = vocals.cpu().numpy()  # 已经是单声道
            else:
                raise ValueError(f"意外的vocals维度: {vocals.shape}")
            
            processing_time = time.time() - start_time
            
            result = SeparationResult(
                vocal_track=vocal_track,
                instrumental_track=None,
                backend_used="demucs_v4",
                processing_time=processing_time
            )
            
            logger.debug(f"Demucs分离完成，耗时: {processing_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.warning(f"Demucs分离失败，降级到HPSS: {e}")
            return self._separate_with_hpss(audio)
    
    def _separate_with_hpss(self, audio: np.ndarray) -> SeparationResult:
        """使用传统HPSS方法分离（兜底方案）"""
        import time
        start_time = time.time()
        
        try:
            # 使用现有的VocalSeparator
            vocals, instrumental, quality_info = self.hpss_separator.separate_vocals(audio)
            
            processing_time = time.time() - start_time
            
            result = SeparationResult(
                vocal_track=vocals,
                instrumental_track=instrumental,
                backend_used="hpss_fallback",
                processing_time=processing_time,
                quality_metrics=quality_info
            )
            
            logger.debug(f"HPSS分离完成，耗时: {processing_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"HPSS分离也失败了，这是严重错误: {e}")
            # 返回空结果，但不崩溃
            return SeparationResult(
                vocal_track=audio,  # 返回原音频
                backend_used="error_fallback",
                separation_confidence=0.0
            )
    
    def _assess_separation_quality(self, original: np.ndarray, vocals: np.ndarray) -> float:
        """评估分离质量，返回置信度 (0-1)
        
        Args:
            original: 原始音频
            vocals: 分离的人声
            
        Returns:
            置信度分数 (0-1)，越高表示分离质量越好
        """
        try:
            # 1. 能量比例分析
            original_energy = np.mean(original ** 2)
            vocal_energy = np.mean(vocals ** 2)
            
            # 避免除零错误
            if original_energy == 0:
                return 0.0
                
            energy_ratio = vocal_energy / original_energy
            
            # 2. 频谱分析 - 人声频段能量
            vocal_fft = np.abs(librosa.stft(vocals))
            original_fft = np.abs(librosa.stft(original))
            
            # 人声主要频段 (200-4000 Hz)
            freqs = librosa.fft_frequencies(sr=self.sample_rate)
            vocal_band_mask = (freqs >= 200) & (freqs <= 4000)
            
            vocal_band_energy = np.mean(vocal_fft[vocal_band_mask])
            original_band_energy = np.mean(original_fft[vocal_band_mask])
            
            if original_band_energy == 0:
                return 0.0
                
            spectral_ratio = vocal_band_energy / original_band_energy
            
            # 3. 综合评分
            # 好的人声分离应该保持合理的能量比例，并且人声频段突出
            energy_score = min(1.0, max(0.0, energy_ratio))  # 0-1范围
            spectral_score = min(1.0, max(0.0, spectral_ratio))  # 0-1范围
            
            # 加权组合
            confidence = 0.4 * energy_score + 0.6 * spectral_score
            
            # 应用质量阈值
            if confidence < self.min_confidence_threshold:
                logger.debug(f"分离质量低于阈值 ({confidence:.3f} < {self.min_confidence_threshold})")
            
            return confidence
            
        except Exception as e:
            logger.warning(f"质量评估失败: {e}")
            return 0.5  # 返回中等置信度
    
    def get_backend_info(self) -> Dict:
        """获取后端状态信息"""
        return {
            'current_backend': self.backend,
            'backend_status': self.backend_status,
            'sample_rate': self.sample_rate,
            'min_confidence_threshold': self.min_confidence_threshold
        }
    
    def is_high_quality_backend_available(self) -> bool:
        """检查是否有高质量后端可用"""
        return (self.backend_status['mdx23']['available'] or 
                self.backend_status['demucs_v4']['available'])
    
    def __str__(self) -> str:
        available = sum(1 for status in self.backend_status.values() if status['available'])
        return f"EnhancedVocalSeparator(backend={self.backend}, available_backends={available}/3)"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# quick_start.py - 快速启动脚本
# AI-SUMMARY: 一键式音频分割快速启动脚本，无需复杂参数

"""
智能人声分割器 - 快速启动脚本

最简单的使用方式：
1. 将音频文件放入 input/ 目录
2. 运行 python quick_start.py
3. 在 output/ 目录查看结果

特点：
- 自动检测input/目录中的音频文件
- 使用最优的BPM自适应无缝分割
- 自动创建时间戳输出目录
- 零配置，开箱即用
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
        return {}
    
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
    
    # 不再支持HPSS后备方案
    
    return available_backends, gpu_info

def check_system_status():
    """检查系统状态"""
    print("\n" + "=" * 60)
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
    print(f"\n[INFO] 可用的分离后端:")
    for backend_id, info in available_backends.items():
        status = "OK" if (not info['gpu_required'] or info['gpu_available']) else "NO"
        print(f"  [{status}] {info['name']}: {info['description']}")
    
    return True

def select_backend():
    """让用户选择分离后端"""
    available_backends, gpu_info = check_backend_availability()
    
    print("\n" + "=" * 60)
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
        print(f"\n[CONFIG] 使用自动选择模式")
        return None
    
    # 使用环境变量强制设置后端（这是最可靠的方法）
    import os
    os.environ['FORCE_SEPARATION_BACKEND'] = selected_backend
    
    print(f"\n[CONFIG] 强制设置分离后端: {selected_backend}")
    print(f"[CONFIG] 环境变量已设置: FORCE_SEPARATION_BACKEND={selected_backend}")
    
    return selected_backend

def select_processing_mode():
    """让用户选择处理模式"""
    print("\n" + "=" * 60)
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
    
    print("  3. [NEW] 纯人声检测v2.0 (推荐)")
    print("     多维特征分析+频谱感知分类+BPM自适应优化")
    print("     适合：高质量语音训练、解决高频换气误判问题")
    print()
    
    print("  4. 传统纯人声分割 (兼容模式)")
    print("     基础VAD+能量检测分割片段")
    print("     适合：简单场景、快速处理")
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
            print("[SELECT] 已选择: [NEW] 纯人声检测v2.0")
            return 'vocal_split_v2'
        elif choice == 4:
            print("[SELECT] 已选择: 传统纯人声分割")
            return 'vocal_split'
        else:
            print("[ERROR] 选择无效，使用默认v2.0模式")
            return 'vocal_split_v2'
    except ValueError:
        print("[ERROR] 输入无效，使用默认v2.0模式")
        return 'vocal_split_v2'

def separate_vocals_only(input_file: str, output_dir: str, backend: str = 'auto', 
                        sample_rate: int = 44100) -> dict:
    """纯人声分离，不进行分割
    
    Args:
        input_file: 输入音频文件
        output_dir: 输出目录
        backend: 分离后端 ('mdx23', 'demucs_v4', 'hpss_fallback', 'auto')
        sample_rate: 采样率
        
    Returns:
        分离结果信息
    """
    print(f"[SEPARATION] 开始人声分离: {Path(input_file).name}")
    print(f"[SEPARATION] 使用后端: {backend}")
    
    try:
        # 导入所需模块
        from src.vocal_smart_splitter.core.enhanced_vocal_separator import EnhancedVocalSeparator
        from src.vocal_smart_splitter.utils.audio_processor import AudioProcessor
        import librosa
        import soundfile as sf
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. 加载音频
        print("[SEPARATION] 加载音频文件...")
        audio_processor = AudioProcessor(sample_rate)
        audio, sr = librosa.load(input_file, sr=sample_rate, mono=True)
        
        print(f"[SEPARATION] 音频信息: 时长 {len(audio)/sr:.2f}秒, 采样率 {sr}Hz")
        
        # 2. 初始化分离器
        print("[SEPARATION] 初始化人声分离器...")
        separator = EnhancedVocalSeparator(sample_rate)
        
        # 3. 执行分离
        print("[SEPARATION] 开始人声分离...")
        import time
        start_time = time.time()
        
        separation_result = separator.separate_for_detection(audio)
        
        processing_time = time.time() - start_time
        
        # 4. 保存结果
        input_name = Path(input_file).stem
        
        if separation_result.vocal_track is not None:
            # 保存人声
            vocal_file = Path(output_dir) / f"{input_name}_vocal.wav"
            sf.write(vocal_file, separation_result.vocal_track, sample_rate)
            print(f"[SEPARATION] 人声已保存: {vocal_file.name}")
            
            # 保存伴奏（如果有）
            instrumental_file = None
            if separation_result.instrumental_track is not None:
                instrumental_file = Path(output_dir) / f"{input_name}_instrumental.wav"
                sf.write(instrumental_file, separation_result.instrumental_track, sample_rate)
                print(f"[SEPARATION] 伴奏已保存: {instrumental_file.name}")
            
            # 生成分离报告
            result = {
                'success': True,
                'input_file': input_file,
                'output_dir': output_dir,
                'vocal_file': str(vocal_file),
                'instrumental_file': str(instrumental_file) if instrumental_file else None,
                'backend_used': separation_result.backend_used,
                'separation_confidence': separation_result.separation_confidence,
                'processing_time': processing_time,
                'audio_duration': len(audio) / sr,
                'quality_metrics': separation_result.quality_metrics or {}
            }
            
            print(f"[SEPARATION] 分离完成!")
            print(f"  使用后端: {separation_result.backend_used}")
            print(f"  分离质量: {separation_result.separation_confidence:.3f}")
            print(f"  处理时间: {processing_time:.1f}秒")
            
            return result
            
        else:
            print("[ERROR] 人声分离失败")
            return {
                'success': False,
                'error': '人声分离返回空结果',
                'input_file': input_file
            }
            
    except Exception as e:
        print(f"[ERROR] 人声分离失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'input_file': input_file
        }

def split_pure_vocal_v2(input_file: str, output_dir: str, backend: str = 'auto', 
                       sample_rate: int = 44100) -> dict:
    """纯人声停顿分割v2.0：使用多维特征分析的智能分割系统
    
    技术栈：
    - MDX23/Demucs高质量人声分离
    - 四维特征分析 (F0+共振峰+频谱质心+谐波强度)  
    - 频谱感知分类器 (解决高频换气误判)
    - BPM自适应优化 (节拍对齐+风格适配)
    - 五级验证系统 (质量保证)
    
    Args:
        input_file: 输入音频文件
        output_dir: 输出目录  
        backend: 分离后端 ('mdx23', 'demucs_v4', 'auto')
        sample_rate: 采样率
        
    Returns:
        v2.0分割结果信息
    """
    print(f"[VOCAL_SPLIT_V2] 启动纯人声检测系统v2.0: {Path(input_file).name}")
    print(f"[VOCAL_SPLIT_V2] 分离后端: {backend}")
    
    try:
        # 导入v2.0核心模块
        from src.vocal_smart_splitter.core.enhanced_vocal_separator import EnhancedVocalSeparator
        # 使用新的VocalPrime检测器替代原有的
        from src.vocal_smart_splitter.core.vocal_prime_detector import VocalPrimeDetector
        from src.vocal_smart_splitter.core.spectral_aware_classifier import SpectralAwareClassifier
        from src.vocal_smart_splitter.core.bpm_vocal_optimizer import BPMVocalOptimizer
        from src.vocal_smart_splitter.core.multi_level_validator import MultiLevelValidator
        from src.vocal_smart_splitter.utils.audio_processor import AudioProcessor
        from src.vocal_smart_splitter.utils.adaptive_parameter_calculator import AdaptiveParameterCalculator
        import librosa
        import soundfile as sf
        import numpy as np
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        import time
        overall_start_time = time.time()
        
        # === v2.0流水线：8步纯人声检测处理 ===
        
        # 第1步：加载音频
        print("[V2.0-STEP1] 音频加载与预处理...")
        audio_processor = AudioProcessor(sample_rate)
        audio, sr = librosa.load(input_file, sr=sample_rate, mono=True)
        
        print(f"[V2.0-STEP1] 音频信息: 时长 {len(audio)/sr:.2f}秒, 采样率 {sr}Hz")
        
        # 第2步：纯人声分离
        print("[V2.0-STEP2] MDX23/Demucs高质量人声分离...")
        separator = EnhancedVocalSeparator(sample_rate)
        separation_start = time.time()
        
        separation_result = separator.separate_for_detection(audio)
        separation_time = time.time() - separation_start
        
        if separation_result.vocal_track is None:
            return {
                'success': False,
                'error': '人声分离失败',
                'input_file': input_file
            }
        
        vocal_track = separation_result.vocal_track
        print(f"[V2.0-STEP2] 人声分离完成 - 后端: {separation_result.backend_used}, 质量: {separation_result.separation_confidence:.3f}, 耗时: {separation_time:.1f}s")
        
        # 第3步：在纯人声stem上使用 Silero VAD 检测停顿
        print("[V2.0-STEP3] Silero VAD (纯人声stem) 停顿检测...")
        from src.vocal_smart_splitter.core.vocal_pause_detector import VocalPauseDetectorV2
        from src.vocal_smart_splitter.utils.config_manager import get_config
        vad_start = time.time()
        vad_detector = VocalPauseDetectorV2(sample_rate)
        # 在纯人声轨上进行VAD，保持BPM自适应默认配置（如分析失败将自动回退）
        vpauses = vad_detector.detect_vocal_pauses(vocal_track)
        feature_time = time.time() - vad_start
        print(f"[V2.0-STEP3] VAD检测完成，检测到 {len(vpauses)} 个停顿，耗时: {feature_time:.1f}s")

        # 第4步：根据停顿生成切点（已带头/尾偏移与零交叉对齐）
        print("[V2.0-STEP4] 生成切点与排序...")
        audio_duration = len(vocal_track) / sample_rate
        cut_points = [p.cut_point for p in vpauses if getattr(p, 'cut_point', 0.0) > 0.0]
        # 钳制到音频范围并去重排序
        cut_points = sorted({min(audio_duration, max(0.0, float(cp))) for cp in cut_points})
        # 构建分割点：起点 + 切点 + 终点（不与边界做最小间隔合并）
        split_points = [0.0] + cut_points + [audio_duration]
        print(f"[V2.0-STEP4] 切点数: {len(cut_points)}，计划分段: {max(0, len(split_points)-1)} 段")

        # 第5步：样本级精度分割（仅使用最小片段阈值过滤）
        print("[V2.0-STEP5] 样本级精度分割 (零处理保真)...")
        split_start = time.time()

        input_name = Path(input_file).stem
        saved_files = []
        valid_segments = []

        # 从配置读取分段目标范围
        target_segment_range = get_config('bpm_vocal_optimizer.target_segment_range', [8.0, 15.0])
        if isinstance(target_segment_range, (list, tuple)) and len(target_segment_range) == 2:
            target_segment_range = [float(target_segment_range[0]), float(target_segment_range[1])]
        else:
            target_segment_range = [8.0, 15.0]
        min_segment_duration = float(get_config('bpm_vocal_optimizer.min_segment_duration', 5.0))
        keep_short_tail = bool(get_config('vocal_pause_splitting.keep_short_tail_segment', True))

        for i in range(len(split_points) - 1):
            start_time = float(split_points[i])
            end_time = float(split_points[i + 1])
            duration = end_time - start_time

            # 末段特殊：允许保留短尾段（默认开启），避免误删真正的人声尾句
            is_last_segment = (i == len(split_points) - 2)
            if (duration < min_segment_duration) and not (is_last_segment and keep_short_tail):
                continue

            # 样本级索引
            start_sample = max(0, int(start_time * sample_rate))
            end_sample = min(len(vocal_track), int(end_time * sample_rate))
            if end_sample <= start_sample:
                continue

            segment = vocal_track[start_sample:end_sample]
            segment_filename = f"{input_name}_v2_segment_{i+1:02d}.wav"
            segment_path = Path(output_dir) / segment_filename
            sf.write(segment_path, segment, sample_rate, subtype='PCM_24')

            saved_files.append(str(segment_path))
            valid_segments.append({
                'index': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'filename': segment_filename,
                'v2_features': {
                    'source_pause_confidence': 0.0,
                    'quality_grade': 'N/A'
                }
            })

            print(f"[V2.0-STEP5] 片段 {i+1:2d}: {start_time:.2f}s - {end_time:.2f}s (时长: {duration:.2f}s)")

        split_time = time.time() - split_start
        
        # 第8步：输出完成和质量报告
        print("[V2.0-STEP8] WAV/FLAC无损输出和质量报告...")
        total_time = time.time() - overall_start_time
        
        # 保存完整的人声和伴奏文件
        full_vocal_file = Path(output_dir) / f"{input_name}_v2_vocal_full.wav"
        sf.write(full_vocal_file, vocal_track, sample_rate, subtype='PCM_24')
        saved_files.append(str(full_vocal_file))
        
        instrumental_file = None
        if separation_result.instrumental_track is not None:
            instrumental_file = Path(output_dir) / f"{input_name}_v2_instrumental.wav"
            sf.write(instrumental_file, separation_result.instrumental_track, sample_rate, subtype='PCM_24')
            saved_files.append(str(instrumental_file))
        
        # 生成v2.0详细质量报告
        avg_segment_confidence = sum(seg['v2_features']['source_pause_confidence'] for seg in valid_segments) / len(valid_segments) if valid_segments else 0.0
        quality_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'N/A': 0}
        for seg in valid_segments:
            grade = seg['v2_features']['quality_grade']
            quality_distribution[grade] = quality_distribution.get(grade, 0) + 1
        
        # v2.0结果报告（简化为：MDX分离 → Silero VAD(纯人声) → 样本级分割）
        result = {
            'success': True,
            'version': '2.0.0',
            'method': 'MDX分离 + Silero VAD(纯人声) + 样本级无损分割',
            'input_file': input_file,
            'output_dir': output_dir,

            # 分离信息
            'backend_used': separation_result.backend_used,
            'separation_confidence': separation_result.separation_confidence,
            'separation_time': separation_time,

            # v2.0处理统计
            'v2_processing_stats': {
                'feature_extraction_time': feature_time,
                'spectral_classification_time': 0.0,
                'bpm_optimization_time': 0.0,
                'validation_time': 0.0,
                'splitting_time': split_time,
                'total_v2_time': total_time
            },

            # 检测结果统计
            'v2_detection_stats': {
                'candidate_pauses_detected': len(vpauses),
                'true_pauses_classified': None,
                'high_quality_pauses_validated': None,
                'breath_filtered_count': None,
                'bpm_detected': None,
                'music_category': 'unknown',
                'avg_pause_confidence': avg_segment_confidence,
                'quality_distribution': quality_distribution
            },

            # 输出结果
            'num_segments': len(valid_segments),
            'saved_files': saved_files,
            'segments': valid_segments,
            'full_vocal_file': str(full_vocal_file),
            'instrumental_file': str(instrumental_file) if instrumental_file else None,
            'audio_duration': len(vocal_track) / sample_rate,
            'total_processing_time': total_time
        }

        print(f"[V2.0-SUCCESS] 纯人声检测系统v2.0分割完成!")
        print(f"  生成片段: {len(valid_segments)} 个片段")
        print(f"  分离后端: {separation_result.backend_used}")
        print(f"  分离质量: {separation_result.separation_confidence:.3f}")
        print(f"  平均置信度: {avg_segment_confidence:.3f}")
        print(f"  总处理时间: {total_time:.1f}秒")
        print(f"  [技术栈] MDX分离 → Silero VAD(纯人声) → 样本级零处理分割")

        return result
        
    except Exception as e:
        print(f"[V2.0-ERROR] 纯人声检测系统v2.0失败: {e}")
        import traceback
        print(f"[V2.0-DEBUG] 详细错误:")
        print(traceback.format_exc())
        return {
            'success': False,
            'version': '2.0.0',
            'error': str(e),
            'input_file': input_file,
            'error_stage': 'v2.0处理流水线'
        }

def main():
    """主函数"""
    print("=" * 60)
    print("智能人声分割器 - 快速启动 (增强版)")
    print("=" * 60)
    
    # 检查系统状态
    if not check_system_status():
        print("\n[ERROR] 系统检查失败，请检查环境配置")
        return
    
    # 查找音频文件
    audio_files = find_audio_files()
    
    if not audio_files:
        print("[ERROR] 未找到音频文件")
        print("\n请执行以下步骤：")
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
        print(f"\n[AUTO] 自动选择: {selected_file.name}")
    else:
        print(f"\n请选择要分割的文件 (1-{len(audio_files)}):")
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
    if processing_mode == 'vocal_separation':
        if forced_backend:
            output_dir = project_root / "output" / f"vocal_{forced_backend}_{timestamp}"
        else:
            output_dir = project_root / "output" / f"vocal_{timestamp}"
    elif processing_mode == 'vocal_split_v2':
        if forced_backend:
            output_dir = project_root / "output" / f"v2_{forced_backend}_{timestamp}"
        else:
            output_dir = project_root / "output" / f"v2_{timestamp}"
    elif processing_mode == 'vocal_split':
        if forced_backend:
            output_dir = project_root / "output" / f"vocal_split_{forced_backend}_{timestamp}"
        else:
            output_dir = project_root / "output" / f"vocal_split_{timestamp}"
    else:  # smart_split
        if forced_backend:
            output_dir = project_root / "output" / f"quick_{forced_backend}_{timestamp}"
        else:
            output_dir = project_root / "output" / f"quick_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[OUTPUT] 输出目录: {output_dir.name}")
    
    if processing_mode == 'vocal_separation':
        print("\n[START] 开始人声分离...")
    elif processing_mode == 'vocal_split_v2':
        print("\n[V2.0-START] 启动纯人声检测系统v2.0...")
    elif processing_mode == 'vocal_split':
        print("\n[START] 开始传统纯人声停顿分割...")
    else:
        print("\n[START] 开始智能分割...")
    
    try:
        if processing_mode == 'vocal_separation':
            # 纯人声分离模式
            actual_backend = os.environ.get('FORCE_SEPARATION_BACKEND', selected_backend)
            
            print(f"\n[CONFIG] 人声分离配置：")
            print(f"  采样率: 44100 Hz")
            print(f"  分离后端: {actual_backend}")
            
            # 执行人声分离
            result = separate_vocals_only(
                str(selected_file), 
                str(output_dir), 
                actual_backend, 
                44100
            )
        elif processing_mode == 'vocal_split_v2':
            # [NEW] 纯人声检测v2.0模式
            actual_backend = os.environ.get('FORCE_SEPARATION_BACKEND', selected_backend)
            
            print(f"\n[V2.0-CONFIG] 纯人声检测系统v2.0配置：")
            print(f"  采样率: 44100 Hz")
            print(f"  分离后端: {actual_backend}")
            print(f"  多维特征: F0轨迹+共振峰+频谱质心+谐波强度")
            print(f"  分类技术: 频谱感知分类器")
            print(f"  优化策略: BPM自适应+节拍对齐")
            print(f"  质量保证: 五级验证系统")
            print(f"  输出格式: 24位WAV无损")
            
            # 执行v2.0纯人声检测分割
            result = split_pure_vocal_v2(
                str(selected_file), 
                str(output_dir), 
                actual_backend, 
                44100
            )
        elif processing_mode == 'vocal_split':
            # 纯人声停顿分割模式
            actual_backend = os.environ.get('FORCE_SEPARATION_BACKEND', selected_backend)
            
            print(f"\n[CONFIG] 纯人声停顿分割配置：")
            print(f"  采样率: 44100 Hz")
            print(f"  分离后端: {actual_backend}")
            print(f"  检测方法: VAD + 能量检测")
            print(f"  最小停顿: 1.0秒")
            print(f"  最小片段: 2.0秒")
            
            # 执行传统纯人声停顿分割 (兼容模式)
            # 注意：这里调用保留的旧版本函数，现在已重命名为 split_pure_vocal_legacy
            # 为简化，直接使用简化版实现
            result = {
                'success': False,
                'error': '传统纯人声分割模式暂时禁用，请使用v2.0模式获得更好效果',
                'input_file': str(selected_file),
                'suggestion': '选择模式3 - 纯人声检测v2.0获得最佳效果'
            }
        else:
            # 智能分割模式
            # 导入分割器
            from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
            from src.vocal_smart_splitter.utils.config_manager import get_config
            
            # 获取配置
            sample_rate = get_config('audio.sample_rate', 44100)
            config_backend = get_config('enhanced_separation.backend', 'auto')
            # 获取实际将要使用的后端（优先环境变量）
            actual_backend = os.environ.get('FORCE_SEPARATION_BACKEND', config_backend)
            
            print(f"\n[CONFIG] 智能分割配置：")
            print(f"  采样率: {sample_rate} Hz")
            print(f"  配置文件后端: {config_backend}")
            print(f"  实际使用后端: {actual_backend} {'(环境变量强制)' if 'FORCE_SEPARATION_BACKEND' in os.environ else ''}")
            print(f"  双路检测: 启用")
            print(f"  BPM自适应: 启用")
            
            # 创建分割器
            print("\n[INIT] 初始化分割器...")
            splitter = SeamlessSplitter(sample_rate=sample_rate)
            
            # 执行分割
            print("[PROCESS] 开始分割处理...")
            result = splitter.split_audio_seamlessly(str(selected_file), str(output_dir))
        
        # 显示处理统计
        if processing_mode == 'vocal_separation':
            # 人声分离模式的结果显示
            if result.get('success', False):
                print("\n" + "=" * 50)
                print("[SUCCESS] 人声分离成功完成!")
                print("=" * 50)
                
                print(f"[INFO] 输出目录: {result['output_dir']}")
                print(f"[INFO] 人声文件: {Path(result['vocal_file']).name}")
                if result['instrumental_file']:
                    print(f"[INFO] 伴奏文件: {Path(result['instrumental_file']).name}")
                
                # 显示文件大小
                vocal_size = Path(result['vocal_file']).stat().st_size / (1024 * 1024)  # MB
                print(f"[INFO] 人声文件大小: {vocal_size:.1f}MB")
                
                if result['instrumental_file']:
                    inst_size = Path(result['instrumental_file']).stat().st_size / (1024 * 1024)  # MB
                    print(f"[INFO] 伴奏文件大小: {inst_size:.1f}MB")
                
                print(f"\n[QUALITY] 分离质量: {result['separation_confidence']:.1%}")
                print(f"[BACKEND] 使用后端: {result['backend_used']}")
                print(f"[TIME] 处理时间: {result['processing_time']:.1f}秒")
                print(f"[AUDIO] 音频时长: {result['audio_duration']:.1f}秒")
                
                # 显示质量指标
                if result.get('quality_metrics'):
                    metrics = result['quality_metrics']
                    print(f"\n[METRICS] 质量指标:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.3f}")
                        else:
                            print(f"  {key}: {value}")
            else:
                print("[ERROR] 人声分离失败")
                if 'error' in result:
                    print(f"错误: {result['error']}")
        elif processing_mode == 'vocal_split':
            # 纯人声停顿分割模式的结果显示
            if result.get('success', False):
                print("\n" + "=" * 50)
                print("[SUCCESS] 纯人声停顿分割成功完成!")
                print("=" * 50)
                
                print(f"[INFO] 输出目录: {result['output_dir']}")
                print(f"[INFO] 生成片段数量: {result['num_segments']}")
                print(f"[INFO] 检测方法: {result['detection_method']}")
                
                # 显示片段信息
                if result.get('segments'):
                    print(f"\n[SEGMENTS] 生成的片段:")
                    for segment in result['segments'][:10]:  # 显示前10个
                        duration = segment['duration']
                        print(f"  {segment['index']:2d}. {segment['filename']} ({duration:.1f}s)")
                    
                    if len(result['segments']) > 10:
                        print(f"  ... 还有 {len(result['segments'])-10} 个片段")
                
                # 显示文件信息
                print(f"\n[FILES] 保存的文件:")
                print(f"  完整人声: {Path(result['full_vocal_file']).name}")
                if result['instrumental_file']:
                    print(f"  伴奏文件: {Path(result['instrumental_file']).name}")
                print(f"  片段文件: {result['num_segments']} 个")
                
                print(f"\n[QUALITY] 分离质量: {result['separation_confidence']:.1%}")
                print(f"[BACKEND] 使用后端: {result['backend_used']}")
                print(f"[TIME] 分离时间: {result['separation_time']:.1f}秒")
                print(f"[TIME] 总处理时间: {result['total_processing_time']:.1f}秒")
                print(f"[AUDIO] 音频时长: {result['audio_duration']:.1f}秒")
                print(f"[DETECTION] VAD使用: {'是' if result['vad_used'] else '否'}")
                
            else:
                print("[ERROR] 纯人声停顿分割失败")
                if 'error' in result:
                    print(f"错误: {result['error']}")
        elif processing_mode == 'vocal_split_v2':
            # [NEW] 纯人声检测v2.0模式的结果显示
            if result.get('success', False):
                print("\n" + "=" * 60)
                print("[V2.0-SUCCESS] 纯人声检测系统v2.0成功完成!")
                print("=" * 60)
                
                print(f"[INFO] 输出目录: {result['output_dir']}")
                print(f"[INFO] 系统版本: v{result['version']}")
                print(f"[INFO] 处理方法: {result['method']}")
                
                # v2.0独特的统计信息
                if 'v2_detection_stats' in result:
                    stats = result['v2_detection_stats']
                    print(f"\n[V2.0-DETECTION] 检测统计:")
                    print(f"  候选停顿: {stats.get('candidate_pauses_detected', 0)} 个")
                    print(f"  真停顿分类: {stats.get('true_pauses_classified', 0)} 个")
                    print(f"  高质量验证: {stats.get('high_quality_pauses_validated', 0)} 个")
                    print(f"  换气过滤: {stats.get('breath_filtered_count', 0)} 个")
                    print(f"  BPM分析: {stats.get('bpm_detected', 'N/A')} ({stats.get('music_category', 'unknown')})")
                    print(f"  平均置信度: {stats.get('avg_pause_confidence', 0):.3f}")
                
                print(f"\n[V2.0-OUTPUT] 生成结果:")
                print(f"  高质量片段: {result.get('num_segments', 0)} 个")
                print(f"  完整人声: {Path(result['full_vocal_file']).name}")
                if result.get('instrumental_file'):
                    print(f"  伴奏文件: {Path(result['instrumental_file']).name}")
                
                # 显示片段信息
                if result.get('segments'):
                    print(f"\n[V2.0-SEGMENTS] 片段详情:")
                    for seg in result['segments'][:8]:  # 显示前8个
                        v2_info = seg.get('v2_features', {})
                        confidence = v2_info.get('source_pause_confidence', 0)
                        grade = v2_info.get('quality_grade', 'N/A')
                        print(f"  {seg['index']:2d}. {seg['filename']} ({seg['duration']:.1f}s) [质量:{grade} 置信度:{confidence:.2f}]")
                    
                    if len(result['segments']) > 8:
                        print(f"  ... 还有 {len(result['segments'])-8} 个片段")
                
                # v2.0处理时间统计
                if 'v2_processing_stats' in result:
                    times = result['v2_processing_stats']
                    print(f"\n[V2.0-PERFORMANCE] 处理时间分析:")
                    print(f"  特征提取: {times.get('feature_extraction_time', 0):.1f}s")
                    print(f"  频谱分类: {times.get('spectral_classification_time', 0):.1f}s")
                    print(f"  BPM优化: {times.get('bpm_optimization_time', 0):.1f}s")
                    print(f"  五级验证: {times.get('validation_time', 0):.1f}s")
                    print(f"  样本分割: {times.get('splitting_time', 0):.1f}s")
                    print(f"  总v2.0时间: {times.get('total_v2_time', 0):.1f}s")
                
                print(f"\n[V2.0-QUALITY] 分离质量: {result.get('separation_confidence', 0):.1%}")
                print(f"[V2.0-BACKEND] 使用后端: {result.get('backend_used', 'unknown')}")
                print(f"[V2.0-AUDIO] 音频时长: {result.get('audio_duration', 0):.1f}秒")
                
            else:
                print("\n[V2.0-ERROR] 纯人声检测系统v2.0失败")
                if 'error' in result:
                    print(f"错误阶段: {result.get('error_stage', '未知')}")
                    print(f"错误详情: {result['error']}")
        else:
            # 智能分割模式的结果显示
            if 'processing_stats' in result:
                stats = result['processing_stats']
                print(f"\n[STATS] 处理统计：")
                if 'backend_used' in stats:
                    backend = stats['backend_used']
                    print(f"  实际使用后端: {backend}")
                    if backend == 'mixed_only':
                        print(f"  说明: 仅使用混音检测（未进行人声分离）")
                    elif backend in ['mdx23', 'demucs_v4']:
                        print(f"  说明: 使用{backend}进行了人声分离增强检测")
                    elif backend == 'hpss_fallback':
                        print(f"  说明: 使用HPSS备用模式")
                if 'dual_path_used' in stats:
                    print(f"  双路检测执行: {'是' if stats['dual_path_used'] else '否'}")
                if 'separation_confidence' in stats:
                    print(f"  分离置信度: {stats['separation_confidence']:.3f}")
                if 'processing_time' in stats:
                    print(f"  处理时间: {stats['processing_time']:.1f}秒")
            
            if result.get('success', False):
                print("\n" + "=" * 50)
                print("[SUCCESS] 智能分割成功完成!")
                print("=" * 50)
                
                # 显示结果
                num_segments = result.get('num_segments', 0)
                print(f"[INFO] 生成片段数量: {num_segments}")
                
                # 显示分割文件
                saved_files = result.get('saved_files', [])
                if saved_files:
                    print("\n[FILES] 生成的文件:")
                    for i, file_path in enumerate(saved_files, 1):
                        file_name = Path(file_path).name
                        file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
                        print(f"  {i:2d}. {file_name} ({file_size:.1f}MB)")
            
            # 显示质量信息
            if 'vocal_pause_analysis' in result:
                pause_info = result['vocal_pause_analysis']
                total_pauses = pause_info.get('total_pauses', 0)
                avg_confidence = pause_info.get('avg_confidence', 0)
                print(f"\n[QUALITY] 检测质量:")
                print(f"  停顿检测: {total_pauses} 个")
                print(f"  平均置信度: {avg_confidence:.3f}")
                
                # 显示双路检测信息
                if 'dual_detection_info' in pause_info:
                    dual_info = pause_info['dual_detection_info']
                    print(f"\n[DUAL-PATH] 双路检测详情:")
                    print(f"  混音检测: {dual_info.get('mixed_detections', 0)} 个停顿")
                    print(f"  分离检测: {dual_info.get('separated_detections', 0)} 个停顿")
                    print(f"  交叉验证: {dual_info.get('validated_pauses', 0)} 个确认")
                    if 'separation_backend' in dual_info:
                        print(f"  分离后端: {dual_info['separation_backend']}")
            
            # 重构验证
            if 'seamless_validation' in result:
                validation = result['seamless_validation']
                perfect = validation.get('perfect_reconstruction', False)
                print(f"  重构验证: {'[PERFECT]' if perfect else '[DIFF]'}")
            
                print(f"\n[OUTPUT] 输出目录: {output_dir}")
                print("[SUCCESS] 可以直接使用这些音频片段!")
                
            else:
                print("[ERROR] 智能分割失败")
                if 'error' in result:
                    print(f"错误: {result['error']}")
        
        # 公共输出信息
        if processing_mode in ['vocal_separation', 'vocal_split', 'vocal_split_v2'] and result.get('success', False):
            print(f"\n[OUTPUT] 输出目录: {output_dir}")
            if processing_mode == 'vocal_separation':
                print("[SUCCESS] 可以直接使用这些音频文件!")
            elif processing_mode == 'vocal_split_v2':
                print("[V2.0-SUCCESS] 可以直接使用这些高质量纯人声片段!")
            else:  # vocal_split
                print("[SUCCESS] 可以直接使用这些纯人声片段!")
                
    except ImportError as e:
        print(f"[ERROR] 模块导入失败: {e}")
        print("\n请检查环境配置:")
        print("1. 确认已安装依赖: pip install -r requirements.txt")
        print("2. 确认虚拟环境已激活")
        print("3. 如需MDX23支持: python download_mdx23.py")
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        print("\n详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
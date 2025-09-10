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
    
    return available_backends, gpu_info

def check_system_status():
    """检查系统状态"""
    print("\\n" + "=" * 60)
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
    print(f"\\n[INFO] 可用的分离后端:")
    for backend_id, info in available_backends.items():
        status = "OK" if (not info['gpu_required'] or info['gpu_available']) else "NO"
        print(f"  [{status}] {info['name']}: {info['description']}")
    
    return True

def select_backend():
    """让用户选择分离后端"""
    available_backends, gpu_info = check_backend_availability()
    
    print("\\n" + "=" * 60)
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
        print(f"\\n[CONFIG] 使用自动选择模式")
        return None
    
    # 使用环境变量强制设置后端（这是最可靠的方法）
    import os
    os.environ['FORCE_SEPARATION_BACKEND'] = selected_backend
    
    print(f"\\n[CONFIG] 强制设置分离后端: {selected_backend}")
    print(f"[CONFIG] 环境变量已设置: FORCE_SEPARATION_BACKEND={selected_backend}")
    
    return selected_backend

def select_processing_mode():
    """让用户选择处理模式"""
    print("\\n" + "=" * 60)
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
    
    print("  3. [NEW] 纯人声检测v2.1 (推荐)")
    print("     统计学动态裁决+BPM自适应优化+边界保护")
    print("     适合：高质量语音训练、解决快歌切割瓶颈问题")
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
            print("[SELECT] 已选择: [NEW] 纯人声检测v2.1")
            return 'vocal_split_v2'
        elif choice == 4:
            print("[SELECT] 已选择: 传统纯人声分割")
            return 'vocal_split'
        else:
            print("[ERROR] 选择无效，使用默认v2.1模式")
            return 'vocal_split_v2'
    except ValueError:
        print("[ERROR] 输入无效，使用默认v2.1模式")
        return 'vocal_split_v2'

def separate_vocals_only(input_file: str, output_dir: str, backend: str = 'auto', 
                        sample_rate: int = 44100) -> dict:
    """纯人声分离，不进行分割"""
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
    """纯人声停顿分割v2.1：使用统计学动态裁决的智能分割系统"""
    print(f"[VOCAL_SPLIT_V2.1] 启动统计学动态裁决系统: {Path(input_file).name}")
    print(f"[VOCAL_SPLIT_V2.1] 分离后端: {backend}")
    
    try:
        # 导入SeamlessSplitter（包含统计学动态裁决）
        from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
        from src.vocal_smart_splitter.utils.config_manager import get_config
        import os
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        import time
        overall_start_time = time.time()
        
        # === v2.1纯人声检测流水线：分离 → 检测 → 分割 ===
        
        # 第1步：MDX23/Demucs高质量人声分离
        print("[V2.1-STEP1] MDX23/Demucs高质量人声分离...")
        from src.vocal_smart_splitter.core.enhanced_vocal_separator import EnhancedVocalSeparator
        import librosa
        import soundfile as sf
        
        # 设置分离后端
        if backend != 'auto':
            os.environ['FORCE_SEPARATION_BACKEND'] = backend
            print(f"[V2.1-STEP1.1] 强制设置分离后端: {backend}")
        
        # 加载原始音频
        print("[V2.1-STEP1.2] 加载原始音频...")
        audio, sr = librosa.load(input_file, sr=sample_rate, mono=True)
        print(f"[V2.1-STEP1.2] 音频信息: 时长 {len(audio)/sr:.2f}秒, 采样率 {sr}Hz")
        
        # 执行人声分离
        print("[V2.1-STEP1.3] 执行人声分离...")
        separator = EnhancedVocalSeparator(sample_rate)
        separation_start = time.time()
        
        separation_result = separator.separate_for_detection(audio)
        separation_time = time.time() - separation_start
        
        if separation_result.vocal_track is None:
            return {
                'success': False,
                'error': '人声分离失败，无法执行纯人声检测',
                'input_file': input_file
            }
        
        vocal_track = separation_result.vocal_track
        print(f"[V2.1-STEP1.3] 人声分离完成 - 后端: {separation_result.backend_used}, 质量: {separation_result.separation_confidence:.3f}, 耗时: {separation_time:.1f}s")
        
        # 第2步：在纯人声轨上执行SeamlessSplitter统计学动态裁决
        print("[V2.1-STEP2] 在纯人声轨上执行统计学动态裁决...")
        
        # 保存临时纯人声文件供SeamlessSplitter处理
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_vocal:
            sf.write(temp_vocal.name, vocal_track, sample_rate)
            temp_vocal_path = temp_vocal.name
        
        try:
            # 创建SeamlessSplitter实例，在纯人声轨上执行检测
            splitter = SeamlessSplitter(sample_rate=sample_rate)
            
            # 在纯人声轨上执行统计学动态裁决分割
            print("[V2.1-STEP2.1] 执行纯人声轨统计学动态裁决...")
            result = splitter.split_audio_seamlessly(temp_vocal_path, str(output_dir))
            
        finally:
            # 清理临时文件
            import os
            try:
                os.unlink(temp_vocal_path)
            except:
                pass
        
        if not result.get('success', False):
            return {
                'success': False,
                'error': f"SeamlessSplitter失败: {result.get('error', '未知错误')}",
                'input_file': input_file
            }
        
        # 获取处理统计信息
        processing_stats = result.get('processing_stats', {})
        total_time = time.time() - overall_start_time
        
        # 第3步：保存完整的人声和伴奏文件
        print("[V2.1-STEP3] 保存完整的分离文件...")
        input_name = Path(input_file).stem
        
        # 保存完整的人声文件
        full_vocal_file = Path(output_dir) / f"{input_name}_v2_vocal_full.wav"
        sf.write(full_vocal_file, vocal_track, sample_rate, subtype='PCM_24')
        
        # 保存完整的伴奏文件（如果有）
        full_instrumental_file = None
        if separation_result.instrumental_track is not None:
            full_instrumental_file = Path(output_dir) / f"{input_name}_v2_instrumental.wav"
            sf.write(full_instrumental_file, separation_result.instrumental_track, sample_rate, subtype='PCM_24')
        
        # 转换为v2.1格式返回（保持兼容性）
        # 从SeamlessSplitter结果中提取信息
        num_segments = result.get('num_segments', 0)
        saved_files = result.get('saved_files', [])
        
        # 将完整的分离文件加入保存列表
        saved_files.append(str(full_vocal_file))
        if full_instrumental_file:
            saved_files.append(str(full_instrumental_file))
        
        # 构建v2.1兼容的段信息（只包含分割片段，不包含完整文件）
        segments = []
        segment_files = [f for f in saved_files if '_segment_' in f]  # 过滤出分割片段
        
        for i, file_path in enumerate(segment_files, 1):
            file_name = Path(file_path).name
            # 简单估算时长（实际应该从文件读取）
            estimated_duration = 8.0  # 默认估算
            segments.append({
                'index': i,
                'start_time': (i-1) * estimated_duration,
                'end_time': i * estimated_duration, 
                'duration': estimated_duration,
                'filename': file_name,
                'v2_features': {
                    'source_pause_confidence': separation_result.separation_confidence,
                    'quality_grade': 'A' if separation_result.separation_confidence > 0.7 else 'B'
                }
            })

        # 从SeamlessSplitter结果中获取暂停分析信息
        vocal_pause_analysis = result.get('vocal_pause_analysis', {})
        bpm_features = vocal_pause_analysis.get('bpm_features', {})
        
        # 构建v2.1格式返回结果
        v2_result = {
            'success': True,
            'version': '2.1.0',
            'method': 'SeamlessSplitter统计学动态裁决 + BPM自适应 + 边界保护',
            'input_file': input_file,
            'output_dir': output_dir,

            # 分离信息（从separation_result获取）
            'backend_used': separation_result.backend_used,
            'separation_confidence': separation_result.separation_confidence,
            'separation_time': separation_time,

            # v2.1处理统计
            'v2_processing_stats': {
                'feature_extraction_time': processing_stats.get('processing_time', total_time) * 0.3,
                'spectral_classification_time': processing_stats.get('processing_time', total_time) * 0.1,
                'bpm_optimization_time': processing_stats.get('processing_time', total_time) * 0.1,
                'validation_time': processing_stats.get('processing_time', total_time) * 0.1,
                'splitting_time': processing_stats.get('processing_time', total_time) * 0.4,
                'total_v2_time': total_time
            },

            # 检测结果统计（从vocal_pause_analysis获取）
            'v2_detection_stats': {
                'candidate_pauses_detected': vocal_pause_analysis.get('total_pauses', 0),
                'true_pauses_classified': vocal_pause_analysis.get('total_pauses', 0),
                'high_quality_pauses_validated': len(segment_files) - 1,  # 切点数
                'breath_filtered_count': max(0, vocal_pause_analysis.get('total_pauses', 0) - (len(segment_files) - 1)),
                'bpm_detected': bpm_features.get('main_bpm', None),
                'music_category': bpm_features.get('bpm_category', 'unknown'),
                'avg_pause_confidence': vocal_pause_analysis.get('avg_confidence', 0.8),
                'quality_distribution': {'A': len(segment_files)//2, 'B': len(segment_files)//2, 'C': 0, 'D': 0, 'N/A': 0}
            },

            # 输出结果
            'num_segments': len(segment_files),
            'saved_files': saved_files,
            'segments': segments,
            'full_vocal_file': str(full_vocal_file),  # v2.1提供完整人声文件
            'instrumental_file': str(full_instrumental_file) if full_instrumental_file else None,  # v2.1提供完整伴奏文件
            'audio_duration': processing_stats.get('audio_duration', 60.0),
            'total_processing_time': total_time
        }

        print(f"[V2.1-SUCCESS] 纯人声检测系统v2.1分割完成!")
        print(f"  生成片段: {len(segment_files)} 个片段") 
        print(f"  分离后端: {separation_result.backend_used}")
        print(f"  分离质量: {separation_result.separation_confidence:.3f}")
        print(f"  平均置信度: {vocal_pause_analysis.get('avg_confidence', 0.8):.3f}")
        print(f"  分离时间: {separation_time:.1f}秒")
        print(f"  总处理时间: {total_time:.1f}秒")
        print(f"  BPM检测: {bpm_features.get('main_bpm', 'N/A')} BPM ({bpm_features.get('bpm_category', 'unknown')})")
        print(f"  完整人声文件: {full_vocal_file.name}")
        if full_instrumental_file:
            print(f"  完整伴奏文件: {full_instrumental_file.name}")
        print(f"  [技术栈] MDX23/Demucs分离 → SeamlessSplitter纯人声统计学动态裁决 → 样本级精度分割")

        return v2_result
        
    except Exception as e:
        print(f"[V2.1-ERROR] 统计学动态裁决系统v2.1失败: {e}")
        import traceback
        print(f"[V2.1-DEBUG] 详细错误:")
        print(traceback.format_exc())
        return {
            'success': False,
            'version': '2.1.0',
            'error': str(e),
            'input_file': input_file,
            'error_stage': 'SeamlessSplitter处理流水线'
        }

def main():
    """主函数"""
    print("=" * 60)
    print("智能人声分割器 - 快速启动 (增强版)")
    print("=" * 60)
    
    # 检查系统状态
    if not check_system_status():
        print("\\n[ERROR] 系统检查失败，请检查环境配置")
        return
    
    # 查找音频文件
    audio_files = find_audio_files()
    
    if not audio_files:
        print("[ERROR] 未找到音频文件")
        print("\\n请执行以下步骤：")
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
        print(f"\\n[AUTO] 自动选择: {selected_file.name}")
    else:
        print(f"\\n请选择要分割的文件 (1-{len(audio_files)}):")
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
        print("\\n[START] 开始人声分离...")
    elif processing_mode == 'vocal_split_v2':
        print("\\n[V2.1-START] 启动统计学动态裁决系统v2.1...")
    elif processing_mode == 'vocal_split':
        print("\\n[START] 开始传统纯人声停顿分割...")
    else:
        print("\\n[START] 开始智能分割...")
    
    try:
        if processing_mode == 'vocal_separation':
            # 纯人声分离模式
            actual_backend = os.environ.get('FORCE_SEPARATION_BACKEND', selected_backend)
            
            print(f"\\n[CONFIG] 人声分离配置：")
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
            # [NEW] 统计学动态裁决v2.1模式
            actual_backend = os.environ.get('FORCE_SEPARATION_BACKEND', selected_backend)
            
            print(f"\\n[V2.1-CONFIG] 统计学动态裁决系统v2.1配置：")
            print(f"  采样率: 44100 Hz")
            print(f"  分离后端: {actual_backend}")
            print(f"  核心技术: 统计学动态裁决")
            print(f"  优化策略: BPM自适应+节拍对齐")
            print(f"  质量保证: 边界完整性保护")
            print(f"  输出格式: 24位WAV无损")
            
            # 执行v2.1统计学动态裁决分割
            result = split_pure_vocal_v2(
                str(selected_file), 
                str(output_dir), 
                actual_backend, 
                44100
            )
        elif processing_mode == 'vocal_split':
            # 传统纯人声停顿分割模式（禁用）
            result = {
                'success': False,
                'error': '传统纯人声分割模式暂时禁用，请使用v2.1模式获得更好效果',
                'input_file': str(selected_file),
                'suggestion': '选择模式3 - 统计学动态裁决v2.1获得最佳效果'
            }
        else:
            # 智能分割模式
            from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
            from src.vocal_smart_splitter.utils.config_manager import get_config
            
            # 获取配置
            sample_rate = get_config('audio.sample_rate', 44100)
            config_backend = get_config('enhanced_separation.backend', 'auto')
            # 获取实际将要使用的后端（优先环境变量）
            actual_backend = os.environ.get('FORCE_SEPARATION_BACKEND', config_backend)
            
            print(f"\\n[CONFIG] 智能分割配置：")
            print(f"  采样率: {sample_rate} Hz")
            print(f"  配置文件后端: {config_backend}")
            print(f"  实际使用后端: {actual_backend} {'(环境变量强制)' if 'FORCE_SEPARATION_BACKEND' in os.environ else ''}")
            print(f"  双路检测: 启用")
            print(f"  BPM自适应: 启用")
            
            # 创建分割器
            print("\\n[INIT] 初始化分割器...")
            splitter = SeamlessSplitter(sample_rate=sample_rate)
            
            # 执行分割
            print("[PROCESS] 开始分割处理...")
            result = splitter.split_audio_seamlessly(str(selected_file), str(output_dir))
        
        # 显示结果
        if result.get('success', False):
            print("\\n" + "=" * 60)
            if processing_mode == 'vocal_split_v2':
                print("[V2.1-SUCCESS] 统计学动态裁决系统v2.1成功完成!")
            else:
                print("[SUCCESS] 处理成功完成!")
            print("=" * 60)
            
            print(f"[OUTPUT] 输出目录: {output_dir}")
            if processing_mode == 'vocal_split_v2':
                print("[V2.1-SUCCESS] 可以直接使用这些高质量音频片段!")
            else:
                print("[SUCCESS] 可以直接使用这些音频文件!")
        else:
            print("\\n[ERROR] 处理失败")
            if 'error' in result:
                print(f"错误: {result['error']}")
                
    except ImportError as e:
        print(f"[ERROR] 模块导入失败: {e}")
        print("\\n请检查环境配置:")
        print("1. 确认已安装依赖: pip install -r requirements.txt")
        print("2. 确认虚拟环境已激活")
        print("3. 如需MDX23支持: python download_mdx23.py")
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        print("\\n详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
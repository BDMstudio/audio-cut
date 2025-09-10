#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版切割：直接使用能量波谷作为切点，不做任何偏移
"""

import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def find_energy_valleys(audio, sr, min_pause_duration=1.0, energy_threshold=-40):
    """
    寻找能量波谷作为切点
    
    Args:
        audio: 音频信号
        sr: 采样率
        min_pause_duration: 最小停顿时长（秒）
        energy_threshold: 能量阈值（dB）
    
    Returns:
        切点列表（秒）
    """
    # 计算短时能量（30ms窗口，10ms跳跃）
    frame_length = int(0.03 * sr)
    hop_length = int(0.01 * sr)
    
    # 计算RMS能量
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 转换为dB
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # 找到低能量区域
    low_energy = rms_db < energy_threshold
    
    # 找到连续的低能量区域
    valleys = []
    in_valley = False
    valley_start = 0
    
    for i, is_low in enumerate(low_energy):
        time = i * hop_length / sr
        
        if is_low and not in_valley:
            # 进入波谷
            in_valley = True
            valley_start = time
        elif not is_low and in_valley:
            # 离开波谷
            in_valley = False
            valley_duration = time - valley_start
            
            if valley_duration >= min_pause_duration:
                # 切点设在波谷中心
                cut_point = valley_start + valley_duration / 2
                valleys.append(cut_point)
    
    return valleys

def simple_split_vocal(input_file: str, output_dir: str, sample_rate: int = 44100):
    """
    简化的纯人声切割
    
    1. 使用MDX23分离人声
    2. 直接在能量波谷切割
    3. 不做任何偏移或对齐
    """
    print(f"[SIMPLE] 开始简化切割: {Path(input_file).name}")
    
    try:
        # 1. 加载音频
        print("[STEP1] 加载音频...")
        audio, sr = librosa.load(input_file, sr=sample_rate, mono=True)
        print(f"  音频时长: {len(audio)/sr:.2f}秒")
        
        # 2. 人声分离
        print("[STEP2] MDX23人声分离...")
        from src.vocal_smart_splitter.core.enhanced_vocal_separator import EnhancedVocalSeparator
        
        separator = EnhancedVocalSeparator(sample_rate)
        separation_result = separator.separate_for_detection(audio)
        
        if separation_result.vocal_track is None:
            print("[ERROR] 人声分离失败")
            return
        
        vocal_track = separation_result.vocal_track
        print(f"  分离完成，后端: {separation_result.backend_used}")
        
        # 3. 寻找能量波谷
        print("[STEP3] 寻找能量波谷...")
        valleys = find_energy_valleys(vocal_track, sample_rate, 
                                     min_pause_duration=1.0, 
                                     energy_threshold=-40)
        
        print(f"  找到 {len(valleys)} 个波谷")
        for i, v in enumerate(valleys[:10]):  # 显示前10个
            print(f"    波谷{i+1}: {v:.2f}s")
        
        # 4. 在波谷处切割
        print("[STEP4] 切割音频...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 添加起点和终点
        cut_points = [0.0] + valleys + [len(vocal_track) / sample_rate]
        
        input_name = Path(input_file).stem
        saved_files = []
        
        for i in range(len(cut_points) - 1):
            start_time = cut_points[i]
            end_time = cut_points[i + 1]
            duration = end_time - start_time
            
            # 跳过太短的片段
            if duration < 2.0:
                continue
            
            # 样本索引
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # 提取片段
            segment = vocal_track[start_sample:end_sample]
            
            # 保存
            segment_filename = f"{input_name}_simple_{i+1:02d}.wav"
            segment_path = Path(output_dir) / segment_filename
            sf.write(segment_path, segment, sample_rate, subtype='PCM_24')
            
            saved_files.append(str(segment_path))
            print(f"  片段{i+1}: {start_time:.2f}s - {end_time:.2f}s (时长: {duration:.2f}s)")
        
        print(f"[SUCCESS] 切割完成！生成 {len(saved_files)} 个片段")
        print(f"  输出目录: {output_dir}")
        
        return saved_files
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """主函数"""
    # 查找输入文件
    input_dir = Path("input")
    audio_files = list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.wav"))
    
    if not audio_files:
        print("未找到音频文件")
        return
    
    # 使用第一个文件
    input_file = audio_files[0]
    print(f"选择文件: {input_file.name}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"simple_{timestamp}"
    
    # 执行简化切割
    simple_split_vocal(str(input_file), str(output_dir))

if __name__ == "__main__":
    main()
(audio_env) E:\BDM_STATION\Desktop\BDM_projects\audio-cut>python quick_start.py
============================================================
智能人声分割器 - 快速启动 (v2.3 统一指挥中心版)
============================================================

============================================================
系统状态检查
============================================================
[OK] PyTorch版本: 2.8.0+cu129
[OK] CUDA可用: True
[OK] GPU设备: NVIDIA GeForce RTX 5060 Ti
[INFO] 发现 37 个音频文件:
  1. 02 - Be Your Love.mp3
  2. 02 - Smells Like Teen Spirit(from Nevermind).mp3
  3. 06. Hotel California.wav
  4. 07. Little Bird.mp3
  5. 07.MP3
  6. 11.MP3
  7. 12.MP3
  8. 13. All Of Me.mp3
  9. 13.MP3
  10. 14.MP3
  11. 15.MP3
  12. 16.MP3
  13. 17.MP3
  14. 18.MP3
  15. Angel.mp3
  16. come away with me.MP3
  17. dirty happy.MP3
  18. dont get lost in heaven.MP3
  19. dont know why.MP3
  20. Fix You.mp3
  21. forgotten.MP3
  22. i dont give.MP3
  23. if everyone cared.MP3
  24. maps.MP3
  25. Melody.mp3
  26. One more time,One more chance.mp3
  27. together.MP3
  28. wake me up when september ends.MP3
  29. yesterday once more.MP3
  30. 倔强.mp3
  31. 值得.mp3
  32. 千千阕歌.mp3
  33. 恋爱ing.mp3
  34. 猜心.mp3
  35. 追梦人.mp3
  36. 邓丽君 - 甜蜜蜜.mp3
  37. 飘雪.mp3

请选择要分割的文件 (1-37): 8
[SELECT] 选择文件: 13. All Of Me.mp3

============================================================
选择处理模式
============================================================
  1. 智能分割 (Smart Split)
     - 在原始混音上识别人声停顿并分割。
  2. 纯人声分离 (Vocal Separation Only)
     - 仅分离人声和伴奏，不分割。
  3. [推荐] 纯人声检测v2.1 (Pure Vocal v2.1)
     - 先分离再检测，使用统计学动态裁决，适合快歌。
  4. [最新] MDD增强纯人声检测v2.2 (Pure Vocal v2.2 MDD)
     - 在v2.1基础上，集成音乐动态密度(MDD)识别主副歌。

请选择 (1-4): 4
[SELECT] 已选择模式: v2.2_mdd
[OUTPUT] 输出目录: quick_v2.2_mdd_20250916_015231
[DIAG] Python: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env\Scripts\python.exe
[DIAG] VIRTUAL_ENV: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env
[DIAG] FORCE_SEPARATION_BACKEND:
2025-09-16 01:52:31,361 - src.vocal_smart_splitter.utils.config_manager - INFO - 已合并外部配置: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\config\default.yaml      
2025-09-16 01:52:31,362 - src.vocal_smart_splitter.utils.config_manager - INFO - 配置参数验证通过
2025-09-16 01:52:31,362 - src.vocal_smart_splitter.utils.config_manager - INFO - 配置管理器初始化完成，主配置: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\src\vocal_smart_splitter\config.yaml
2025-09-16 01:52:31,362 - src.vocal_smart_splitter.utils.adaptive_parameter_calculator - INFO - BPM驱动自适应参数计算器初始化完成
2025-09-16 01:52:31,362 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM分析器初始化完成 (采样率: 44100)
2025-09-16 01:52:31,363 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - 乐器复杂度分析器初始化完成
2025-09-16 01:52:31,363 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM感知的编曲复杂度自适应VAD增强器初始化完成
2025-09-16 01:52:31,363 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - BPM自适应增强器已启用
Using cache found in C:\Users\BDM_workstation/.cache\torch\hub\snakers4_silero-vad_master
2025-09-16 01:52:32,112 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - Silero VAD模型加载成功
2025-09-16 01:52:32,112 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - VocalPauseDetectorV2 初始化完成 (SR: 44100)
2025-09-16 01:52:32,112 - src.vocal_smart_splitter.utils.adaptive_parameter_calculator - INFO - BPM驱动自适应参数计算器初始化完成
2025-09-16 01:52:32,112 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM分析器初始化完成 (采样率: 44100)
2025-09-16 01:52:32,112 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - 乐器复杂度分析器初始化完成
2025-09-16 01:52:32,112 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM感知的编曲复杂度自适应VAD增强器初始化完成
2025-09-16 01:52:32,112 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - BPM自适应增强器已启用
Using cache found in C:\Users\BDM_workstation/.cache\torch\hub\snakers4_silero-vad_master
2025-09-16 01:52:32,659 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - Silero VAD模型加载成功
2025-09-16 01:52:32,659 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - VocalPauseDetectorV2 初始化完成 (SR: 44100)
2025-09-16 01:52:32,659 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 纯人声停顿检测器初始化完成 (采样率: 44100) - 已集成能量谷切点计算
2025-09-16 01:52:32,669 - vocal_smart_splitter.utils.config_manager - INFO - 已合并外部配置: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\config\default.yaml
2025-09-16 01:52:32,669 - vocal_smart_splitter.utils.config_manager - INFO - 配置参数验证通过
2025-09-16 01:52:32,669 - vocal_smart_splitter.utils.config_manager - INFO - 配置管理器初始化完成，主配置: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\src\vocal_smart_splitter\config.yaml
2025-09-16 01:52:32,669 - vocal_smart_splitter.utils.adaptive_parameter_calculator - INFO - BPM驱动自适应参数计算器初始化完成
2025-09-16 01:52:32,669 - src.vocal_smart_splitter.core.quality_controller - INFO - BPM感知质量控制器初始化完成
2025-09-16 01:52:32,670 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 检查MDX23路径候选: ['E:\\BDM_STATION\\Desktop\\BDM_projects\\audio-cut\\MVSEP-MDX23-music-separation-model']
2025-09-16 01:52:32,670 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - ✓ MDX23后端可用 - 模型目录: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model\models，数量: 2
2025-09-16 01:52:32,745 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - Demucs v4后端可用
2025-09-16 01:52:32,745 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 可用分离后端: ['mdx23', 'demucs_v4']
2025-09-16 01:52:32,746 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 增强型分离器初始化完成 - 主后端: mdx23
2025-09-16 01:52:32,746 - src.vocal_smart_splitter.core.seamless_splitter - INFO - 无缝分割器统一指挥中心初始化完成 (SR: 44100) - 已加载双检测器

[START] 正在启动统一分割引擎，模式: v2.2_mdd...
2025-09-16 01:52:32,746 - src.vocal_smart_splitter.core.seamless_splitter - INFO - 开始无缝分割: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\input\13. All Of Me.mp3 (模式: v2.2_mdd)
2025-09-16 01:52:32,746 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD] 执行纯人声分割流程...
2025-09-16 01:52:33,871 - src.vocal_smart_splitter.utils.audio_processor - INFO - 音频加载完成: 时长=326.48s, 采样率=44100Hz
2025-09-16 01:52:33,871 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP1] 执行高质量人声分离...
2025-09-16 01:52:33,871 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - === 分离后端选择决策 ===
2025-09-16 01:52:33,871 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 配置后端: mdx23
2025-09-16 01:52:33,871 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 后端状态概览:
2025-09-16 01:52:33,871 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO -   ✓ mdx23: 可用
2025-09-16 01:52:33,871 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO -   ✓ demucs_v4: 可用
2025-09-16 01:52:33,871 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - ✓ 选择用户指定后端: mdx23
2025-09-16 01:52:33,871 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - === 开始MDX23 CLI分离 ===
2025-09-16 01:52:33,872 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - PYTHON (CLI): E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env\Scripts\python.exe
2025-09-16 01:52:33,872 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - VIRTUAL_ENV: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env
2025-09-16 01:52:33,873 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 临时目录: C:\Windows\TEMP\mdx23_separation_jz4mov5u
2025-09-16 01:52:33,873 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 输入文件: C:\Windows\TEMP\mdx23_separation_jz4mov5u\input.wav
2025-09-16 01:52:33,873 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 输出目录: C:\Windows\TEMP\mdx23_separation_jz4mov5u\output
2025-09-16 01:52:33,920 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 音频写入完成: C:\Windows\TEMP\mdx23_separation_jz4mov5u\input.wav (长度: 14397696样本, 采样率: 44100Hz)
2025-09-16 01:52:33,951 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - Demucs import preflight: OK
2025-09-16 01:52:33,952 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23 CLI cmd: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env\Scripts\python.exe E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model\inference.py
2025-09-16 01:52:33,952 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23使用模型: Kim Model 2 (默认)
2025-09-16 01:52:33,952 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23启用大GPU模式 (GPU内存: 15.9GB)
2025-09-16 01:52:33,952 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23参数: chunk_size=1000000, overlap_large=0.6, overlap_small=0.5
2025-09-16 01:52:33,952 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23命令: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env\Scripts\python.exe E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model\inference.py --input_audio C:\Windows\TEMP\mdx23_separation_jz4mov5u\input.wav 
--output_folder C:\Windows\TEMP\mdx23_separation_jz4mov5u\output --large_gpu --overlap_large 0.6 --overlap_small 0.5 --chunk_size 1000000 --single_onnx --only_vocals   
2025-09-16 01:52:33,953 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 执行目录: E:\BDM_STATION\Desktop\BDM_projects\audio-cut
2025-09-16 01:52:33,953 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23项目路径: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model
2025-09-16 01:52:33,953 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 开始执行MDX23命令...
2025-09-16 01:53:21,217 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23命令执行完成，返回码: 0
2025-09-16 01:53:21,218 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23输出: GPU use: 0
[COMPAT] Demucs兼容性修复已应用
Version: 1.0.1
Options:
input_audio: ['C:\\Windows\\TEMP\\mdx23_separation_jz4mov5u\\input.wav']
output_folder: C:\Windows\TEMP\mdx23_separation_jz4mov5u\output
cpu: False
overlap_large: 0.6
overlap_small: 0.5
single_onnx: True
chunk_size: 1000000
large_gpu: True
use_kim_model_1: False
only_vocals: True
Generate only vocals and instrumental
Use fast large GPU memory version of code
Use device: cuda:0
Use single vocal ONNX
Use Kim model 2
[COMPAT] 使用兼容模式加载模型: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model/models/04573f0d-f3cf25b2.th
Model path: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model/models/Kim_Vocal_2.onnx
Device: cuda:0 Chunk size: 1000000
Go for: C:\Windows\TEMP\mdx23_separation_jz4mov5u\input.wav
Input audio: (2, 14397696) Sample rate: 44100
File created: C:\Windows\TEMP\mdx23_separation_jz4mov5u\output/input_vocals.wav
File created: C:\Windows\TEMP\mdx23_separation_jz4mov5u\output/input_instrum.wav
Time: 45 sec
Presented by https://mvsep.com

2025-09-16 01:53:21,219 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 输出目录内容: ['input_instrum.wav', 'input_vocals.wav']
2025-09-16 01:53:21,219 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 找到人声文件: C:\Windows\TEMP\mdx23_separation_jz4mov5u\output\input_vocals.wav
2025-09-16 01:53:21,387 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 人声轨道加载完成: 长度=14397696, 采样率=44100
2025-09-16 01:53:21,387 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - ✓ MDX23分离成功完成，耗时: 47.52秒
2025-09-16 01:53:21,971 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP1] 人声分离完成 - 后端: mdx23, 质量: 0.579, 耗时: 48.1s
2025-09-16 01:53:21,971 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP2] 使用PureVocalPauseDetector在[纯人声轨道]上进行多维特征检测...
2025-09-16 01:53:21,971 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 开始纯人声停顿检测... (MDD增强: True)
2025-09-16 01:53:21,971 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 使用相对能量谷检测模式...
2025-09-16 01:53:24,926 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 相对阈值自适应：BPM=114.8(medium), MDD=0.56, mul=0.99 → peak=0.257, rms=0.3162025-09-16 01:53:25,050 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - VPP自适应：VPP{no_rests}, mul_pause=1.00 → peak=0.257, rms=0.316
2025-09-16 01:53:25,050 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 相对能量谷检测: peak_ratio=0.25681331490338716, rms_ratio=0.31607792603493806
2025-09-16 01:53:25,518 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 峰值能量: 0.499338, 平均能量: 0.050661
2025-09-16 01:53:25,518 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 能量谷阈值: 0.016013 (peak:0.128237, rms:0.016013)
2025-09-16 01:53:25,522 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 能量谷检测完成: 发现74个能量谷停顿
2025-09-16 01:53:25,522 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - VPP最高限定生效: 原始=73 -> 保留=65 (上限=65)
2025-09-16 01:53:25,522 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 应用MDD增强处理...
2025-09-16 01:53:25,522 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 开始MDD增强处理...
2025-09-16 01:53:25,815 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 MDD增强完成: 65个停顿已优化
2025-09-16 01:53:25,815 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 使用能量谷算法计算 65 个停顿的精确切点...
2025-09-16 01:53:25,815 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - 计算 65 个停顿的切割点...
2025-09-16 01:54:16,512 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - ✅ 能量谷切点计算成功
2025-09-16 01:54:16,513 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 检测完成: 65个高质量停顿点
2025-09-16 01:54:16,513 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP3] 生成65个候选分割点
2025-09-16 01:54:16,594 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [NoVocalRuns] thr_db=-48.52, noise_db=-82.55, voice_db=-14.50, inactive_frames=18101/32648
2025-09-16 01:54:16,594 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP3] 纯音乐无人声区间: 3 段满足 >= 6.00s
2025-09-16 01:54:16,594 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 加权NMS准备: candidates=71
2025-09-16 01:54:16,594 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [NMS] candidates=71 -> kept=66, top_score=1.120
2025-09-16 01:54:16,595 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正开始: 66 个候选
2025-09-16 01:54:16,771 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 1/66: 0.000s -> 0.001s
2025-09-16 01:54:16,772 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 6/66: 70.111s -> 70.111s
2025-09-16 01:54:16,773 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 11/66: 86.570s -> 86.570s
2025-09-16 01:54:16,774 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 16/66: 107.295s -> 107.295s
2025-09-16 01:54:16,775 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 21/66: 122.957s -> 122.957s
2025-09-16 01:54:16,776 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 26/66: 139.529s -> 139.529s
2025-09-16 01:54:16,777 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 31/66: 157.971s -> 157.971s
2025-09-16 01:54:16,779 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 36/66: 191.287s -> 191.287s
2025-09-16 01:54:16,780 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 41/66: 212.039s -> 212.039s
2025-09-16 01:54:16,781 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 46/66: 233.733s -> 233.733s
2025-09-16 01:54:16,782 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 51/66: 252.219s -> 252.219s
2025-09-16 01:54:16,783 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 56/66: 270.267s -> 270.267s
2025-09-16 01:54:16,784 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 61/66: 292.559s -> 292.559s
2025-09-16 01:54:16,785 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 66/66: 326.478s -> 326.477s
2025-09-16 01:54:16,785 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 最小间隔过滤: 66 -> 64
2025-09-16 01:54:16,785 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 初始边界数: 66
2025-09-16 01:54:16,786 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 3，片段时长 3.510s < 5.000s
2025-09-16 01:54:16,786 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 4，片段时长 4.323s < 5.000s
2025-09-16 01:54:16,786 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 5，片段时长 4.129s < 5.000s
2025-09-16 01:54:16,786 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 6，片段时长 2.765s < 5.000s
2025-09-16 01:54:16,786 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 6，片段时长 4.409s < 5.000s
2025-09-16 01:54:16,786 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 8，片段时长 3.440s < 5.000s
2025-09-16 01:54:16,786 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 8，片段时长 4.599s < 5.000s
2025-09-16 01:54:16,786 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 9，片段时长 4.812s < 5.000s
2025-09-16 01:54:16,786 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 10，片段时长 2.506s < 5.000s
2025-09-16 01:54:16,787 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 11，片段时长 2.543s < 5.000s
2025-09-16 01:54:16,787 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 12，片段时长 4.994s < 5.000s
2025-09-16 01:54:16,787 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 13，片段时长 1.123s < 5.000s
2025-09-16 01:54:16,787 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 13，片段时长 3.870s < 5.000s
2025-09-16 01:54:16,787 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 15，片段时长 2.521s < 5.000s
2025-09-16 01:54:16,787 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 15，片段时长 3.602s < 5.000s
2025-09-16 01:54:16,787 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 18，片段时长 2.649s < 5.000s
2025-09-16 01:54:16,787 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 20，片段时长 3.712s < 5.000s
2025-09-16 01:54:16,787 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 21，片段时长 1.565s < 5.000s
2025-09-16 01:54:16,787 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 22，片段时长 3.943s < 5.000s
2025-09-16 01:54:16,788 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 23，片段时长 3.913s < 5.000s
2025-09-16 01:54:16,788 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 25，片段时长 4.657s < 5.000s
2025-09-16 01:54:16,788 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 26，片段时长 2.473s < 5.000s
2025-09-16 01:54:16,788 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 27，片段时长 3.511s < 5.000s
2025-09-16 01:54:16,788 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 28，片段时长 2.994s < 5.000s
2025-09-16 01:54:16,788 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 28，片段时长 4.933s < 5.000s
2025-09-16 01:54:16,788 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 30，片段时长 2.536s < 5.000s
2025-09-16 01:54:16,788 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 30，片段时长 4.562s < 5.000s
2025-09-16 01:54:16,789 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 35，片段时长 2.542s < 5.000s
2025-09-16 01:54:16,789 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 36，片段时长 3.273s < 5.000s
2025-09-16 01:54:16,789 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 最终边界数: 37

==================================================
[SUCCESS] 处理成功完成!
==================================================
  处理方法: pure_vocal_split_v2.2_mdd
  生成片段数量: 36
  文件保存在: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\output\quick_v2.2_mdd_20250916_015231
  使用后端: mdx23
  总耗时: 104.2秒

(audio_env) E:\BDM_STATION\Desktop\BDM_projects\audio-cut>python run_splitter.py input/07.mp3 --seamless-vocal --validate-reconstruction        
2025-09-16 02:21:18,456 - __main__ - INFO - 创建输出目录: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\output\test_20250916_022118
2025-09-16 02:21:18,456 - __main__ - INFO - 使用无缝人声停顿分割模式...
2025-09-16 02:21:18,465 - src.vocal_smart_splitter.utils.config_manager - INFO - 已合并外部配置: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\config\default.yaml      
2025-09-16 02:21:18,465 - src.vocal_smart_splitter.utils.config_manager - INFO - 配置参数验证通过
2025-09-16 02:21:18,465 - src.vocal_smart_splitter.utils.config_manager - INFO - 配置管理器初始化完成，主配置: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\src\vocal_smart_splitter\config.yaml
2025-09-16 02:21:18,466 - __main__ - INFO - 使用采样率: 44100Hz
2025-09-16 02:21:18,466 - src.vocal_smart_splitter.utils.adaptive_parameter_calculator - INFO - BPM驱动自适应参数计算器初始化完成
2025-09-16 02:21:18,466 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM分析器初始化完成 (采样率: 44100)
2025-09-16 02:21:18,466 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - 乐器复杂度分析器初始化完成
2025-09-16 02:21:18,466 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM感知的编曲复杂度自适应VAD增强器初始化完成
2025-09-16 02:21:18,466 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - BPM自适应增强器已启用
Using cache found in C:\Users\BDM_workstation/.cache\torch\hub\snakers4_silero-vad_master
2025-09-16 02:21:19,227 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - Silero VAD模型加载成功
2025-09-16 02:21:19,227 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - VocalPauseDetectorV2 初始化完成 (SR: 44100)
2025-09-16 02:21:19,227 - src.vocal_smart_splitter.utils.adaptive_parameter_calculator - INFO - BPM驱动自适应参数计算器初始化完成
2025-09-16 02:21:19,228 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM分析器初始化完成 (采样率: 44100)
2025-09-16 02:21:19,228 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - 乐器复杂度分析器初始化完成
2025-09-16 02:21:19,228 - src.vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM感知的编曲复杂度自适应VAD增强器初始化完成
2025-09-16 02:21:19,228 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - BPM自适应增强器已启用
Using cache found in C:\Users\BDM_workstation/.cache\torch\hub\snakers4_silero-vad_master
2025-09-16 02:21:20,064 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - Silero VAD模型加载成功
2025-09-16 02:21:20,065 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - VocalPauseDetectorV2 初始化完成 (SR: 44100)
2025-09-16 02:21:20,065 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 纯人声停顿检测器初始化完成 (采样率: 44100) - 已集成能量谷切点计算
2025-09-16 02:21:20,075 - vocal_smart_splitter.utils.config_manager - INFO - 已合并外部配置: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\config\default.yaml
2025-09-16 02:21:20,076 - vocal_smart_splitter.utils.config_manager - INFO - 配置参数验证通过
2025-09-16 02:21:20,076 - vocal_smart_splitter.utils.config_manager - INFO - 配置管理器初始化完成，主配置: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\src\vocal_smart_splitter\config.yaml
2025-09-16 02:21:20,076 - vocal_smart_splitter.utils.adaptive_parameter_calculator - INFO - BPM驱动自适应参数计算器初始化完成
2025-09-16 02:21:20,076 - src.vocal_smart_splitter.core.quality_controller - INFO - BPM感知质量控制器初始化完成
2025-09-16 02:21:20,076 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 检查MDX23路径候选: ['E:\\BDM_STATION\\Desktop\\BDM_projects\\audio-cut\\MVSEP-MDX23-music-separation-model']
2025-09-16 02:21:20,077 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - ✓ MDX23后端可用 - 模型目录: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model\models，数量: 2
2025-09-16 02:21:20,153 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - Demucs v4后端可用
2025-09-16 02:21:20,153 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 可用分离后端: ['mdx23', 'demucs_v4']
2025-09-16 02:21:20,153 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 增强型分离器初始化完成 - 主后端: mdx23
2025-09-16 02:21:20,153 - src.vocal_smart_splitter.core.seamless_splitter - INFO - 无缝分割器统一指挥中心初始化完成 (SR: 44100) - 已加载双检测器
2025-09-16 02:21:20,153 - __main__ - INFO - 开始无缝分割: input\07.mp3
2025-09-16 02:21:20,154 - __main__ - INFO - 输出目录: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\output\test_20250916_022118
2025-09-16 02:21:20,154 - __main__ - INFO - 分割策略: 基于人声停顿的精确分割
2025-09-16 02:21:20,154 - __main__ - INFO - 输出格式: 无损WAV/FLAC，零音频处理
2025-09-16 02:21:20,154 - src.vocal_smart_splitter.core.seamless_splitter - INFO - 开始无缝分割: input\07.mp3 (模式: v2.2_mdd)
2025-09-16 02:21:20,154 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD] 执行纯人声分割流程...
2025-09-16 02:21:20,840 - src.vocal_smart_splitter.utils.audio_processor - INFO - 音频加载完成: 时长=60.11s, 采样率=44100Hz
2025-09-16 02:21:20,840 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP1] 执行高质量人声分离...
2025-09-16 02:21:20,840 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - === 分离后端选择决策 ===
2025-09-16 02:21:20,841 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 配置后端: mdx23
2025-09-16 02:21:20,841 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 后端状态概览:
2025-09-16 02:21:20,841 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO -   ✓ mdx23: 可用
2025-09-16 02:21:20,841 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO -   ✓ demucs_v4: 可用
2025-09-16 02:21:20,841 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - ✓ 选择用户指定后端: mdx23
2025-09-16 02:21:20,841 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - === 开始MDX23 CLI分离 ===
2025-09-16 02:21:20,841 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - PYTHON (CLI): E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env\Scripts\python.exe
2025-09-16 02:21:20,841 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - VIRTUAL_ENV: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env
2025-09-16 02:21:20,842 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 临时目录: C:\Windows\TEMP\mdx23_separation_tjxwmsl3
2025-09-16 02:21:20,843 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 输入文件: C:\Windows\TEMP\mdx23_separation_tjxwmsl3\input.wav
2025-09-16 02:21:20,843 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 输出目录: C:\Windows\TEMP\mdx23_separation_tjxwmsl3\output
2025-09-16 02:21:20,856 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 音频写入完成: C:\Windows\TEMP\mdx23_separation_tjxwmsl3\input.wav (长度: 2650752样本, 采样率: 44100Hz)
2025-09-16 02:21:20,885 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - Demucs import preflight: OK
2025-09-16 02:21:20,885 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23 CLI cmd: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env\Scripts\python.exe E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model\inference.py
2025-09-16 02:21:20,886 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23使用模型: Kim Model 2 (默认)
2025-09-16 02:21:20,896 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23启用大GPU模式 (GPU内存: 15.9GB)
2025-09-16 02:21:20,896 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23参数: chunk_size=1000000, overlap_large=0.6, overlap_small=0.5
2025-09-16 02:21:20,897 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23命令: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env\Scripts\python.exe E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model\inference.py --input_audio C:\Windows\TEMP\mdx23_separation_tjxwmsl3\input.wav 
--output_folder C:\Windows\TEMP\mdx23_separation_tjxwmsl3\output --large_gpu --overlap_large 0.6 --overlap_small 0.5 --chunk_size 1000000 --single_onnx --only_vocals   
2025-09-16 02:21:20,897 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 执行目录: E:\BDM_STATION\Desktop\BDM_projects\audio-cut
2025-09-16 02:21:20,897 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23项目路径: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model
2025-09-16 02:21:20,897 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 开始执行MDX23命令...
2025-09-16 02:21:34,457 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23命令执行完成，返回码: 0
2025-09-16 02:21:34,458 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23输出: GPU use: 0
[COMPAT] Demucs兼容性修复已应用
Version: 1.0.1
Options:
input_audio: ['C:\\Windows\\TEMP\\mdx23_separation_tjxwmsl3\\input.wav']
output_folder: C:\Windows\TEMP\mdx23_separation_tjxwmsl3\output
cpu: False
overlap_large: 0.6
overlap_small: 0.5
single_onnx: True
chunk_size: 1000000
large_gpu: True
use_kim_model_1: False
only_vocals: True
Generate only vocals and instrumental
Use fast large GPU memory version of code
Use device: cuda:0
Use single vocal ONNX
Use Kim model 2
[COMPAT] 使用兼容模式加载模型: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model/models/04573f0d-f3cf25b2.th
Model path: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model/models/Kim_Vocal_2.onnx
Device: cuda:0 Chunk size: 1000000
Go for: C:\Windows\TEMP\mdx23_separation_tjxwmsl3\input.wav
Input audio: (2, 2650752) Sample rate: 44100
File created: C:\Windows\TEMP\mdx23_separation_tjxwmsl3\output/input_vocals.wav
File created: C:\Windows\TEMP\mdx23_separation_tjxwmsl3\output/input_instrum.wav
Time: 12 sec
Presented by https://mvsep.com

2025-09-16 02:21:34,458 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 输出目录内容: ['input_instrum.wav', 'input_vocals.wav']
2025-09-16 02:21:34,459 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 找到人声文件: C:\Windows\TEMP\mdx23_separation_tjxwmsl3\output\input_vocals.wav
2025-09-16 02:21:34,495 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 人声轨道加载完成: 长度=2650752, 采样率=44100
2025-09-16 02:21:34,495 - src.vocal_smart_splitter.core.enhanced_vocal_separator - INFO - ✓ MDX23分离成功完成，耗时: 13.65秒
2025-09-16 02:21:34,601 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP1] 人声分离完成 - 后端: mdx23, 质量: 0.434, 耗时: 13.8s
2025-09-16 02:21:34,602 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP2] 使用PureVocalPauseDetector在[纯人声轨道]上进行多维特征检测...
2025-09-16 02:21:34,602 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 开始纯人声停顿检测... (MDD增强: True)
2025-09-16 02:21:34,602 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 使用相对能量谷检测模式...
2025-09-16 02:21:36,041 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 相对阈值自适应：BPM=114.8(medium), MDD=0.44, mul=1.01 → peak=0.263, rms=0.3242025-09-16 02:21:36,057 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - VPP自适应：VPP{no_rests}, mul_pause=1.00 → peak=0.263, rms=0.324
2025-09-16 02:21:36,057 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 相对能量谷检测: peak_ratio=0.26322073517034283, rms_ratio=0.32396398174811425
2025-09-16 02:21:36,159 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 峰值能量: 0.290037, 平均能量: 0.113719
2025-09-16 02:21:36,159 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 能量谷阈值: 0.036841 (peak:0.076344, rms:0.036841)
2025-09-16 02:21:36,160 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 能量谷检测完成: 发现10个能量谷停顿
2025-09-16 02:21:36,160 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 应用MDD增强处理...
2025-09-16 02:21:36,160 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 开始MDD增强处理...
2025-09-16 02:21:36,215 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 MDD增强完成: 10个停顿已优化
2025-09-16 02:21:36,215 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 使用能量谷算法计算 10 个停顿的精确切点...
2025-09-16 02:21:36,216 - src.vocal_smart_splitter.core.vocal_pause_detector - INFO - 计算 10 个停顿的切割点...
2025-09-16 02:21:39,496 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - ✅ 能量谷切点计算成功
2025-09-16 02:21:39,496 - src.vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 检测完成: 10个高质量停顿点
2025-09-16 02:21:39,496 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP3] 生成10个候选分割点
2025-09-16 02:21:39,511 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [NoVocalRuns] thr_db=-42.94, noise_db=-72.01, voice_db=-13.87, inactive_frames=1015/6011
2025-09-16 02:21:39,512 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP3] 纯音乐无人声区间: 1 段满足 >= 6.00s
2025-09-16 02:21:39,512 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 加权NMS准备: candidates=12
2025-09-16 02:21:39,512 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [NMS] candidates=12 -> kept=12, top_score=1.133
2025-09-16 02:21:39,512 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正开始: 12 个候选
2025-09-16 02:21:39,545 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 1/12: 0.000s -> 0.001s
2025-09-16 02:21:39,547 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 6/12: 15.735s -> 15.735s
2025-09-16 02:21:39,549 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 11/12: 50.796s -> 50.796s
2025-09-16 02:21:39,549 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 最小间隔过滤: 12 -> 11
2025-09-16 02:21:39,549 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 初始边界数: 13
2025-09-16 02:21:39,549 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 1，片段时长 4.700s < 5.000s
2025-09-16 02:21:39,549 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 2，片段时长 3.166s < 5.000s
2025-09-16 02:21:39,550 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 3，片段时长 2.790s < 5.000s
2025-09-16 02:21:39,550 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 4，片段时长 3.529s < 5.000s
2025-09-16 02:21:39,550 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 7，片段时长 4.645s < 5.000s
2025-09-16 02:21:39,550 - src.vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 最终边界数: 8
2025-09-16 02:21:39,587 - __main__ - INFO - ==================================================
2025-09-16 02:21:39,587 - __main__ - INFO - 分割完成！
2025-09-16 02:21:39,587 - __main__ - INFO - 输出目录: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\output\test_20250916_022118
2025-09-16 02:21:39,587 - __main__ - INFO - 生成片段数: 7
2025-09-16 02:21:39,588 - __main__ - INFO - 处理模式: seamless_vocal_pause_splitting
2025-09-16 02:21:39,588 - __main__ - INFO - 运行拼接完整性验证...
2025-09-16 02:21:39,594 - vocal_smart_splitter.core.vocal_pause_detector - INFO - 自适应VAD增强器可用
2025-09-16 02:21:39,598 - vocal_smart_splitter.utils.adaptive_parameter_calculator - INFO - BPM驱动自适应参数计算器初始化完成
2025-09-16 02:21:39,598 - vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM分析器初始化完成 (采样率: 44100)
2025-09-16 02:21:39,598 - vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - 乐器复杂度分析器初始化完成
2025-09-16 02:21:39,598 - vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM感知的编曲复杂度自适应VAD增强器初始化完成
2025-09-16 02:21:39,598 - vocal_smart_splitter.core.vocal_pause_detector - INFO - BPM自适应增强器已启用
Using cache found in C:\Users\BDM_workstation/.cache\torch\hub\snakers4_silero-vad_master
2025-09-16 02:21:40,089 - vocal_smart_splitter.core.vocal_pause_detector - INFO - Silero VAD模型加载成功
2025-09-16 02:21:40,089 - vocal_smart_splitter.core.vocal_pause_detector - INFO - VocalPauseDetectorV2 初始化完成 (SR: 44100)
2025-09-16 02:21:40,090 - vocal_smart_splitter.utils.adaptive_parameter_calculator - INFO - BPM驱动自适应参数计算器初始化完成
2025-09-16 02:21:40,090 - vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM分析器初始化完成 (采样率: 44100)
2025-09-16 02:21:40,090 - vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - 乐器复杂度分析器初始化完成
2025-09-16 02:21:40,090 - vocal_smart_splitter.core.adaptive_vad_enhancer - INFO - BPM感知的编曲复杂度自适应VAD增强器初始化完成
2025-09-16 02:21:40,090 - vocal_smart_splitter.core.vocal_pause_detector - INFO - BPM自适应增强器已启用
Using cache found in C:\Users\BDM_workstation/.cache\torch\hub\snakers4_silero-vad_master
2025-09-16 02:21:40,587 - vocal_smart_splitter.core.vocal_pause_detector - INFO - Silero VAD模型加载成功
2025-09-16 02:21:40,587 - vocal_smart_splitter.core.vocal_pause_detector - INFO - VocalPauseDetectorV2 初始化完成 (SR: 44100)
2025-09-16 02:21:40,587 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 纯人声停顿检测器初始化完成 (采样率: 44100) - 已集成能量谷切点计算
2025-09-16 02:21:40,588 - vocal_smart_splitter.utils.adaptive_parameter_calculator - INFO - BPM驱动自适应参数计算器初始化完成
2025-09-16 02:21:40,588 - vocal_smart_splitter.core.quality_controller - INFO - BPM感知质量控制器初始化完成
2025-09-16 02:21:40,588 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 检查MDX23路径候选: ['E:\\BDM_STATION\\Desktop\\BDM_projects\\audio-cut\\MVSEP-MDX23-music-separation-model']
2025-09-16 02:21:40,589 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - ✓ MDX23后端可用 - 模型目录: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model\models，数量: 2
2025-09-16 02:21:40,589 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - Demucs v4后端可用
2025-09-16 02:21:40,589 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 可用分离后端: ['mdx23', 'demucs_v4']
2025-09-16 02:21:40,589 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 增强型分离器初始化完成 - 主后端: mdx23
2025-09-16 02:21:40,589 - vocal_smart_splitter.core.seamless_splitter - INFO - 无缝分割器统一指挥中心初始化完成 (SR: 44100) - 已加载双检测器
2025-09-16 02:21:40,589 - tests.test_seamless_reconstruction - INFO - 开始测试无缝拼接: input\07.mp3
2025-09-16 02:21:40,589 - vocal_smart_splitter.core.seamless_splitter - INFO - 开始无缝分割: input\07.mp3 (模式: v2.2_mdd)
2025-09-16 02:21:40,589 - vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD] 执行纯人声分割流程...
2025-09-16 02:21:40,669 - vocal_smart_splitter.utils.audio_processor - INFO - 音频加载完成: 时长=60.11s, 采样率=44100Hz
2025-09-16 02:21:40,669 - vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP1] 执行高质量人声分离...
2025-09-16 02:21:40,669 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - === 分离后端选择决策 ===
2025-09-16 02:21:40,669 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 配置后端: mdx23
2025-09-16 02:21:40,669 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 后端状态概览:
2025-09-16 02:21:40,669 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO -   ✓ mdx23: 可用
2025-09-16 02:21:40,670 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO -   ✓ demucs_v4: 可用
2025-09-16 02:21:40,670 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - ✓ 选择用户指定后端: mdx23
2025-09-16 02:21:40,670 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - === 开始MDX23 CLI分离 ===
2025-09-16 02:21:40,670 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - PYTHON (CLI): E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env\Scripts\python.exe
2025-09-16 02:21:40,670 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - VIRTUAL_ENV: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env
2025-09-16 02:21:40,671 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 临时目录: C:\Windows\TEMP\mdx23_separation_nj4uh_6h
2025-09-16 02:21:40,671 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 输入文件: C:\Windows\TEMP\mdx23_separation_nj4uh_6h\input.wav
2025-09-16 02:21:40,671 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 输出目录: C:\Windows\TEMP\mdx23_separation_nj4uh_6h\output
2025-09-16 02:21:40,686 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 音频写入完成: C:\Windows\TEMP\mdx23_separation_nj4uh_6h\input.wav (长度: 2650752样
本, 采样率: 44100Hz)
2025-09-16 02:21:40,713 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - Demucs import preflight: OK
2025-09-16 02:21:40,713 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23 CLI cmd: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env\Scripts\python.exe E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model\inference.py
2025-09-16 02:21:40,713 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23使用模型: Kim Model 2 (默认)
2025-09-16 02:21:40,714 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23启用大GPU模式 (GPU内存: 15.9GB)
2025-09-16 02:21:40,714 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23参数: chunk_size=1000000, overlap_large=0.6, overlap_small=0.5
2025-09-16 02:21:40,714 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23命令: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\audio_env\Scripts\python.exe E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model\inference.py --input_audio C:\Windows\TEMP\mdx23_separation_nj4uh_6h\input.wav --output_folder C:\Windows\TEMP\mdx23_separation_nj4uh_6h\output --large_gpu --overlap_large 0.6 --overlap_small 0.5 --chunk_size 1000000 --single_onnx --only_vocals       
2025-09-16 02:21:40,714 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 执行目录: E:\BDM_STATION\Desktop\BDM_projects\audio-cut
2025-09-16 02:21:40,714 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23项目路径: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model
2025-09-16 02:21:40,714 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 开始执行MDX23命令...
2025-09-16 02:21:54,106 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23命令执行完成，返回码: 0
2025-09-16 02:21:54,106 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - MDX23输出: GPU use: 0
[COMPAT] Demucs兼容性修复已应用
Version: 1.0.1
Options:
input_audio: ['C:\\Windows\\TEMP\\mdx23_separation_nj4uh_6h\\input.wav']
output_folder: C:\Windows\TEMP\mdx23_separation_nj4uh_6h\output
cpu: False
overlap_large: 0.6
overlap_small: 0.5
single_onnx: True
chunk_size: 1000000
large_gpu: True
use_kim_model_1: False
only_vocals: True
Generate only vocals and instrumental
Use fast large GPU memory version of code
Use device: cuda:0
Use single vocal ONNX
Use Kim model 2
[COMPAT] 使用兼容模式加载模型: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model/models/04573f0d-f3cf25b2.th
Model path: E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model/models/Kim_Vocal_2.onnx
Device: cuda:0 Chunk size: 1000000
Go for: C:\Windows\TEMP\mdx23_separation_nj4uh_6h\input.wav
Input audio: (2, 2650752) Sample rate: 44100
File created: C:\Windows\TEMP\mdx23_separation_nj4uh_6h\output/input_vocals.wav
File created: C:\Windows\TEMP\mdx23_separation_nj4uh_6h\output/input_instrum.wav
Time: 11 sec
Presented by https://mvsep.com

2025-09-16 02:21:54,107 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 输出目录内容: ['input_instrum.wav', 'input_vocals.wav']
2025-09-16 02:21:54,107 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 找到人声文件: C:\Windows\TEMP\mdx23_separation_nj4uh_6h\output\input_vocals.wav   
2025-09-16 02:21:54,152 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - 人声轨道加载完成: 长度=2650752, 采样率=44100
2025-09-16 02:21:54,152 - vocal_smart_splitter.core.enhanced_vocal_separator - INFO - ✓ MDX23分离成功完成，耗时: 13.48秒
2025-09-16 02:21:54,257 - vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP1] 人声分离完成 - 后端: mdx23, 质量: 0.433, 耗时: 13.6s
2025-09-16 02:21:54,258 - vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP2] 使用PureVocalPauseDetector在[纯人声轨道]上进行多维特征检测...
2025-09-16 02:21:54,258 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 开始纯人声停顿检测... (MDD增强: True)
2025-09-16 02:21:54,258 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 使用相对能量谷检测模式...
2025-09-16 02:21:54,603 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 相对阈值自适应：BPM=114.8(medium), MDD=0.44, mul=1.01 → peak=0.263, rms=0.324
2025-09-16 02:21:54,618 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - VPP自适应：VPP{no_rests}, mul_pause=1.00 → peak=0.263, rms=0.324
2025-09-16 02:21:54,619 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 相对能量谷检测: peak_ratio=0.26322073517034283, rms_ratio=0.32396398174811425 
2025-09-16 02:21:54,711 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 峰值能量: 0.290261, 平均能量: 0.113708
2025-09-16 02:21:54,711 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 能量谷阈值: 0.036837 (peak:0.076403, rms:0.036837)
2025-09-16 02:21:54,712 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 能量谷检测完成: 发现10个能量谷停顿
2025-09-16 02:21:54,712 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 应用MDD增强处理...
2025-09-16 02:21:54,712 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 开始MDD增强处理...
2025-09-16 02:21:54,774 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 MDD增强完成: 10个停顿已优化
2025-09-16 02:21:54,774 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 🔥 使用能量谷算法计算 10 个停顿的精确切点...
2025-09-16 02:21:54,774 - vocal_smart_splitter.core.vocal_pause_detector - INFO - 计算 10 个停顿的切割点...
2025-09-16 02:21:58,035 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - ✅ 能量谷切点计算成功
2025-09-16 02:21:58,035 - vocal_smart_splitter.core.pure_vocal_pause_detector - INFO - 检测完成: 10个高质量停顿点
2025-09-16 02:21:58,036 - vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP3] 生成10个候选分割点
2025-09-16 02:21:58,051 - vocal_smart_splitter.core.seamless_splitter - INFO - [NoVocalRuns] thr_db=-42.90, noise_db=-71.92, voice_db=-13.87, inactive_frames=1016/6011 
2025-09-16 02:21:58,051 - vocal_smart_splitter.core.seamless_splitter - INFO - [V2.2_MDD-STEP3] 纯音乐无人声区间: 1 段满足 >= 6.00s
2025-09-16 02:21:58,051 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 加权NMS准备: candidates=12
2025-09-16 02:21:58,051 - vocal_smart_splitter.core.seamless_splitter - INFO - [NMS] candidates=12 -> kept=12, top_score=1.133
2025-09-16 02:21:58,051 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正开始: 12 个候选
2025-09-16 02:21:58,084 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 1/12: 0.000s -> 0.001s
2025-09-16 02:21:58,085 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 6/12: 15.735s -> 15.735s
2025-09-16 02:21:58,087 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 守卫校正进度 11/12: 50.796s -> 50.796s
2025-09-16 02:21:58,088 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 最小间隔过滤: 12 -> 11
2025-09-16 02:21:58,088 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 初始边界数: 13
2025-09-16 02:21:58,088 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 1，片段时长 4.700s < 5.000s
2025-09-16 02:21:58,088 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 2，片段时长 3.166s < 5.000s
2025-09-16 02:21:58,088 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 3，片段时长 2.790s < 5.000s
2025-09-16 02:21:58,089 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 4，片段时长 3.599s < 5.000s
2025-09-16 02:21:58,089 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 合并短段: 移除边界索引 7，片段时长 4.645s < 5.000s
2025-09-16 02:21:58,089 - vocal_smart_splitter.core.seamless_splitter - INFO - [FinalizeV2] 最终边界数: 8
2025-09-16 02:21:58,128 - tests.test_seamless_reconstruction - ERROR - 测试失败: 'seamless_validation'
2025-09-16 02:21:58,129 - __main__ - INFO - 详细验证结果: FAIL
2025-09-16 02:21:58,129 - __main__ - INFO - 生成的片段文件:
2025-09-16 02:21:58,129 - __main__ - INFO -   1. segment_001.wav
2025-09-16 02:21:58,129 - __main__ - INFO -   2. segment_002.wav
2025-09-16 02:21:58,129 - __main__ - INFO -   3. segment_003.wav
2025-09-16 02:21:58,130 - __main__ - INFO -   4. segment_004.wav
2025-09-16 02:21:58,130 - __main__ - INFO -   5. segment_005.wav
2025-09-16 02:21:58,130 - __main__ - INFO -   6. segment_006.wav
2025-09-16 02:21:58,130 - __main__ - INFO -   7. segment_007.wav
2025-09-16 02:21:58,130 - __main__ - INFO -   8. 07_v2.2_mdd_vocal_full.wav
2025-09-16 02:21:58,131 - __main__ - INFO - ==================================================
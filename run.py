#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run.py - 智能人声分割器统一运行脚本
# AI-SUMMARY: 高质量统一项目入口，提供完整的命令行界面和功能选择

"""
智能人声分割器统一运行脚本 v1.1.4

这是智能人声分割器的统一入口点，提供所有功能的便捷访问。

主要功能：
1. 🎵 音频分割 - 基于BPM自适应的无缝人声分割
2. 🧪 系统测试 - 运行质量验证测试
3. ⚙️ 环境检查 - 验证系统依赖和配置
4. 📊 项目状态 - 显示系统状态和信息

使用方法：
    python run.py split input/audio.mp3              # 快速分割
    python run.py test                               # 运行测试
    python run.py status                            # 系统状态
    python run.py --help                            # 显示帮助
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging(verbose=False):
    """设置日志系统"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_environment():
    """检查运行环境"""
    logger = logging.getLogger(__name__)
    
    logger.info("环境检查中...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version < (3, 8):
        logger.error(f"Python版本过低: {python_version.major}.{python_version.minor}, 需要 >= 3.8")
        return False
    logger.info(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查核心依赖
    required_packages = ['numpy', 'librosa', 'torch', 'torchaudio']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"依赖包 {package}: 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"依赖包 {package}: 未安装")
    
    if missing_packages:
        logger.error("缺少依赖包，请运行: pip install -r requirements.txt")
        return False
    
    # 检查项目结构
    required_dirs = ['src/vocal_smart_splitter', 'tests', 'input']
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)
            logger.error(f"目录缺失: {dir_path}")
    
    if missing_dirs:
        logger.error("项目结构不完整")
        return False
    
    logger.info("环境检查通过!")
    return True

def run_audio_splitting(input_file, output_dir=None, **kwargs):
    """运行音频分割"""
    logger = logging.getLogger(__name__)
    
    # 验证输入文件
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        return False
    
    # 创建输出目录
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "output" / f"split_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"输入文件: {input_path}")
    logger.info(f"输出目录: {output_path}")
    
    try:
        # 导入并运行分割器
        from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
        
        # 从配置获取采样率
        try:
            from src.vocal_smart_splitter.utils.config_manager import get_config
            sample_rate = get_config('audio.sample_rate', 44100)
        except:
            sample_rate = 44100
            logger.warning("无法读取配置，使用默认采样率: 44100")
        
        logger.info(f"使用采样率: {sample_rate}Hz")
        logger.info("使用无缝BPM自适应分割模式...")
        
        splitter = SeamlessSplitter(sample_rate=sample_rate)
        
        # 执行分割
        logger.info("开始音频分割...")
        result = splitter.split_audio_seamlessly(str(input_path), str(output_path))
        
        if result.get('success', False):
            logger.info("=" * 50)
            logger.info("分割成功完成!")
            logger.info(f"生成片段数: {result.get('num_segments', 0)}")
            
            # 显示分割文件
            saved_files = result.get('saved_files', [])
            if saved_files:
                logger.info("生成的分割文件:")
                for i, file_path in enumerate(saved_files, 1):
                    file_name = Path(file_path).name
                    logger.info(f"  {i}. {file_name}")
            
            # 显示质量信息
            if 'vocal_pause_analysis' in result:
                pause_info = result['vocal_pause_analysis']
                logger.info(f"检测到停顿: {pause_info.get('total_pauses', 0)} 个")
                logger.info(f"平均置信度: {pause_info.get('avg_confidence', 0):.3f}")
            
            # 显示重构验证
            if 'seamless_validation' in result:
                validation = result['seamless_validation']
                perfect = validation.get('perfect_reconstruction', False)
                logger.info(f"重构验证: {'完美' if perfect else '有差异'}")
                if 'max_difference' in validation:
                    logger.info(f"最大差异: {validation['max_difference']:.2e}")
            
            logger.info("=" * 50)
            return True
        else:
            logger.error("分割失败")
            if 'error' in result:
                logger.error(f"错误信息: {result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"分割过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_tests():
    """运行系统测试"""
    logger = logging.getLogger(__name__)
    
    logger.info("运行系统质量验证测试...")
    
    try:
        test_script = project_root / "tests" / "run_tests.py"
        if not test_script.exists():
            logger.error("测试脚本不存在")
            return False
        
        result = subprocess.run([
            sys.executable, str(test_script)
        ], capture_output=True, text=True, timeout=300)
        
        # 输出测试结果
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode == 0:
            logger.info("所有测试通过!")
            return True
        else:
            logger.error("部分测试失败")
            return False
            
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        return False

def show_status():
    """显示项目状态"""
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("智能人声分割器 v1.1.4 - 系统状态")
    print("=" * 60)
    
    # 项目信息
    print("[项目信息]")
    print(f"  项目路径: {project_root}")
    print(f"  配置文件: {project_root / 'src/vocal_smart_splitter/config.yaml'}")
    print(f"  Python版本: {sys.version.split()[0]}")
    
    # 核心组件状态
    print("\n[核心组件]")
    try:
        from src.vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
        print("  [OK] 无缝分割器: 已加载")
    except ImportError as e:
        print(f"  [FAIL] 无缝分割器: 加载失败 ({e})")
    
    try:
        from src.vocal_smart_splitter.core.dual_path_detector import DualPathVocalDetector
        print("  [OK] 双路检测器: 已加载")
    except ImportError as e:
        print(f"  [FAIL] 双路检测器: 加载失败 ({e})")
    
    try:
        from src.vocal_smart_splitter.utils.config_manager import get_config
        sample_rate = get_config('audio.sample_rate', 'N/A')
        print(f"  [OK] 配置管理器: 已加载 (采样率: {sample_rate})")
    except Exception as e:
        print(f"  [FAIL] 配置管理器: 加载失败 ({e})")
    
    # 测试状态
    print("\n[测试状态]")
    test_dir = project_root / "tests"
    if test_dir.exists():
        test_files = list(test_dir.glob("test_*.py"))
        print(f"  测试文件数: {len(test_files)}")
        for test_file in test_files:
            print(f"    - {test_file.name}")
    else:
        print("  测试目录不存在")
    
    # 输入输出目录
    print("\n[目录状态]")
    input_dir = project_root / "input"
    output_dir = project_root / "output"
    
    if input_dir.exists():
        input_files = list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.wav"))
        print(f"  输入目录: {len(input_files)} 个音频文件")
    else:
        print("  输入目录: 不存在")
    
    if output_dir.exists():
        output_subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        print(f"  输出目录: {len(output_subdirs)} 个输出文件夹")
    else:
        print("  输出目录: 不存在")
    
    print("=" * 60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="智能人声分割器统一运行脚本 v1.1.4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run.py split input/song.mp3           # 分割音频文件
  python run.py split input/song.mp3 -o output/custom/  # 指定输出目录
  python run.py test                           # 运行质量测试
  python run.py status                         # 显示系统状态
  python run.py check                          # 检查环境配置

支持的音频格式: MP3, WAV, FLAC, M4A
        """
    )
    
    parser.add_argument(
        'command',
        choices=['split', 'test', 'status', 'check'],
        help='命令类型: split=音频分割, test=运行测试, status=系统状态, check=环境检查'
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        help='输入音频文件路径 (split命令必需)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出目录路径'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细日志'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='运行重构验证 (仅split命令)'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.verbose)
    
    # 根据命令执行对应功能
    if args.command == 'check':
        success = check_environment()
        sys.exit(0 if success else 1)
        
    elif args.command == 'status':
        show_status()
        sys.exit(0)
        
    elif args.command == 'test':
        success = run_tests()
        sys.exit(0 if success else 1)
        
    elif args.command == 'split':
        if not args.input_file:
            logger.error("split命令需要指定输入文件")
            parser.print_help()
            sys.exit(1)
        
        success = run_audio_splitting(
            args.input_file,
            args.output,
            validate=args.validate
        )
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
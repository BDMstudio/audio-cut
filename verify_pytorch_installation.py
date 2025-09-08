#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch 2.8.0 安装验证和CUDA兼容性检查脚本
"""

import sys
import torch
import time

def print_separator(title):
    """打印分隔符"""
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_pytorch_version():
    """检查PyTorch版本信息"""
    print_separator("PyTorch版本信息")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 检查是否在虚拟环境中
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"虚拟环境: {'是' if in_venv else '否'}")
    
    if in_venv and 'audio_env' in sys.executable:
        print("✅ 正在使用audio_env虚拟环境")
    else:
        print("⚠️  未使用预期的虚拟环境")

def check_cuda_support():
    """检查CUDA支持"""
    print_separator("CUDA支持检查")
    
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA支持已启用")
        print(f"PyTorch编译的CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  总内存: {props.total_memory / 1024**3:.1f}GB")
            print(f"  计算能力: {props.major}.{props.minor}")
            print(f"  多处理器数量: {props.multi_processor_count}")
        
        # 检查支持的架构
        if hasattr(torch.cuda, 'get_arch_list'):
            archs = torch.cuda.get_arch_list()
            print(f"支持的CUDA架构: {archs}")
            
            # 检查RTX 5060 Ti的sm_120架构
            if 'sm_120' in archs:
                print("✅ 支持RTX 5060 Ti (sm_120架构)")
            else:
                print("⚠️  可能不完全支持RTX 5060 Ti")
    else:
        print("❌ CUDA支持未启用")
        return False
    
    return True

def test_gpu_computation():
    """测试GPU计算功能"""
    print_separator("GPU计算测试")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过GPU测试")
        return False
    
    try:
        # 创建测试张量
        print("创建测试张量...")
        device = torch.device('cuda:0')
        
        # 测试矩阵运算
        size = 1000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        print(f"执行 {size}x{size} 矩阵乘法...")
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # 等待GPU计算完成
        end_time = time.time()
        
        print(f"✅ GPU矩阵运算成功")
        print(f"计算时间: {(end_time - start_time)*1000:.2f}ms")
        print(f"结果形状: {c.shape}")
        print(f"结果数据类型: {c.dtype}")
        
        # 检查内存使用
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"GPU内存使用: {memory_allocated:.1f}MB (已分配) / {memory_reserved:.1f}MB (已预留)")
        
        # 清理内存
        del a, b, c
        torch.cuda.empty_cache()
        print("GPU内存已清理")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU计算测试失败: {e}")
        return False

def check_version_compatibility():
    """检查版本兼容性"""
    print_separator("版本兼容性检查")
    
    # 检查PyTorch版本
    pytorch_version = torch.__version__
    if pytorch_version.startswith('2.8.0'):
        print("✅ PyTorch版本正确: 2.8.0")
    else:
        print(f"⚠️  PyTorch版本不匹配: {pytorch_version} (期望: 2.8.0)")
    
    # 检查CUDA版本
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        if cuda_version == '12.9':
            print("✅ CUDA版本完全匹配: 12.9")
        elif cuda_version and cuda_version.startswith('12.'):
            print(f"✅ CUDA版本兼容: {cuda_version} (支持12.9)")
        else:
            print(f"⚠️  CUDA版本可能不兼容: {cuda_version}")
    
    # 检查cuDNN
    if torch.backends.cudnn.is_available():
        print("✅ cuDNN可用")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    else:
        print("⚠️  cuDNN不可用")

def main():
    """主函数"""
    print("🔍 PyTorch 2.8.0 安装验证和CUDA兼容性检查")
    print(f"检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 检查PyTorch版本
    check_pytorch_version()
    print()
    
    # 2. 检查CUDA支持
    cuda_available = check_cuda_support()
    print()
    
    # 3. 测试GPU计算
    if cuda_available:
        gpu_test_passed = test_gpu_computation()
        print()
    else:
        gpu_test_passed = False
    
    # 4. 检查版本兼容性
    check_version_compatibility()
    print()
    
    # 5. 总结
    print_separator("总结")
    
    if torch.__version__.startswith('2.8.0') and cuda_available and gpu_test_passed:
        print("🎉 完美！PyTorch 2.8.0+cu129安装成功，GPU加速完全可用")
        print("✅ 版本兼容性: 完全兼容")
        print("✅ GPU支持: 完全支持")
        print("✅ 计算测试: 全部通过")
        print()
        print("🚀 可以开始使用GPU加速的深度学习功能！")
    else:
        print("⚠️  安装存在问题，请检查以上输出")
        
        if not torch.__version__.startswith('2.8.0'):
            print("- PyTorch版本不正确")
        if not cuda_available:
            print("- CUDA支持不可用")
        if not gpu_test_passed:
            print("- GPU计算测试失败")

if __name__ == "__main__":
    main()

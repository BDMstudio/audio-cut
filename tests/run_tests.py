#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/run_tests.py
# AI-SUMMARY: 统一测试运行脚本

"""
测试运行器
统一运行所有测试用例
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_test(test_file):
    """运行单个测试"""
    logger = logging.getLogger(__name__)
    logger.info(f"运行测试: {test_file}")
    
    try:
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info(f"[PASS] {test_file} - 测试通过")
            return True
        else:
            logger.error(f"[FAIL] {test_file} - 测试失败")
            logger.error(f"错误输出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"[TIMEOUT] {test_file} - 测试超时")
        return False
    except Exception as e:
        logger.error(f"[ERROR] {test_file} - 测试异常: {e}")
        return False

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 获取测试目录
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))
    
    if not test_files:
        logger.warning("未找到测试文件")
        return
    
    logger.info(f"找到 {len(test_files)} 个测试文件")
    
    # 运行所有测试
    passed = 0
    failed = 0
    
    for test_file in test_files:
        if run_test(test_file):
            passed += 1
        else:
            failed += 1
    
    # 输出结果
    logger.info("=" * 50)
    logger.info(f"测试完成: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        logger.info("所有测试通过!")
    else:
        logger.warning(f"有 {failed} 个测试失败")

if __name__ == "__main__":
    main()
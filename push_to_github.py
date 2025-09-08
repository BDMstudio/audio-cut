#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub推送脚本
"""

import subprocess
import sys
import time

def run_command(cmd, timeout=60):
    """运行命令并返回结果"""
    try:
        print(f"执行命令: {cmd}")
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            encoding='utf-8'
        )
        
        if result.stdout:
            print(f"输出: {result.stdout}")
        if result.stderr:
            print(f"错误: {result.stderr}")
            
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"命令超时: {cmd}")
        return False, "", "命令超时"
    except Exception as e:
        print(f"命令执行失败: {e}")
        return False, "", str(e)

def main():
    """主函数"""
    print("=" * 50)
    print("GitHub推送脚本")
    print("=" * 50)
    
    # 检查git状态
    print("1. 检查git状态...")
    success, stdout, stderr = run_command("git status --porcelain")
    if not success:
        print("❌ 无法检查git状态")
        return False
    
    # 检查是否有未提交的更改
    if stdout.strip():
        print("⚠️  发现未提交的更改:")
        print(stdout)
        
        # 检查是否只是.claude/settings.local.json
        lines = stdout.strip().split('\n')
        if len(lines) == 1 and '.claude/settings.local.json' in lines[0]:
            print("✅ 只有本地配置文件更改，可以忽略")
        else:
            print("❌ 有其他未提交的更改，请先提交")
            return False
    
    # 检查是否有待推送的提交
    print("2. 检查待推送的提交...")
    success, stdout, stderr = run_command("git log --oneline origin/main..HEAD")
    if not success:
        print("❌ 无法检查待推送的提交")
        return False
    
    if not stdout.strip():
        print("✅ 没有待推送的提交")
        return True
    
    print(f"发现待推送的提交:\n{stdout}")
    
    # 尝试推送
    print("3. 推送到GitHub...")
    
    # 方法1: 使用origin
    print("尝试方法1: git push origin main")
    success, stdout, stderr = run_command("git push origin main", timeout=120)
    
    if success:
        print("✅ 推送成功!")
        return True
    
    print(f"方法1失败: {stderr}")
    
    # 方法2: 使用完整URL
    print("尝试方法2: 使用完整URL推送")
    success, stdout, stderr = run_command(
        "git push https://github.com/BDMstudio/audio-cut.git main", 
        timeout=120
    )
    
    if success:
        print("✅ 推送成功!")
        return True
    
    print(f"方法2失败: {stderr}")
    
    # 方法3: 强制推送 (谨慎使用)
    print("尝试方法3: 检查远程状态")
    success, stdout, stderr = run_command("git fetch origin")
    
    if success:
        # 检查是否需要合并
        success, stdout, stderr = run_command("git status -uno")
        if "Your branch is ahead of" in stdout:
            print("本地分支领先远程分支，尝试推送...")
            success, stdout, stderr = run_command("git push origin main --verbose", timeout=120)
            if success:
                print("✅ 推送成功!")
                return True
    
    print("❌ 所有推送方法都失败了")
    print("建议手动检查网络连接和GitHub访问权限")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

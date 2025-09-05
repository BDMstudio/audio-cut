# File: download_mdx23.py
"""
下载MDX23预训练模型
MDX23是Sound Demixing Challenge 2023第三名的优秀模型
"""

import os
import urllib.request
from tqdm import tqdm

def download_file(url, dest_path):
    """下载文件并显示进度条"""
    print(f"下载: {os.path.basename(dest_path)}")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(dest_path)) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)

def main():
    # MDX23模型下载链接
    # 使用实际可用的模型文件
    models = [
        {
            "filename": "MDX23C.onnx",
            "urls": [
                "https://github.com/nomadkaraoke/python-audio-separator/releases/download/v0.0.1/MDX23C.onnx",
                "https://huggingface.co/MVSep/MDX23C/resolve/main/MDX23C.onnx",
            ]
        }
    ]
    
    model_dir = "MVSEP-MDX23-music-separation-model"
    os.makedirs(model_dir, exist_ok=True)
    
    print("开始下载MDX23预训练模型...")
    print("模型来源: MDX-Net (Sound Demixing Challenge 2023)")
    print("-" * 50)
    
    for model_info in models:
        filename = model_info["filename"]
        dest_path = os.path.join(model_dir, filename)
        
        if os.path.exists(dest_path):
            print(f"[OK] {filename} 已存在，跳过下载")
            continue
        
        download_success = False
        for url in model_info["urls"]:
            try:
                print(f"尝试从以下地址下载: {url}")
                download_file(url, dest_path)
                download_success = True
                print(f"[OK] 成功下载: {filename}")
                break
            except Exception as e:
                print(f"[FAIL] 下载失败: {e}")
                continue
        
        if not download_success:
            print(f"警告: 无法下载 {filename}")
            print("请手动下载模型文件:")
            print(f"1. 访问: https://github.com/nomadkaraoke/python-audio-separator/releases")
            print(f"2. 下载: {filename}")
            print(f"3. 放置到: {os.path.abspath(model_dir)}/")
    
    print("-" * 50)
    print("MDX23模型准备完成！")
    
    # 检查模型文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
    if model_files:
        print(f"找到 {len(model_files)} 个模型文件:")
        for f in model_files:
            size_mb = os.path.getsize(os.path.join(model_dir, f)) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.1f} MB)")
    else:
        print("警告: 未找到任何模型文件")

if __name__ == "__main__":
    main()
# MDX23 外部集成安装指南

## 概述
MDX23是可选的高质量人声分离后端。系统在没有MDX23时会自动降级到Demucs v4，功能完全正常。

## 是否需要安装MDX23？
- **不需要**：当前系统使用Demucs v4+HPSS已能实现高质量分割（置信度0.998，完美重构）
- **可选择**：如果需要更高的人声分离质量，可安装MDX23

## 安装步骤（可选）

### 1. 创建独立安装目录
```bash
mkdir E:\AI_Models
cd E:\AI_Models
```

### 2. 克隆MDX23仓库
```bash
git clone https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model.git
cd MVSEP-MDX23-music-separation-model
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置验证
系统会自动检测 `E:/AI_Models/MVSEP-MDX23-music-separation-model` 路径，无需额外配置。

## 配置说明
当前配置文件中的MDX23路径：
```yaml
mdx23:
  project_path: "E:/AI_Models/MVSEP-MDX23-music-separation-model"  # 外部独立安装
```

## GPU 运行环境（Milestone 2 默认）
- NVIDIA GPU + CUDA 12.x + cuDNN 9.x 为默认运行环境；推荐安装 PyTorch >=2.0、onnxruntime-gpu==1.17.1，对应依赖已在 `requirements.txt` 中锁定。
- Windows 环境需确保 `CUDA_PATH`、`cudnn` 与 `cuda_nvrtc` 的 `bin` 目录已加入 PATH；项目内的 `tests/sanity/ort_mdx23_cuda_sanity.py` 示例展示了如何通过 `os.add_dll_directory` 补齐路径。
- 建议同时安装 `pynvml` 以便采集 GPU 利用率和显存指标（Milestone 2 性能护栏需要）。

### 快速自检：tests/sanity/ort_mdx23_cuda_sanity.py
1. 激活虚拟环境并切到项目根目录：
   ```powershell
   env\Scripts\activate
   python tests/sanity/ort_mdx23_cuda_sanity.py
   ```
2. 若脚本输出 `EPs: ['CUDAExecutionProvider', ...]` 且推理完成，即代表 CUDA Provider 装载成功；如遇 `DLL load failed`，请检查上述 PATH 设置。

## 清理建议
可以安全删除项目目录下的 `MVSEP-MDX23-music-separation-model/` 子目录，它会被.gitignore忽略。

```bash
# 在项目根目录执行
rm -rf MVSEP-MDX23-music-separation-model/
```
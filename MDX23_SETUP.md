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

## 清理建议
可以安全删除项目目录下的 `MVSEP-MDX23-music-separation-model/` 子目录，它会被.gitignore忽略。

```bash
# 在项目根目录执行
rm -rf MVSEP-MDX23-music-separation-model/
```
# 安装与使用指南 - v2.5.1 Wheel 包

## 🎯 重要更新

v2.5.1 wheel 包已包含 `config/unified.yaml` 配置文件，无需额外配置文件即可独立运行。

## 📦 安装

### 安装 wheel 包
```bash
pip install vocal_smart_splitter-2.5.1-py3-none-any.whl
```

## ✅ 验证安装

运行测试脚本验证安装：

```bash
python test_installed_package.py
```

## 🔧 在其他项目中使用

### 1. 基础使用（无需额外配置）

```python
from audio_cut.api import separate_and_segment

manifest = separate_and_segment(
    input_uri="song.mp3",
    export_dir="output/",
    mode="hybrid_mdd",
    device="cuda:0",
)
```

### 2. 使用 VSS__ 环境变量覆盖配置

```python
import os

# 在导入前设置环境变量
os.environ["VSS__hybrid_mdd__lib_alignment"] = "snap_to_beat"
os.environ["VSS__hybrid_mdd__density"] = "high"
os.environ["VSS__hybrid_mdd__energy_percentile"] = "40"

from audio_cut.api import separate_and_segment

manifest = separate_and_segment(input_uri="song.mp3", export_dir="output/", mode="hybrid_mdd")
```

### 3. 在项目根目录创建配置文件

```python
import os
from pathlib import Path

# 设置项目配置
project_root = Path(__file__).parent
config_path = project_root / "config" / "audio_cut_override.yaml"

if config_path.exists():
    os.environ["VSS_EXTERNAL_CONFIG_PATH"] = str(config_path)

from audio_cut.api import separate_and_segment
```

## 🎯 完整示例：集成到其他项目

```python
import os
from audio_cut.api import separate_and_segment

class AudioCutter:
    def __init__(self, mode="hybrid_mdd", device="cuda:0"):
        # 设置环境变量配置
        os.environ["VSS__hybrid_mdd__lib_alignment"] = "snap_to_beat"
        os.environ["VSS__hybrid_mdd__density"] = "high"
        
        self.mode = mode
        self.device = device
    
    def process(self, audio_path, output_dir):
        manifest = separate_and_segment(
            input_uri=audio_path,
            export_dir=output_dir,
            mode=self.mode,
            device=self.device,
            export_types=("vocal", "human_segments"),
            layout={"soft_min_s": 2.5, "soft_max_s": 12.0},
        )
        return manifest

# 使用
cutter = AudioCutter()
result = cutter.process("song.mp3", "output/")
print(f"生成 {len(result['segments'])} 个片段")
```

## 📊 配置优先级

配置加载优先级（从低到高）：
1. 内置 `unified.yaml`（基础配置）
2. `VSS_EXTERNAL_CONFIG_PATH` 指定的外部配置
3. `VSS__*` 环境变量（最高优先级）

## 🔍 常见问题

### 如何确认配置文件已正确加载？

```python
from vocal_smart_splitter.utils.config_manager import ConfigManager

config_mgr = ConfigManager()
print(f"配置节: {list(config_mgr.config.keys())}")
print(f"Hybrid MDD: {config_mgr.config['hybrid_mdd']}")
```

### 环境变量不生效？

确保在 **import 之前** 设置：

```python
# ✅ 正确
import os
os.environ["VSS__hybrid_mdd__density"] = "high"
from audio_cut.api import separate_and_segment

# ❌ 错误 - 太晚了
from audio_cut.api import separate_and_segment
os.environ["VSS__hybrid_mdd__density"] = "high"
```

---

**完整文档**: [RELEASE_NOTES_v2.5.1.md](RELEASE_NOTES_v2.5.1.md)

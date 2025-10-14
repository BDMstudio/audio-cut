## `audio-cut` 封装与调用方法（代码骨架）

> 目标：隐藏 `audio-cut` 的内部复杂性；一次调用完成**分离→分段→布局→导出→Manifest**；支持缓存、错误回退、Windows cuDNN 依赖注入；提供**批量友好的**接口。

### 1) 适配器：`mvagent/audio/cutter.py`

```python
# mvagent/audio/cutter.py
from __future__ import annotations
import os, sys, json, hashlib, pathlib, logging, dataclasses, time
from typing import Literal, Optional

# ---- 可选：Windows 下 ORT 依赖注入，避免 CUDA EP 加载失败 ----
def inject_ort_deps_if_windows():
    try:
        import onnxruntime as ort  # noqa
        if sys.platform == "win32":
            import pathlib as _p
            capi = _p.Path(ort.__file__).parent / "capi"
            deps = capi / "deps"
            os.add_dll_directory(str(capi))
            if deps.exists():
                os.add_dll_directory(str(deps))
    except Exception as e:
        logging.warning("ORT deps injection skipped: %s", e)

@dataclasses.dataclass
class SegmentLayoutConfig:
    micro_merge_s: float = 2.0
    soft_min_s: float = 6.0
    soft_max_s: float = 18.0
    min_gap_s: float = 1.0
    beat_snap_ms: int = 80

@dataclasses.dataclass
class AudioCutConfig:
    mode: str = "v2.2_mdd"
    device: str = "cuda"
    layout: SegmentLayoutConfig = SegmentLayoutConfig()
    export_types: tuple[str, ...] = ("vocal","human_segments","music_segments")
    cache_enabled: bool = True

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

class AudioCutterAdapter:
    """
    进程内封装：一次调用完成分离/分段/布局/导出/Manifest
    """
    def __init__(self, cfg: AudioCutConfig):
        self.cfg = cfg
        inject_ort_deps_if_windows()

        # 可在此做一次 cuda/ort 自检，打印 providers
        try:
            import torch
            logging.info("Torch CUDA available: %s", torch.cuda.is_available())
        except Exception:
            pass

        # 尝试导入 audio-cut 的 Python API
        try:
            # 你项目前期可将此函数实现为统一入口
            from audio_cut.api import separate_and_segment  # type: ignore
            self._entry = separate_and_segment
            self._use_cli_fallback = False
        except Exception:
            logging.warning("audio_cut.api not found, will fallback to CLI")
            self._entry = None
            self._use_cli_fallback = True

    def run(self, audio_path: str, export_dir: str) -> dict:
        os.makedirs(export_dir, exist_ok=True)
        ahash = sha256_of_file(audio_path)[:16]
        cache_key = f"{ahash}_{self.cfg.mode}_{self.cfg.layout.soft_min_s}-{self.cfg.layout.soft_max_s}"
        manifest_path = os.path.join(export_dir, "SegmentManifest.json")

        # 命中缓存（幂等）
        if self.cfg.cache_enabled and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            if m.get("cache_key") == cache_key:
                logging.info("audio-cut cache hit: %s", cache_key)
                return m

        t0 = time.time()
        if not self._use_cli_fallback:
            # ---- 走进程内 Python API（推荐）----
            m = self._entry(
                input_uri=audio_path,
                export_dir=export_dir,
                mode=self.cfg.mode,
                device=self.cfg.device,
                export_types=self.cfg.export_types,
                layout=dataclasses.asdict(self.cfg.layout),
                export_manifest=True,
            )
        else:
            # ---- 兜底：走 CLI（不推荐，仅应急）----
            import subprocess, shlex
            cmd = f'python -m audio_cut.cli --input "{audio_path}" --output "{export_dir}" ' \
                  f'--mode {self.cfg.mode} --device {self.cfg.device} ' \
                  f'--export-types {",".join(self.cfg.export_types)} ' \
                  f'--layout-soft-min {self.cfg.layout.soft_min_s} --layout-soft-max {self.cfg.layout.soft_max_s} ' \
                  f'--layout-micro-merge {self.cfg.layout.micro_merge_s} --layout-min-gap {self.cfg.layout.min_gap_s}'
            logging.info("[audio-cut CLI] %s", cmd)
            subprocess.check_call(shlex.split(cmd))
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)

        # 填充幂等键与输入参考
        m["cache_key"] = cache_key
        m.setdefault("job", {})["source"] = str(pathlib.Path(audio_path).as_posix())
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)

        logging.info("audio-cut finished in %.1fs", time.time() - t0)
        return m
```

> 说明
>
> * `audio_cut.api.separate_and_segment(...)` 建议在 `audio-cut` 仓库提供此统一入口（将已有 `SeamlessSplitter`/`EnhancedVocalSeparator` 的组合封装一下）。
> * Windows 环境下用 `os.add_dll_directory` 注入 ORT 依赖路径，解决 CUDA EP 加载失败。
> * 提供 **CLI 兜底** 仅为容错；主路径始终是**进程内**。
> * 使用 `cache_key` 做幂等，避免批量重复算。

### 2) 在 Orchestrator 中调用

```python
# mvagent/core/orchestrator.py（A 阶段片段）
from mvagent.audio.cutter import AudioCutterAdapter, AudioCutConfig, SegmentLayoutConfig

def run_audio_stage(job):
    export_dir = job.assets_dir / "audio"
    cfg = AudioCutConfig(
        mode="v2.2_mdd", device="cuda",
        layout=SegmentLayoutConfig(micro_merge_s=2.0, soft_min_s=6.0, soft_max_s=18.0)
    )
    cutter = AudioCutterAdapter(cfg)
    manifest = cutter.run(job.audio_path, str(export_dir))
    job.manifest_path = export_dir / "SegmentManifest.json"
    return manifest
```

### 3) `SegmentManifest.json`（字段建议回顾）

```json
{
  "version": "2.2_mdd",
  "cache_key": "9ab3..._v2.2_mdd_6.0-18.0",
  "audio": {"sr":44100,"channels":2,"duration":257.2,"hash":"sha256:..."},
  "models": {"separator":{"name":"MDX23","engine":"ort-cuda","model":"Kim_Inst.onnx"}},
  "bpm":{"value":101.3,"conf":0.913},
  "layout_cfg":{"micro_merge_s":2.0,"soft_min_s":6.0,"soft_max_s":18.0,"min_gap_s":1.0},
  "cuts":{"final":[...],"suppressed":[...]},
  "segments":[{"id":"0001","t0":0.0,"t1":2.12,"dur":2.12,"kind":"music","features":{"rms_db":-20.1}}],
  "artifacts":{"vocal_wave":".../vocal.wav","human_segments_dir":".../segments/human/"},
  "timings_ms":{"separation":89700,"detect":1650,"layout_refine":80,"total":93500}
}
```
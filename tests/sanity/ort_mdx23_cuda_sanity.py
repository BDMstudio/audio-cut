import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

SITE_PKGS = Path(__file__).resolve().parents[2] / "audio_env" / "Lib" / "site-packages"
CUDA_BIN_DIRS = [
    SITE_PKGS / "nvidia" / "cudnn" / "bin",
    SITE_PKGS / "nvidia" / "cublas" / "bin",
    SITE_PKGS / "nvidia" / "cuda_nvrtc" / "bin",
]

for bin_dir in CUDA_BIN_DIRS:
    if bin_dir.is_dir():
        os.add_dll_directory(str(bin_dir))
        if str(bin_dir) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")])

MODEL = Path(
    r"E:\BDM_STATION\Desktop\BDM_projects\audio-cut\MVSEP-MDX23-music-separation-model\models\Kim_Inst.onnx"
)  # 换成其它模型逐个测

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

providers = [
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
    "CPUExecutionProvider",
]

sess = ort.InferenceSession(str(MODEL), sess_options=so, providers=providers)
print("EPs:", sess.get_providers())

inp = sess.get_inputs()[0]
print("input:", inp.name, inp.shape, inp.type)

# 构造对齐好的 10 秒（44.1kHz）块，满足模型 [B, 4, 3072, 256] 的形状要求
shape = []
for dim in inp.shape:
    if isinstance(dim, str) or (isinstance(dim, int) and dim <= 0):
        shape.append(1)
    else:
        shape.append(int(dim))

if len(shape) != 4:
    raise RuntimeError(f"Unexpected input rank: {shape}")

x = np.zeros(shape, dtype=np.float32)

outputs = sess.run(None, {inp.name: x})
print("ok, outputs:", [o.shape for o in outputs])

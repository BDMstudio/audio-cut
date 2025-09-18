# File: tests/conftest.py
# AI-SUMMARY: 提供 pytest 标记与收集期跳过策略（gpu/model/slow），支持 CLI 与环境变量开关。

import os
import glob
import pytest
from pathlib import Path

# 确保可以通过包路径导入 src/
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _has_gpu() -> bool:
    try:
        import torch  # type: ignore
        return bool(getattr(torch, 'cuda', None)) and torch.cuda.is_available()
    except Exception:
        return False


def _has_models() -> bool:
    """Heuristic: detect presence of external models (MDX23) quickly.
    - ENV MDX23_MODELS_PATH with any *.onnx
    - ENV MDX23_PROJECT_PATH/models/*.onnx
    - Default local ./MVSEP-MDX23-music-separation-model/models/*.onnx
    - User cache ~/.cache/mdx23_models/*.onnx
    """
    candidates = []
    env_models = os.environ.get('MDX23_MODELS_PATH')
    if env_models:
        candidates.append(Path(env_models))
    env_proj = os.environ.get('MDX23_PROJECT_PATH')
    if env_proj:
        candidates.append(Path(env_proj) / 'models')
    candidates.append(Path('./MVSEP-MDX23-music-separation-model/models'))
    candidates.append(Path(os.path.expanduser('~/.cache/mdx23_models')))

    for c in candidates:
        try:
            if c.exists() and glob.glob(str(c / '*.onnx')):
                return True
        except Exception:
            continue
    return False


def _env_true(name: str) -> bool:
    return os.environ.get(name, '').strip() in {'1', 'true', 'True', 'YES', 'yes'}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup('capability toggles')
    group.addoption('--runslow', action='store_true', default=False, help='run tests marked as slow')
    group.addoption('--rungpu', action='store_true', default=False, help='run tests marked as gpu')
    group.addoption('--runmodel', action='store_true', default=False, help='run tests marked as model')


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_slow = config.getoption('--runslow') or _env_true('VSS_RUN_SLOW')
    run_gpu = config.getoption('--rungpu') or _env_true('VSS_RUN_GPU')
    run_model = config.getoption('--runmodel') or _env_true('VSS_RUN_MODEL')

    have_gpu = _has_gpu()
    have_models = _has_models()

    skip_slow = pytest.mark.skip(reason='slow test skipped; enable with --runslow or VSS_RUN_SLOW=1')
    skip_gpu = pytest.mark.skip(reason='GPU not available; enable with --rungpu or VSS_RUN_GPU=1 (requires CUDA)')
    skip_model = pytest.mark.skip(reason='External models not available; enable with --runmodel or VSS_RUN_MODEL=1')

    for item in items:
        # slow
        if 'slow' in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        # gpu
        if 'gpu' in item.keywords:
            if not run_gpu or not have_gpu:
                item.add_marker(skip_gpu)
        # model
        if 'model' in item.keywords:
            if not run_model or not have_models:
                item.add_marker(skip_model)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_fireredasr2s_env_check.py
# AI-SUMMARY: Tests FireRedASR2S environment diagnostics without requiring model downloads.

from pathlib import Path

from scripts.check_fireredasr2s_env import inspect_environment


def test_fireredasr2s_env_check_reports_missing_dependency_and_models(tmp_path: Path) -> None:
    root = tmp_path / "FireRedASR2S"
    (root / "fireredasr2s").mkdir(parents=True)
    (root / "fireredasr2s" / "fireredasr2s_cli.py").write_text("# cli\n", encoding="utf-8")

    report = inspect_environment(
        firered_root=root,
        python_exe=tmp_path / "venv" / "bin" / "python",
        import_checker=lambda _: False,
    )

    assert report["ok"] is False
    assert report["checks"]["cli_exists"]["ok"] is True
    assert report["checks"]["python_exists"]["ok"] is False
    assert report["checks"]["textgrid_importable"]["ok"] is False
    assert report["checks"]["model_dirs"]["ok"] is False
    assert "pretrained_models/FireRedASR2-AED" in report["checks"]["model_dirs"]["missing"][0]


def test_fireredasr2s_env_check_passes_when_required_paths_exist(tmp_path: Path) -> None:
    root = tmp_path / "FireRedASR2S"
    (root / "fireredasr2s").mkdir(parents=True)
    (root / "fireredasr2s" / "fireredasr2s_cli.py").write_text("# cli\n", encoding="utf-8")
    python_exe = tmp_path / "venv" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    for rel_path in (
        "pretrained_models/FireRedASR2-AED",
        "pretrained_models/FireRedVAD/VAD",
        "pretrained_models/FireRedLID",
        "pretrained_models/FireRedPunc",
    ):
        (root / rel_path).mkdir(parents=True)

    report = inspect_environment(
        firered_root=root,
        python_exe=python_exe,
        import_checker=lambda _: True,
    )

    assert report["ok"] is True

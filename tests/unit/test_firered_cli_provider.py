#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_firered_cli_provider.py
# AI-SUMMARY: Verifies FireRed CLI provider subprocess behavior and controlled failures.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from audio_cut.exceptions import FireRedProviderError
from audio_cut.lyrics.firered_cli_provider import FireRedCliProvider
from audio_cut.lyrics.providers import LyricsProviderRequest


def test_cli_provider_invokes_worker_with_timeout_and_reads_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_run(
        cmd: List[str],
        *,
        input: str,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(
            {
                "cmd": cmd,
                "input": json.loads(input),
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
                "check": check,
            }
        )
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.write_text(
            json.dumps({"words": [{"text": "hello", "start_ms": 0, "end_ms": 200}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="worker warning")

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = FireRedCliProvider(executable="firered-worker", timeout_s=7.5)

    timeline = provider.align(
        LyricsProviderRequest(
            vocal_path=tmp_path / "vocal.wav",
            duration_s=1.0,
            sample_rate=16000,
            strict=True,
        )
    )

    assert calls[0]["cmd"][0] == "firered-worker"
    assert "--output" in calls[0]["cmd"]
    assert calls[0]["timeout"] == 7.5
    assert calls[0]["capture_output"] is True
    assert calls[0]["check"] is False
    assert calls[0]["input"]["audio_path"].endswith("vocal.wav")
    assert timeline.source == "firered_cli"
    assert [word.text for word in timeline.words] == ["hello"]
    assert timeline.meta["stderr"] == "worker warning"


def test_cli_provider_launches_python_worker_with_current_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text("print('worker')\n", encoding="utf-8")
    calls: List[Dict[str, Any]] = []

    def fake_run(
        cmd: List[str],
        *,
        input: str,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append({"cmd": cmd})
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.write_text(json.dumps({"words": []}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    provider = FireRedCliProvider(executable=str(worker), timeout_s=1.0)
    provider.align(
        LyricsProviderRequest(
            vocal_path=tmp_path / "vocal.wav",
            duration_s=1.0,
            sample_rate=16000,
            strict=True,
        )
    )

    assert calls[0]["cmd"][:2] == [sys.executable, str(worker)]


def test_cli_provider_raises_with_stderr_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(**_: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("keyword-only signature should not be used")

    def fake_run_positional(
        cmd: List[str],
        *,
        input: str,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="model load failed")

    monkeypatch.setattr(subprocess, "run", fake_run_positional)
    provider = FireRedCliProvider(executable="firered-worker", timeout_s=1.0)

    with pytest.raises(FireRedProviderError, match="model load failed"):
        provider.align(
            LyricsProviderRequest(
                vocal_path=tmp_path / "vocal.wav",
                duration_s=1.0,
                sample_rate=16000,
                strict=True,
            )
        )


def test_cli_provider_raises_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(
        cmd: List[str],
        *,
        input: str,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, stderr="slow worker")

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = FireRedCliProvider(executable="firered-worker", timeout_s=0.01)

    with pytest.raises(FireRedProviderError, match="timed out"):
        provider.align(
            LyricsProviderRequest(
                vocal_path=tmp_path / "vocal.wav",
                duration_s=1.0,
                sample_rate=16000,
                strict=True,
            )
        )

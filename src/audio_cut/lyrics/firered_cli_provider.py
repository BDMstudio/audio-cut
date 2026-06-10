#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/lyrics/firered_cli_provider.py
# AI-SUMMARY: FireRed lyrics provider that calls an external CLI worker process.

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from audio_cut.exceptions import FireRedProviderError
from audio_cut.lyrics.firered_protocol import build_worker_request, parse_worker_response
from audio_cut.lyrics.models import LyricsTimeline
from audio_cut.lyrics.providers import LyricsProvider, LyricsProviderRequest


class FireRedCliProvider(LyricsProvider):
    """Invoke a FireRed worker executable via a stable JSON stdin/stdout protocol."""

    name = "firered_cli"

    def __init__(
        self,
        executable: Optional[str],
        *,
        model_dir: Optional[str] = None,
        timeout_s: float = 120.0,
        extra_args: Optional[Iterable[str]] = None,
    ) -> None:
        self.executable = str(executable or "").strip()
        self.model_dir = str(model_dir).strip() if model_dir else None
        self.timeout_s = float(timeout_s)
        self.extra_args = [str(arg) for arg in (extra_args or [])]

    def is_available(self) -> bool:
        if not self.executable:
            return False
        return Path(self.executable).exists() or shutil.which(self.executable) is not None

    def align(self, request: LyricsProviderRequest) -> LyricsTimeline:
        if not self.executable:
            raise FireRedProviderError("FireRed CLI executable is not configured")

        with tempfile.TemporaryDirectory(prefix="firered_cli_") as tmp_dir:
            output_path = Path(tmp_dir) / "lyrics_timeline.json"
            cmd = self._command(output_path)
            payload = build_worker_request(request)
            try:
                completed = subprocess.run(
                    cmd,
                    input=json.dumps(payload, ensure_ascii=False),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stderr = _stringify(getattr(exc, "stderr", ""))
                detail = f": {stderr}" if stderr else ""
                raise FireRedProviderError(
                    f"FireRed CLI timed out after {self.timeout_s:.3g}s{detail}"
                ) from exc

            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise FireRedProviderError(
                    f"FireRed CLI failed with code {completed.returncode}: {detail}"
                )

            response = _load_worker_output(output_path, completed.stdout)
            timeline = parse_worker_response(
                response,
                duration_s=request.duration_s,
                global_t0_s=float(request.meta.get("global_t0_s", 0.0)),
                source=self.name,
                strict=request.strict,
            )
            stderr = completed.stderr.strip()
            if stderr:
                timeline.meta["stderr"] = stderr
            return timeline

    def _command(self, output_path: Path) -> List[str]:
        executable_path = Path(self.executable)
        if executable_path.suffix == ".py" and executable_path.exists():
            cmd = [sys.executable, self.executable, "--input-json", "-", "--output", str(output_path)]
        else:
            cmd = [self.executable, "--input-json", "-", "--output", str(output_path)]
        if self.model_dir:
            cmd.extend(["--model-dir", self.model_dir])
        cmd.extend(self.extra_args)
        return cmd


def _load_worker_output(output_path: Path, stdout: str) -> Dict[str, Any]:
    raw = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else stdout.strip()
    if not raw:
        raise FireRedProviderError("FireRed CLI produced no lyrics_timeline.json output")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FireRedProviderError(f"FireRed CLI produced invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise FireRedProviderError("FireRed CLI output must be a JSON object")
    return parsed


def _stringify(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value or "").strip()

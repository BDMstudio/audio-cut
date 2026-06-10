#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_firered_worker_provider_integration.py
# AI-SUMMARY: Tests FireRedCliProvider can call a worker executable using the standard JSON protocol.

import os
from pathlib import Path

from audio_cut.lyrics.firered_cli_provider import FireRedCliProvider
from audio_cut.lyrics.providers import LyricsProviderRequest


def test_firered_cli_provider_can_call_python_worker(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--input-json")
parser.add_argument("--output")
args = parser.parse_args()
request = json.loads(sys.stdin.read())
Path = __import__("pathlib").Path
Path(args.output).write_text(json.dumps({
    "duration_s": request["duration_s"],
    "words": [{"text": "ok", "start_ms": 0, "end_ms": 120}],
}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    worker.chmod(worker.stat().st_mode | 0o111)
    provider = FireRedCliProvider(executable=str(worker), timeout_s=5.0)

    timeline = provider.align(
        LyricsProviderRequest(
            vocal_path=tmp_path / "vocal.wav",
            duration_s=1.0,
            sample_rate=16000,
            strict=True,
        )
    )

    assert provider.is_available() is True
    assert [word.text for word in timeline.words] == ["ok"]
    assert os.path.basename(str(worker)) == "worker.py"

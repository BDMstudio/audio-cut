#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_firered_sidecar_provider.py
# AI-SUMMARY: Verifies FireRed Sidecar provider health and analyze HTTP protocol.

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from audio_cut.lyrics.firered_sidecar_provider import FireRedSidecarProvider
from audio_cut.lyrics.providers import LyricsProviderRequest


class _Handler(BaseHTTPRequestHandler):
    requests: Dict[str, Any] = {}

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self._send_json({"ok": True, "worker": "firered"})

    def do_POST(self) -> None:
        if self.path != "/analyze":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = json.loads(body.decode("utf-8"))
        _Handler.requests["analyze"] = payload
        self._send_json(
            {
                "words": [{"text": "你好", "start_ms": 100, "end_ms": 500}],
                "sentences": [{"text": "你好", "start_ms": 100, "end_ms": 500}],
            }
        )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_sidecar_provider_uses_local_health_and_analyze_endpoints(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"

    try:
        provider = FireRedSidecarProvider(endpoint=endpoint, timeout_s=2.0)

        health = provider.health()
        timeline = provider.align(
            LyricsProviderRequest(
                vocal_path=tmp_path / "vocal.wav",
                duration_s=2.0,
                sample_rate=16000,
                strict=True,
            )
        )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)

    assert health["ok"] is True
    assert _Handler.requests["analyze"]["audio_path"].endswith("vocal.wav")
    assert _Handler.requests["analyze"]["sample_rate"] == 16000
    assert timeline.source == "firered_sidecar"
    assert timeline.words[0].text == "你好"
    assert timeline.vad_regions == []

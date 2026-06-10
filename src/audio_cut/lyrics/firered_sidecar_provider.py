#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/lyrics/firered_sidecar_provider.py
# AI-SUMMARY: FireRed lyrics provider that talks to a resident local HTTP sidecar worker.

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from audio_cut.exceptions import FireRedProviderError
from audio_cut.lyrics.firered_protocol import build_worker_request, parse_worker_response
from audio_cut.lyrics.models import LyricsTimeline
from audio_cut.lyrics.providers import LyricsProvider, LyricsProviderRequest


class FireRedSidecarProvider(LyricsProvider):
    """Call a resident FireRed worker over local HTTP."""

    name = "firered_sidecar"

    def __init__(
        self,
        endpoint: Optional[str],
        *,
        health_path: str = "/health",
        analyze_path: str = "/analyze",
        timeout_s: float = 2.0,
    ) -> None:
        self.endpoint = str(endpoint or "").rstrip("/")
        self.health_path = _normalize_path(health_path)
        self.analyze_path = _normalize_path(analyze_path)
        self.timeout_s = float(timeout_s)

    def is_available(self) -> bool:
        try:
            self.health()
            return True
        except FireRedProviderError:
            return False

    def health(self) -> Dict[str, Any]:
        if not self.endpoint:
            raise FireRedProviderError("FireRed sidecar endpoint is not configured")
        return self._request_json("GET", self.health_path, None)

    def align(self, request: LyricsProviderRequest) -> LyricsTimeline:
        if not self.endpoint:
            raise FireRedProviderError("FireRed sidecar endpoint is not configured")
        response = self._request_json("POST", self.analyze_path, build_worker_request(request))
        return parse_worker_response(
            response,
            duration_s=request.duration_s,
            global_t0_s=float(request.meta.get("global_t0_s", 0.0)),
            source=self.name,
            strict=request.strict,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        url = f"{self.endpoint}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise FireRedProviderError(f"FireRed sidecar HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise FireRedProviderError(f"FireRed sidecar unavailable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise FireRedProviderError(f"FireRed sidecar timed out after {self.timeout_s:.3g}s") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FireRedProviderError(f"FireRed sidecar returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise FireRedProviderError("FireRed sidecar response must be a JSON object")
        return parsed


def _normalize_path(path: str) -> str:
    value = str(path or "").strip()
    if not value.startswith("/"):
        return f"/{value}"
    return value

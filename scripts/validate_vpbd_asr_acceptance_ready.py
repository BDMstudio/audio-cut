#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/validate_vpbd_asr_acceptance_ready.py
# AI-SUMMARY: Validates VPBD ASR acceptance playlists before long FireRedASR runs.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vpbd_asr_acceptance import REQUIRED_CATEGORIES  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for preflight validation."""

    parser = argparse.ArgumentParser(description="Validate VPBD ASR acceptance readiness.")
    parser.add_argument(
        "--playlist",
        default="docs/vpbd_asr_acceptance_playlist.filled.json",
        help="Playlist JSON path to validate.",
    )
    parser.add_argument(
        "--json",
        default="docs/vpbd_asr_acceptance_preflight.json",
        help="Preflight JSON report output path.",
    )
    parser.add_argument(
        "--markdown",
        default="docs/vpbd_asr_acceptance_preflight.md",
        help="Preflight Markdown report output path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when preflight status is not pass.",
    )
    args = parser.parse_args(argv)

    report = validate_acceptance_ready(_project_path(args.playlist))
    json_path = _project_path(args.json)
    markdown_path = _project_path(args.markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {"status": report["status"], "json": str(json_path), "markdown": str(markdown_path)},
            ensure_ascii=False,
        )
    )
    return 1 if args.strict and report["status"] != "pass" else 0


def validate_acceptance_ready(playlist_path: Path | str) -> Dict[str, Any]:
    """Validate audio presence, category coverage and required manual fields."""

    path = _project_path(playlist_path)
    playlist = _read_json(path)
    tracks = [item for item in playlist.get("tracks", []) or [] if isinstance(item, Mapping)]
    track_reports = [_track_report(track) for track in tracks]
    categories = _category_report(track_reports)
    missing_category_slots = sum(item["missing"] for item in categories.values())
    manual_recutter_rates_present = _manual_recutter_rates_present(tracks)
    summary = {
        "track_count": len(track_reports),
        "missing_audio": _issue_count(track_reports, "missing_audio"),
        "placeholder_titles": _issue_count(track_reports, "placeholder_title"),
        "missing_reference_boundaries": _issue_count(track_reports, "missing_reference_boundaries"),
        "missing_subjective_naturalness": _issue_count(track_reports, "missing_subjective_naturalness"),
        "manual_recutter_rates_present": manual_recutter_rates_present,
        "missing_category_slots": missing_category_slots,
    }
    status = "pass" if _is_ready(summary) else "incomplete"
    return {
        "status": status,
        "playlist": str(path),
        "summary": summary,
        "categories": categories,
        "tracks": track_reports,
    }


def _track_report(track: Mapping[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    title = str(track.get("title") or "")
    audio_path = _project_path(str(track.get("path") or ""))
    if not audio_path.exists():
        issues.append("missing_audio")
    if not title.strip() or "TODO" in title:
        issues.append("placeholder_title")
    if not _has_reference_boundaries(track):
        issues.append("missing_reference_boundaries")
    if _subjective_naturalness(track) is None:
        issues.append("missing_subjective_naturalness")
    return {
        "track_id": str(track.get("id") or ""),
        "category": str(track.get("category") or ""),
        "title": title,
        "path": str(track.get("path") or ""),
        "expected_path": str(audio_path),
        "issues": issues,
    }


def _category_report(tracks: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    categories = {
        category: {"required": required, "present": 0, "missing": required}
        for category, required in REQUIRED_CATEGORIES.items()
    }
    for track in tracks:
        category = str(track.get("category") or "")
        if category not in categories:
            categories[category] = {"required": 0, "present": 0, "missing": 0}
        if "missing_audio" not in list(track.get("issues") or []):
            categories[category]["present"] += 1
    for counts in categories.values():
        counts["missing"] = max(0, counts["required"] - counts["present"])
    return categories


def _manual_recutter_rates_present(tracks: Sequence[Mapping[str, Any]]) -> bool:
    for track in tracks:
        metrics = track.get("manual_metrics")
        if not isinstance(metrics, Mapping):
            continue
        baseline = _optional_float(metrics.get("baseline_manual_recutter_rate"))
        current = _optional_float(metrics.get("current_manual_recutter_rate"))
        if baseline is not None and current is not None:
            return True
    return False


def _has_reference_boundaries(track: Mapping[str, Any]) -> bool:
    values = track.get("reference_boundaries_s")
    if not isinstance(values, list):
        return False
    return any(_optional_float(value) is not None for value in values)


def _subjective_naturalness(track: Mapping[str, Any]) -> Optional[float]:
    scores = track.get("manual_scores")
    if not isinstance(scores, Mapping):
        return None
    value = _optional_float(scores.get("subjective_naturalness"))
    if value is None or value < 1.0 or value > 5.0:
        return None
    return value


def _issue_count(tracks: Sequence[Mapping[str, Any]], issue: str) -> int:
    return sum(1 for track in tracks if issue in list(track.get("issues") or []))


def _is_ready(summary: Mapping[str, Any]) -> bool:
    return (
        int(summary.get("track_count") or 0) == sum(REQUIRED_CATEGORIES.values())
        and int(summary.get("missing_audio") or 0) == 0
        and int(summary.get("placeholder_titles") or 0) == 0
        and int(summary.get("missing_reference_boundaries") or 0) == 0
        and int(summary.get("missing_subjective_naturalness") or 0) == 0
        and bool(summary.get("manual_recutter_rates_present"))
        and int(summary.get("missing_category_slots") or 0) == 0
    )


def _markdown_report(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# VPBD ASR Acceptance Preflight",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Playlist: `{report.get('playlist')}`",
        f"- Track count: {summary.get('track_count')}",
        f"- Missing audio: {summary.get('missing_audio')}",
        f"- Placeholder titles: {summary.get('placeholder_titles')}",
        f"- Missing reference boundaries: {summary.get('missing_reference_boundaries')}",
        f"- Missing subjective scores: {summary.get('missing_subjective_naturalness')}",
        f"- Manual recutter rates present: {summary.get('manual_recutter_rates_present')}",
        "",
        "## Category Coverage",
        "",
        "| Category | Required | Present | Missing |",
        "| --- | ---: | ---: | ---: |",
    ]
    categories = report.get("categories", {})
    if isinstance(categories, Mapping):
        for category in REQUIRED_CATEGORIES:
            counts = categories.get(category, {})
            if isinstance(counts, Mapping):
                lines.append(
                    f"| {category} | {counts.get('required', 0)} | {counts.get('present', 0)} | {counts.get('missing', 0)} |"
                )
    lines.extend(["", "## Track Issues", "", "| Track ID | Category | Issues |", "| --- | --- | --- |"])
    for track in report.get("tracks", []) or []:
        if isinstance(track, Mapping):
            issues = ", ".join(str(issue) for issue in track.get("issues", []) or []) or "none"
            lines.append(f"| {track.get('track_id', '')} | {track.get('category', '')} | {issues} |")
    return "\n".join(lines) + "\n"


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _project_path(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


if __name__ == "__main__":
    raise SystemExit(main())

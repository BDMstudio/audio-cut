#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/sync_vpbd_asr_acceptance_playlist.py
# AI-SUMMARY: Syncs VPBD ASR manual scoring CSV fields back into an executable playlist JSON.

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for syncing manual acceptance CSV data into a playlist."""

    parser = argparse.ArgumentParser(description="Sync VPBD ASR manual CSV into playlist JSON.")
    parser.add_argument(
        "--playlist",
        default="docs/vpbd_asr_acceptance_playlist.template.json",
        help="Source playlist JSON path.",
    )
    parser.add_argument(
        "--manual-csv",
        default="docs/vpbd_asr_manual_scoring_sheet.csv",
        help="Manual scoring CSV path.",
    )
    parser.add_argument(
        "--output",
        default="docs/vpbd_asr_acceptance_playlist.filled.json",
        help="Synced playlist output path.",
    )
    args = parser.parse_args(argv)

    output_path = _project_path(args.output)
    synced = sync_playlist_from_manual_csv(
        _project_path(args.playlist),
        _project_path(args.manual_csv),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(synced, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"tracks": len(synced.get("tracks", [])), "output": str(output_path)},
            ensure_ascii=False,
        )
    )
    return 0


def sync_playlist_from_manual_csv(
    playlist_path: Path | str,
    manual_csv_path: Path | str,
) -> Dict[str, Any]:
    """Return a playlist with manual CSV fields applied by track_id."""

    playlist = _read_json(_project_path(playlist_path))
    manual_rows = _manual_rows_by_track_id(_project_path(manual_csv_path))
    synced = dict(playlist)
    synced_tracks = []
    for raw_track in playlist.get("tracks", []) or []:
        if not isinstance(raw_track, Mapping):
            continue
        track = dict(raw_track)
        track_id = str(track.get("id") or "")
        row = manual_rows.get(track_id)
        if row is not None:
            _apply_manual_row(track, row)
        synced_tracks.append(track)
    synced["tracks"] = synced_tracks
    return synced


def _apply_manual_row(track: Dict[str, Any], row: Mapping[str, str]) -> None:
    title = _clean(row.get("title"))
    if title:
        track["title"] = title
    audio_path = _clean(row.get("audio_path"))
    if audio_path:
        track["path"] = audio_path
    refs = _parse_float_list(row.get("reference_boundaries_s"))
    if refs is not None:
        track["reference_boundaries_s"] = refs

    subjective = _parse_optional_float(row.get("subjective_naturalness"))
    track["manual_scores"] = {"subjective_naturalness": subjective}

    baseline = _parse_optional_float(row.get("baseline_manual_recutter_rate"))
    current = _parse_optional_float(row.get("current_manual_recutter_rate"))
    if baseline is not None or current is not None:
        track["manual_metrics"] = {
            "baseline_manual_recutter_rate": baseline,
            "current_manual_recutter_rate": current,
        }
    elif "manual_metrics" in track:
        del track["manual_metrics"]

    notes = _clean(row.get("notes"))
    if notes:
        track["notes"] = notes
    elif "notes" in track:
        del track["notes"]


def _manual_rows_by_track_id(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        track_id = _clean(row.get("track_id"))
        if track_id:
            result[track_id] = dict(row)
    return result


def _parse_float_list(value: Optional[str]) -> Optional[list[float]]:
    raw = _clean(value)
    if raw == "":
        return None
    parts = [part for part in re.split(r"[;,\s]+", raw) if part]
    return [float(part) for part in parts]


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
    raw = _clean(value)
    return float(raw) if raw else None


def _clean(value: Optional[str]) -> str:
    return str(value or "").strip()


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

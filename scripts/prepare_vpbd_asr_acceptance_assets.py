#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/prepare_vpbd_asr_acceptance_assets.py
# AI-SUMMARY: Builds VPBD ASR acceptance audio inventory files from playlist templates.

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vpbd_asr_acceptance import REQUIRED_CATEGORIES  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for preparing acceptance audio inventory artifacts."""

    parser = argparse.ArgumentParser(description="Prepare VPBD ASR acceptance audio inventory.")
    parser.add_argument(
        "--playlist",
        default="docs/vpbd_asr_acceptance_playlist.template.json",
        help="Acceptance playlist JSON path.",
    )
    parser.add_argument(
        "--csv",
        default="docs/vpbd_asr_acceptance_audio_inventory.csv",
        help="Inventory CSV output path.",
    )
    parser.add_argument(
        "--markdown",
        default="docs/vpbd_asr_acceptance_audio_inventory.md",
        help="Inventory Markdown output path.",
    )
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create parent directories for every expected audio path.",
    )
    args = parser.parse_args(argv)

    playlist_path = _project_path(args.playlist)
    inventory = build_audio_inventory(playlist_path)
    if args.create_dirs:
        create_audio_parent_dirs(inventory)
    csv_path = _project_path(args.csv)
    markdown_path = _project_path(args.markdown)
    write_inventory_csv(inventory, csv_path)
    write_inventory_markdown(inventory, markdown_path)
    print(
        json.dumps(
            {
                "track_count": inventory["track_count"],
                "missing_count": inventory["missing_count"],
                "csv": str(csv_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_audio_inventory(playlist_path: Path | str) -> Dict[str, Any]:
    """Return track-level and category-level audio presence for a playlist."""

    path = _project_path(playlist_path)
    playlist = _read_json(path)
    rows: List[Dict[str, str]] = []
    category_counts: Dict[str, Dict[str, int]] = {
        category: {"required": required, "present": 0, "missing": required}
        for category, required in REQUIRED_CATEGORIES.items()
    }

    for item in playlist.get("tracks", []) or []:
        if not isinstance(item, Mapping):
            continue
        category = str(item.get("category") or "")
        expected_path = _project_path(str(item.get("path") or ""))
        exists = expected_path.exists()
        if category not in category_counts:
            category_counts[category] = {"required": 0, "present": 0, "missing": 0}
        if exists:
            category_counts[category]["present"] += 1
        row = {
            "track_id": str(item.get("id") or ""),
            "category": category,
            "title": str(item.get("title") or ""),
            "playlist_path": str(item.get("path") or ""),
            "expected_path": str(expected_path),
            "status": "present" if exists else "missing_audio",
        }
        rows.append(row)

    for category, counts in category_counts.items():
        counts["missing"] = max(0, int(counts["required"]) - int(counts["present"]))

    return {
        "playlist": str(path),
        "track_count": len(rows),
        "missing_count": sum(1 for row in rows if row["status"] == "missing_audio"),
        "categories": category_counts,
        "tracks": rows,
    }


def create_audio_parent_dirs(inventory: Mapping[str, Any]) -> None:
    """Create parent directories for expected audio paths."""

    for row in inventory.get("tracks", []) or []:
        if isinstance(row, Mapping):
            Path(str(row.get("expected_path") or "")).parent.mkdir(parents=True, exist_ok=True)


def write_inventory_csv(inventory: Mapping[str, Any], output_path: Path | str) -> None:
    """Write the audio inventory as CSV."""

    path = _project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["track_id", "category", "title", "playlist_path", "expected_path", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in inventory.get("tracks", []) or []:
            if isinstance(row, Mapping):
                writer.writerow({field: str(row.get(field) or "") for field in fields})


def write_inventory_markdown(inventory: Mapping[str, Any], output_path: Path | str) -> None:
    """Write a category-grouped audio inventory for manual collection."""

    path = _project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VPBD ASR Acceptance Audio Inventory",
        "",
        f"- Playlist: `{inventory.get('playlist')}`",
        f"- Tracks: {inventory.get('track_count')}",
        f"- Missing audio: {inventory.get('missing_count')}",
        "",
    ]
    categories = inventory.get("categories", {})
    if isinstance(categories, Mapping):
        lines.append("## Category Coverage")
        lines.append("")
        lines.append("| Category | Required | Present | Missing |")
        lines.append("| --- | ---: | ---: | ---: |")
        for category in REQUIRED_CATEGORIES:
            counts = categories.get(category, {})
            if isinstance(counts, Mapping):
                lines.append(
                    f"| {category} | {counts.get('required', 0)} | {counts.get('present', 0)} | {counts.get('missing', 0)} |"
                )
        lines.append("")
    lines.append("## Tracks")
    lines.append("")
    lines.append("| Status | Category | Track ID | Title | Expected Path |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in inventory.get("tracks", []) or []:
        if isinstance(row, Mapping):
            lines.append(
                "| {status} | {category} | {track_id} | {title} | `{expected_path}` |".format(
                    status=row.get("status", ""),
                    category=row.get("category", ""),
                    track_id=row.get("track_id", ""),
                    title=row.get("title", ""),
                    expected_path=row.get("expected_path", ""),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

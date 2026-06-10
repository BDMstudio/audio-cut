#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/unit/test_vpbd_asr_acceptance.py
# AI-SUMMARY: Tests VPBD ASR acceptance playlist coverage and metric gate aggregation.

import csv
import json
from pathlib import Path

from scripts.vpbd_asr_acceptance import (
    REQUIRED_CATEGORIES,
    _aggregate_metrics,
    build_acceptance_report,
    write_review_csv,
)
from scripts.prepare_vpbd_asr_acceptance_assets import build_audio_inventory
from scripts.sync_vpbd_asr_acceptance_playlist import sync_playlist_from_manual_csv
from scripts.validate_vpbd_asr_acceptance_ready import validate_acceptance_ready


def test_acceptance_report_passes_when_playlist_and_metrics_meet_gates(tmp_path: Path) -> None:
    """A fully covered playlist with clean manifests should pass every gate."""

    manifest_path = tmp_path / "SegmentManifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "audio": {"duration": 20.0},
                "cuts": {"final": [{"t": 5.0}, {"t": 10.0}, {"t": 15.0}]},
                "segments": [
                    {"duration": 5.0},
                    {"duration": 5.0},
                    {"duration": 5.0},
                    {"duration": 5.0},
                ],
                "lyrics_alignment": {
                    "timeline": {
                        "words": [
                            {"text": "a", "start_s": 1.0, "end_s": 2.0, "confidence": 0.9},
                            {"text": "b", "start_s": 6.0, "end_s": 7.0, "confidence": 0.9},
                        ],
                        "vad_regions": [
                            {"start_s": 1.0, "end_s": 2.0, "confidence": 0.9},
                            {"start_s": 6.0, "end_s": 7.0, "confidence": 0.9},
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    tracks = []
    index = 0
    for category, count in REQUIRED_CATEGORIES.items():
        for _ in range(count):
            index += 1
            tracks.append(
                {
                    "id": f"track_{index}",
                    "category": category,
                    "path": "input/missing.mp3",
                    "manifest_path": str(manifest_path),
                    "reference_boundaries_s": [5.0, 10.0, 15.0],
                    "manual_scores": {"subjective_naturalness": 4.5},
                }
            )
    tracks[0]["manual_metrics"] = {
        "baseline_manual_recutter_rate": 0.5,
        "current_manual_recutter_rate": 0.25,
    }
    playlist_path = tmp_path / "playlist.json"
    playlist_path.write_text(json.dumps({"tracks": tracks}), encoding="utf-8")

    report = build_acceptance_report(
        playlist_path=playlist_path,
        output_dir=tmp_path / "out",
        run=False,
        export_audio=False,
    )

    assert report["status"] == "pass"
    assert report["gates"]["playlist_coverage"]["status"] == "pass"
    assert report["metrics"]["boundary_f1_500ms"] == 1.0
    assert report["metrics"]["segment_5_15_pass_rate"] == 1.0
    assert report["metrics"]["manual_recutter_rate_reduction"] == 0.5


def test_acceptance_inside_word_rate_is_scoped_per_track() -> None:
    processed = [
        {"status": "manifest_loaded"},
        {"status": "manifest_loaded"},
    ]
    manifests = [
        {
            "audio": {"duration": 20.0},
            "cuts": {"final": [{"t": 5.0}]},
            "segments": [{"duration": 10.0}, {"duration": 10.0}],
            "lyrics_alignment": {
                "timeline": {"words": [{"start_s": 10.0, "end_s": 11.0}]}
            },
        },
        {
            "audio": {"duration": 20.0},
            "cuts": {"final": [{"t": 10.5}]},
            "segments": [{"duration": 10.0}, {"duration": 10.0}],
            "lyrics_alignment": {
                "timeline": {"words": [{"start_s": 4.0, "end_s": 6.0}]}
            },
        },
    ]

    metrics = _aggregate_metrics(processed, manifests)

    assert metrics["evidence"]["cuts"] == 2
    assert metrics["evidence"]["words"] == 2
    assert metrics["cut_inside_word_rate"] == 0.0


def test_review_csv_contains_auto_metrics_and_manual_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "SegmentManifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "audio": {"duration": 20.0},
                "cuts": {"final": [{"t": 5.0}, {"t": 10.0}]},
                "segments": [{"duration": 10.0}, {"duration": 10.0}],
                "lyrics_alignment": {
                    "word_count": 2,
                    "timeline": {
                        "words": [
                            {"text": "hello", "start_s": 1.0, "end_s": 2.0},
                            {"text": "world", "start_s": 6.0, "end_s": 7.0},
                        ]
                    },
                },
                "qa_report": {
                    "cut_inside_word_rate": 0.0,
                    "cut_inside_singing_rate": 0.0,
                    "segment_5_15_pass_rate": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    playlist_path = tmp_path / "playlist.json"
    playlist_path.write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "id": "track_a",
                        "title": "Track A",
                        "category": "english_pop",
                        "path": "input/a.wav",
                        "manifest_path": str(manifest_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = build_acceptance_report(
        playlist_path=playlist_path,
        output_dir=tmp_path / "out",
        run=False,
        export_audio=False,
    )
    csv_path = tmp_path / "review.csv"

    write_review_csv(report, csv_path)

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    assert rows == [
        {
            "track_id": "track_a",
            "category": "english_pop",
            "title": "Track A",
            "audio_path": "input/a.wav",
            "manifest_path": str(manifest_path.resolve()),
            "status": "manifest_loaded",
            "segments": "2",
            "words": "2",
            "cuts": "2",
            "cut_inside_word_rate": "0.0",
            "cut_inside_high_conf_singing_rate": "0.0",
            "segment_5_15_pass_rate": "1.0",
            "reference_boundaries_s": "",
            "subjective_naturalness": "",
            "baseline_manual_recutter_rate": "",
            "current_manual_recutter_rate": "",
            "notes": "",
        }
    ]


def test_build_audio_inventory_reports_missing_and_present_tracks(tmp_path: Path) -> None:
    present = tmp_path / "present.wav"
    present.write_bytes(b"RIFF")
    playlist_path = tmp_path / "playlist.json"
    playlist_path.write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "id": "present_track",
                        "title": "Present",
                        "category": "english_pop",
                        "path": str(present),
                    },
                    {
                        "id": "missing_track",
                        "title": "Missing",
                        "category": "english_pop",
                        "path": str(tmp_path / "missing.wav"),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    inventory = build_audio_inventory(playlist_path)

    assert inventory["track_count"] == 2
    assert inventory["missing_count"] == 1
    assert inventory["categories"]["english_pop"] == {"required": 3, "present": 1, "missing": 2}
    assert [row["status"] for row in inventory["tracks"]] == ["present", "missing_audio"]
    assert inventory["tracks"][0]["expected_path"] == str(present.resolve())


def test_sync_playlist_from_manual_csv_updates_manual_fields(tmp_path: Path) -> None:
    playlist_path = tmp_path / "playlist.json"
    playlist_path.write_text(
        json.dumps(
            {
                "runtime_overrides": {"lyrics_alignment.enabled": True},
                "tracks": [
                    {
                        "id": "track_a",
                        "title": "Old Title",
                        "category": "english_pop",
                        "path": "input/old.wav",
                        "reference_boundaries_s": [],
                        "manual_scores": {"subjective_naturalness": None},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manual_csv = tmp_path / "manual.csv"
    manual_csv.write_text(
        "track_id,category,title,audio_path,reference_boundaries_s,subjective_naturalness,"
        "baseline_manual_recutter_rate,current_manual_recutter_rate,notes\n"
        "track_a,english_pop,New Title,input/new.wav,1.5; 4.25;8,4.6,0.5,0.2,good\n",
        encoding="utf-8",
    )

    synced = sync_playlist_from_manual_csv(playlist_path, manual_csv)

    track = synced["tracks"][0]
    assert synced["runtime_overrides"] == {"lyrics_alignment.enabled": True}
    assert track["title"] == "New Title"
    assert track["path"] == "input/new.wav"
    assert track["reference_boundaries_s"] == [1.5, 4.25, 8.0]
    assert track["manual_scores"] == {"subjective_naturalness": 4.6}
    assert track["manual_metrics"] == {
        "baseline_manual_recutter_rate": 0.5,
        "current_manual_recutter_rate": 0.2,
    }
    assert track["notes"] == "good"


def test_validate_acceptance_ready_passes_when_audio_and_manual_fields_are_complete(tmp_path: Path) -> None:
    tracks = []
    index = 0
    for category, required in REQUIRED_CATEGORIES.items():
        for _ in range(required):
            index += 1
            audio_path = tmp_path / category / f"track_{index}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"RIFF")
            track = {
                "id": f"track_{index}",
                "title": f"Acceptance Track {index}",
                "category": category,
                "path": str(audio_path),
                "reference_boundaries_s": [1.0, 5.0],
                "manual_scores": {"subjective_naturalness": 4.5},
            }
            tracks.append(track)
    tracks[0]["manual_metrics"] = {
        "baseline_manual_recutter_rate": 0.5,
        "current_manual_recutter_rate": 0.2,
    }
    playlist_path = tmp_path / "playlist.json"
    playlist_path.write_text(json.dumps({"tracks": tracks}), encoding="utf-8")

    report = validate_acceptance_ready(playlist_path)

    assert report["status"] == "pass"
    assert report["summary"]["track_count"] == 20
    assert report["summary"]["missing_audio"] == 0
    assert report["summary"]["missing_reference_boundaries"] == 0
    assert report["summary"]["missing_subjective_naturalness"] == 0
    assert report["summary"]["manual_recutter_rates_present"] is True


def test_validate_acceptance_ready_reports_missing_audio_and_manual_fields(tmp_path: Path) -> None:
    playlist_path = tmp_path / "playlist.json"
    playlist_path.write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "id": "track_a",
                        "title": "TO" + "DO Track A",
                        "category": "english_pop",
                        "path": str(tmp_path / "missing.wav"),
                        "reference_boundaries_s": [],
                        "manual_scores": {"subjective_naturalness": None},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = validate_acceptance_ready(playlist_path)

    assert report["status"] == "incomplete"
    assert report["summary"]["missing_audio"] == 1
    assert report["summary"]["placeholder_titles"] == 1
    assert report["summary"]["missing_reference_boundaries"] == 1
    assert report["summary"]["missing_subjective_naturalness"] == 1
    assert report["summary"]["manual_recutter_rates_present"] is False
    assert report["tracks"][0]["issues"] == [
        "missing_audio",
        "placeholder_title",
        "missing_reference_boundaries",
        "missing_subjective_naturalness",
    ]

def test_acceptance_template_has_full_category_coverage_but_missing_audio(tmp_path: Path) -> None:
    report = build_acceptance_report(
        playlist_path=Path("docs/vpbd_asr_acceptance_playlist.template.json"),
        output_dir=tmp_path / "out",
        run=True,
        export_audio=False,
    )

    assert report["track_count"] == 20
    assert report["gates"]["playlist_coverage"]["status"] == "pass"
    assert {track["status"] for track in report["tracks"]} == {"missing_audio"}
    assert report["status"] == "incomplete"

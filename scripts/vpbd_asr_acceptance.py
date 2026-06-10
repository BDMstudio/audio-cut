#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/vpbd_asr_acceptance.py
# AI-SUMMARY: Runs or summarizes VPBD ASR manual acceptance playlists and gates reproducible QA metrics.

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio_cut.api import separate_and_segment  # noqa: E402
from audio_cut.qa_report import build_qa_report  # noqa: E402


REQUIRED_CATEGORIES: Dict[str, int] = {
    "chinese_ballad": 3,
    "chinese_fast_rap": 3,
    "english_pop": 3,
    "folk_low_dynamic": 3,
    "strong_chorus": 3,
    "harmony_adlib": 3,
    "long_intro_outro": 2,
}

THRESHOLDS: Dict[str, float] = {
    "boundary_f1_500ms": 0.82,
    "cut_inside_word_rate": 0.01,
    "cut_inside_high_conf_singing_rate": 0.03,
    "segment_5_15_pass_rate": 0.90,
    "subjective_naturalness": 4.2,
    "manual_recutter_rate_reduction": 0.40,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for reproducible VPBD ASR acceptance checks."""

    parser = argparse.ArgumentParser(description="Run or summarize VPBD ASR acceptance playlists.")
    parser.add_argument(
        "--playlist",
        default="docs/vpbd_asr_acceptance_playlist.local.json",
        help="Acceptance playlist JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/vpbd_asr_acceptance",
        help="Directory for generated manifests and acceptance_report.json.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run audio_cut.api.separate_and_segment for tracks without manifest_path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any gate fails or has insufficient data.",
    )
    parser.add_argument(
        "--export-audio",
        action="store_true",
        help="Export audio segments while running. Defaults to manifest-only execution.",
    )
    parser.add_argument(
        "--review-csv",
        default=None,
        help="Optional manual review CSV path. Relative paths are written under output-dir.",
    )
    args = parser.parse_args(argv)

    playlist_path = (PROJECT_ROOT / args.playlist).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    report = build_acceptance_report(
        playlist_path=playlist_path,
        output_dir=output_dir,
        run=bool(args.run),
        export_audio=bool(args.export_audio),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "acceptance_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    response = {"status": report["status"], "report_path": str(report_path)}
    if args.review_csv:
        review_path = Path(args.review_csv)
        if not review_path.is_absolute():
            review_path = output_dir / review_path
        write_review_csv(report, review_path)
        response["review_csv_path"] = str(review_path)
    print(json.dumps(response, ensure_ascii=False))
    return 1 if args.strict and report["status"] != "pass" else 0


def build_acceptance_report(
    *,
    playlist_path: Path,
    output_dir: Path,
    run: bool,
    export_audio: bool,
) -> Dict[str, Any]:
    """Build a complete acceptance report from a playlist and optional generated manifests."""

    playlist = _read_json(playlist_path)
    tracks = [item for item in playlist.get("tracks", []) if isinstance(item, Mapping)]
    runtime_overrides = _runtime_overrides(playlist)
    processed: List[Dict[str, Any]] = []
    manifests: List[Mapping[str, Any]] = []

    for index, track in enumerate(tracks, start=1):
        item = _track_base(track, index)
        audio_path = _audio_path(track)
        item["audio_exists"] = audio_path.exists()
        manifest_path = _manifest_path(track, output_dir, item["id"])
        item["manifest_path"] = manifest_path.as_posix()
        manifest = _load_or_run_manifest(
            track=track,
            output_dir=output_dir,
            runtime_overrides=runtime_overrides,
            run=run,
            export_audio=export_audio,
        )
        if manifest is None:
            item["status"] = "missing_audio" if not audio_path.exists() else "missing_manifest"
        else:
            item["status"] = "manifest_loaded"
            qa_report = dict(manifest.get("qa_report") or build_qa_report(manifest))
            item["qa_report"] = qa_report
            item["auto_metrics"] = _track_auto_metrics(manifest, qa_report)
            manifests.append(manifest)
        processed.append(item)

    category_report = _category_report(processed)
    metrics = _aggregate_metrics(processed, manifests)
    gates = _build_gates(category_report, metrics)
    status = "pass" if all(gate["status"] == "pass" for gate in gates.values()) else "incomplete"

    return {
        "status": status,
        "playlist": str(playlist_path),
        "track_count": len(processed),
        "manifest_count": len(manifests),
        "required_categories": dict(REQUIRED_CATEGORIES),
        "categories": category_report,
        "metrics": metrics,
        "gates": gates,
        "tracks": processed,
    }


REVIEW_CSV_FIELDS: Tuple[str, ...] = (
    "track_id",
    "category",
    "title",
    "audio_path",
    "manifest_path",
    "status",
    "segments",
    "words",
    "cuts",
    "cut_inside_word_rate",
    "cut_inside_high_conf_singing_rate",
    "segment_5_15_pass_rate",
    "reference_boundaries_s",
    "subjective_naturalness",
    "baseline_manual_recutter_rate",
    "current_manual_recutter_rate",
    "notes",
)


def write_review_csv(report: Mapping[str, Any], path: Path) -> None:
    """Write one manual review row per acceptance track."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_CSV_FIELDS))
        writer.writeheader()
        for track in report.get("tracks", []) or []:
            if isinstance(track, Mapping):
                writer.writerow(_review_csv_row(track))


def _review_csv_row(track: Mapping[str, Any]) -> Dict[str, str]:
    auto = track.get("auto_metrics") if isinstance(track.get("auto_metrics"), Mapping) else {}
    manual_scores = track.get("manual_scores") if isinstance(track.get("manual_scores"), Mapping) else {}
    manual_metrics = track.get("manual_metrics") if isinstance(track.get("manual_metrics"), Mapping) else {}
    return {
        "track_id": _cell(track.get("id")),
        "category": _cell(track.get("category")),
        "title": _cell(track.get("title")),
        "audio_path": _cell(track.get("path")),
        "manifest_path": _cell(track.get("manifest_path")),
        "status": _cell(track.get("status")),
        "segments": _cell(auto.get("segments")),
        "words": _cell(auto.get("words")),
        "cuts": _cell(auto.get("cuts")),
        "cut_inside_word_rate": _cell(auto.get("cut_inside_word_rate")),
        "cut_inside_high_conf_singing_rate": _cell(auto.get("cut_inside_high_conf_singing_rate")),
        "segment_5_15_pass_rate": _cell(auto.get("segment_5_15_pass_rate")),
        "reference_boundaries_s": _json_cell(track.get("reference_boundaries_s")),
        "subjective_naturalness": _cell(manual_scores.get("subjective_naturalness")),
        "baseline_manual_recutter_rate": _cell(manual_metrics.get("baseline_manual_recutter_rate")),
        "current_manual_recutter_rate": _cell(manual_metrics.get("current_manual_recutter_rate")),
        "notes": _cell(track.get("notes")),
    }


def _load_or_run_manifest(
    *,
    track: Mapping[str, Any],
    output_dir: Path,
    runtime_overrides: Mapping[str, Any],
    run: bool,
    export_audio: bool,
) -> Optional[Mapping[str, Any]]:
    manifest_path = _manifest_path(track, output_dir, str(track.get("id") or "track"))
    if manifest_path.exists() and not run:
        return _read_json(manifest_path)
    if not run:
        return None

    audio_path = _audio_path(track)
    if not audio_path.exists():
        return None
    track_output_dir = manifest_path.parent
    export_types: Optional[Sequence[str]] = None if export_audio else ("none",)
    manifest = separate_and_segment(
        input_uri=str(audio_path),
        export_dir=str(track_output_dir),
        mode=str(track.get("mode", "vpbd_asr")),
        export_types=export_types,
        export_manifest=True,
        manifest_filename=manifest_path.name,
        runtime_overrides=runtime_overrides,
    )
    return manifest


def _aggregate_metrics(
    processed: Sequence[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    cut_count = 0
    word_count = 0
    high_conf_singing_count = 0
    inside_word_count = 0
    inside_high_conf_singing_count = 0
    segment_durations: List[float] = []
    boundary_tp = boundary_fp = boundary_fn = 0

    for track, manifest in zip((item for item in processed if item.get("status") == "manifest_loaded"), manifests):
        duration = _duration_s(manifest)
        track_cuts = _internal_cuts(manifest, duration)
        track_words = _intervals(_timeline_items(manifest, "words"))
        track_singing_regions = _intervals(
            _timeline_items(manifest, "vad_regions"), min_confidence=0.8
        )
        cut_count += len(track_cuts)
        word_count += len(track_words)
        high_conf_singing_count += len(track_singing_regions)
        inside_word_count += _inside_count(track_cuts, track_words)
        inside_high_conf_singing_count += _inside_count(track_cuts, track_singing_regions)
        segment_durations.extend(_segment_durations(manifest))
        refs = [float(value) for value in track.get("reference_boundaries_s", []) or []]
        if refs:
            tp, fp, fn = _boundary_counts(track_cuts, refs, tolerance_s=0.5)
            boundary_tp += tp
            boundary_fp += fp
            boundary_fn += fn

    subjective_scores = [
        _float(value)
        for value in (
            _nested(item, ("manual_scores", "subjective_naturalness"))
            for item in processed
        )
        if value is not None
    ]
    baseline_rate = _first_number(processed, ("manual_metrics", "baseline_manual_recutter_rate"))
    current_rate = _first_number(processed, ("manual_metrics", "current_manual_recutter_rate"))

    return {
        "boundary_f1_500ms": _f1(boundary_tp, boundary_fp, boundary_fn),
        "cut_inside_word_rate": _rate(inside_word_count, cut_count),
        "cut_inside_high_conf_singing_rate": _rate(inside_high_conf_singing_count, cut_count),
        "segment_5_15_pass_rate": _rate(
            sum(1 for value in segment_durations if 5.0 <= value <= 15.0),
            len(segment_durations),
        ),
        "subjective_naturalness": _average(subjective_scores),
        "manual_recutter_rate_reduction": _recutter_reduction(baseline_rate, current_rate),
        "evidence": {
            "cuts": cut_count,
            "words": word_count,
            "high_conf_singing_regions": high_conf_singing_count,
            "segments": len(segment_durations),
            "reference_boundaries": boundary_tp + boundary_fn,
            "subjective_scores": len(subjective_scores),
            "manual_recutter_rates_present": baseline_rate is not None and current_rate is not None,
        },
    }


def _track_auto_metrics(manifest: Mapping[str, Any], qa_report: Mapping[str, Any]) -> Dict[str, Any]:
    duration = _duration_s(manifest)
    cuts = _internal_cuts(manifest, duration)
    words = _intervals(_timeline_items(manifest, "words"))
    singing_regions = _intervals(_timeline_items(manifest, "vad_regions"), min_confidence=0.8)
    return {
        "segments": len(_segment_durations(manifest)),
        "words": len(words),
        "cuts": len(cuts),
        "cut_inside_word_rate": qa_report.get("cut_inside_word_rate"),
        "cut_inside_high_conf_singing_rate": _inside_rate(cuts, singing_regions),
        "segment_5_15_pass_rate": qa_report.get("segment_5_15_pass_rate"),
    }


def _build_gates(
    category_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    gates: Dict[str, Dict[str, Any]] = {}
    missing = category_report.get("missing", {})
    gates["playlist_coverage"] = {
        "status": "pass" if not missing else "insufficient_data",
        "missing": missing,
    }
    for key, threshold in THRESHOLDS.items():
        value = metrics.get(key)
        if value is None:
            status = "insufficient_data"
        elif key in {"cut_inside_word_rate", "cut_inside_high_conf_singing_rate"}:
            status = "pass" if float(value) <= threshold else "fail"
        else:
            status = "pass" if float(value) >= threshold else "fail"
        gates[key] = {"status": status, "value": value, "threshold": threshold}
    return gates


def _category_report(tracks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {key: 0 for key in REQUIRED_CATEGORIES}
    extra: Dict[str, int] = {}
    for track in tracks:
        category = str(track.get("category", "")).strip()
        if category in counts:
            counts[category] += 1
        elif category:
            extra[category] = extra.get(category, 0) + 1
    missing = {
        category: required - counts.get(category, 0)
        for category, required in REQUIRED_CATEGORIES.items()
        if counts.get(category, 0) < required
    }
    return {"counts": counts, "extra": extra, "missing": missing}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _runtime_overrides(playlist: Mapping[str, Any]) -> Dict[str, Any]:
    value = playlist.get("runtime_overrides", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _track_base(track: Mapping[str, Any], index: int) -> Dict[str, Any]:
    track_id = str(track.get("id") or f"track_{index:03d}")
    item = {
        "id": track_id,
        "title": str(track.get("title") or track_id),
        "category": str(track.get("category") or ""),
        "path": str(track.get("path") or ""),
    }
    for key in ("reference_boundaries_s", "manual_scores", "manual_metrics"):
        if key in track:
            item[key] = track[key]
    return item


def _manifest_path(track: Mapping[str, Any], output_dir: Path, track_id: str) -> Path:
    configured = track.get("manifest_path")
    if configured:
        return (PROJECT_ROOT / str(configured)).resolve()
    return output_dir / _safe_id(track_id) / "SegmentManifest.json"


def _audio_path(track: Mapping[str, Any]) -> Path:
    return (PROJECT_ROOT / str(track.get("path", ""))).resolve()


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "track"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _json_cell(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False)


def _duration_s(manifest: Mapping[str, Any]) -> Optional[float]:
    return _float(_nested(manifest, ("audio", "duration")))


def _internal_cuts(manifest: Mapping[str, Any], duration_s: Optional[float]) -> List[float]:
    cuts = manifest.get("cuts")
    if not isinstance(cuts, Mapping):
        return []
    values: List[float] = []
    for item in cuts.get("final", []) or []:
        raw = item.get("t") if isinstance(item, Mapping) else item
        t = _float(raw)
        if t is None or t <= 0.0:
            continue
        if duration_s is not None and t >= duration_s:
            continue
        values.append(t)
    return values


def _timeline_items(manifest: Mapping[str, Any], key: str) -> Iterable[Mapping[str, Any]]:
    timeline = _nested(manifest, ("lyrics_alignment", "timeline"))
    if not isinstance(timeline, Mapping):
        return []
    items = timeline.get(key, [])
    return [item for item in items if isinstance(item, Mapping)]


def _intervals(
    items: Iterable[Mapping[str, Any]],
    *,
    min_confidence: Optional[float] = None,
) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    for item in items:
        if min_confidence is not None:
            confidence = _float(item.get("confidence"))
            if confidence is None or confidence < min_confidence:
                continue
        start = _float(item.get("start_s"))
        end = _float(item.get("end_s"))
        if start is not None and end is not None and end > start:
            ranges.append((start, end))
    return ranges


def _segment_durations(manifest: Mapping[str, Any]) -> List[float]:
    segments = manifest.get("segments", [])
    durations: List[float] = []
    for item in segments:
        if not isinstance(item, Mapping):
            continue
        value = _float(item.get("duration"))
        if value is not None:
            durations.append(value)
    return durations


def _boundary_counts(
    predicted: Sequence[float],
    reference: Sequence[float],
    *,
    tolerance_s: float,
) -> Tuple[int, int, int]:
    unused = set(range(len(reference)))
    tp = 0
    for cut in predicted:
        best_idx = None
        best_dist = tolerance_s
        for idx in unused:
            dist = abs(cut - reference[idx])
            if dist <= best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is not None:
            unused.remove(best_idx)
            tp += 1
    fp = len(predicted) - tp
    fn = len(unused)
    return tp, fp, fn


def _inside_count(cuts: Sequence[float], intervals: Sequence[Tuple[float, float]]) -> int:
    return sum(1 for cut in cuts if any(start < cut < end for start, end in intervals))


def _inside_rate(cuts: Sequence[float], intervals: Sequence[Tuple[float, float]]) -> Optional[float]:
    if not cuts:
        return None
    inside = sum(1 for cut in cuts if any(start < cut < end for start, end in intervals))
    return _rate(inside, len(cuts))


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 12)


def _f1(tp: int, fp: int, fn: int) -> Optional[float]:
    denominator = (2 * tp) + fp + fn
    if denominator <= 0:
        return None
    return round((2 * tp) / denominator, 12)


def _average(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 12)


def _recutter_reduction(
    baseline_rate: Optional[float],
    current_rate: Optional[float],
) -> Optional[float]:
    if baseline_rate is None or current_rate is None or baseline_rate <= 0.0:
        return None
    return round((baseline_rate - current_rate) / baseline_rate, 12)


def _first_number(
    items: Sequence[Mapping[str, Any]],
    path: Sequence[str],
) -> Optional[float]:
    for item in items:
        value = _float(_nested(item, path))
        if value is not None:
            return value
    return None


def _nested(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

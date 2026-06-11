#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/vpbd_rollback_diff_gate.py
# AI-SUMMARY: Verifies VPBD legacy candidate-pool rollback output against the v2.6 baseline.

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.legacy_mode_diff_gate import SEGMENT_PATH_RE, ensure_baseline_worktree


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for VPBD rollback diff verification."""

    parser = argparse.ArgumentParser(description="Diff VPBD legacy rollback output against v2.6 baseline.")
    parser.add_argument("--baseline-ref", default="8271984", help="Git ref used as v2.6 baseline.")
    parser.add_argument(
        "--baseline-worktree",
        default="/tmp/audio-cut-v2_6_baseline",
        help="Detached worktree path for the baseline ref.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input audio used for the VPBD rollback smoke diff.",
    )
    parser.add_argument(
        "--lyrics-fixture",
        default="tests/fixtures/lyrics/simple_song_timeline.json",
        help="Fake lyrics timeline fixture used for vpbd_asr.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/v2_7_h_vpbd_rollback_diff",
        help="Directory for generated current/baseline outputs and report.",
    )
    parser.add_argument("--skip-run", action="store_true", help="Only diff existing output manifests.")
    args = parser.parse_args(argv)

    output_dir = _project_path(args.output_dir)
    input_path = _project_path(args.input)
    lyrics_fixture = _project_path(args.lyrics_fixture)
    if not input_path.exists():
        raise FileNotFoundError(f"input audio not found: {input_path}")
    if not lyrics_fixture.exists():
        raise FileNotFoundError(f"lyrics fixture not found: {lyrics_fixture}")

    baseline_worktree = Path(args.baseline_worktree).expanduser().resolve()
    if not args.skip_run:
        baseline_repo = ensure_baseline_worktree(args.baseline_ref, baseline_worktree)
        run_vpbd(
            repo_path=baseline_repo,
            input_path=input_path,
            export_dir=output_dir / "baseline",
            lyrics_fixture=lyrics_fixture,
            current=False,
        )
        run_vpbd(
            repo_path=PROJECT_ROOT,
            input_path=input_path,
            export_dir=output_dir / "current",
            lyrics_fixture=lyrics_fixture,
            current=True,
        )

    baseline = _read_manifest(output_dir / "baseline" / "SegmentManifest.json")
    current = _read_manifest(output_dir / "current" / "SegmentManifest.json")
    issues = diff_vpbd_rollback_manifests(baseline, current)
    report = {
        "status": "pass" if not issues else "fail",
        "baseline_ref": args.baseline_ref,
        "baseline_manifest": str(output_dir / "baseline" / "SegmentManifest.json"),
        "current_manifest": str(output_dir / "current" / "SegmentManifest.json"),
        "baseline_segments": len(baseline.get("segments", []) or []),
        "current_segments": len(current.get("segments", []) or []),
        "baseline_cuts_samples": baseline.get("cuts", {}).get("samples", []),
        "current_cuts_samples": current.get("cuts", {}).get("samples", []),
        "issues": issues,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "vpbd_rollback_diff_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path)}, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


def run_vpbd(
    *,
    repo_path: Path,
    input_path: Path,
    export_dir: Path,
    lyrics_fixture: Path,
    current: bool,
) -> None:
    """Run vpbd_asr through the public API with pop-profile rollback settings."""

    export_dir.mkdir(parents=True, exist_ok=True)
    extra_overrides: Dict[str, Any] = {
        "lyrics_alignment.enabled": True,
        "lyrics_alignment.provider": "fake",
        "lyrics_alignment.fixture_path": str(lyrics_fixture),
    }
    if current:
        extra_overrides.update(
            {
                "smart_cut.profile": "pop",
                "vpbd.candidate_pool": "legacy",
            }
        )
    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "repo=Path(sys.argv[1]).resolve()\n"
        "sys.path[:0]=[str(repo), str(repo/'src')]\n"
        "from audio_cut.api import separate_and_segment\n"
        "from audio_cut.config.derive import apply_profile_overrides\n"
        "_, profile_overrides = apply_profile_overrides('pop')\n"
        "runtime_overrides = dict(profile_overrides)\n"
        "runtime_overrides.update(json.loads(sys.argv[4]))\n"
        "manifest=separate_and_segment(\n"
        "    input_uri=sys.argv[2], export_dir=sys.argv[3], mode='vpbd_asr',\n"
        "    device='cpu', export_types=('music_segments',), export_manifest=True,\n"
        "    runtime_overrides=runtime_overrides,\n"
        ")\n"
        "print(json.dumps({'success': manifest.get('success'), 'segments': len(manifest.get('segments', []))}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_path}:{repo_path / 'src'}"
    subprocess.run(
        [sys.executable, "-c", code, str(repo_path), str(input_path), str(export_dir), json.dumps(extra_overrides)],
        cwd=repo_path,
        env=env,
        check=True,
    )


def diff_vpbd_rollback_manifests(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> List[str]:
    """Return compatibility issues between baseline and current VPBD rollback manifests."""

    issues: List[str] = []
    baseline_keys = set(baseline.keys()) - {"manifest_path"}
    current_keys = set(current.keys()) - {"manifest_path"}
    if baseline_keys != current_keys:
        issues.append(f"top-level keys differ: missing={sorted(baseline_keys - current_keys)} extra={sorted(current_keys - baseline_keys)}")
    if "auto_profile" in current:
        issues.append("current rollback manifest unexpectedly includes auto_profile")
    if (current.get("boundary_detection") or {}).get("candidate_pool") != "legacy":
        issues.append("current boundary_detection.candidate_pool is not legacy")
    if baseline.get("cuts", {}).get("samples") != current.get("cuts", {}).get("samples"):
        issues.append("cut samples differ")
    baseline_segments = list(baseline.get("segments", []) or [])
    current_segments = list(current.get("segments", []) or [])
    if len(baseline_segments) != len(current_segments):
        issues.append(f"segment count changed: {len(baseline_segments)} -> {len(current_segments)}")
    for index, (baseline_segment, current_segment) in enumerate(zip(baseline_segments, current_segments), start=1):
        baseline_signature = (baseline_segment.get("label"), baseline_segment.get("mix_path"))
        current_signature = (current_segment.get("label"), current_segment.get("mix_path"))
        if baseline_signature != current_signature:
            issues.append(f"segment {index} output differs: {baseline_signature} -> {current_signature}")
        path = current_segment.get("mix_path")
        if (
            path is not None
            and baseline_signature != current_signature
            and not SEGMENT_PATH_RE.match(str(path))
        ):
            issues.append(f"segment {index} mix_path has unexpected name: {path}")
    return issues


def _read_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"manifest root must be an object: {path}")
    return value


def _project_path(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


if __name__ == "__main__":
    raise SystemExit(main())

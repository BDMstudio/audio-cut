#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/legacy_mode_diff_gate.py
# AI-SUMMARY: Runs legacy split modes against a baseline ref and diffs Manifest/naming contracts.

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = ("v2.2_mdd", "hybrid_mdd", "librosa_onset")
FORBIDDEN_LEGACY_FIELDS = ("lyrics_alignment", "boundary_detection", "auto_profile")
SEGMENT_PATH_RE = re.compile(
    r"^(?:segments_vocal/)?segment_\d{3}_(?:human|music)(?:_lib)?(?:_vocal)?_\d+(?:\.\d+)?\.(?:wav|mp3)$"
)


@dataclass(frozen=True)
class GateIssue:
    """One legacy diff gate violation."""

    mode: str
    message: str


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the legacy mode diff gate."""

    parser = argparse.ArgumentParser(description="Run and diff legacy mode Manifest contracts.")
    parser.add_argument("--baseline-ref", default="8271984", help="Git ref used as v2.6 baseline.")
    parser.add_argument(
        "--baseline-worktree",
        default="/tmp/audio-cut-v2_6_baseline",
        help="Detached worktree path for the baseline ref.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input audio used for the legacy mode smoke diff.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/v2_7_h_legacy_diff",
        help="Directory for generated current/baseline outputs and report.",
    )
    parser.add_argument("--mode", action="append", choices=DEFAULT_MODES, help="Mode to run. Repeatable.")
    parser.add_argument(
        "--export-types",
        default="music_segments",
        help="Comma-separated export types passed to audio_cut.api.separate_and_segment.",
    )
    parser.add_argument("--skip-run", action="store_true", help="Only diff existing output manifests.")
    args = parser.parse_args(argv)

    modes = tuple(args.mode or DEFAULT_MODES)
    input_path = _project_path(args.input)
    output_dir = _project_path(args.output_dir)
    baseline_worktree = Path(args.baseline_worktree).expanduser().resolve()
    export_types = tuple(item.strip() for item in args.export_types.split(",") if item.strip())

    if not input_path.exists():
        raise FileNotFoundError(f"input audio not found: {input_path}")

    baseline_repo = baseline_worktree
    if not args.skip_run:
        baseline_repo = ensure_baseline_worktree(args.baseline_ref, baseline_worktree)
        for repo_label, repo_path in (("baseline", baseline_repo), ("current", PROJECT_ROOT)):
            for mode in modes:
                run_mode(
                    repo_path=repo_path,
                    input_path=input_path,
                    export_dir=output_dir / repo_label / mode,
                    mode=mode,
                    export_types=export_types,
                )

    report = build_report(output_dir=output_dir, modes=modes, baseline_ref=args.baseline_ref)
    report_path = output_dir / "legacy_mode_diff_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path)}, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


def ensure_baseline_worktree(ref: str, worktree: Path) -> Path:
    """Ensure a detached baseline worktree exists at the requested ref."""

    if worktree.exists():
        head = _git(["rev-parse", "HEAD"], cwd=worktree).strip()
        expected = _git(["rev-parse", ref], cwd=PROJECT_ROOT).strip()
        if head != expected:
            raise RuntimeError(f"baseline worktree {worktree} is at {head}, expected {expected}")
        _ensure_baseline_model_assets(worktree)
        return worktree

    worktree.parent.mkdir(parents=True, exist_ok=True)
    _git(["worktree", "add", "--detach", str(worktree), ref], cwd=PROJECT_ROOT)
    _ensure_baseline_model_assets(worktree)
    return worktree


def _ensure_baseline_model_assets(worktree: Path) -> None:
    """Expose local MDX model assets to a detached baseline worktree."""

    source_root = PROJECT_ROOT / "MVSEP-MDX23-music-separation-model"
    target_root = worktree / "MVSEP-MDX23-music-separation-model"
    source_models = source_root / "models"
    target_models = target_root / "models"
    if not source_models.exists():
        return
    target_models.mkdir(parents=True, exist_ok=True)
    for name in ("Kim_Vocal_1.onnx", "Kim_Vocal_2.onnx", "Kim_Inst.onnx"):
        source = source_models / name
        target = target_models / name
        if not source.exists() or target.exists():
            continue
        target.symlink_to(source)
    for name in ("inference.py",):
        source = source_root / name
        target = target_root / name
        if not source.exists() or target.exists():
            continue
        target.symlink_to(source)


def run_mode(
    *,
    repo_path: Path,
    input_path: Path,
    export_dir: Path,
    mode: str,
    export_types: Sequence[str],
) -> None:
    """Run one split mode through the public API and write SegmentManifest.json."""

    export_dir.mkdir(parents=True, exist_ok=True)
    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "repo=Path(sys.argv[1]).resolve()\n"
        "sys.path[:0]=[str(repo), str(repo/'src')]\n"
        "from audio_cut.api import separate_and_segment\n"
        "export_types=tuple(item for item in sys.argv[5].split(',') if item)\n"
        "manifest=separate_and_segment(\n"
        "    input_uri=sys.argv[2], export_dir=sys.argv[3], mode=sys.argv[4],\n"
        "    device='cpu', export_types=export_types, export_manifest=True,\n"
        ")\n"
        "print(json.dumps({'success': manifest.get('success'), 'segments': len(manifest.get('segments', []))}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_path}:{repo_path / 'src'}"
    subprocess.run(
        [sys.executable, "-c", code, str(repo_path), str(input_path), str(export_dir), mode, ",".join(export_types)],
        cwd=repo_path,
        env=env,
        check=True,
    )


def build_report(*, output_dir: Path, modes: Sequence[str], baseline_ref: str) -> Dict[str, Any]:
    """Build a diff report for generated baseline/current manifests."""

    mode_reports: Dict[str, Any] = {}
    issues: List[GateIssue] = []
    for mode in modes:
        baseline_path = output_dir / "baseline" / mode / "SegmentManifest.json"
        current_path = output_dir / "current" / mode / "SegmentManifest.json"
        baseline = _read_manifest(baseline_path)
        current = _read_manifest(current_path)
        mode_issues = diff_manifests(mode, baseline, current)
        issues.extend(mode_issues)
        mode_reports[mode] = {
            "baseline_manifest": str(baseline_path),
            "current_manifest": str(current_path),
            "baseline_segments": len(baseline.get("segments", []) or []),
            "current_segments": len(current.get("segments", []) or []),
            "issues": [issue.message for issue in mode_issues],
        }
    return {
        "status": "pass" if not issues else "fail",
        "baseline_ref": baseline_ref,
        "modes": mode_reports,
        "issues": [{"mode": issue.mode, "message": issue.message} for issue in issues],
    }


def diff_manifests(mode: str, baseline: Mapping[str, Any], current: Mapping[str, Any]) -> List[GateIssue]:
    """Compare legacy Manifest shape and filename contracts."""

    issues: List[GateIssue] = []
    baseline_keys = set(baseline.keys()) - {"manifest_path"}
    current_keys = set(current.keys()) - {"manifest_path"}
    if baseline_keys != current_keys:
        issues.append(
            GateIssue(
                mode,
                f"top-level keys differ: missing={sorted(baseline_keys - current_keys)} extra={sorted(current_keys - baseline_keys)}",
            )
        )

    for field in FORBIDDEN_LEGACY_FIELDS:
        if field in current:
            issues.append(GateIssue(mode, f"legacy mode unexpectedly includes {field}"))

    if current.get("version") != baseline.get("version"):
        issues.append(GateIssue(mode, f"version changed: {baseline.get('version')} -> {current.get('version')}"))

    baseline_segments = list(baseline.get("segments", []) or [])
    current_segments = list(current.get("segments", []) or [])
    if len(baseline_segments) != len(current_segments):
        issues.append(GateIssue(mode, f"segment count changed: {len(baseline_segments)} -> {len(current_segments)}"))

    issues.extend(_diff_segment_shape(mode, baseline_segments, current_segments))
    issues.extend(_diff_artifact_shape(mode, baseline.get("artifacts", {}), current.get("artifacts", {})))
    return issues


def _diff_segment_shape(
    mode: str,
    baseline_segments: Sequence[Mapping[str, Any]],
    current_segments: Sequence[Mapping[str, Any]],
) -> List[GateIssue]:
    issues: List[GateIssue] = []
    for index, (baseline, current) in enumerate(zip(baseline_segments, current_segments), start=1):
        baseline_keys = set(baseline.keys())
        current_keys = set(current.keys())
        if baseline_keys != current_keys:
            issues.append(
                GateIssue(
                    mode,
                    f"segment {index} keys differ: missing={sorted(baseline_keys - current_keys)} extra={sorted(current_keys - baseline_keys)}",
                )
            )
        for key in ("mix_path", "vocal_path"):
            baseline_value = baseline.get(key)
            path_value = current.get(key)
            if mode != "hybrid_mdd" and baseline_value != path_value:
                issues.append(GateIssue(mode, f"segment {index} {key} changed: {baseline_value} -> {path_value}"))
            if (
                path_value is not None
                and baseline_value != path_value
                and not SEGMENT_PATH_RE.match(str(path_value))
            ):
                issues.append(GateIssue(mode, f"segment {index} {key} has unexpected name: {path_value}"))
    return issues


def _diff_artifact_shape(mode: str, baseline_artifacts: Any, current_artifacts: Any) -> List[GateIssue]:
    if not isinstance(baseline_artifacts, Mapping) or not isinstance(current_artifacts, Mapping):
        return [GateIssue(mode, "artifacts must be manifest objects")]

    issues: List[GateIssue] = []
    baseline_keys = set(baseline_artifacts.keys()) - {"output_dir"}
    current_keys = set(current_artifacts.keys()) - {"output_dir"}
    if baseline_keys != current_keys:
        issues.append(
            GateIssue(
                mode,
                f"artifact keys differ: missing={sorted(baseline_keys - current_keys)} extra={sorted(current_keys - baseline_keys)}",
            )
        )
    for key, values in current_artifacts.items():
        if key == "output_dir" or not isinstance(values, list):
            continue
        baseline_values = baseline_artifacts.get(key)
        if mode != "hybrid_mdd" and baseline_values != values:
            issues.append(GateIssue(mode, f"artifact {key} changed: {baseline_values} -> {values}"))
        for value in values:
            if baseline_values != values and not SEGMENT_PATH_RE.match(str(value)):
                issues.append(GateIssue(mode, f"artifact {key} has unexpected name: {value}"))
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


def _git(args: Sequence[str], *, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


if __name__ == "__main__":
    raise SystemExit(main())

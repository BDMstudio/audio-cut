# audio-cut v2.7 Unified Cut Plan

## Objective
Move audio-cut toward v2.7's unified cut engine while preserving v2.6 behavior behind explicit rollback switches.

## Current Scope
- Branch: `codex/v2.7-unified-cut`.
- Source specs:
  - `docs/audio_cut_v2_7_unified_cut_proposal.md`
  - `docs/audio_cut_v2_7_unified_cut_todo.md`
- Immediate milestone: A -> B, shipping the v2.6.1 hybrid_mdd bugfix before unified VPBD candidate-pool work.

## Non-Goals For This Checkpoint
- Do not start C/D unified candidate-pool work before B has tests and rollback switches.
- Do not mark A5/A6 complete without manual listening and a reproducible ASR Manifest.
- Do not commit temporary real song names or artist names into user-facing docs.

## Execution Order
1. Freeze automated v2.6 baseline.
2. Preserve 2-3 real hybrid_mdd before outputs when original chorus-dense inputs are available. Done: `output/v2_7_a4_before/sample_01_hybrid_mdd`, `sample_02_hybrid_mdd`, `sample_03_hybrid_mdd`.
3. Add failing tests for hybrid_mdd VAD protection.
4. Implement minimal B1-B5 fixes:
   - pass `vocal_track` into hybrid strategies;
   - make `vad_protection` block unsafe beat snaps using vocal-track energy;
   - guard high-density beat insertions;
   - add `chorus_force_snap` rollback;
   - clamp/default `snap_tolerance_ms`.
5. Run B/H verification gates before any merge.

## Verification
- A baseline:
  - `venv/bin/python -m pytest -s -m "not slow and not gpu and not firered"` -> 108 passed, 1 deselected in 11.54s.
  - `venv/bin/python -m pytest -s tests/contracts/test_config_contracts.py` -> 2 passed in 0.83s.
  - `venv/bin/python -m pytest -s tests/unit/test_cpu_baseline_perfect_reconstruction.py` -> 1 passed in 2.39s.
  - A4 before manifests: sample_01 -> 29 segments, max 15.16s; sample_02 -> 12 segments, max 21.10s; sample_03 -> 15 segments, max 54.53s.
  - A5 stopgap manifests: sample_01 -> 29 segments, max 15.16s; sample_02 -> 12 segments, max 20.95s; sample_03 -> 15 segments, max 54.53s. Manual listening still pending.
  - A6 VPBD ASR CLI strict manifest: `output/v2_7_a6_vpbd_asr_cli/sample_02_vpbd_asr_cli/SegmentManifest.json` -> provider `firered_cli`, fallback_reason null, 12 segments, 9 segments with lyrics; sidecar health refused connection.
  - B after manifests: sample_01 -> 29 segments, max 14.37s, guard_count 28; sample_02 -> 12 segments, max 21.59s, guard_count 11; sample_03 -> 15 segments, max 54.87s, guard_count 14. Precision guard is still false on these smokes, so manual/quality acceptance is not complete.
- Required before B completion:
  - `venv/bin/python -m pytest -s tests/unit/test_snap_to_beat_vad_guard.py`
  - `venv/bin/python -m pytest -s tests/unit/test_cutting_consistency.py`
  - `venv/bin/python -m pytest -s tests/unit/test_legacy_mode_regression.py`
  - `venv/bin/python -m pytest -s -m "not slow and not gpu and not firered"`

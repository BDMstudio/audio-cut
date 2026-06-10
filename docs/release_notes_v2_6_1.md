<!-- File: docs/release_notes_v2_6_1.md -->
<!-- AI-SUMMARY: v2.6.1 hybrid_mdd bugfix release notes, documenting vocal-safe beat snapping and rollback switches. -->

# Release Notes v2.6.1 Draft

## Scope

This draft is a focused `hybrid_mdd` bugfix release. It does not introduce the v2.7 unified candidate pool yet.

## Changes

- `hybrid_mdd.vad_protection` now affects beat snapping decisions. `snap_to_beat` and `beat_only` check the separated vocal track before accepting beat-aligned cuts in high-energy regions.
- `snap_to_beat` now clamps `snap_tolerance_ms` to at most `0.4 * beat_interval`; the default is reduced from 500ms to 200ms.
- `hybrid_mdd.chorus_force_snap` is added as an explicit rollback switch. Set it to `true` to restore the legacy chorus force-snap behavior.
- `hybrid_mdd` strategy output now passes through the shared guard/refine chain before export. Manifest output includes guard shift stats and adjustments.
- `_lib` segment markers are remapped from raw strategy cut points to the nearest guard-refined cut boundary.

## Compatibility

- Public API signatures are unchanged.
- Existing mode names are unchanged.
- Users who need the old aggressive beat alignment can set:

```yaml
hybrid_mdd:
  vad_protection: false
  chorus_force_snap: true
```

## Verification

- `venv/bin/python -m pytest -s tests/unit/test_snap_to_beat_vad_guard.py` -> 6 passed.
- `venv/bin/python -m pytest -s tests/contracts/test_config_contracts.py` -> 2 passed.
- `venv/bin/python -m pytest -s tests/unit/test_cutting_consistency.py tests/unit/test_legacy_mode_regression.py` -> passed.
- `venv/bin/python -m pytest -s tests/unit/test_cpu_baseline_perfect_reconstruction.py` -> passed.
- `venv/bin/python -m pytest -s -m "not slow and not gpu and not firered"` -> 114 passed, 1 deselected.

## Known Follow-Up

The anonymous A/B smoke outputs confirm the guard chain now records post-snap guard adjustments, but long segment layout can still require v2.7 unified candidate-pool work. Do not treat v2.6.1 as the final v2.7 segmentation quality target.

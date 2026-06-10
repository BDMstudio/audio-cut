<!-- File: docs/release_notes_v2_7_draft.md -->
<!-- AI-SUMMARY: Draft release notes for v2.7 unified cut engine, AutoProfile, and configuration migration. -->

# Release Notes v2.7 Draft

## Status

This is a draft, not a release sign-off. `v2.7.0-beta` still requires the M2 acceptance gates, and final `v2.7.0` still requires the full 20-track manual playlist, AutoProfile accuracy review, and H release gate.

Current blocking evidence:

- `docs/vpbd_asr_acceptance_preflight.md` reports `status=incomplete`.
- The 20-track playlist has full category slots but `missing_audio=20`, `missing_reference_boundaries=20`, and `missing_subjective_scores=20`.
- `output/v2_7_i_acceptance_preflight/acceptance_report.json` has `manifest_count=0`; all metric gates except playlist coverage are `insufficient_data`.
- The coverage form of the H quick regression still requires `pytest-cov`; in this environment pytest rejects `--cov=src --cov-report=term-missing` as unrecognized arguments.

## v2.7.0-beta Scope

The beta scope is the unified candidate-pool and planning behavior that replaces ad hoc cut-point priority with scored candidates and explicit rollback switches.

### Changes

- VPBD now uses a unified candidate pool for acoustic pauses, VPBD-only breath candidates, ASR lyrics gap / sentence-end / mVAD candidates, MDD-affinity candidates, and weak beat candidates in high-energy regions.
- Near-duplicate candidates are fused within the candidate pool; `meta.sources` preserves all contributing sources for debugging.
- `phrase_boundary.weights` now includes `breath`; `inside_word_penalty` is stronger and word-edge aware.
- `vocal_cut_risk`, `mdd_affinity`, and `beat_conflict` are wired into scoring/planning instead of remaining dead configuration.
- QA report metrics now include `breath_cut_ratio` and `beat_aligned_ratio` for manual listening review.
- `vpbd.candidate_pool=legacy` is the explicit rollback switch for v2.6-style acoustic-only candidate planning.

### Compatibility

- Existing mode names remain available.
- `vpbd.candidate_pool=legacy` restores the acoustic-only VPBD candidate pool.
- FireRed remains optional; when sidecar/CLI providers are unavailable and strict mode is false, the pipeline falls back to `vpbd_acoustic`.
- FireRed dependencies are not added to base `requirements.txt` or `setup.py`.

### Beta Verification Checklist

- `venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_asr_fake_provider.py`
- `venv/bin/python -m pytest -s tests/unit/test_qa_report.py`
- `venv/bin/python -m pytest -s tests/contracts/test_config_contracts.py`
- `venv/bin/python -m pytest -s tests/integration/test_pipeline_vpbd_acoustic_fallback.py` (last recorded: 6 passed in 7.33s)
- `venv/bin/python -m pytest -s -m "not slow and not gpu and not firered"` (last recorded after config slimming: 153 passed, 1 deselected)

### Beta Blockers

- M2 acceptance remains incomplete because the local playlist lacks real audio, reference boundaries, words/cuts evidence, and subjective review scores.
- `boundary_f1_500ms`, `cut_inside_word_rate`, `cut_inside_high_conf_singing_rate`, and `segment_5_15_pass_rate` are currently `insufficient_data` on the full 20-track playlist.

## v2.7.0 Final Scope

The final scope adds AutoProfile and configuration migration on top of the beta unified candidate pool.

### Changes

- `smart_cut.profile=auto` estimates style from BPM, global MDD, energy CV, and vocal coverage.
- Manual `--profile ballad|pop|edm|rap` values override auto and remain the rollback path for profile selection.
- Profile anchors are interpolated to avoid hard threshold jumps between styles.
- Manifest output can include optional `auto_profile` metadata with style, confidence, features, and applied overrides.
- `smart_cut.target_duration_s` is the user-facing duration entry point and derives planner/layout/quality duration knobs.
- `config/unified.yaml` is now a slim user surface; `config/expert.yaml` holds advanced defaults and is loaded automatically before unified config.
- Deprecated default keys `bpm_adaptive_core.*` and `vocal_pause_splitting.bpm_adaptive_settings` were removed. VPP pause-stat multipliers live under `pure_vocal_detection.relative_threshold_adaptation.pause_stats_multipliers`.
- `migrate_v2_to_v3.py` warns when removed legacy keys are present.

### Compatibility

- `VSS__...` environment overrides and `set_runtime_config` still use the same dotted config paths.
- Advanced hidden defaults can still be overridden through external config or `VSS__...`.
- Existing output naming and Manifest base fields remain unchanged for legacy modes.

### Final Verification Checklist

- `venv/bin/python -m pytest -s tests/unit/test_auto_profile.py`
- `venv/bin/python -m pytest -s tests/unit/test_smart_cut_duration_derivation.py tests/unit/test_seamless_splitter_auto_profile.py` (last recorded: 5 passed)
- `venv/bin/python -m pytest -s tests/unit/test_config_migration.py tests/unit/test_pause_stats_adaptation_config.py` (last recorded: 3 passed)
- `wc -l config/unified.yaml` (last recorded: 62)
- `venv/bin/python -m pytest -s tests/unit/test_cpu_baseline_perfect_reconstruction.py` (last recorded: 1 passed)
- `rg -n "TODO|FIXME" src tests` (last recorded: no matches)

### Final Blockers

- Full I-stage manual acceptance has not run because the repository does not include the required 20 local acceptance tracks or manual labels.
- AutoProfile accuracy against human style labels is not measured yet.
- Subjective naturalness, manual recutter-rate reduction, `breath_cut_ratio`, and rhythmic `beat_aligned_ratio` still require human review.
- The H gate still needs a successful coverage-enabled quick regression once `pytest-cov` is installed.

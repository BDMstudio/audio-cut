# AutoProfile Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add v2.7 AutoProfile so `--profile auto` estimates a style from cached track features, derives runtime overrides, and records optional Manifest metadata.

**Architecture:** Keep AutoProfile as a small config module that converts `TrackFeatureCache`-like stats into a `StyleEstimate`, interpolates existing Schema v3 profile anchors, and derives the user-facing `smart_cut.target_duration_s` into planner/layout duration settings. CLI and API callers pass the selected profile through runtime overrides; manual profiles still call existing `apply_profile_overrides` and take precedence over auto.

**Tech Stack:** Python dataclasses, existing `audio_cut.config.derive` helpers, `set_runtime_config`, pytest unit/contract tests.

---

### Task 1: Add AutoProfile Estimator

**Files:**
- Create: `src/audio_cut/config/auto_profile.py`
- Test: `tests/unit/test_auto_profile.py`

**Steps:**
1. Write failing tests for four typical feature vectors: ballad, pop, edm, rap.
2. Write failing tests for low-confidence fallback to pop.
3. Implement `StyleEstimate` and `estimate_style(cache)`.
4. Run `pytest tests/unit/test_auto_profile.py -q`.

### Task 2: Add Anchor Interpolation And Style Weights

**Files:**
- Modify: `src/audio_cut/config/auto_profile.py`
- Test: `tests/unit/test_auto_profile.py`

**Steps:**
1. Write failing tests that 95 BPM interpolates between ballad and pop without a step jump.
2. Write failing tests that rhythmic styles increase `beat_affinity`/`breath` while ballad increases `acoustic_pause`/`sentence_end`.
3. Implement `build_auto_profile_overrides(estimate, cut_style="natural")`.
4. Run `pytest tests/unit/test_auto_profile.py -q`.

### Task 3: Add smart_cut Duration Derivation

**Files:**
- Modify: `config/unified.yaml`
- Modify: `src/audio_cut/config/auto_profile.py`
- Test: `tests/unit/test_smart_cut_duration_derivation.py`
- Test: `tests/contracts/test_config_contracts.py`

**Steps:**
1. Write failing tests for `target_duration_s: [5, 12]` deriving planner hard/target, layout soft, and quality max duration values.
2. Add `smart_cut` defaults to `unified.yaml`.
3. Implement `derive_smart_cut_overrides`.
4. Run duration and contract tests.

### Task 4: Wire CLI/API Manifest Metadata

**Files:**
- Modify: `run_splitter.py`
- Modify: `quick_start.py`
- Modify: `src/audio_cut/api.py`
- Test: `tests/unit/test_run_splitter_cli.py`
- Test: `tests/unit/test_quick_start_vpbd.py`
- Test: `tests/unit/test_api_manifest.py`

**Steps:**
1. Extend `--profile` choices to include `auto`; keep manual values as explicit overrides.
2. Add result metadata `auto_profile: {style, confidence, bpm, mdd, applied_overrides}` when auto is applied.
3. Include optional `auto_profile` in Manifest only when present.
4. Run affected CLI/API tests.

### Task 5: Docs, TODO, Regression, Commit

**Files:**
- Modify: `README.md`
- Modify: `development.md`
- Modify: `docs/audio_cut_v2_7_unified_cut_todo.md`

**Steps:**
1. Document `--profile auto`, `smart_cut`, and manual profile priority.
2. Run F section tests and required H gate subset.
3. Commit as one AutoProfile feature commit.

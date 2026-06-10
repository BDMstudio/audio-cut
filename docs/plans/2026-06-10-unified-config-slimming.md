# Unified Config Slimming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Slim `config/unified.yaml` to the v2.7 user-facing surface while preserving runtime behavior through automatically loaded expert defaults and migration coverage.

**Architecture:** Keep `config/unified.yaml` small and user-facing. Move advanced defaults into `config/expert.yaml`; `ConfigManager` loads `expert.yaml` first, then overlays `unified.yaml`, so existing `get_config(...)`, `VSS__...`, and `set_runtime_config(...)` behavior is preserved. Migration tests guard legacy v2 keys and deprecation warnings.

**Tech Stack:** YAML config, Python `ConfigManager`, pytest contracts.

---

### Task 1: Red Tests For Slim Config

**Files:**
- Test: `tests/contracts/test_config_contracts.py`

**Steps:**
1. Add assertions that `config/unified.yaml` has ≤120 lines.
2. Assert user-facing config does not expose `bpm_adaptive_core`, `vocal_pause_splitting.bpm_adaptive_settings`, `valley_scoring`, `advanced_vad`, `enforce_quiet_cut`, or `gpu_pipeline.ort`.
3. Assert `ConfigManager` still resolves expert keys after load.
4. Run contract tests and confirm failure.

### Task 2: Add Expert Defaults Loader

**Files:**
- Create: `config/expert.yaml`
- Modify: `src/vocal_smart_splitter/utils/config_manager.py`

**Steps:**
1. Move expert defaults into `config/expert.yaml`.
2. Load expert config before unified config in `_load_config`.
3. Keep direct merge and `v2_mdd` flattening behavior unchanged.
4. Run contract tests.

### Task 3: Slim Main Unified Config

**Files:**
- Modify: `config/unified.yaml`
- Modify: `src/audio_cut/config/unified.yaml`

**Steps:**
1. Reduce root config to smart_cut, audio, output, logging, basic gpu_pipeline, lyrics_alignment, fire_red, and essential mode rollback switches.
2. Add comments that show each retained parameter's effective mode/code owner.
3. Keep packaged fallback config aligned.
4. Run `wc -l config/unified.yaml`.

### Task 4: Migration Tests

**Files:**
- Test: `tests/unit/test_config_migration.py`
- Modify: `src/audio_cut/config/migrate_v2_to_v3.py`

**Steps:**
1. Add tests that legacy v2 config emits `DeprecationWarning` and maps key old fields.
2. Add a test documenting removed dead config behavior.
3. Implement minimal migration warning/mapping changes if needed.

### Task 5: Docs, Regression, Commit

**Files:**
- Modify: `README.md`
- Modify: `development.md`
- Modify: `audio-cut封装为模块.md`
- Modify: `docs/audio_cut_v2_7_unified_cut_todo.md`

**Steps:**
1. Document the slim user config and expert defaults.
2. Run G validation and H regression subset.
3. Commit as one config slimming change.

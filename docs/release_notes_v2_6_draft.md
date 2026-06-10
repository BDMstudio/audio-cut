<!-- File: docs/release_notes_v2_6_draft.md -->
<!-- AI-SUMMARY: Draft release notes for the v2.6 VPBD + FireRedASR optional path. -->

# v2.6 Draft Release Notes

## Highlights

- Added experimental `vpbd_acoustic` and `vpbd_asr` modes.
- Added lyrics timeline models, fake provider, FireRed sidecar provider and FireRed CLI provider.
- Added VPBD candidate scoring and global cut planning metadata for Manifest consumers.
- Added CLI flags for VPBD ASR:
  - `--lyrics-provider`
  - `--firered-endpoint`
  - `--asr-chunk-s`
  - `--asr-overlap-s`
  - `--asr-strict`
  - `--lyrics-fixture`
- Added quick-start menu entry for VPBD + FireRedASR2S.
- Added `scripts/fireredasr2s_worker.py` to adapt the official FireRedASR2S CLI `result.jsonl` into `lyrics_timeline.json`.
- Added `scripts/check_fireredasr2s_env.py` to report missing FireRedASR2S dependencies and model directories before real smoke tests.

## Compatibility

- Existing `v2.2_mdd`, `hybrid_mdd`, `librosa_onset` and `vocal_separation` modes remain unchanged.
- FireRed dependencies are not part of base `requirements.txt`; sidecar/CLI workers run in an external environment.
- Real FireRed smoke tests are gated by `firered` and `gpu` pytest markers.

## Verification

- `venv/bin/python -m pytest -s tests/unit/test_firered_protocol.py tests/unit/test_firered_cli_provider.py tests/unit/test_firered_sidecar_provider.py tests/unit/test_firered_provider_selection.py`
- `venv/bin/python -m pytest -s tests/unit/test_run_splitter_cli.py tests/unit/test_quick_start_vpbd.py`
- `venv/bin/python -m pytest -s -m "not slow and not gpu"`

## Remaining Work

- Run real FireRed validation against `/home/ubuntu/asr_test` after installing missing `textgrid` and restoring FireRedASR2S pretrained model directories.
- Add QA report metrics for lyrics coverage and cut-inside-word rates.
- Complete manual acceptance playlist before marking v2.6 stable.

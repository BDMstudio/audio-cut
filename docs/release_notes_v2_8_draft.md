# audio-cut v2.8.0-beta Draft Release Notes

## Product Surface Reduction

- `quick_start.py` now asks only file scope plus three user-facing questions: split or separate, segment density, and lyric-to-beat cut style.
- Internal provider/profile/mode branching is hidden from the default user path; old `--mode` values remain available for expert compatibility.
- CLI adds `--segments` and `--align`; Python API adds optional `segments` and `alignment` parameters.

## Intent Dual Track

- `segments`: `few` -> `[10,18]`, `medium` -> `[5,12]`, `many` -> `[3,8]`, or direct numeric ranges.
- `alignment`: `lyric`, `lyric_lean`, `balanced`, `beat_lean`, `beat`, or `0.0-1.0`.
- `alignment=0.5` is an identity point and returns no alignment override.
- Beat preference increases beat candidate/base conflict weights without weakening inside-word or singing-risk penalties.

## Compatibility Commitments

- Existing `separate_and_segment()` parameters remain optional and compatible.
- Calling the API without intent parameters still resolves to the legacy `v2.2_mdd` default.
- Explicit `--mode` / `mode=` always wins over intent routing.
- Legacy Manifest fields are preserved; v2.8 only adds `intent`.
- `smart_cut.target_duration_s` still works and wins when explicitly set with `segments`.
- Deprecated `smart_cut.cut_style` is still read and mapped to the new axes; planned removal is v3.0.

## Agent Contract

Manifests now expose `intent`, optional `segments[*].lyrics`, and `qa_report` together so mvagent / hermes / openclaw style workflows can verify how the request was interpreted and whether the output is usable.

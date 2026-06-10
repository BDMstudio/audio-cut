# VPBD ASR Soft-Prior Split Plan

## Objective
Make VPBD ASR follow the Hybrid MDD ownership model: acoustic/MDD evidence decides candidate cut points, FireRedASR only improves ranking and protects word boundaries.

## Minimal Scope
- Add layout-level secondary split selection for overlong segments:
  - prefer existing acoustic suppressed candidates inside the segment;
  - otherwise search local RMS valleys from `TrackFeatureCache`;
  - boost candidates near ASR word/sentence boundaries;
  - do not use midpoint fallback when no acoustic valley exists.
- Pass ASR boundary times from `SeamlessSplitter` to `refine_layout` only in `vpbd_asr` mode.
- Add focused unit tests for soft-max secondary splitting.

## Verification
- New layout tests must fail before implementation and pass after.
- Run VPBD integration tests.
- Run quick non-GPU regression.
- Re-run the target song with FireRed CLI and inspect Manifest durations/cuts.

# eval300_top8_candidate_evidence_cache_2026_06_26

Candidate-local evidence cache for top-8 assignment diagnostics.
This is infrastructure for future rankers, not a compression result.

## Summary

- rows: `632`
- patch shape: `[632, 6, 64, 192]`
- patch dtype: `float16`
- image size `[width, height]`: `[192, 64]`
- channels: `source, nearest, candidate, candidate_minus_nearest, source_minus_nearest, source_minus_candidate`
- code feature count: `24`
- glyph feature count: `26`
- oracle rows: `77`

## Target Counts

- `keep_current_good_change`: `40`
- `keep_noop`: `480`
- `recover_shortlist_oracle_change`: `104`
- `reject_current_bad_change`: `8`

## Next Use

- Train a candidate scorer over non-nearest rows using patch channels plus code/glyph auxiliary features.
- Keep no-op acceptance as a separate head; this cache is not sufficient by itself for policy promotion.

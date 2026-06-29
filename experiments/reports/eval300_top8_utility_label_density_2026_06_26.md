# eval300_top8_utility_label_density_2026_06_26

Top-8 assignment label-density audit comparing exact assignment oracle labels with a non-deployable Tesseract/PARSeq-safe utility oracle.

| partition | groups | exact changes | groups with safe T cand | safe rows | exact T | safe-best T | safe-best changed | safe-best exact | safe-best different | false vs exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 225 | 63 | 63 | 218 | -87 | -87 | 63 | 225 | 0 | 0 |
| val | 75 | 17 | 17 | 40 | -24 | -24 | 17 | 75 | 0 | 0 |

## Interpretation

Exact oracle-code supervision and direct OCR-utility supervision are not the same target. The safe-best oracle is a non-deployable held-out Tesseract/PARSeq upper control, but it quantifies how often useful utility exists outside the exact assignment oracle label. Use this to design future marginal-utility targets, not as a selector or publication result.

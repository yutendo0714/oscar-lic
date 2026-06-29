# eval300_top8_trainval_source_robust_feature_ablation_2026_06_26

Source leave-out feature/loss ablation for Eval300 top-8 candidate ranking.
This is a diagnostic only: no validation labels are used for promotion and no counted `.oscr` stream is exported.

## Variant Summary

| feature family | loss | features | rank<=1 | rank<=4 | exact changed | wrong | Tesseract delta | PARSeq delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `all_tabular` | `standard` | 218 | 23 | 57 | 23 | 57 | -34 | 0 |
| `all_tabular` | `source_balanced` | 218 | 27 | 53 | 27 | 53 | -42 | -1 |
| `continuous_no_onehot` | `standard` | 90 | 25 | 54 | 25 | 55 | -44 | -1 |
| `continuous_no_onehot` | `source_balanced` | 90 | 22 | 57 | 22 | 58 | -42 | -1 |
| `codegeom_no_onehot` | `standard` | 26 | 26 | 66 | 26 | 54 | -38 | -1 |
| `codegeom_no_onehot` | `source_balanced` | 26 | 23 | 64 | 23 | 57 | -44 | -1 |
| `imgdiff_only` | `standard` | 66 | 29 | 59 | 29 | 51 | -48 | 0 |
| `imgdiff_only` | `source_balanced` | 66 | 31 | 57 | 31 | 49 | -51 | 0 |
| `topk_error_only` | `standard` | 2 | 36 | 67 | 36 | 44 | -44 | -1 |
| `topk_error_only` | `source_balanced` | 2 | 36 | 66 | 36 | 44 | -44 | -1 |

## Interpretation

If source-balanced loss or removing code one-hot improves leave-source-out rank1/rank4, the next candidate scorer should reduce code-identity shortcuts and include source-balanced calibration. If image-only features win, richer OCR-aware image/latent patches are the right evidence path. If all variants remain weak, the bottleneck is representation depth or label diversity rather than loss weighting.

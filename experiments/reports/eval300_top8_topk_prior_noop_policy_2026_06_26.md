# eval300_top8_topk_prior_noop_policy_2026_06_26

Train-tuned no-op policies over the top non-nearest first-stage prior candidate.
This is a table-level diagnostic and does not export counted `.oscr` streams.

## Candidate Prior

- train top-prior exact changed: `30`
- train top-prior false+wrong: `195`
- train top-prior Tesseract delta: `-7`
- val top-prior exact changed: `6`
- val top-prior false+wrong: `69`
- val top-prior Tesseract delta: `3`

## Best Train-Tuned Policies

| score | policy | train changed | train exact changed | train false+wrong | train T | val changed | val exact changed | val false+wrong | val T | val P |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `feature_le_img_source_nearest_edge_density_delta` | `best_train_tesseract_parseq_safe` | 18 | 5 | 13 | -7 | 7 | 1 | 6 | -2 | 0 |
| `feature_le_img_source_variant_edge_density_delta` | `best_train_tesseract_parseq_safe` | 32 | 8 | 24 | -7 | 8 | 1 | 7 | -2 | 0 |
| `feature_ge_img_variant_nearest_signed_abs_ratio` | `best_train_tesseract_parseq_safe` | 80 | 12 | 68 | -10 | 26 | 5 | 21 | -2 | 0 |
| `feature_ge_img_source_nearest_grad_abs_delta` | `best_train_tesseract_parseq_safe` | 143 | 19 | 124 | -6 | 35 | 3 | 32 | -2 | 0 |
| `feature_ge_img_source_variant_grad_abs_delta` | `best_train_tesseract_parseq_safe` | 145 | 19 | 126 | -6 | 36 | 3 | 33 | -2 | 0 |
| `feature_le_img_source_nearest_signed_abs_ratio` | `best_train_tesseract_parseq_safe` | 10 | 2 | 8 | -1 | 5 | 1 | 4 | -1 | 0 |
| `feature_le_img_nearest_max` | `best_train_tesseract_parseq_safe` | 57 | 6 | 51 | -4 | 9 | 1 | 8 | -1 | 0 |
| `feature_le_img_variant_max` | `best_train_tesseract_parseq_safe` | 57 | 6 | 51 | -4 | 9 | 1 | 8 | -1 | 0 |
| `feature_le_img_variant_nearest_signed_mean` | `best_train_tesseract_parseq_safe` | 43 | 10 | 33 | -9 | 12 | 2 | 10 | -1 | 0 |
| `feature_le_codebook_code_index_norm` | `best_train_tesseract_parseq_safe` | 57 | 5 | 52 | -2 | 20 | 2 | 18 | -1 | 0 |
| `feature_le_img_variant_nearest_grad_abs_delta` | `best_train_tesseract_parseq_safe` | 101 | 19 | 82 | -14 | 50 | 4 | 46 | -1 | 0 |
| `feature_ge_codebook_code_abs_mean` | `zero_error_safe` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `feature_ge_codebook_code_abs_mean` | `one_error_parseq_safe` | 2 | 1 | 1 | -3 | 0 | 0 | 0 | 0 | 0 |
| `feature_ge_codebook_code_abs_mean` | `best_train_tesseract_parseq_safe` | 2 | 1 | 1 | -3 | 0 | 0 | 0 | 0 | 0 |
| `feature_ge_codebook_code_index_norm` | `zero_error_safe` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `feature_ge_codebook_code_index_norm` | `one_error_parseq_safe` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `feature_ge_codebook_code_l2` | `zero_error_safe` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `feature_ge_codebook_code_l2` | `one_error_parseq_safe` | 2 | 1 | 1 | -3 | 0 | 0 | 0 | 0 | 0 |
| `feature_ge_codebook_code_l2` | `best_train_tesseract_parseq_safe` | 2 | 1 | 1 | -3 | 0 | 0 | 0 | 0 | 0 |
| `feature_ge_codebook_code_mean` | `zero_error_safe` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Interpretation

This tests whether the source-robust top-k prior from N097 becomes deployable with train-only no-op calibration. Promotion still requires beating the current actual-bitstream -8 Tesseract floor with low false/wrong changes and PARSeq no worse.

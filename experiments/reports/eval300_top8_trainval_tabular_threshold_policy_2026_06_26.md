# eval300_top8_trainval_tabular_threshold_policy_2026_06_26

Train-tuned threshold policies for the Eval300 top-8 tabular ranker.
This is a deployable-style policy diagnostic, not a counted `.oscr` promotion.

## Policy Results

| policy | train changed | train exact changed | train false+wrong | train T | val changed | val exact changed | val false+wrong | val T | val P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero_error_safe | 20 | 20 | 0 | -27 | 1 | 0 | 1 | 0 | 0 |
| one_error_parseq_safe | 36 | 35 | 1 | -46 | 3 | 0 | 3 | 0 | 0 |
| best_train_tesseract_parseq_safe | 68 | 54 | 14 | -71 | 5 | 0 | 5 | 1 | 0 |
| lowest_threshold | 225 | 63 | 162 | -57 | 72 | 7 | 65 | -7 | 0 |
| highest_threshold | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Interpretation

- A deployable policy must beat the current actual-bitstream `-8` Tesseract floor with low false/wrong changes before any `.oscr` promotion attempt.
- This diagnostic reuses N091's ranker family and tunes thresholds only on train-partition groups.

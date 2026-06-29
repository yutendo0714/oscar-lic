# eval300_top8_trainval_group_acceptor_2026_06_26

Group-level no-op acceptors over N091 tabular ranker score features.
This is a policy-calibration diagnostic, not a counted `.oscr` promotion.

## Results

| model | policy | train changed | train exact changed | train false+wrong | train T | val changed | val exact changed | val false+wrong | val T | val P |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_balanced | zero_error_safe | 63 | 63 | 0 | -87 | 2 | 0 | 2 | 0 | 0 |
| logistic_balanced | one_error_parseq_safe | 63 | 63 | 0 | -87 | 2 | 0 | 2 | 0 | 0 |
| logistic_balanced | best_train_tesseract_parseq_safe | 63 | 63 | 0 | -87 | 2 | 0 | 2 | 0 | 0 |
| hist_gradient_boosting | zero_error_safe | 63 | 63 | 0 | -87 | 3 | 0 | 3 | 0 | 0 |
| hist_gradient_boosting | one_error_parseq_safe | 63 | 63 | 0 | -87 | 3 | 0 | 3 | 0 | 0 |
| hist_gradient_boosting | best_train_tesseract_parseq_safe | 63 | 63 | 0 | -87 | 3 | 0 | 3 | 0 | 0 |
| random_forest_balanced | zero_error_safe | 63 | 63 | 0 | -87 | 2 | 0 | 2 | 0 | 0 |
| random_forest_balanced | one_error_parseq_safe | 63 | 63 | 0 | -87 | 2 | 0 | 2 | 0 | 0 |
| random_forest_balanced | best_train_tesseract_parseq_safe | 63 | 63 | 0 | -87 | 2 | 0 | 2 | 0 | 0 |

## Interpretation

- Promotion remains blocked unless a val policy beats the counted current `-8` Tesseract floor with low false/wrong changes.
- Inputs exclude OCR deltas, oracle labels and OCR prediction text; labels are used only for train supervision and evaluation.

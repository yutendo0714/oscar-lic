# Exception Counterfactual Probe Set

This is a selection manifest for future actual compact stream/OCR counterfactual labeling.
Rows with `training_use=false` are validation references and must not be used as training labels.

## Summary

Rows: `61`; training rows: `53`; audit-only rows: `8`.

Tags: `{"train_close_current_only": 30, "train_iam_current_only_near": 8, "train_iam_proxy_true_control": 22, "train_replacement_proxy_false": 1, "val_harmful_mixed_reference": 8}`.

Sources: `{"iam_words": 37, "icdar2013": 6, "icdar2015": 7, "icdar_mlt2019_words": 3, "synthtext_words": 5, "textocr": 3}`.

## Next Action

Generate compact current/drop/replacement streams for the training-use probe rows and run OCR to obtain current-relative hard negative labels. Keep validation reference rows audit-only.

# Exception Counterfactual Probe Set

This is a selection manifest for future actual compact stream/OCR counterfactual labeling.
Rows with `training_use=false` are validation references and must not be used as training labels.

## Summary

Rows: `71`; training rows: `63`; audit-only rows: `8`.

Tags: `{"train_close_current_only": 40, "train_iam_current_only_near": 12, "train_iam_proxy_true_control": 22, "train_replacement_proxy_false": 1, "val_harmful_mixed_reference": 8}`.

Sources: `{"iam_words": 41, "icdar2013": 7, "icdar2015": 7, "icdar_mlt2019_words": 3, "synthtext_words": 6, "textocr": 7}`.

## Next Action

Generate compact current/drop/replacement streams for the training-use probe rows and run OCR to obtain current-relative hard negative labels. Keep validation reference rows audit-only.

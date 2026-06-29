# PaddleOCR E2E Eval300 Actual OSCR Evaluation

Evaluator role: held-out PaddleOCR detector+recognizer; not used for selection, training, utility labels, feature extraction or threshold tuning.

## Conditions

- `nearest_k64`: samples=225, unicode CER=0.156743, exact=0.760000, zero-det=12, multi-det=5, mean_s=0.1049
- `current_best`: samples=225, unicode CER=0.156743, exact=0.760000, zero-det=12, multi-det=5, mean_s=0.1045
- `top4_shortlist_oracle`: samples=225, unicode CER=0.156743, exact=0.760000, zero-det=12, multi-det=5, mean_s=0.1046

## Comparisons

- `nearest_to_current`: delta_char_errors=0, improved=0, worsened=0, unchanged=225
- `nearest_to_shortlist`: delta_char_errors=0, improved=0, worsened=0, unchanged=225
- `current_to_shortlist`: delta_char_errors=0, improved=0, worsened=0, unchanged=225

## Notes

PaddleOCR E2E is stricter than recognizer-only on these word crops because detection can miss the crop; the current-best and shortlist-oracle actual reconstructions remain exactly neutral versus nearest under this evaluator.

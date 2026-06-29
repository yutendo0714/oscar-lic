# PaddleOCR E2E Eval300 Original Upper Bound

Evaluator role: held-out PaddleOCR detector+recognizer; not used for selection, training, utility labels, feature extraction or threshold tuning.

## Conditions

- `original`: samples=225, unicode CER=0.143608, unicode exact=0.755556, latin CER=0.124101, zero-det=7, multi-det=5, mean_s=0.1056
- `nearest_k64`: samples=225, unicode CER=0.156743, unicode exact=0.760000, latin CER=0.144784, zero-det=12, multi-det=5, mean_s=0.1049
- `current_best`: samples=225, unicode CER=0.156743, unicode exact=0.760000, latin CER=0.144784, zero-det=12, multi-det=5, mean_s=0.1045
- `top4_shortlist_oracle`: samples=225, unicode CER=0.156743, unicode exact=0.760000, latin CER=0.144784, zero-det=12, multi-det=5, mean_s=0.1046

## Original To Reconstruction

- `original_to_nearest`: unicode_delta_chars=15, unicode_improved=16, unicode_worsened=16, latin_delta_chars=23
- `original_to_current`: unicode_delta_chars=15, unicode_improved=16, unicode_worsened=16, latin_delta_chars=23
- `original_to_shortlist`: unicode_delta_chars=15, unicode_improved=16, unicode_worsened=16, latin_delta_chars=23

## Notes

All actual OSCR reconstruction conditions are identical under PaddleOCR E2E, so the measured degradation relative to original crops is attributable to the frozen base/reconstruction path rather than the current OSCAR assignment variants. This is no-regression evidence only, not a deployable selection signal.

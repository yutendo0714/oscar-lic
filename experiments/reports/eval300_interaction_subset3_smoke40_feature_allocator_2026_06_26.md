# eval300_interaction_subset3_smoke40_feature_allocator_2026_06_26

This is a small diagnostic over OCR-derived conditional edge labels.
Feature models are cross-fitted by `index % folds`; threshold selection uses train folds only.

## Edge Baselines

| policy | selected | TP/FP/FN | precision | recall | F1 | Tess delta | PARSeq delta | selected transitions |
|---|---:|---|---:|---:|---:|---:|---:|---|
| none | 0 | 0/0/153 | n/a | 0.0000 | n/a | 0 | 0 | `{'0_to_1': 0, '1_to_2': 0, '2_to_3': 0}` |
| all | 480 | 153/327/0 | 0.3187 | 1.0000 | 0.4834 | -7 | -13 | `{'0_to_1': 120, '1_to_2': 240, '2_to_3': 120}` |
| first_only | 120 | 116/4/37 | 0.9667 | 0.7582 | 0.8498 | -179 | -13 | `{'0_to_1': 120, '1_to_2': 0, '2_to_3': 0}` |
| first_or_third | 240 | 142/98/11 | 0.5917 | 0.9281 | 0.7226 | -185 | -17 | `{'0_to_1': 120, '1_to_2': 0, '2_to_3': 120}` |

## Policy Baselines

| policy | changed | cardinalities | policies | Tess delta | PARSeq delta | mean enh bpp | improved/worse/unchanged |
|---|---:|---|---|---:|---:|---:|---|
| base | 0 | `{'0': 40}` | `{'base': 40}` | 0 | 0 | 0.0000 | `[0, 0, 40]` |
| single_a | 40 | `{'1': 40}` | `{'a': 40}` | -63 | -5 | 2.1017 | `[39, 0, 1]` |
| single_b | 40 | `{'1': 40}` | `{'b': 40}` | -58 | -5 | 2.1017 | `[38, 0, 2]` |
| single_c | 40 | `{'1': 40}` | `{'c': 40}` | -58 | -3 | 2.1017 | `[39, 0, 1]` |
| subset_oracle | 39 | `{'0': 1, '1': 35, '2': 4}` | `{'a': 35, 'ab': 1, 'ac': 2, 'base': 1, 'bc': 1}` | -72 | -2 | 2.2575 | `[39, 0, 1]` |
| best_single_train_cv | 40 | `{'1': 40}` | `{'a': 40}` | -63 | -5 | 2.1017 | `[39, 0, 1]` |

## Feature Models

| mode | edge objective | selected | precision | recall | F1 | Tess delta | PARSeq delta |
|---|---|---:|---:|---:|---:|---:|---:|
| transition | max_f1 | 120 | 0.9667 | 0.7582 | 0.8498 | -179 | -13 |
| transition | max_gain_precision90 | 120 | 0.9667 | 0.7582 | 0.8498 | -179 | -13 |
| transition_candidate_rate_psnr | max_f1 | 126 | 0.9127 | 0.7516 | 0.8244 | -177 | -13 |
| transition_candidate_rate_psnr | max_gain_precision90 | 127 | 0.9055 | 0.7516 | 0.8214 | -176 | -13 |
| transition_candidate_rate_psnr_source | max_f1 | 131 | 0.8855 | 0.7582 | 0.8169 | -177 | -13 |
| transition_candidate_rate_psnr_source | max_gain_precision90 | 135 | 0.8667 | 0.7647 | 0.8125 | -176 | -13 |
| transition_candidate_rate_psnr_ocr_state_upper | max_f1 | 133 | 0.9098 | 0.7908 | 0.8462 | -190 | -13 |
| transition_candidate_rate_psnr_ocr_state_upper | max_gain_precision90 | 131 | 0.9160 | 0.7843 | 0.8451 | -189 | -13 |

## Greedy Stop/Continue Policies

| mode | changed | cardinalities | policies | Tess delta | PARSeq delta | mean enh bpp | improved/worse/unchanged |
|---|---:|---|---|---:|---:|---:|---|
| transition | 40 | `{'1': 40}` | `{'a': 40}` | -63 | -5 | 2.1017 | `[39, 0, 1]` |
| transition_candidate_rate_psnr | 40 | `{'1': 40}` | `{'a': 4, 'c': 36}` | -59 | -3 | 2.1017 | `[39, 0, 1]` |
| transition_candidate_rate_psnr_source | 40 | `{'1': 40}` | `{'a': 4, 'c': 36}` | -59 | -3 | 2.1017 | `[39, 0, 1]` |
| transition_candidate_rate_psnr_ocr_state_upper | 40 | `{'1': 38, '3': 2}` | `{'a': 5, 'abc': 2, 'c': 33}` | -54 | -3 | 2.2122 | `[39, 0, 1]` |

## Interpretation

Best feature edge model by F1: `transition` / `max_f1` with F1 `0.8498` and Tesseract delta `-179`.
Best feature greedy policy by Tesseract delta: `transition` with delta `-63` and mean enhancement bpp `2.1017`.

Promotion rule: a learned multi-candidate allocator must beat both the N074 edge first-only diagnostic and the fixed one-candidate subset policy under policy-level OCR, not just row metrics.

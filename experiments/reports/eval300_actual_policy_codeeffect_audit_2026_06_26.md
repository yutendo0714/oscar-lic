# Eval300 Actual-Policy Code-Effect Audit

This is a diagnostic-only audit. It categorizes already-evaluated actual `.oscr` policies with held-out Tesseract labels, but does not define a deployable selector.

## Inputs

- current policy: `experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_model1_threshold05_explicit_codes_2026_06_25.jsonl`
- shortlist-oracle policy: `experiments/results/eval300_assignment_shortlist_oracle_top4_scoreseed2_explicit_codes_2026_06_26.jsonl`
- top-8 structured feature artifact: `experiments/features/eval300_assignment_top8_codeeffect_latentctx_windows_parseqfeat_seed0_1_2_2026_06_25.npz`

## Counts

- audited groups: 75
- current policy: 4 exact oracle changes, 1 false changes, 13 missed oracle changes
- current Tesseract/PARSeq deltas: -8 / 0
- shortlist Tesseract/PARSeq deltas: -20 / 0
- shortlist extra exact changes over current: 11
- shortlist missed oracle groups: 2

## Feature Signals

| feature | current exact | current false | missed oracle candidate |
|---|---:|---:|---:|
| topk_rank | 1.75 med 1.5 (n=4) | 2 med 2 (n=1) | 3.385 med 2 (n=13) |
| assignment_relative_error_from_core | 1.055 med 1.055 (n=4) | 1.028 med 1.028 (n=1) | 1.116 med 1.098 (n=13) |
| candidate_quant_error_l2 | 1.13 med 1.134 (n=4) | 1.134 med 1.134 (n=1) | 1.069 med 1.04 (n=13) |
| relative_quant_error | 1.055 med 1.055 (n=4) | 1.028 med 1.028 (n=1) | 1.116 med 1.098 (n=13) |
| code_delta_l2 | 0.9333 med 0.9794 (n=4) | 1.023 med 1.023 (n=1) | 0.878 med 0.814 (n=13) |
| residual_delta_cosine | 0.08081 med 0.1187 (n=4) | -0.07622 med -0.07622 (n=1) | 0.00401 med -0.008593 (n=13) |
| window_residual_l2 | 1.527 med 1.417 (n=4) | 1.057 med 1.057 (n=1) | 1.432 med 1.458 (n=13) |
| center_window_l2_fraction | 0.7528 med 0.7606 (n=4) | 1 med 1 (n=1) | 0.6548 med 0.5609 (n=13) |
| core_tesseract_nearest_distance | 3.5 med 3 (n=4) | 4 med 4 (n=1) | 5.154 med 5 (n=13) |

## Notable Cases

| category | seed/source/cand | source | ref | nearest -> current / oracle / shortlist | Tesseract deltas current/shortlist | ranks current/oracle | rel-error current/oracle |
|---|---|---|---|---|---:|---:|---:|
| exact_oracle_change | 2/6/5 | icdar2013 | SLUSH | 34 -> 62 / 62 / 62 | -2/-2 | 1/1 | 1.013/1.013 |
| exact_oracle_change | 2/13/1 | iam_words | Mauro's | 46 -> 18 / 18 / 18 | -3/-3 | 2/2 | 1.104/1.104 |
| false_change | 2/51/1 | iam_words | from | 2 -> 48 / 2 / 2 | 1/0 | 2/0 | 1.028/1 |
| exact_oracle_change | 1/63/2 | icdar2015 | heart | 53 -> 19 / 19 / 19 | -3/-3 | 1/1 | 1.006/1.006 |
| exact_oracle_change | 2/64/0 | icdar2013 | JOHN | 18 -> 54 / 54 / 54 | -1/-1 | 3/3 | 1.098/1.098 |
| missed_oracle | 1/12/1 | iam_words | Lawrence | 2 -> 2 / 49 / 2 | 0/0 | 0/6 | 1/1.158 |
| missed_oracle | 2/69/1 | icdar2013 | COSTA | 1 -> 1 / 25 / 1 | 0/0 | 0/6 | 1/1.12 |

## Interpretation

- The one current false change is not separable by a simple low residual-error or rank rule; its rank and relative quantization error sit inside the useful-change range.
- Many missed useful oracle candidates have non-nearest ranks up to 6 or 7 and often worse residual-fidelity error than nearest, so residual-MSE or top-1 confidence is the wrong objective.
- The actual-counted top-4 shortlist oracle mostly converts current missed groups at unchanged payload size, while the two remaining misses are first-stage shortlist coverage failures.
- Next work should change the candidate-local code-effect evidence or policy-level verifier. This audit should not be used to tune held-out-OCR thresholds.

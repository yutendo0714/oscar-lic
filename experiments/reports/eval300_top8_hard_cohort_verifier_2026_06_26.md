# eval300_top8_hard_cohort_verifier_2026_06_26

W&B: `98v72lq6`

Train-only N127 hard-cohort verifier over the N117 OOF shortlist. This is a table-level diagnostic; no counted `.oscr` stream is exported.

## Feature Scrub

- Selected features: `227`
- Leak-like selected features: `[]`
- Suspicious selected features: `[]`

## Validation Policies

| model | rank1/rank<=4 | budget | train OOF | validation |
|---|---:|---|---|---|
| logistic_c03 | 3/10 | train_oof_budget0 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| logistic_c03 | 3/10 | train_oof_budget1 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| logistic_c03 | 3/10 | train_oof_budget3 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| hist_gradient_l2 | 2/10 | train_oof_budget0 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| hist_gradient_l2 | 2/10 | train_oof_budget1 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| hist_gradient_l2 | 2/10 | train_oof_budget3 | 0/63 exact, false 1, wrong 2, miss 61, T -1, P 0 | 0/17 exact, false 3, wrong 0, miss 17, T 0, P 0 |
| random_forest_d5 | 2/10 | train_oof_budget0 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| random_forest_d5 | 2/10 | train_oof_budget1 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| random_forest_d5 | 2/10 | train_oof_budget3 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |

## Best Observed Validation Policy

{"model": "logistic_c03", "budget": "train_oof_budget0", "exact_changed_groups": 0, "false_wrong": 0, "tesseract_delta_vs_nearest": 0, "parseq_delta_vs_nearest": 0}

## Interpretation

N128 tests whether the N127 hard cohorts provide a cleaner verifier objective under strict feature scrubbing and train-only/source-OOF thresholding. A positive result would still need counted `.oscr` export; a negative result means hard-cohort supervision alone does not solve N126's no-headroom boundary.

# eval300_top8_hard_cohort_pairwise_verifier_2026_06_26

W&B: `y61bs9uu`

Pairwise verifier trained from N127 matched hard cohorts. This is a table-level diagnostic; no counted `.oscr` stream is exported.

## Validation Policies

| model | pair acc | rank1/rank<=4 | budget | train OOF | validation |
|---|---:|---:|---|---|---|
| pairwise_logistic_c0p05 | 1.000 | 3/10 | train_oof_budget0 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| pairwise_logistic_c0p05 | 1.000 | 3/10 | train_oof_budget1 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| pairwise_logistic_c0p05 | 1.000 | 3/10 | train_oof_budget3 | 0/63 exact, false 1, wrong 1, miss 62, T -1, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| pairwise_logistic_c0p2 | 1.000 | 3/10 | train_oof_budget0 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| pairwise_logistic_c0p2 | 1.000 | 3/10 | train_oof_budget1 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| pairwise_logistic_c0p2 | 1.000 | 3/10 | train_oof_budget3 | 0/63 exact, false 2, wrong 1, miss 62, T -1, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| pairwise_logistic_c1p0 | 1.000 | 3/10 | train_oof_budget0 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| pairwise_logistic_c1p0 | 1.000 | 3/10 | train_oof_budget1 | 0/63 exact, false 0, wrong 0, miss 63, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| pairwise_logistic_c1p0 | 1.000 | 3/10 | train_oof_budget3 | 1/63 exact, false 2, wrong 1, miss 61, T -2, P 0 | 0/17 exact, false 1, wrong 0, miss 17, T 0, P 0 |

## Best Observed Validation Policy

{"model": "pairwise_logistic_c0p05", "budget": "train_oof_budget0", "exact_changed_groups": 0, "false_wrong": 0, "tesseract_delta_vs_nearest": 0, "parseq_delta_vs_nearest": 0}

## Interpretation

N129 tests whether the matched structure in N127 is better used as pairwise utility ranking than row classification. A positive result would still need counted `.oscr` export; a negative result closes pairwise linear scoring over the same scrubbed feature family.

# eval300_top8_oof_two_stage_risk_miss_audit_2026_06_26

Diagnostic audit of N120 two-stage risk misses. No `.oscr` stream is exported.

W&B offline run: `ll9iwhpy`.

## Summary

| proposer | risk | policy | accepted exact | risk-abstained exact | proposal wrong | oracle missing | val zero-bad exact ceiling |
|---|---|---|---:|---:|---:|---:|---:|
| proposal_logistic_c05 | risk_logistic_c05 | 0/17 exact, false 7, wrong 3, miss 14, T 4, P 0 | 0 | 4 | 11 | 2 | 0 |
| proposal_logistic_c05 | risk_hist_gradient | 0/17 exact, false 1, wrong 0, miss 17, T 0, P 0 | 0 | 4 | 11 | 2 | 0 |
| proposal_logistic_c05 | risk_random_forest | 0/17 exact, false 2, wrong 0, miss 17, T 1, P 0 | 0 | 4 | 11 | 2 | 0 |
| proposal_hist_gradient | risk_logistic_c05 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0 | 6 | 9 | 2 | 2 |
| proposal_hist_gradient | risk_hist_gradient | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2 | 4 | 9 | 2 | 2 |
| proposal_hist_gradient | risk_random_forest | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2 | 4 | 9 | 2 | 2 |
| proposal_random_forest | risk_logistic_c05 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0 | 4 | 11 | 2 | 0 |
| proposal_random_forest | risk_hist_gradient | 0/17 exact, false 1, wrong 1, miss 16, T 2, P 0 | 0 | 4 | 11 | 2 | 0 |
| proposal_random_forest | risk_random_forest | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0 | 4 | 11 | 2 | 0 |

## Interpretation

N120's safe gain is limited by two bottlenecks: the proposal stage still fails many oracle groups, and the risk gate abstains some exact proposals to keep false/wrong at zero. Use this audit before adding hard-positive mining, richer proposal evidence, or a monotone/conformal abstention rule.

# eval300_top8_oof_shortlist_two_stage_risk_2026_06_26

Two-stage diagnostic: non-nearest proposal followed by separate risk-abstention over the N117 OOF shortlist. No `.oscr` stream is exported.

W&B offline run: `kf6ekni7`.

## Validation

| proposer | risk | all proposals | budget0 | budget1 | budget2 | nonnearest rank1/rank<=4 |
|---|---|---|---|---|---|---:|
| proposal_logistic_c05 | risk_logistic_c05 | 4/17 exact, false 58, wrong 13, miss 0, T 4, P 0 | 0/17 exact, false 7, wrong 3, miss 14, T 4, P 0 | 0/17 exact, false 7, wrong 3, miss 14, T 4, P 0 | 0/17 exact, false 7, wrong 3, miss 14, T 4, P 0 | 4/13 |
| proposal_logistic_c05 | risk_hist_gradient | 4/17 exact, false 58, wrong 13, miss 0, T 4, P 0 | 0/17 exact, false 1, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 1, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 9, wrong 0, miss 17, T 1, P 0 | 4/13 |
| proposal_logistic_c05 | risk_random_forest | 4/17 exact, false 58, wrong 13, miss 0, T 4, P 0 | 0/17 exact, false 2, wrong 0, miss 17, T 1, P 0 | 0/17 exact, false 2, wrong 0, miss 17, T 1, P 0 | 0/17 exact, false 3, wrong 0, miss 17, T 1, P 0 | 4/13 |
| proposal_hist_gradient | risk_logistic_c05 | 6/17 exact, false 58, wrong 11, miss 0, T 2, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 6/14 |
| proposal_hist_gradient | risk_hist_gradient | 6/17 exact, false 58, wrong 11, miss 0, T 2, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 6/14 |
| proposal_hist_gradient | risk_random_forest | 6/17 exact, false 58, wrong 11, miss 0, T 2, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 6/14 |
| proposal_random_forest | risk_logistic_c05 | 4/17 exact, false 58, wrong 13, miss 0, T 3, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 4/13 |
| proposal_random_forest | risk_hist_gradient | 4/17 exact, false 58, wrong 13, miss 0, T 3, P 0 | 0/17 exact, false 1, wrong 1, miss 16, T 2, P 0 | 0/17 exact, false 1, wrong 1, miss 16, T 2, P 0 | 0/17 exact, false 1, wrong 1, miss 16, T 2, P 0 | 4/13 |
| proposal_random_forest | risk_random_forest | 4/17 exact, false 58, wrong 13, miss 0, T 3, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 4/13 |

## Interpretation

This diagnostic tests the N119 recommendation by separating non-nearest proposal from no-op risk. It is not promotable unless a validation policy beats the current actual-bitstream floor before counted `.oscr` export.

# eval300_top8_oof_weighted_proposal_risk_2026_06_26

Hard-positive weighted non-nearest proposal diagnostic over the N117 OOF shortlist. No `.oscr` stream is exported.

W&B offline run: `87vmfylo`.

## Validation

| pos_w | hard_pos_w | noop_neg_w | all proposals | risk | budget0 | budget1 | rank1/rank<=4 |
|---:|---:|---:|---|---|---|---|---:|
| 1.0 | 1.0 | 0.25 | 4/17 exact, false 58, wrong 13, miss 0, T 2, P 0 | risk_hist_gradient | 1/17 exact, false 1, wrong 0, miss 16, T -3, P 0 | 1/17 exact, false 1, wrong 0, miss 16, T -3, P 0 | 4/14 |
| 1.0 | 1.0 | 0.25 | 4/17 exact, false 58, wrong 13, miss 0, T 2, P 0 | risk_random_forest | 1/17 exact, false 1, wrong 0, miss 16, T -3, P 0 | 1/17 exact, false 1, wrong 0, miss 16, T -3, P 0 | 4/14 |
| 1.0 | 1.0 | 0.5 | 5/17 exact, false 58, wrong 12, miss 0, T 4, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 0, miss 15, T -4, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -4, P 0 | 5/15 |
| 1.0 | 1.0 | 0.5 | 5/17 exact, false 58, wrong 12, miss 0, T 4, P 0 | risk_random_forest | 2/17 exact, false 1, wrong 0, miss 15, T -4, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -4, P 0 | 5/15 |
| 1.0 | 1.0 | 1.0 | 5/17 exact, false 58, wrong 12, miss 0, T 4, P 0 | risk_hist_gradient | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 5/14 |
| 1.0 | 1.0 | 1.0 | 5/17 exact, false 58, wrong 12, miss 0, T 4, P 0 | risk_random_forest | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 5/14 |
| 1.0 | 2.0 | 0.25 | 5/17 exact, false 58, wrong 12, miss 0, T 4, P 0 | risk_hist_gradient | 2/17 exact, false 2, wrong 0, miss 15, T -4, P 0 | 2/17 exact, false 2, wrong 0, miss 15, T -4, P 0 | 5/15 |
| 1.0 | 2.0 | 0.25 | 5/17 exact, false 58, wrong 12, miss 0, T 4, P 0 | risk_random_forest | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 5/15 |
| 1.0 | 2.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 3, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 1.0 | 2.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 3, P 0 | risk_random_forest | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 1.0 | 2.0 | 1.0 | 5/17 exact, false 58, wrong 12, miss 0, T 2, P 0 | risk_hist_gradient | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 5/15 |
| 1.0 | 2.0 | 1.0 | 5/17 exact, false 58, wrong 12, miss 0, T 2, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 5/15 |
| 1.0 | 4.0 | 0.25 | 6/17 exact, false 58, wrong 11, miss 0, T 5, P 0 | risk_hist_gradient | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 6/13 |
| 1.0 | 4.0 | 0.25 | 6/17 exact, false 58, wrong 11, miss 0, T 5, P 0 | risk_random_forest | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 6/13 |
| 1.0 | 4.0 | 0.5 | 7/17 exact, false 58, wrong 10, miss 0, T 2, P 0 | risk_hist_gradient | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 7/13 |
| 1.0 | 4.0 | 0.5 | 7/17 exact, false 58, wrong 10, miss 0, T 2, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 7/13 |
| 1.0 | 4.0 | 1.0 | 6/17 exact, false 58, wrong 11, miss 0, T -1, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 1.0 | 4.0 | 1.0 | 6/17 exact, false 58, wrong 11, miss 0, T -1, P 0 | risk_random_forest | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 2.0 | 1.0 | 0.25 | 4/17 exact, false 58, wrong 13, miss 0, T 5, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 4/14 |
| 2.0 | 1.0 | 0.25 | 4/17 exact, false 58, wrong 13, miss 0, T 5, P 0 | risk_random_forest | 1/17 exact, false 1, wrong 0, miss 16, T -3, P 0 | 1/17 exact, false 1, wrong 0, miss 16, T -3, P 0 | 4/14 |
| 2.0 | 1.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 4, P 0 | risk_hist_gradient | 2/17 exact, false 2, wrong 0, miss 15, T -4, P 0 | 2/17 exact, false 2, wrong 0, miss 15, T -4, P 0 | 6/14 |
| 2.0 | 1.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 4, P 0 | risk_random_forest | 2/17 exact, false 2, wrong 0, miss 15, T -4, P 0 | 2/17 exact, false 2, wrong 0, miss 15, T -4, P 0 | 6/14 |
| 2.0 | 1.0 | 1.0 | 6/17 exact, false 58, wrong 11, miss 0, T 2, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 2.0 | 1.0 | 1.0 | 6/17 exact, false 58, wrong 11, miss 0, T 2, P 0 | risk_random_forest | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 2.0 | 2.0 | 0.25 | 6/17 exact, false 58, wrong 11, miss 0, T 4, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 2.0 | 2.0 | 0.25 | 6/17 exact, false 58, wrong 11, miss 0, T 4, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 0, miss 16, T -3, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -3, P 0 | 6/14 |
| 2.0 | 2.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 0, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 2.0 | 2.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 0, P 0 | risk_random_forest | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 2.0 | 2.0 | 1.0 | 6/17 exact, false 58, wrong 11, miss 0, T 0, P 0 | risk_hist_gradient | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 2.0 | 2.0 | 1.0 | 6/17 exact, false 58, wrong 11, miss 0, T 0, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 6/14 |
| 2.0 | 4.0 | 0.25 | 6/17 exact, false 58, wrong 11, miss 0, T 5, P 0 | risk_hist_gradient | 1/17 exact, false 1, wrong 1, miss 15, T 0, P 0 | 1/17 exact, false 1, wrong 1, miss 15, T 0, P 0 | 6/13 |
| 2.0 | 4.0 | 0.25 | 6/17 exact, false 58, wrong 11, miss 0, T 5, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 6/13 |
| 2.0 | 4.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 2, P 0 | risk_hist_gradient | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 2.0 | 4.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 2, P 0 | risk_random_forest | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 2.0 | 4.0 | 1.0 | 5/17 exact, false 58, wrong 12, miss 0, T 2, P 0 | risk_hist_gradient | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 5/14 |
| 2.0 | 4.0 | 1.0 | 5/17 exact, false 58, wrong 12, miss 0, T 2, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 5/14 |
| 4.0 | 1.0 | 0.25 | 4/17 exact, false 58, wrong 13, miss 0, T 2, P 0 | risk_hist_gradient | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 4/14 |
| 4.0 | 1.0 | 0.25 | 4/17 exact, false 58, wrong 13, miss 0, T 2, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 0, miss 16, T -3, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -3, P 0 | 4/14 |
| 4.0 | 1.0 | 0.5 | 5/17 exact, false 58, wrong 12, miss 0, T 2, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 5/14 |
| 4.0 | 1.0 | 0.5 | 5/17 exact, false 58, wrong 12, miss 0, T 2, P 0 | risk_random_forest | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 5/14 |
| 4.0 | 1.0 | 1.0 | 5/17 exact, false 58, wrong 12, miss 0, T 4, P 0 | risk_hist_gradient | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 5/14 |
| 4.0 | 1.0 | 1.0 | 5/17 exact, false 58, wrong 12, miss 0, T 4, P 0 | risk_random_forest | 2/17 exact, false 1, wrong 1, miss 14, T -2, P 0 | 2/17 exact, false 1, wrong 1, miss 14, T -2, P 0 | 5/14 |
| 4.0 | 2.0 | 0.25 | 6/17 exact, false 58, wrong 11, miss 0, T 4, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 4.0 | 2.0 | 0.25 | 6/17 exact, false 58, wrong 11, miss 0, T 4, P 0 | risk_random_forest | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -5, P 0 | 6/14 |
| 4.0 | 2.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 1, P 0 | risk_hist_gradient | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 6/14 |
| 4.0 | 2.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 1, P 0 | risk_random_forest | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 6/14 |
| 4.0 | 2.0 | 1.0 | 7/17 exact, false 58, wrong 10, miss 0, T 0, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 0, miss 15, T -4, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -4, P 0 | 7/14 |
| 4.0 | 2.0 | 1.0 | 7/17 exact, false 58, wrong 10, miss 0, T 0, P 0 | risk_random_forest | 2/17 exact, false 1, wrong 0, miss 15, T -4, P 0 | 2/17 exact, false 1, wrong 0, miss 15, T -4, P 0 | 7/14 |
| 4.0 | 4.0 | 0.25 | 8/17 exact, false 58, wrong 9, miss 0, T 2, P 0 | risk_hist_gradient | 1/17 exact, false 0, wrong 1, miss 15, T 0, P 0 | 1/17 exact, false 0, wrong 1, miss 15, T 0, P 0 | 8/14 |
| 4.0 | 4.0 | 0.25 | 8/17 exact, false 58, wrong 9, miss 0, T 2, P 0 | risk_random_forest | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 8/14 |
| 4.0 | 4.0 | 0.5 | 7/17 exact, false 58, wrong 10, miss 0, T -2, P 0 | risk_hist_gradient | 2/17 exact, false 0, wrong 1, miss 14, T -3, P 0 | 2/17 exact, false 0, wrong 1, miss 14, T -3, P 0 | 7/14 |
| 4.0 | 4.0 | 0.5 | 7/17 exact, false 58, wrong 10, miss 0, T -2, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 7/14 |
| 4.0 | 4.0 | 1.0 | 6/17 exact, false 58, wrong 11, miss 0, T 2, P 0 | risk_hist_gradient | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 6/14 |
| 4.0 | 4.0 | 1.0 | 6/17 exact, false 58, wrong 11, miss 0, T 2, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 6/14 |
| 8.0 | 1.0 | 0.25 | 3/17 exact, false 58, wrong 14, miss 0, T 6, P 0 | risk_hist_gradient | 2/17 exact, false 3, wrong 1, miss 14, T -2, P 0 | 2/17 exact, false 3, wrong 1, miss 14, T -2, P 0 | 3/14 |
| 8.0 | 1.0 | 0.25 | 3/17 exact, false 58, wrong 14, miss 0, T 6, P 0 | risk_random_forest | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 3/14 |
| 8.0 | 1.0 | 0.5 | 3/17 exact, false 58, wrong 14, miss 0, T 6, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 1, miss 14, T -3, P 0 | 2/17 exact, false 1, wrong 1, miss 14, T -3, P 0 | 3/14 |
| 8.0 | 1.0 | 0.5 | 3/17 exact, false 58, wrong 14, miss 0, T 6, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 3/14 |
| 8.0 | 1.0 | 1.0 | 8/17 exact, false 58, wrong 9, miss 0, T -1, P 0 | risk_hist_gradient | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 8/14 |
| 8.0 | 1.0 | 1.0 | 8/17 exact, false 58, wrong 9, miss 0, T -1, P 0 | risk_random_forest | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 2/17 exact, false 2, wrong 1, miss 14, T -2, P 0 | 8/14 |
| 8.0 | 2.0 | 0.25 | 5/17 exact, false 58, wrong 12, miss 0, T 4, P 0 | risk_hist_gradient | 1/17 exact, false 0, wrong 0, miss 16, T -3, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -3, P 0 | 5/13 |
| 8.0 | 2.0 | 0.25 | 5/17 exact, false 58, wrong 12, miss 0, T 4, P 0 | risk_random_forest | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 5/13 |
| 8.0 | 2.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 3, P 0 | risk_hist_gradient | 2/17 exact, false 3, wrong 1, miss 14, T -1, P 0 | 2/17 exact, false 3, wrong 1, miss 14, T -1, P 0 | 6/14 |
| 8.0 | 2.0 | 0.5 | 6/17 exact, false 58, wrong 11, miss 0, T 3, P 0 | risk_random_forest | 2/17 exact, false 2, wrong 0, miss 15, T -4, P 0 | 2/17 exact, false 2, wrong 0, miss 15, T -4, P 0 | 6/14 |
| 8.0 | 2.0 | 1.0 | 5/17 exact, false 58, wrong 12, miss 0, T 0, P 0 | risk_hist_gradient | 2/17 exact, false 3, wrong 0, miss 15, T -3, P 0 | 2/17 exact, false 3, wrong 0, miss 15, T -3, P 0 | 5/15 |
| 8.0 | 2.0 | 1.0 | 5/17 exact, false 58, wrong 12, miss 0, T 0, P 0 | risk_random_forest | 2/17 exact, false 3, wrong 0, miss 15, T -3, P 0 | 2/17 exact, false 3, wrong 0, miss 15, T -3, P 0 | 5/15 |
| 8.0 | 4.0 | 0.25 | 7/17 exact, false 58, wrong 10, miss 0, T 4, P 0 | risk_hist_gradient | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 2/17 exact, false 0, wrong 0, miss 15, T -5, P 0 | 7/14 |
| 8.0 | 4.0 | 0.25 | 7/17 exact, false 58, wrong 10, miss 0, T 4, P 0 | risk_random_forest | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 7/14 |
| 8.0 | 4.0 | 0.5 | 7/17 exact, false 58, wrong 10, miss 0, T 5, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 1, miss 14, T -3, P 0 | 2/17 exact, false 1, wrong 1, miss 14, T -3, P 0 | 7/14 |
| 8.0 | 4.0 | 0.5 | 7/17 exact, false 58, wrong 10, miss 0, T 5, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 1, miss 15, T -1, P 0 | 1/17 exact, false 0, wrong 1, miss 15, T -1, P 0 | 7/14 |
| 8.0 | 4.0 | 1.0 | 6/17 exact, false 58, wrong 11, miss 0, T 2, P 0 | risk_hist_gradient | 2/17 exact, false 1, wrong 1, miss 14, T -3, P 0 | 2/17 exact, false 1, wrong 1, miss 14, T -3, P 0 | 6/14 |
| 8.0 | 4.0 | 1.0 | 6/17 exact, false 58, wrong 11, miss 0, T 2, P 0 | risk_random_forest | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 1/17 exact, false 0, wrong 0, miss 16, T -2, P 0 | 6/14 |

## Interpretation

This diagnostic tests whether simple hard-positive/no-op-negative weighting improves the N120 non-nearest proposal stage. A promotable result would still need to beat the current actual-bitstream floor before any counted `.oscr` export.

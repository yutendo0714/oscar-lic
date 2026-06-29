# eval300_top8_oof_deterministic_proposal_risk_2026_06_26

Deterministic proposal anchors over the N117 source-modulo OOF shortlist plus N120-style risk abstention. Diagnostic only; no `.oscr` stream is exported.

W&B: `lfs66wt4`

| anchor | all proposals | best budget0 | best budget1 | rank1/rank<=4 |
|---|---|---|---|---:|
| topk_rank_anchor | 6/17 exact, false 58, wrong 11, miss 0, T 3, P 0 | 1/17 exact, false 7, wrong 3, miss 13, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 6/12 |
| relative_error_anchor | 6/17 exact, false 58, wrong 11, miss 0, T 3, P 0 | 1/17 exact, false 7, wrong 3, miss 13, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 6/12 |
| oof_score_anchor | 5/17 exact, false 58, wrong 12, miss 0, T -1, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 5/12 |
| oof_rank_anchor | 5/17 exact, false 58, wrong 12, miss 0, T 1, P 0 | 2/17 exact, false 3, wrong 1, miss 14, T -2, P 0 | 2/17 exact, false 3, wrong 1, miss 14, T -2, P 0 | 5/14 |
| topk_oof_rrf_anchor | 4/17 exact, false 58, wrong 13, miss 0, T 7, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 4/13 |
| union_membership_anchor | 4/17 exact, false 58, wrong 13, miss 0, T 7, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 4/13 |
| z_fused_anchor | 5/17 exact, false 58, wrong 12, miss 0, T 2, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 1/17 exact, false 2, wrong 1, miss 15, T -1, P 0 | 5/12 |

## Interpretation

This low-variance diagnostic checks whether N123's failure was merely high-capacity overfit. A promotable result would still need to beat the current actual-bitstream floor before `.oscr` export; otherwise deterministic anchors should serve only as simple controls for future verifier objectives.

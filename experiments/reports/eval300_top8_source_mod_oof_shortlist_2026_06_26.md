# eval300_top8_source_mod_oof_shortlist_2026_06_26

Diagnostic-only source-modulo OOF shortlist audit. Each candidate score is taken only from the model whose source-modulo validation slice contains that candidate's source index.

W&B offline run: `lz93tl1o`.

## Validation Summary

| shortlist | size mean/max | summary |
|---|---:|---|
| oof_score_top1 | 1.00/1 | 5/17 exact, false 0, wrong 0, miss 12, T -10, P 0 |
| oof_score_top2 | 2.00/2 | 7/17 exact, false 0, wrong 0, miss 10, T -12, P 0 |
| oof_score_top4 | 4.00/4 | 14/17 exact, false 0, wrong 0, miss 3, T -21, P 0 |
| oof_score_top8 | 7.00/7 | 17/17 exact, false 0, wrong 0, miss 0, T -24, P 0 |
| topk4_union_oof_top1 | 4.19/5 | 12/17 exact, false 0, wrong 0, miss 5, T -17, P 0 |
| topk4_union_oof_top2 | 4.55/6 | 13/17 exact, false 0, wrong 0, miss 4, T -18, P 0 |
| topk4_union_oof_top4 | 5.16/6 | 15/17 exact, false 0, wrong 0, miss 2, T -22, P 0 |
| deterministic_topk_rank_le4 | 4.00/4 | 12/17 exact, false 0, wrong 0, miss 5, T -17, P 0 |
| full_top8_candidate_set | 7.00/7 | 17/17 exact, false 0, wrong 0, miss 0, T -24, P 0 |

## Unsafe Top-Score Policy

5/17 exact, false 58, wrong 12, miss 0, T 1, P 0

## Group Score Separation

AUC/AP for max non-nearest OOF score predicting an oracle-change group: 0.731237322515213/0.4676966716886586.

## Interpretation

Source-modulo OOF scores are valid as held-out score evidence for each source-index slice, but this diagnostic still uses an oracle chooser inside each candidate set. Therefore a shortlist improvement is first-stage evidence only, not a deployable policy. Promotion remains blocked until a no-op-aware selector beats the current counted actual-bitstream floor.

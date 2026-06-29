# eval300_top8_rankcap4_candidate_chooser_2026_06_26

Leakage-controlled second-stage chooser over non-nearest `topk_rank <= K` candidates.
This is a candidate-chooser diagnostic only; no `.oscr` policy is promoted.

## Validation

| model | rank1 | rank<=2 | rank<=4 | covered-only policy | all-change policy |
|---|---:|---:|---:|---|---|
| logistic_l2_c02 | 6 | 10 | 12 | 6/17 exact, false 0, wrong 6, miss 5, T -8, P 0 | 6/17 exact, false 0, wrong 11, miss 0, T -7, P 0 |
| hist_gradient_l2 | 6 | 11 | 12 | 6/17 exact, false 0, wrong 6, miss 5, T -9, P 0 | 6/17 exact, false 0, wrong 11, miss 0, T -7, P 0 |
| random_forest_d4 | 7 | 9 | 12 | 7/17 exact, false 0, wrong 5, miss 5, T -13, P 0 | 7/17 exact, false 0, wrong 10, miss 0, T -12, P 0 |

## Interpretation

This uses the clean, deterministic topk-rankcap shortlist rather than the provenance-sensitive source_mod5_r4 scores. If it cannot improve rank1 inside the rank<=4 candidate set, the next step should change candidate-local representation or add teacher-auxiliary representation learning, not threshold this chooser into an actual `.oscr` policy.

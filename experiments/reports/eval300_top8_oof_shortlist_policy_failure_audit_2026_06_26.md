# eval300_top8_oof_shortlist_policy_failure_audit_2026_06_26

Diagnostic audit of the N118 selector failure over the N117 source-modulo OOF shortlist. No `.oscr` stream is exported.

W&B offline run: `p5okugfr`.

## Shortlist Coverage

{
  "oracle_change_groups": 17,
  "covered_oracle_changes": 15,
  "missing_oracle_changes": 2,
  "covered_by_topk_cap": 12,
  "covered_by_oof_top": 14,
  "covered_by_both": 11,
  "oracle_topk_rank_stats": {
    "count": 17,
    "mean": 3.0,
    "std": 2.1693045781865616,
    "min": 1.0,
    "p25": 1.0,
    "median": 2.0,
    "p75": 6.0,
    "max": 7.0
  },
  "covered_oracle_oof_rank_stats": {
    "count": 15,
    "mean": 2.6666666666666665,
    "std": 1.6193277068654826,
    "min": 1.0,
    "p25": 1.0,
    "median": 3.0,
    "p75": 3.5,
    "max": 7.0
  }
}

## Model Margins

| model | N118 argmax | oracle>nearest | oracle>wrong nonnearest | best nonnearest>nearest | oracle rank<=1/4 all | oracle rank<=1/4 nonnearest |
|---|---|---:|---:|---:|---:|---:|
| logistic_l2_c02 | 0/17 exact, false 1, wrong 0, miss 17, T 0, P 0 | 0/15 | 2/15 | 1/75 | 0/12 | 2/14 |
| hist_gradient_l2 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/15 | 5/15 | 0/75 | 0/12 | 5/14 |
| random_forest_d5 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/15 | 8/15 | 0/75 | 0/11 | 8/14 |

## Interpretation

N117's union shortlist is a real first-stage improvement, but N118 fails mostly at the no-op/risk boundary and at ranking the useful non-nearest above nearby wrong non-nearest codes. This audit should be used to design a materially different risk objective or richer candidate-local evidence, not to seed/capacity sweep the shallow selector family.

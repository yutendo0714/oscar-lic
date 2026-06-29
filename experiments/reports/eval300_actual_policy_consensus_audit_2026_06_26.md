# Eval300 Actual-Policy Consensus Audit

This is a diagnostic-only audit. It compares already-exported actual `.oscr` assignment policies and does not define a deployable selector.

## Inputs

- policy A: `experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_model1_threshold05_explicit_codes_2026_06_25.jsonl`
- policy B: `experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_model2_threshold08_explicit_codes_2026_06_25.jsonl`
- shortlist oracle: `experiments/results/eval300_assignment_shortlist_oracle_top4_scoreseed2_explicit_codes_2026_06_26.jsonl`
- top-8 metadata: `experiments/features/eval300_assignment_top8_codeeffect_latentctx_windows_parseqfeat_seed0_1_2_2026_06_25.npz.meta.json`

## Agreement

- common groups: 75
- relationship counts: `{'both_nearest': 56, 'both_nearest_miss_oracle': 12, 'both_same_change': 5, 'policy_b_only_change': 2}`

## Policy Summary

| policy | changed | exact | false | wrong | missed | Tesseract Δ | PARSeq Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled_imgdiff_model1_threshold05 | 5 | 4 | 1 | 0 | 13 | -8 | 0 |
| pooled_imgdiff_model2_threshold08 | 7 | 4 | 2 | 1 | 12 | -8 | 0 |
| intersection_same_change | 5 | 4 | 1 | 0 | 13 | -8 | 0 |
| union_prefer_a | 7 | 4 | 2 | 1 | 12 | -8 | 0 |
| union_prefer_b | 7 | 4 | 2 | 1 | 12 | -8 | 0 |
| policy_b_extra_only | 2 | 0 | 1 | 1 | 16 | 0 | 0 |
| shortlist_oracle | 15 | 15 | 0 | 0 | 2 | -20 | 0 |

## Notable Changed Or Oracle-Headroom Cases

| relation | seed/source/cand | source | ref | nearest -> A / B / oracle / shortlist | categories A/B/short | Tesseract Δ A/B/short | ranks A/B/short |
|---|---|---|---|---|---|---:|---:|
| policy_b_only_change | 0/73/0 | iam_words | for | 34 -> 34 / 53 / 34 / 34 | correct_nearest/false_change/correct_nearest | 0/0/0 | 0/5/0 |
| both_nearest_miss_oracle | 1/9/0 | iam_words | own | 62 -> 62 / 62 / 31 / 31 | missed_oracle/missed_oracle/exact_oracle_change | 0/0/-1 | 0/0/1 |
| both_nearest_miss_oracle | 1/12/1 | iam_words | Lawrence | 2 -> 2 / 2 / 49 / 2 | missed_oracle/missed_oracle/missed_oracle | 0/0/0 | 0/0/0 |
| both_nearest_miss_oracle | 1/32/0 | iam_words | predetermined | 62 -> 62 / 62 / 50 / 50 | missed_oracle/missed_oracle/exact_oracle_change | 0/0/-1 | 0/0/2 |
| both_nearest_miss_oracle | 1/51/1 | iam_words | THE | 59 -> 59 / 59 / 56 / 56 | missed_oracle/missed_oracle/exact_oracle_change | 0/0/-1 | 0/0/7 |
| both_nearest_miss_oracle | 1/53/1 | icdar2015 | Accessories | 32 -> 32 / 32 / 43 / 43 | missed_oracle/missed_oracle/exact_oracle_change | 0/0/-1 | 0/0/2 |
| both_same_change | 1/63/2 | icdar2015 | heart | 53 -> 19 / 19 / 19 / 19 | exact_oracle_change/exact_oracle_change/exact_oracle_change | -3/-3/-3 | 1/1/1 |
| both_nearest_miss_oracle | 2/4/0 | iam_words | own | 18 -> 18 / 18 / 36 / 36 | missed_oracle/missed_oracle/exact_oracle_change | 0/0/-1 | 0/0/3 |
| both_nearest_miss_oracle | 2/5/4 | iam_words | Lawrence | 34 -> 34 / 34 / 2 / 2 | missed_oracle/missed_oracle/exact_oracle_change | 0/0/-1 | 0/0/6 |
| both_same_change | 2/6/5 | icdar2013 | SLUSH | 34 -> 62 / 62 / 62 / 62 | exact_oracle_change/exact_oracle_change/exact_oracle_change | -2/-2/-2 | 1/1/1 |
| both_same_change | 2/13/1 | iam_words | Mauro's | 46 -> 18 / 18 / 18 / 18 | exact_oracle_change/exact_oracle_change/exact_oracle_change | -3/-3/-3 | 2/2/2 |
| both_nearest_miss_oracle | 2/34/0 | icdar2015 | finest | 18 -> 18 / 18 / 1 / 1 | missed_oracle/missed_oracle/exact_oracle_change | 0/0/-1 | 0/0/6 |
| policy_b_only_change | 2/39/1 | iam_words | predetermined | 18 -> 18 / 30 / 1 / 1 | missed_oracle/wrong_change/exact_oracle_change | 0/0/-1 | 0/2/1 |
| both_nearest_miss_oracle | 2/43/4 | iam_words | understand | 7 -> 7 / 7 / 21 / 21 | missed_oracle/missed_oracle/exact_oracle_change | 0/0/-1 | 0/0/1 |
| both_nearest_miss_oracle | 2/50/0 | iam_words | text | 18 -> 18 / 18 / 46 / 46 | missed_oracle/missed_oracle/exact_oracle_change | 0/0/-1 | 0/0/2 |
| both_same_change | 2/51/1 | iam_words | from | 2 -> 48 / 48 / 2 / 2 | false_change/false_change/correct_nearest | 1/1/0 | 2/2/0 |
| both_nearest_miss_oracle | 2/53/1 | synthtext_words | vista" | 45 -> 45 / 45 / 20 / 20 | missed_oracle/missed_oracle/exact_oracle_change | 0/0/-1 | 0/0/1 |
| both_same_change | 2/64/0 | icdar2013 | JOHN | 18 -> 54 / 54 / 54 / 54 | exact_oracle_change/exact_oracle_change/exact_oracle_change | -1/-1/-1 | 3/3/3 |
| both_nearest_miss_oracle | 2/69/1 | icdar2013 | COSTA | 1 -> 1 / 1 / 25 / 1 | missed_oracle/missed_oracle/missed_oracle | 0/0/0 | 0/0/0 |

## Interpretation

- Policy B's two extra changes do not recover shortlist-oracle headroom; they add one false and one wrong/neutral change.
- Same-change consensus collapses to policy A, so this gate family does not offer a hidden high-precision ensemble path.
- The top-4 shortlist oracle remains the useful upper-bound substrate, but the deployable decision evidence needs to change rather than another consensus knob.

# Exception Current-Only Distribution Gap

## Current-Only Summary

| partition | rows | min | median | max | <=1.1 | <=1.5 | <=2.0 | <=10 | <=1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 180 | 1.020390 | 3445.966481 | 4940.269471 | 2 | 14 | 30 | 41 | 44 |
| val | 53 | 1.007230 | 1.073276 | 1.421825 | 32 | 53 | 53 | 53 | 53 |

## Labeled Val Seed1/2 Current-Only Rows

Joined rows: `28`; labels: `{"beneficial_no_profile_harm": 4, "harmful_any_profile": 5, "mixed_harm_and_benefit": 1, "neutral_all_profiles": 18}`.

| threshold | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|
| <= 1.1 | 20 | `{"beneficial_no_profile_harm": 4, "harmful_any_profile": 3, "mixed_harm_and_benefit": 1, "neutral_all_profiles": 12}` | 1 | 1 |
| <= 1.5 | 28 | `{"beneficial_no_profile_harm": 4, "harmful_any_profile": 5, "mixed_harm_and_benefit": 1, "neutral_all_profiles": 18}` | 3 | 4 |
| <= 2.0 | 28 | `{"beneficial_no_profile_harm": 4, "harmful_any_profile": 5, "mixed_harm_and_benefit": 1, "neutral_all_profiles": 18}` | 3 | 4 |
| <= 10.0 | 28 | `{"beneficial_no_profile_harm": 4, "harmful_any_profile": 5, "mixed_harm_and_benefit": 1, "neutral_all_profiles": 18}` | 3 | 4 |
| <= 1000.0 | 28 | `{"beneficial_no_profile_harm": 4, "harmful_any_profile": 5, "mixed_harm_and_benefit": 1, "neutral_all_profiles": 18}` | 3 | 4 |

## Conclusion

Current-only hard negatives are distributionally mismatched: validation current-only rows are almost all near the exception threshold, while train current-only rows are mostly far from exception centers. This explains why proxy/distance training lacks the hard negatives needed to preserve current selections.

## Next Action

Create or mine train hard negatives that resemble validation current-only rows before fitting a replace/drop verifier.

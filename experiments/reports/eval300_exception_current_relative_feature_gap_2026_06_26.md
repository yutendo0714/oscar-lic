# Current-Relative Exception Feature Gap

Joined current-selected validation rows: `50`.

## Main Splits

| subset | rows | labels | tess unicode delta | tess latin delta | parseq unicode delta |
|---|---:|---|---:|---:|---:|
| all_current_selected | 50 | `{"beneficial_no_profile_harm": 5, "harmful_any_profile": 7, "mixed_harm_and_benefit": 1, "neutral_all_profiles": 37}` | 3 | 6 | 0 |
| replace_with_exception | 22 | `{"beneficial_no_profile_harm": 1, "harmful_any_profile": 2, "neutral_all_profiles": 19}` | 0 | 2 | 0 |
| drop_current | 28 | `{"beneficial_no_profile_harm": 4, "harmful_any_profile": 5, "mixed_harm_and_benefit": 1, "neutral_all_profiles": 18}` | 3 | 4 | 0 |
| replacement_proxy_true | 9 | `{"beneficial_no_profile_harm": 1, "harmful_any_profile": 1, "neutral_all_profiles": 7}` | -1 | 1 | 0 |
| replacement_proxy_false | 13 | `{"harmful_any_profile": 1, "neutral_all_profiles": 12}` | 1 | 1 | 0 |
| strict_beneficial | 5 | `{"beneficial_no_profile_harm": 5}` | -4 | -2 | 0 |
| strict_harmful_or_mixed | 8 | `{"harmful_any_profile": 7, "mixed_harm_and_benefit": 1}` | 7 | 8 | 0 |

## Keep-Current-Only Counterfactual

Dropped current rows: `28`.
Dropping them causes exception-minus-current Tesseract deltas `+3` unicode and `+4` latin chars; keeping current would remove those deltas at `1204` transmitted bytes over this seed1/2 subset.

## Source Breakdown

| source | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|
| iam_words | 28 | `{"beneficial_no_profile_harm": 2, "harmful_any_profile": 6, "mixed_harm_and_benefit": 1, "neutral_all_profiles": 19}` | 5 | 6 |
| icdar2013 | 6 | `{"beneficial_no_profile_harm": 2, "harmful_any_profile": 1, "neutral_all_profiles": 3}` | -1 | 0 |
| icdar2015 | 8 | `{"beneficial_no_profile_harm": 1, "neutral_all_profiles": 7}` | -1 | 0 |
| icdar_mlt2019_words | 1 | `{"neutral_all_profiles": 1}` | 0 | 0 |
| synthtext_words | 5 | `{"neutral_all_profiles": 5}` | 0 | 0 |
| textocr | 2 | `{"neutral_all_profiles": 2}` | 0 | 0 |

## Conclusion

N143 proxy/action features explain exception eligibility, but the executed N145 labels show current-relative safety is dominated by whether current-only rows should be kept. Proxy-true replacement is not enough, and drop-current safety needs explicit supervision.

## Next Action

Construct counterfactual current-preserving labels for current-selected units, especially hard negative drop/replacement examples, before training a replace/drop verifier.

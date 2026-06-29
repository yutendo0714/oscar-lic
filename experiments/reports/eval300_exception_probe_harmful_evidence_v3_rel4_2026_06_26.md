# Exception Probe Harmful Evidence Audit

This diagnostic uses actual compact-stream OCR labels only; it is not a deployable verifier.

## Summary

Unique rows: `63`; labels: `{"beneficial_no_profile_harm": 17, "harmful_any_profile": 5, "neutral_all_profiles": 41}`.
Rows with label-set duplicates retained for comparison: `148`.

| subset | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|

## by_label_set

| name | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|
| eval300_exception_counterfactual_probe_actual_ocr_2026_06_26 | 32 | `{"beneficial_no_profile_harm": 7, "harmful_any_profile": 4, "neutral_all_profiles": 21}` | -5 | -2 |
| eval300_exception_counterfactual_probe_actual_ocr_v2_rel2_2026_06_26 | 53 | `{"beneficial_no_profile_harm": 16, "harmful_any_profile": 4, "neutral_all_profiles": 33}` | -15 | -11 |
| eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26 | 63 | `{"beneficial_no_profile_harm": 17, "harmful_any_profile": 5, "neutral_all_profiles": 41}` | -14 | -11 |

## by_action

| name | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|
| drop_current | 40 | `{"beneficial_no_profile_harm": 4, "harmful_any_profile": 3, "neutral_all_profiles": 33}` | 0 | 0 |
| replace_with_exception | 23 | `{"beneficial_no_profile_harm": 13, "harmful_any_profile": 2, "neutral_all_profiles": 8}` | -14 | -11 |

## by_source

| name | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|
| iam_words | 34 | `{"beneficial_no_profile_harm": 13, "harmful_any_profile": 3, "neutral_all_profiles": 18}` | -14 | -11 |
| icdar2013 | 6 | `{"beneficial_no_profile_harm": 1, "harmful_any_profile": 1, "neutral_all_profiles": 4}` | 2 | 1 |
| icdar2015 | 7 | `{"neutral_all_profiles": 7}` | 0 | 0 |
| icdar_mlt2019_words | 3 | `{"beneficial_no_profile_harm": 1, "neutral_all_profiles": 2}` | 0 | -1 |
| synthtext_words | 6 | `{"beneficial_no_profile_harm": 2, "harmful_any_profile": 1, "neutral_all_profiles": 3}` | -2 | 0 |
| textocr | 7 | `{"neutral_all_profiles": 7}` | 0 | 0 |

## Harmful Rows

| set | seed | source_index | source | action | ref | cur | exc | unicode | latin | bytes |
|---|---:|---:|---|---|---|---|---|---:|---:|---:|
| eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26 | 2 | 18 | icdar2013 | replace_with_exception | ESTATE | ESTATE | 2STATB | 2 | 2 | 0 |
| eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26 | 2 | 59 | iam_words | drop_current | with | wth | wh | 1 | 1 | -43 |
| eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26 | 0 | 158 | iam_words | drop_current | job | joe | jee | 1 | 1 | -43 |
| eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26 | 0 | 110 | synthtext_words | drop_current | and | =a | chi | 0 | 1 | -43 |
| eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26 | 2 | 17 | iam_words | replace_with_exception | way | ey) | 9, | 0 | 1 | 0 |

## Conclusion

Actual exception probe labels now show a usable positive replacement signal, but harmful rows remain concentrated in small text, handwriting and dark/background-sensitive cases. These rows should become explicit hard negatives or veto targets before replace/drop verifier training.

## Next Action

Join this evidence table to action/codebook features and train only a diagnostic verifier with strict held-out thresholding; do not promote it until actual compact OCR beats current and harmful rows are rejected.

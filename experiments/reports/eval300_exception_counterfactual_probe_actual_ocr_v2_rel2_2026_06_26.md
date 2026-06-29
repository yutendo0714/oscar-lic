# Exception Counterfactual Probe Actual OCR

Negative deltas mean the exception/drop stream has fewer OCR edit errors than the matched current stream.

## Summary

Rows: `53`; labels: `{"beneficial_no_profile_harm": 16, "harmful_any_profile": 4, "neutral_all_profiles": 33}`.

| model | profile | delta | improved | worsened | equal |
|---|---|---:|---:|---:|---:|
| tesseract | unicode_strict_v1 | -15 | 14 | 2 | 37 |
| tesseract | latin_alnum_ci_v1 | -11 | 12 | 4 | 37 |
| parseq | unicode_strict_v1 | 0 | 0 | 0 | 53 |
| parseq | latin_alnum_ci_v1 | 0 | 0 | 0 | 53 |

## By Action

| action | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|
| drop_current | 30 | `{"beneficial_no_profile_harm": 3, "harmful_any_profile": 2, "neutral_all_profiles": 25}` | -1 | 0 |
| replace_with_exception | 23 | `{"beneficial_no_profile_harm": 13, "harmful_any_profile": 2, "neutral_all_profiles": 8}` | -14 | -11 |

## By Tag

| tag | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|
| train_close_current_only | 30 | `{"beneficial_no_profile_harm": 3, "harmful_any_profile": 2, "neutral_all_profiles": 25}` | -1 | 0 |
| train_iam_current_only_near | 8 | `{"harmful_any_profile": 1, "neutral_all_profiles": 7}` | 1 | 1 |
| train_iam_proxy_true_control | 22 | `{"beneficial_no_profile_harm": 13, "harmful_any_profile": 1, "neutral_all_profiles": 8}` | -16 | -13 |
| train_replacement_proxy_false | 1 | `{"harmful_any_profile": 1}` | 2 | 2 |

## Rate

- `current`: weighted bpp `3.332659`, mean bpp `3.604718`, selected candidates `53`.
- `exception`: weighted bpp `3.248499`, mean bpp `3.505313`, selected candidates `23`.

## Conclusion

The mined probe set contains useful current-relative supervision: drop-current can help on some rows, but it also creates hard harmful cases. These labels are suitable as hard-negative diagnostics, not a standalone training distribution yet.

## Next Action

Use the harmful/mixed probe rows as hard negatives and collect more close current-only probes before training a replace/drop verifier.

# Exception Counterfactual Probe Actual OCR

Negative deltas mean the exception/drop stream has fewer OCR edit errors than the matched current stream.

## Summary

Rows: `32`; labels: `{"beneficial_no_profile_harm": 7, "harmful_any_profile": 4, "neutral_all_profiles": 21}`.

| model | profile | delta | improved | worsened | equal |
|---|---|---:|---:|---:|---:|
| tesseract | unicode_strict_v1 | -5 | 7 | 2 | 23 |
| tesseract | latin_alnum_ci_v1 | -2 | 5 | 4 | 23 |
| parseq | unicode_strict_v1 | 0 | 0 | 0 | 32 |
| parseq | latin_alnum_ci_v1 | 0 | 0 | 0 | 32 |

## By Action

| action | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|
| drop_current | 21 | `{"beneficial_no_profile_harm": 2, "harmful_any_profile": 2, "neutral_all_profiles": 17}` | -1 | 1 |
| replace_with_exception | 11 | `{"beneficial_no_profile_harm": 5, "harmful_any_profile": 2, "neutral_all_profiles": 4}` | -4 | -3 |

## By Tag

| tag | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|
| train_close_current_only | 14 | `{"beneficial_no_profile_harm": 2, "harmful_any_profile": 1, "neutral_all_profiles": 11}` | -2 | 0 |
| train_iam_current_only_near | 8 | `{"harmful_any_profile": 1, "neutral_all_profiles": 7}` | 1 | 1 |
| train_iam_proxy_true_control | 10 | `{"beneficial_no_profile_harm": 5, "harmful_any_profile": 1, "neutral_all_profiles": 4}` | -6 | -5 |
| train_replacement_proxy_false | 1 | `{"harmful_any_profile": 1}` | 2 | 2 |

## Rate

- `current`: weighted bpp `3.314236`, mean bpp `3.644198`, selected candidates `32`.
- `exception`: weighted bpp `3.216254`, mean bpp `3.525384`, selected candidates `11`.

## Conclusion

The mined probe set contains useful current-relative supervision: drop-current can help on some rows, but it also creates hard harmful cases. These labels are suitable as hard-negative diagnostics, not a standalone training distribution yet.

## Next Action

Use the harmful/mixed probe rows as hard negatives and collect more close current-only probes before training a replace/drop verifier.

# Exception Counterfactual Probe Actual OCR

Negative deltas mean the exception/drop stream has fewer OCR edit errors than the matched current stream.

## Summary

Rows: `63`; labels: `{"beneficial_no_profile_harm": 17, "harmful_any_profile": 5, "neutral_all_profiles": 41}`.

| model | profile | delta | improved | worsened | equal |
|---|---|---:|---:|---:|---:|
| tesseract | unicode_strict_v1 | -14 | 14 | 3 | 46 |
| tesseract | latin_alnum_ci_v1 | -11 | 13 | 5 | 45 |
| parseq | unicode_strict_v1 | 0 | 0 | 0 | 63 |
| parseq | latin_alnum_ci_v1 | 0 | 0 | 0 | 63 |

## By Action

| action | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|
| drop_current | 40 | `{"beneficial_no_profile_harm": 4, "harmful_any_profile": 3, "neutral_all_profiles": 33}` | 0 | 0 |
| replace_with_exception | 23 | `{"beneficial_no_profile_harm": 13, "harmful_any_profile": 2, "neutral_all_profiles": 8}` | -14 | -11 |

## By Tag

| tag | rows | labels | tess unicode delta | tess latin delta |
|---|---:|---|---:|---:|
| train_close_current_only | 40 | `{"beneficial_no_profile_harm": 4, "harmful_any_profile": 3, "neutral_all_profiles": 33}` | 0 | 0 |
| train_iam_current_only_near | 12 | `{"harmful_any_profile": 2, "neutral_all_profiles": 10}` | 2 | 2 |
| train_iam_proxy_true_control | 22 | `{"beneficial_no_profile_harm": 13, "harmful_any_profile": 1, "neutral_all_profiles": 8}` | -16 | -13 |
| train_replacement_proxy_false | 1 | `{"harmful_any_profile": 1}` | 2 | 2 |

## Rate

- `current`: weighted bpp `3.340284`, mean bpp `3.653858`, selected candidates `63`.
- `exception`: weighted bpp `3.245654`, mean bpp `3.539775`, selected candidates `23`.

## Conclusion

The mined probe set contains useful current-relative supervision: drop-current can help on some rows, but it also creates hard harmful cases. These labels are suitable as hard-negative diagnostics, not a standalone training distribution yet.

## Next Action

Use the harmful/mixed probe rows as hard negatives and collect more close current-only probes before training a replace/drop verifier.

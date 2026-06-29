# Exception Probe Veto Rule Search

Rules are diagnostic vetoes over actual exception/drop labels: a veto means keep the current stream for that row.
Lower OCR deltas are better. This is not a deployable selector claim.

## Baseline

Rows: `63`; labels: `{"beneficial_no_profile_harm": 17, "harmful_any_profile": 5, "neutral_all_profiles": 41}`; Tesseract delta `-14` unicode / `-11` latin; rate delta bytes `-1720`.

## Best Deployable-Feature Rules

| rule | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |
|---|---:|---:|---:|---:|---:|---:|---:|
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.191964) | 0 | 5 | 7 | -12 | -12 | -344 | 38 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bpp <= -0.191964) | 0 | 5 | 7 | -12 | -12 | -344 | 38 |
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.149306) | 0 | 5 | 7 | -12 | -12 | -172 | 42 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bpp <= -0.149306) | 0 | 5 | 7 | -12 | -12 | -172 | 42 |
| (action == drop_current) OR (current_luma_std <= 46.5541) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (current_luma_std <= 46.5541) OR (rate_delta_bytes <= -43) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (action == drop_current) OR (exception_luma_std <= 46.5555) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bytes <= -43) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.111979) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.0895833) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bpp <= -0.111979) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bpp <= -0.0895833) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |

## Deployable Harm-Budget Frontier

| max retained harmful | rule | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | (action == drop_current) OR (current_luma_std <= 46.5541) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| 1 | (current_luma_mean <= 40.1648) OR (rate_delta_bpp <= -0.191964) | 1 | 4 | 2 | -16 | -15 | -989 | 18 |
| 2 | (current_dark_fraction >= 0.933594) OR (rate_delta_bpp <= -0.223958) | 2 | 3 | 0 | -17 | -15 | -1376 | 9 |
| 3 | (current_dark_fraction >= 0.933594) OR (rate_delta_bpp <= -0.223958) | 2 | 3 | 0 | -17 | -15 | -1376 | 9 |
| 4 | (current_dark_fraction >= 0.933594) OR (rate_delta_bpp <= -0.223958) | 2 | 3 | 0 | -17 | -15 | -1376 | 9 |

## Best Audit-Only Rules

| rule | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |
|---|---:|---:|---:|---:|---:|---:|---:|
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.191964) | 0 | 5 | 7 | -12 | -12 | -344 | 38 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bpp <= -0.191964) | 0 | 5 | 7 | -12 | -12 | -344 | 38 |
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.149306) | 0 | 5 | 7 | -12 | -12 | -172 | 42 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bpp <= -0.149306) | 0 | 5 | 7 | -12 | -12 | -172 | 42 |
| (current_luma_std <= 46.5541) OR (tag contains train_iam_current_only_near) | 0 | 5 | 7 | -12 | -12 | -43 | 45 |
| (exception_luma_std <= 46.5555) OR (tag contains train_iam_current_only_near) | 0 | 5 | 7 | -12 | -12 | -43 | 45 |
| (action == drop_current) OR (current_luma_std <= 46.5541) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (current_luma_std <= 46.5541) OR (rate_delta_bytes <= -43) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (action == drop_current) OR (exception_luma_std <= 46.5555) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bytes <= -43) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.111979) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.0895833) | 0 | 5 | 7 | -12 | -12 | 0 | 46 |

## Conclusion

Simple deployable feature vetoes expose whether the four harmful exception/drop rows are separable without sacrificing too much of the replacement gain. Audit-only source/text rules are upper controls, not deployable policies.

## Next Action

Use the best deployable-feature rule families as features or weak veto targets in a diagnostic verifier, then evaluate with strict held-out thresholding and actual compact OCR.

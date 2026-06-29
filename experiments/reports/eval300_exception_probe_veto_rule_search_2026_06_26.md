# Exception Probe Veto Rule Search

Rules are diagnostic vetoes over actual exception/drop labels: a veto means keep the current stream for that row.
Lower OCR deltas are better. This is not a deployable selector claim.

## Baseline

Rows: `53`; labels: `{"beneficial_no_profile_harm": 16, "harmful_any_profile": 4, "neutral_all_profiles": 33}`; Tesseract delta `-15` unicode / `-11` latin; rate delta bytes `-1290`.

## Best Deployable-Feature Rules

| rule | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |
|---|---:|---:|---:|---:|---:|---:|---:|
| (current_luma_mean <= 213.059) OR (current_exception_mad <= 0.00651042) | 0 | 4 | 6 | -13 | -12 | -301 | 32 |
| (current_luma_mean <= 213.059) OR (current_exception_mse <= 0.00651042) | 0 | 4 | 6 | -13 | -12 | -301 | 32 |
| (exception_luma_mean <= 213.064) OR (current_exception_mad <= 0.00651042) | 0 | 4 | 6 | -13 | -12 | -301 | 32 |
| (exception_luma_mean <= 213.064) OR (current_exception_mse <= 0.00651042) | 0 | 4 | 6 | -13 | -12 | -301 | 32 |
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.191964) | 0 | 4 | 6 | -12 | -12 | -215 | 31 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bpp <= -0.191964) | 0 | 4 | 6 | -12 | -12 | -215 | 31 |
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.149306) | 0 | 4 | 6 | -12 | -12 | -129 | 33 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bpp <= -0.149306) | 0 | 4 | 6 | -12 | -12 | -129 | 33 |
| (action == drop_current) OR (current_luma_std <= 46.5541) | 0 | 4 | 6 | -12 | -12 | 0 | 36 |
| (current_luma_std <= 46.5541) OR (rate_delta_bytes <= -43) | 0 | 4 | 6 | -12 | -12 | 0 | 36 |
| (action == drop_current) OR (exception_luma_std <= 46.5555) | 0 | 4 | 6 | -12 | -12 | 0 | 36 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bytes <= -43) | 0 | 4 | 6 | -12 | -12 | 0 | 36 |

## Deployable Harm-Budget Frontier

| max retained harmful | rule | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | (current_luma_mean <= 213.059) OR (current_exception_mad <= 0.00651042) | 0 | 4 | 6 | -13 | -12 | -301 | 32 |
| 1 | (current_luma_mean <= 40.1648) OR (current_exception_mad >= 0.094308) | 1 | 3 | 1 | -17 | -14 | -774 | 14 |
| 2 | (current_luma_mean <= 40.1648) OR (current_luma_std <= 12.8813) | 2 | 2 | 0 | -17 | -14 | -1118 | 5 |
| 3 | (current_luma_mean <= 40.1648) OR (current_luma_std <= 12.8813) | 2 | 2 | 0 | -17 | -14 | -1118 | 5 |
| 4 | (current_luma_mean <= 40.1648) OR (current_luma_std <= 12.8813) | 2 | 2 | 0 | -17 | -14 | -1118 | 5 |

## Best Audit-Only Rules

| rule | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |
|---|---:|---:|---:|---:|---:|---:|---:|
| (current_luma_mean <= 213.059) OR (current_exception_mad <= 0.00651042) | 0 | 4 | 6 | -13 | -12 | -301 | 32 |
| (current_luma_mean <= 213.059) OR (current_exception_mse <= 0.00651042) | 0 | 4 | 6 | -13 | -12 | -301 | 32 |
| (exception_luma_mean <= 213.064) OR (current_exception_mad <= 0.00651042) | 0 | 4 | 6 | -13 | -12 | -301 | 32 |
| (exception_luma_mean <= 213.064) OR (current_exception_mse <= 0.00651042) | 0 | 4 | 6 | -13 | -12 | -301 | 32 |
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.191964) | 0 | 4 | 6 | -12 | -12 | -215 | 31 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bpp <= -0.191964) | 0 | 4 | 6 | -12 | -12 | -215 | 31 |
| (current_luma_std <= 46.5541) OR (rate_delta_bpp <= -0.149306) | 0 | 4 | 6 | -12 | -12 | -129 | 33 |
| (exception_luma_std <= 46.5555) OR (rate_delta_bpp <= -0.149306) | 0 | 4 | 6 | -12 | -12 | -129 | 33 |
| (current_luma_std <= 46.5541) OR (tag contains train_iam_current_only_near) | 0 | 4 | 6 | -12 | -12 | -43 | 35 |
| (exception_luma_std <= 46.5555) OR (tag contains train_iam_current_only_near) | 0 | 4 | 6 | -12 | -12 | -43 | 35 |
| (action == drop_current) OR (current_luma_std <= 46.5541) | 0 | 4 | 6 | -12 | -12 | 0 | 36 |
| (current_luma_std <= 46.5541) OR (rate_delta_bytes <= -43) | 0 | 4 | 6 | -12 | -12 | 0 | 36 |

## Conclusion

Simple deployable feature vetoes expose whether the four harmful exception/drop rows are separable without sacrificing too much of the replacement gain. Audit-only source/text rules are upper controls, not deployable policies.

## Next Action

Use the best deployable-feature rule families as features or weak veto targets in a diagnostic verifier, then evaluate with strict held-out thresholding and actual compact OCR.

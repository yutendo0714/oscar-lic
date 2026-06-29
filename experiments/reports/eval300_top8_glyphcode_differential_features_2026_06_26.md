# eval300_top8_glyphcode_differential_features_2026_06_26

Diagnostic glyph-code differential features for top-8 assignment candidates.
The labels come from held-out OCR failure-bank strata; these OCR strings are not deployable features.

## Summary

- Groups: `75`
- Candidate rows: `632`
- Target counts: `{'keep_current_good_change': 4, 'keep_noop': 57, 'recover_shortlist_oracle_change': 13, 'reject_current_bad_change': 1}`

## Oracle Rank By Glyph Score

Ranks are shown by target because keep-noop groups use the nearest code as the oracle target.

| target | score | oracle rank mean | rank<=1 | rank<=2 | rank<=4 | bad rank mean | bad rank<=1 |
|---|---|---:|---:|---:|---:|---:|---:|
| keep_current_good_change | `source_mse_delta` | 5.25 | 0 | 0 | 1 | None | 0 |
| keep_current_good_change | `source_mse_delta_changed` | 2.75 | 1 | 3 | 3 | None | 0 |
| keep_current_good_change | `toward_source` | 1.75 | 1 | 4 | 4 | None | 0 |
| keep_current_good_change | `sign_alignment` | 3.75 | 1 | 3 | 3 | None | 0 |
| keep_current_good_change | `dark_action_consistency` | 2.0 | 2 | 2 | 4 | None | 0 |
| keep_current_good_change | `adjacent_nearest_ink` | 3.0 | 1 | 2 | 3 | None | 0 |
| keep_current_good_change | `bridge_horizontal` | 5.5 | 0 | 1 | 3 | None | 0 |
| keep_current_good_change | `diff_largest_component` | 4.0 | 2 | 3 | 3 | None | 0 |
| keep_noop | `source_mse_delta` | 4.5964912280701755 | 4 | 9 | 28 | None | 0 |
| keep_noop | `source_mse_delta_changed` | 3.1052631578947367 | 14 | 25 | 42 | None | 0 |
| keep_noop | `toward_source` | 4.4035087719298245 | 10 | 15 | 33 | None | 0 |
| keep_noop | `sign_alignment` | 4.56140350877193 | 10 | 15 | 32 | None | 0 |
| keep_noop | `dark_action_consistency` | 4.982456140350878 | 9 | 15 | 28 | None | 0 |
| keep_noop | `adjacent_nearest_ink` | 5.9298245614035086 | 3 | 7 | 20 | None | 0 |
| keep_noop | `bridge_horizontal` | 3.508771929824561 | 20 | 29 | 37 | None | 0 |
| keep_noop | `diff_largest_component` | 5.947368421052632 | 3 | 7 | 18 | None | 0 |
| recover_shortlist_oracle_change | `source_mse_delta` | 5.076923076923077 | 2 | 3 | 6 | None | 0 |
| recover_shortlist_oracle_change | `source_mse_delta_changed` | 4.923076923076923 | 1 | 2 | 6 | None | 0 |
| recover_shortlist_oracle_change | `toward_source` | 4.461538461538462 | 2 | 4 | 6 | None | 0 |
| recover_shortlist_oracle_change | `sign_alignment` | 4.461538461538462 | 2 | 4 | 6 | None | 0 |
| recover_shortlist_oracle_change | `dark_action_consistency` | 4.076923076923077 | 3 | 4 | 7 | None | 0 |
| recover_shortlist_oracle_change | `adjacent_nearest_ink` | 3.6923076923076925 | 3 | 4 | 9 | None | 0 |
| recover_shortlist_oracle_change | `bridge_horizontal` | 4.076923076923077 | 2 | 4 | 8 | None | 0 |
| recover_shortlist_oracle_change | `diff_largest_component` | 3.3846153846153846 | 3 | 5 | 10 | None | 0 |
| reject_current_bad_change | `source_mse_delta` | 2.0 | 0 | 1 | 1 | 4.0 | 0 |
| reject_current_bad_change | `source_mse_delta_changed` | 6.0 | 0 | 0 | 0 | 2.0 | 0 |
| reject_current_bad_change | `toward_source` | 7.0 | 0 | 0 | 0 | 2.0 | 0 |
| reject_current_bad_change | `sign_alignment` | 7.0 | 0 | 0 | 0 | 2.0 | 0 |
| reject_current_bad_change | `dark_action_consistency` | 7.0 | 0 | 0 | 0 | 3.0 | 0 |
| reject_current_bad_change | `adjacent_nearest_ink` | 8.0 | 0 | 0 | 0 | 2.0 | 0 |
| reject_current_bad_change | `bridge_horizontal` | 4.0 | 0 | 0 | 1 | 6.0 | 0 |
| reject_current_bad_change | `diff_largest_component` | 8.0 | 0 | 0 | 0 | 1.0 | 1 |

## Target Feature Means

| target | feature | mean | min | max |
|---|---|---:|---:|---:|
| keep_current_good_change | `glyph_source_mse_delta` | -5.523434083443135e-07 | -1.730077201500535e-06 | 9.211362339556217e-07 |
| keep_current_good_change | `glyph_source_mse_delta_changed` | -2.08250912692165e-05 | -0.00031910784309729934 | 0.00010765112529043108 |
| keep_current_good_change | `glyph_toward_source_fraction_changed` | 0.7291666666666666 | 0.25 | 1.0 |
| keep_current_good_change | `glyph_sign_alignment_fraction_changed` | 0.7291666666666666 | 0.25 | 1.0 |
| keep_current_good_change | `glyph_dark_action_source_consistency` | 0.7638888888888888 | 0.5 | 1.0 |
| keep_current_good_change | `glyph_changed_adjacent_to_nearest_ink_fraction` | 0.6388888888888888 | 0.0 | 1.0 |
| keep_current_good_change | `glyph_diff_component_largest_fraction` | 0.6805555555555556 | 0.2222222222222222 | 1.0 |
| keep_current_good_change | `glyph_diff_component_largest_bbox_fraction` | 0.0007161458333333334 | 0.0003255208333333333 | 0.0015625 |
| keep_noop | `glyph_source_mse_delta` | -0.0031528355143332887 | -0.16572505235671997 | 0.0 |
| keep_noop | `glyph_source_mse_delta_changed` | -0.005193005931579461 | -0.2680833339691162 | 0.0 |
| keep_noop | `glyph_toward_source_fraction_changed` | 0.0 | 0.0 | 0.0 |
| keep_noop | `glyph_sign_alignment_fraction_changed` | 0.012064615643929714 | 0.0 | 0.4295991778006166 |
| keep_noop | `glyph_dark_action_source_consistency` | 0.001155963188456939 | 0.0 | 0.05124396583735611 |
| keep_noop | `glyph_changed_adjacent_to_nearest_ink_fraction` | 0.021420182199982582 | 0.0 | 0.8920863309352518 |
| keep_noop | `glyph_diff_component_largest_fraction` | 0.02938513684843833 | 0.0 | 0.9922020051986632 |
| keep_noop | `glyph_diff_component_largest_bbox_fraction` | 0.03389830508474576 | 0.0 | 1.0 |
| recover_shortlist_oracle_change | `glyph_source_mse_delta` | -4.904364826730811e-07 | -2.6207126211375e-06 | 7.358903530985117e-07 |
| recover_shortlist_oracle_change | `glyph_source_mse_delta_changed` | 3.97698120370758e-06 | -0.00018454587552696466 | 0.00029878620989620686 |
| recover_shortlist_oracle_change | `glyph_toward_source_fraction_changed` | 0.24981684981684982 | 0.0 | 1.0 |
| recover_shortlist_oracle_change | `glyph_sign_alignment_fraction_changed` | 0.2608058608058608 | 0.0 | 1.0 |
| recover_shortlist_oracle_change | `glyph_dark_action_source_consistency` | 0.3252747252747253 | 0.0 | 1.0 |
| recover_shortlist_oracle_change | `glyph_changed_adjacent_to_nearest_ink_fraction` | 0.5769230769230769 | 0.0 | 1.0 |
| recover_shortlist_oracle_change | `glyph_diff_component_largest_fraction` | 0.3245421245421245 | 0.0 | 1.0 |
| recover_shortlist_oracle_change | `glyph_diff_component_largest_bbox_fraction` | 0.00022401101502664001 | 0.0 | 0.0005208333333333333 |
| reject_current_bad_change | `glyph_source_mse_delta` | 0.0 | 0.0 | 0.0 |
| reject_current_bad_change | `glyph_source_mse_delta_changed` | 0.0 | 0.0 | 0.0 |
| reject_current_bad_change | `glyph_toward_source_fraction_changed` | 0.0 | 0.0 | 0.0 |
| reject_current_bad_change | `glyph_sign_alignment_fraction_changed` | 0.0 | 0.0 | 0.0 |
| reject_current_bad_change | `glyph_dark_action_source_consistency` | 0.0 | 0.0 | 0.0 |
| reject_current_bad_change | `glyph_changed_adjacent_to_nearest_ink_fraction` | 0.0 | 0.0 | 0.0 |
| reject_current_bad_change | `glyph_diff_component_largest_fraction` | 0.0 | 0.0 | 0.0 |
| reject_current_bad_change | `glyph_diff_component_largest_bbox_fraction` | 0.0 | 0.0 | 0.0 |

## Interpretation

- This is a representation diagnostic, not a promoted selector.
- A useful next feature family should rank recover targets high without also ranking the current bad-change candidate high.
- If the simple glyph scores still rank the bad change near the top, the next step should combine them with stricter abstention or richer OCR/text-aware evidence.

# eval300_top8_glyphcode_verifier_score_audit_2026_06_26

Diagnostic score audit for the N085 glyph-code verifier. This is not a promoted selector.

## Policy Reproduction

| val seed | threshold | changed | exact | false | wrong | missed | Tesseract delta | PARSeq delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.21615513748858128 | 1 | 24 | 1 | 0 | 0 | 1 | 0 |
| 1 | -0.27543312552677884 | 1 | 20 | 0 | 0 | 5 | -3 | 0 |
| 2 | -0.4199162912802193 | 1 | 14 | 0 | 1 | 10 | 0 | 0 |

## Target Score-Rank Audit

| target | groups | selected status | oracle score <=1/2/4/8 | oracle topk <=1/2/4/8 | oracle-nearest score mean | best-nonnearest margin mean |
|---|---:|---|---|---|---:|---:|
| keep_current_good_change | 4 | exact:1, missed_oracle:3 | 0/1/2/4 | 2/3/4/4 | -0.7045940769448843 | -0.6565791266137595 |
| keep_noop | 57 | exact:56, false_change:1 | 57/57/57/57 | 57/57/57/57 | 0.0 | -0.6023804785974216 |
| recover_shortlist_oracle_change | 13 | missed_oracle:12, wrong_change:1 | 0/3/7/13 | 4/7/8/13 | -0.8163368308291956 | -0.6889393960183468 |
| reject_current_bad_change | 1 | exact:1 | 1/1/1/1 | 1/1/1/1 | 0.0 | -0.9271544368230046 |

## Mean Standardized Coefficients

### Positive

- `is_nearest`: 1.454223940987447
- `glyph_diff_component_largest_fill`: 0.7904389196253523
- `glyph_ink_component_delta`: 0.5346620531051731
- `glyph_added_bridge_vertical_fraction`: 0.4329727611518384
- `glyph_dark_action_source_consistency`: 0.3927677254249271
- `glyph_y_projection_entropy`: 0.36117964752872395
- `glyph_toward_source_fraction_changed`: 0.1851836200269723
- `glyph_removed_dark_fraction`: 0.15544956780135097
- `glyph_y_projection_max_fraction`: 0.13839041879458927
- `glyph_diff_component_largest_bbox_fraction`: 0.1200663258582753
- `glyph_changed_fraction`: 0.09018467558024308
- `glyph_source_mse_delta_changed`: 0.0688251114471496

### Negative

- `glyph_x_projection_max_fraction`: -0.620055815347408
- `glyph_changed_adjacent_to_variant_ink_fraction`: -0.5411849547378482
- `glyph_changed_adjacent_to_nearest_ink_fraction`: -0.5100837510650206
- `topk_rank_norm`: -0.49041482170727574
- `glyph_diff_component_count`: -0.4327528745019138
- `glyph_diff_component_largest_fraction`: -0.39439195446488023
- `glyph_ink_area_delta_fraction`: -0.24570286047312898
- `glyph_added_bridge_horizontal_fraction`: -0.20670798494601997
- `glyph_removed_dark_source_light_fraction`: -0.1042877588275861
- `glyph_added_dark_fraction`: -0.0611695371586779
- `glyph_sign_alignment_fraction_changed`: -0.01741402725126338
- `glyph_added_dark_source_ink_fraction`: 0.00928000721342015

## Interpretation

- This audit explains why the direct glyph-code verifier is not promotable: useful recover candidates are often not score-rank 1, while no-op pressure remains close enough that train-fold margin tuning abstains.
- Use glyph-code signals as auxiliary local evidence in a richer verifier; do not retry glyphcode-only thresholding or logistic calibration.

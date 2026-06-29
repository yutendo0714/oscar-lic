# eval300_top8_glyphcode_nonnearest_ranker_2026_06_26

Diagnostic non-nearest-only ranker over top-8 glyph-code features.
This isolates candidate ranking from the no-op/nearest decision and is not deployable by itself.

## Oracle-Change-Only Policy

| val seed | changed | exact | false | wrong | missed | Tesseract delta | PARSeq delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 25 | 0 | 0 | 0 | 0 | 0 |
| 1 | 6 | 20 | 0 | 5 | 0 | -3 | 0 |
| 2 | 11 | 15 | 0 | 10 | 0 | 0 | 0 |

## Target Rank Audit

| target | groups | selected status | oracle nonnearest score <=1/2/4/8 | oracle topk <=1/2/4/8 | best-nonnearest Tesseract delta sum |
|---|---:|---|---|---|---:|
| keep_current_good_change | 4 | exact:1, wrong_change:3 | 1/1/2/4 | 2/3/4/4 | -2 |
| keep_noop | 57 | exact:57 | 0/0/0/0 | 57/57/57/57 | 9 |
| recover_shortlist_oracle_change | 13 | exact:1, wrong_change:12 | 1/3/8/13 | 4/7/8/13 | -1 |
| reject_current_bad_change | 1 | exact:1 | 0/0/0/0 | 1/1/1/1 | 1 |

## Mean Standardized Coefficients

### Positive

- `glyph_diff_component_largest_fill`: 1.0887321508091057
- `glyph_ink_component_delta`: 1.0862902397602339
- `glyph_added_bridge_vertical_fraction`: 0.698810020830162
- `glyph_dark_action_source_consistency`: 0.5548012633319781
- `glyph_y_projection_max_fraction`: 0.3356757492974017
- `glyph_toward_source_fraction_changed`: 0.2629667198511411
- `glyph_added_dark_source_ink_fraction`: 0.16506633720980546
- `glyph_abs_change_changed_mean`: 0.11202738244869005
- `glyph_sign_alignment_fraction_changed`: 0.1081079793109976
- `glyph_y_projection_entropy`: 0.0764052527808255
- `glyph_source_mse_delta_changed`: 0.04965168172973252
- `glyph_source_mse_delta`: 0.0440769774874464

### Negative

- `glyph_x_projection_max_fraction`: -0.9900094697713963
- `glyph_diff_component_largest_fraction`: -0.8245177093352604
- `glyph_changed_adjacent_to_variant_ink_fraction`: -0.8053567856969871
- `glyph_changed_adjacent_to_nearest_ink_fraction`: -0.80051069062146
- `glyph_diff_component_count`: -0.7126504925709224
- `glyph_ink_area_delta_fraction`: -0.7045098898742018
- `topk_rank_norm`: -0.42607272511207867
- `glyph_added_bridge_horizontal_fraction`: -0.40309871219080434
- `glyph_added_dark_fraction`: -0.18981673176408795
- `glyph_removed_dark_source_light_fraction`: -0.15973800686540596
- `glyph_changed_fraction`: -0.08500008840459596
- `glyph_abs_change_mean`: -0.06766239120692424

## Interpretation

- If non-nearest-only ranking recovers many oracle candidates, the next model should separate candidate scoring from no-op acceptance.
- If it still misses hard positives, glyph-code scalars alone are insufficient even after removing the nearest prior.

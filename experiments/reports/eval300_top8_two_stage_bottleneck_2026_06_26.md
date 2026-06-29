# eval300_top8_two_stage_bottleneck_2026_06_26

Diagnostic-only two-stage bottleneck audit for the top-8 assignment substrate.

## Upper Bounds

| policy | val summary |
|---|---|
| nearest_no_enhancement | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 |
| perfect_candidate_budget5 | 5/17 exact, false 0, wrong 0, miss 12, T -12, P 0 |
| topk_rankcap_le1 | 6/17 exact, false 0, wrong 0, miss 11, T -9, P 0 |
| topk_rankcap_le2 | 10/17 exact, false 0, wrong 0, miss 7, T -15, P 0 |
| topk_rankcap_le4 | 12/17 exact, false 0, wrong 0, miss 5, T -17, P 0 |
| topk_rankcap_le8 | 17/17 exact, false 0, wrong 0, miss 0, T -24, P 0 |
| full_top8_oracle | 17/17 exact, false 0, wrong 0, miss 0, T -24, P 0 |

## Score-Source Candidate Bottleneck

| score source | val rank1 | val rank<=4 | oracle-acceptor top1 | max-nonnearest AUC/AP |
|---|---:|---:|---|---:|
| group_noop_window_source_mod5_r4_seed0_1_2 | 11 | 16 | 11/17 exact, false 0, wrong 6, miss 0, T -18, P 0 | 0.986/0.971 |
| group_noop_window_tailmargin_r1_hp1_seed0_1_2 | 7 | 12 | 7/17 exact, false 0, wrong 10, miss 0, T -9, P 0 | 0.696/0.366 |
| patch_ranker_codegeom_oraclechoice_combined_seed0_1_2_model0_1_2 | 7 | 11 | 7/17 exact, false 0, wrong 10, miss 0, T -10, P 0 | 0.634/0.355 |
| patch_ranker_codegeom_oraclechoice_listwise_seed2_model0 | 7 | 9 | 7/17 exact, false 0, wrong 4, miss 6, T -8, P 0 | 0.662/0.641 |
| group_noop_window_tailmargin_codeonly_seed0_1_2 | 6 | 15 | 6/17 exact, false 0, wrong 11, miss 0, T -10, P 0 | 0.622/0.364 |
| group_noop_window_default_seed0_1_2 | 6 | 13 | 6/17 exact, false 0, wrong 11, miss 0, T -8, P 0 | 0.689/0.360 |
| group_noop_window_oraclechoice_riskcal_seed0_1_2 | 6 | 13 | 6/17 exact, false 0, wrong 11, miss 0, T -10, P 0 | 0.672/0.373 |
| table_relative_error_only | 6 | 12 | 6/17 exact, false 0, wrong 11, miss 0, T -5, P 0 | 0.552/0.248 |
| table_topk_only | 6 | 12 | 6/17 exact, false 0, wrong 11, miss 0, T -5, P 0 | 0.500/0.156 |
| table_topk_rate_prior | 6 | 12 | 6/17 exact, false 0, wrong 11, miss 0, T -5, P 0 | 0.552/0.248 |
| group_noop_window_tailmargin_reg_rank0heavy_seed0_1_2 | 6 | 11 | 6/17 exact, false 0, wrong 11, miss 0, T -8, P 0 | 0.702/0.393 |
| patch_ranker_codegeom_oraclechoice_bce_calibrated_seed2_model0 | 6 | 9 | 6/17 exact, false 0, wrong 5, miss 6, T -8, P 0 | 0.708/0.579 |
| patch_ranker_codegeom_oraclechoice_bce_listwise_seed2_model0 | 6 | 9 | 6/17 exact, false 0, wrong 5, miss 6, T -8, P 0 | 0.734/0.608 |
| patch_ranker_codegeom_oraclechoice_diffcrop_seed2_model0 | 6 | 9 | 6/17 exact, false 0, wrong 5, miss 6, T -8, P 0 | 0.708/0.575 |
| patch_ranker_codegeom_oraclechoice_listwise_seed2_model1_2 | 6 | 9 | 6/17 exact, false 0, wrong 5, miss 6, T -8, P 0 | 0.724/0.663 |
| codeeffect_window_oraclelabels_nochange2_seed0_1_2 | 5 | 13 | 5/17 exact, false 0, wrong 12, miss 0, T -7, P 0 | 0.671/0.338 |
| codeeffect_window_oraclelabels_seed0_1_2 | 5 | 13 | 5/17 exact, false 0, wrong 12, miss 0, T -10, P 0 | 0.685/0.349 |
| patch_ranker_codegeom_oraclechoice_seed2_model0 | 5 | 9 | 5/17 exact, false 0, wrong 6, miss 6, T -7, P 0 | 0.747/0.689 |
| patch_ranker_codegeom_oraclechoice_seed2_model0_1_2 | 5 | 9 | 5/17 exact, false 0, wrong 6, miss 6, T -7, P 0 | 0.711/0.592 |
| policy_margin_noheadroom_valseed1_probe | 4 | 13 | 4/17 exact, false 0, wrong 13, miss 0, T -10, P 0 | 0.662/0.382 |
| group_noop_window_softutility_strict_seed0_1_2 | 4 | 10 | 4/17 exact, false 0, wrong 13, miss 0, T -6, P 0 | 0.638/0.353 |
| patch_ranker_codegeom_oraclechoice_metadata_difficulty_seed2_model0 | 3 | 9 | 3/17 exact, false 0, wrong 8, miss 6, T -2, P 0 | 0.636/0.529 |
| patch_ranker_codegeom_latentvec_oraclechoice_diffcrop_seed2_model0 | 3 | 8 | 3/17 exact, false 0, wrong 8, miss 6, T -3, P 0 | 0.526/0.449 |
| patch_ranker_codegeom_oraclechoice_metadata_difficulty_tabular_seed2_model0 | 2 | 9 | 2/17 exact, false 0, wrong 9, miss 6, T -3, P 0 | 0.591/0.498 |
| patch_ranker_codegeom_oraclechoice_listwise_seed1_model0 | 2 | 3 | 2/17 exact, false 0, wrong 4, miss 11, T -3, P 0 | 0.430/0.367 |
| patch_ranker_codegeom_oraclechoice_bce_calibrated_seed1_model0 | 1 | 5 | 1/17 exact, false 0, wrong 5, miss 11, T -2, P 0 | 0.561/0.458 |
| patch_ranker_codegeom_oraclechoice_seed1_model0 | 1 | 4 | 1/17 exact, false 0, wrong 5, miss 11, T -2, P 0 | 0.561/0.463 |
| patch_ranker_codegeom_oraclechoice_seed1_model0_1_2 | 1 | 4 | 1/17 exact, false 0, wrong 5, miss 11, T -2, P 0 | 0.500/0.397 |
| patch_ranker_codegeom_oraclechoice_bce_calibrated_seed0_model0 | 0 | 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | n/a |
| patch_ranker_codegeom_oraclechoice_listwise_seed0_model0 | 0 | 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | n/a |

## Provenance Slices

| score source | slice | groups | oracle changes | rank1 | rank<=4 | oracle-acceptor top1 | AUC/AP |
|---|---|---:|---:|---:|---:|---|---:|
| group_noop_window_source_mod5_r4_seed0_1_2 | clean_val_source_mod5_heldout_r4 | 21 | 6 | 0 | 5 | 0/6 exact, false 0, wrong 6, miss 0, T 0, P 0 | 0.944/0.906 |
| group_noop_window_source_mod5_r4_seed0_1_2 | clean_val_source_mod5_in_train_not_r4 | 54 | 11 | 11 | 11 | 11/11 exact, false 0, wrong 0, miss 0, T -18, P 0 | 1.000/1.000 |

## Interpretation

The audit is diagnostic-only. It asks whether existing score families are primarily limited by candidate top-1 ranking or by no-op/change calibration. Any positive score-rankcap number is an upper control with an oracle chooser inside the score shortlist; it is not a deployable policy. The source-modulo provenance slice is especially important: source_mod5_r4 scores are useful as a high-recall candidate-set hint, but clean-val rows outside the held-out source-modulo remainder are not valid held-out evidence for that score source. Promotion still requires an actual counted `.oscr` export and OCR comparison.

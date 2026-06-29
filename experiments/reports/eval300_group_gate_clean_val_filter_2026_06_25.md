# Eval300 Group Gate Clean-Val Filter Diagnostic

**Date:** 2026-06-25  
**W&B:** offline run `36wwlrx8` (`eval300_group_gate_clean_val_filter_negative_2026_06_25`)  
**Scope:** pre-actual promotion check for real-seed cross-fit group-gate policies.

## Purpose

The real-seed cross-fit group gate looked promising at table level, especially with train-tuned thresholds. However, the exported cross-fit policy for each held-out `real_seed` contains a mixed 100-group set: `75` train-partition groups plus `25` clean val-partition groups. Actual `.oscr` promotion for the Eval300 val75 protocol must use only the clean val-compatible rows.

This diagnostic filters each exported policy to:

- `seed == 1`;
- the matching `real_seed`;
- `policy == group_gate_train_tuned` or `policy == group_gate_threshold_0_5`.

No actual `.oscr` run was launched because the clean-val table deltas did not beat the current promoted actual-bitstream baseline.

## Results

Table deltas are summed over the three clean val25 shards, so each model seed covers `75` groups. Negative Tesseract is better. The promoted actual-bitstream baseline is `-8` Tesseract edits with PARSeq unchanged.

| family | gate model seed | groups | changed | PARSeq edits | Tesseract edits | beats promoted table delta |
|---|---:|---:|---:|---:|---:|---|
| train-tuned | 0 | 75 | 10 | 0 | -3 | no |
| train-tuned | 1 | 75 | 11 | 0 | -2 | no |
| train-tuned | 2 | 75 | 10 | 0 | -3 | no |
| fixed threshold 0.5 | 0 | 75 | 7 | 0 | -1 | no |
| fixed threshold 0.5 | 1 | 75 | 5 | 0 | -1 | no |
| fixed threshold 0.5 | 2 | 75 | 5 | 0 | -2 | no |

## Interpretation

This is a valid negative promotion precheck. The apparent cross-fit train-tuned gain is concentrated in rows that are not part of the clean val75 actual-evaluation slice. Once the mixed 100-group export is filtered to the val-compatible rows, the best policy reaches only `-3` Tesseract edits, far below the current promoted `-8` actual-bitstream result.

The operational lesson is important: real-seed cross-fit diagnostics can be useful for calibration, but actual stream promotion must first pass a clean split-compatible table precheck. If it does not beat the promoted table delta, expensive counted `.oscr` generation and OCR evaluation should be skipped.

## Artifacts

```text
0871d050b493447df3f0a11bd320e4a4aca0a595f40c6a4961f13f9e62526560  scripts/filter_assignment_policy_rows.py
e7cab5fb44dc2f4312515997c063b45f724b52444aee2f899cc6ff132bfa498a  experiments/results/eval300_group_gate_clean_val_policy_filter_summary_2026_06_25.json
b0ed1845ae71347e42713ec250f3649b2177697fee1edf528a572b229c918e30  experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_crossfit_realseed0_export_train_tuned_seed0_1_2_2026_06_25.json
4d77b357a65b25735d34125e23d361cac8d0f9a51b63d9233efd5e3fa8b5d1d8  experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_crossfit_realseed1_export_train_tuned_seed0_1_2_2026_06_25.json
3f59a4d3b822630b77423d885f74b288b14b2142beb8ad3d00afabca6ace10d9  experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_crossfit_realseed2_export_train_tuned_seed0_1_2_2026_06_25.json
```

## Decision

Do not run these clean-filtered train-tuned or cross-fit fixed-threshold policies through actual `.oscr`; they cannot beat the promoted baseline on the table precheck. Keep the pooled image-diff model seed 1 threshold `0.5` actual-bitstream policy as the active learned baseline.

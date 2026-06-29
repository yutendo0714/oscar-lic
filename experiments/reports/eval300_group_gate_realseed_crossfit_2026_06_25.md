# Eval300 Group Gate Real-seed Cross-fit Diagnostic

**Date:** 2026-06-25  
**W&B:** offline run `z2cse6yf` (`eval300_group_gate_realseed_crossfit_2026_06_25`)  
**Scope:** table-level calibration diagnostic for the pooled image-diff group gate over codegeom oracle-choice assignment scores. This is not an actual `.oscr` promotion result; promotion still requires explicit policy export, counted stream generation and PARSeq/Tesseract comparison.

## Why This Was Run

The promoted learned baseline uses a pooled image-diff group gate at threshold `0.5`. Earlier diagnostics used `seed` as the train/val split key, but the combined top8 assignment table stores the true held-out split ID in `real_seed`. The group-gate trainer was extended with `--split-key-index`; `0` now holds out `real_seed`, while the default `1` preserves the previous `seed` behavior.

A first attempt with the default split axis was interrupted before output because it did not represent the intended three-way cross-fit; no result artifact was produced.

## Result Summary

Three holdouts were run (`real_seed` 0/1/2), each with model seeds 0/1/2. Values below are means over the three real-seed holdouts and three model seeds per holdout, on the existing assignment table.

| policy | Tesseract edits | PARSeq edits | changed | exact | false | wrong | missed | worsen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed threshold `0.5` | `-10.22` | `0.00` | `9.67` | `7.89` | `1.22` | `0.56` | `18.22` | `0.44` |
| train-tuned threshold | `-16.67` | `0.00` | `15.67` | `12.89` | `2.11` | `0.67` | `13.11` | `0.78` |
| assignment always-best | `-29.33` | `-0.33` | `29.00` | `22.00` | `5.67` | `1.33` | `3.33` | `1.33` |
| oracle | `-37.00` | `-0.33` | `26.67` | `26.67` | `0.00` | `0.00` | `0.00` | `0.00` |

Per-holdout Tesseract edit deltas:

- fixed threshold `0.5`: `-11.33`, `-13.67`, `-5.67`
- train-tuned threshold: `-17.67`, `-18.67`, `-13.67`

## Interpretation

This is a positive calibration diagnostic for the current gate family: the fixed threshold generalizes across real-seed holdouts and beats the promoted `-8` table target on average while preserving PARSeq. However, it is not clean enough to replace the promoted actual-bitstream policy by itself:

- `real_seed=2` fixed-threshold performance is weak (`-5.67`).
- wrong changes and Tesseract-worsen groups are nonzero in cross-fit.
- train-tuned thresholds are much stronger but increase false/wrong/worsen groups.
- the table has 100 groups per real seed and is not identical to the already promoted val75 actual `.oscr` comparison.

## Decision

Keep the existing pooled image-diff threshold `0.5` actual-bitstream result as the learned baseline. Treat real-seed train-tuned thresholds as a promising next actual `.oscr` candidate, not as a promoted result. The next promotion attempt should export cross-fit train-tuned policies per real seed and run counted `.oscr` plus row-wise OCR before claiming an improvement.

## Export Support Smoke

After the diagnostic, `scripts/train_assignment_group_gate.py` was extended to write train-tuned explicit policy JSONL files via `--write-train-tuned-policy`. A smoke run on `real_seed=2` produced per-model train-tuned policies:

| model seed | groups | changed | Tesseract edits | PARSeq edits | threshold |
|---:|---:|---:|---:|---:|---:|
| 0 | `100` | `14` | `-13` | `0` | `0.0027915225` |
| 1 | `100` | `16` | `-14` | `0` | `0.0005553526` |
| 2 | `100` | `15` | `-14` | `0` | `0.0019033208` |

W&B run: `z73ruha2`. This is an export/infrastructure smoke only; the mixed 100-group table is still not an actual `.oscr` promotion split.

## Artifact Hashes

```text
b5d843f123b37f03cc19756df2e5c1c2271db36c75df39682c19677476822ea2  scripts/train_assignment_group_gate.py  # cross-fit metric runs
832d19cce5ab0ba93b669268d59aaefd78f8ae53adccdea974f812891ab3625e  scripts/train_assignment_group_gate.py  # after train-tuned policy export support
7d1bca823ce4b651b4034d3347bd6a3ad674aab7a1de43cf9c4d79962ce35296  experiments/results/eval300_assignment_top8_trainval_table_seed0_1_2_codegeom_imgdiff_2026_06_25.jsonl
cf4b0c27568dc4ba1f82cafbc0aa7f8d5e7e08e1e56feb4ef9592c769a233015  experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_crossfit_realseed0_seed0_1_2_2026_06_25.json
a1f77d6d0c86e2d67508070ca3609b141ba3b7a84490129ac6e311388c8a0210  experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_crossfit_realseed1_seed0_1_2_2026_06_25.json
4c4e629a5f19981dc1b59be4e189d1dbc93555c4aa9dcaf11ce608a37ef8e807  experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_crossfit_realseed2_seed0_1_2_2026_06_25.json
3f59a4d3b822630b77423d885f74b288b14b2142beb8ad3d00afabca6ace10d9  experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_crossfit_realseed2_exportcheck_seed0_1_2_2026_06_25.json
b22c25545ae54590260554de618d357ae8dcc8cb7b92b515c3f5e371b0f9a8c0  experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_crossfit_realseed2_exportcheck_explicit_codes_2026_06_25.model0.train_tuned.jsonl
024fce5255e5b267f09ff0a5af4cd8bfa966f494ab848ba50150c327182c9f92  experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_crossfit_realseed2_exportcheck_explicit_codes_2026_06_25.model1.train_tuned.jsonl
81c4c35210095e09c9faddd0287ee2f197b7207ce49d3db95193a4d2730ae213  experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_crossfit_realseed2_exportcheck_explicit_codes_2026_06_25.model2.train_tuned.jsonl
```

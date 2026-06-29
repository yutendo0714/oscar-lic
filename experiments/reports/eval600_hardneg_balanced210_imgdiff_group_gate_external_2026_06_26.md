# Eval600 Hard-Negative Balanced210 Image-Diff Group Gate External Transfer

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N032 rejected a score-only top-k verifier: the current first-stage score/rank/vote features are anti-calibrated on Eval300. This follow-up tests whether adding deployable reconstructed-image/code-effect evidence at the group gate stage is enough to recover external transfer.

The hypothesis was conservative: image-diff features helped the promoted Eval300 pooled group gate, so they might be a better no-op rejector than score-only features when trained on the larger Eval600 hard-negative-balanced table.

## Data Preparation

Added `scripts/prepare_external_assignment_table.py` to make the table keys compatible with the N031 score file:

- Eval600 tables lack `real_seed`, so they are assigned `real_seed=600`;
- Eval300 rows are filtered to `seed=1`;
- the merged table has `548` groups and `4384` rows;
- the merged group keys exactly match the N031 score JSONL: `548/548`, zero missing either way.

Then `scripts/augment_assignment_table_with_image_diff_features.py` added the existing lightweight `img_*` reconstructed-image difference features.

## Training

The existing `scripts/train_assignment_group_gate.py` was run with:

- table: `experiments/results/eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl`
- assignment scores: `experiments/results/eval600_hardneg_iamsupp20_balanced210_to_eval300_assignment_group_noop_window_oraclechoice_scores_2026_06_26.jsonl`
- assignment model seeds: `0`, `1`, `2`
- gate seeds: `0`, `1`, `2`
- target: `oracle_change`
- features: codegeom + `img_*`
- thresholds: fixed `0.5`, fixed `0.8`, and train-tuned

OCR deltas are labels/evaluation only. No counted `.oscr` stream was produced because the table-level policies are not competitive.

## Results

The group gate itself does not transfer:

- gate val AUC/AP, all assignment model seeds: `0.4980 / 0.2267`;
- train AUC/AP: `1.0 / 1.0`, indicating severe train separation and weak external calibration.

| Assignment scorer seed | Always-best Tesseract delta | Fixed 0.5 Tesseract delta | Train-tuned Tesseract delta | Train-tuned false/wrong | Notes |
|---:|---:|---:|---:|---:|---|
| 0 | -2 | 0 | -2 | 5 / 1 | no exact oracle recovered under fixed gate |
| 1 | +2 | +1 | +2 | 9 / 2 | harmful scorer seed |
| 2 | +2 | +1 | +2 | 6 / 1 | harmful scorer seed |

The Eval300 oracle remains `-24` Tesseract edit characters with PARSeq delta `0`.

## Interpretation

This is a valid negative result for directly reusing the hand-built image-diff group gate in the Eval600->Eval300 top-8 assignment transfer setting.

Important nuance:

- image-diff features are still useful in the earlier Eval300-internal promoted gate;
- but as an external Eval600-trained no-op rejector over the current N031 scorer, they do not recover useful Eval300 assignment changes;
- the scorer itself often ranks the wrong code, so a group gate can only reject bad changes. It cannot promote oracle codes that are not the scorer's best row.

The next useful direction is therefore not another group gate over the same scorer. The scorer/reranker needs stronger candidate-local code-effect evidence that can move oracle assignment codes to rank 1, and calibration must be source/domain-disjoint before any counted `.oscr` promotion.

W&B offline run: `f2omvl4p`

## Decision

Do not promote or rerun this external image-diff group gate as a threshold or seed sweep.

Future attempts may reuse image-diff evidence only if it enters a candidate-level code-effect scorer/reranker, not merely a group-level accept/reject gate over the current externally misranked scores.

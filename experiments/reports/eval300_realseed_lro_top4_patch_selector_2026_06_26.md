# Eval300 Leave-Real-Seed-Out Top-4 Patch Selector Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N039 showed that the score-seed-2 top-4 selector fails even when calibrated on neighboring Eval300 `real_seed` shards. This diagnostic tests whether adding deployable candidate-local visual evidence helps:

- nearest reconstruction patch;
- candidate reconstruction patch;
- absolute difference patch;
- existing scalar code/image/score features.

This is still a non-promotion diagnostic. It uses Eval300 oracle labels for training on two real-seed shards and evaluates table-level assignment choices on the held-out shard. No actual `.oscr` stream is claimed.

## Setup

Rows and score files are the same as N039. Candidate set per group is nearest/no-op plus the union top-4 rows from:

- N034 `oracle_choice` diff-crop candidate scorer;
- N035 `safe_improve` diff-crop candidate scorer.

The model is a small CNN over `96x32` grayscale diff-crops concatenated with normalized tabular/score features. It uses the same listwise oracle-choice target and harmful-candidate margin as N039.

## Results

Means are over 3 validation real-seeds x 3 selector seeds.

| Policy | Changed groups | Tesseract delta | False changes | Wrong changes | Exact oracle changes | Missed oracle changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 5.67 | 0.00 |
| shortlist oracle | 5.00 | -6.67 | 0.00 | 0.00 | 5.00 | 0.67 | 0.00 |
| patch selector argmax | 2.22 | +0.11 | 1.89 | 1.89 | 0.33 | 4.67 | 0.00 |
| patch selector train-tuned margin | 2.22 | +0.11 | 1.89 | 1.89 | 0.33 | 4.67 | 0.00 |

Per split:

| Validation split | Shortlist oracle Tesseract delta | Patch argmax Tesseract delta | Patch exact changes | Patch false changes | Patch missed changes |
|---|---:|---:|---:|---:|---:|
| real_seed 0 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| real_seed 1 | -7.00 | -0.33 | 1.00 | 2.33 | 4.33 |
| real_seed 2 | -13.00 | +0.67 | 0.00 | 2.33 | 9.67 |

## Interpretation

This is a valid negative result for the small patch-CNN selector. Compared with N039, candidate-local diff-crop evidence recovers slightly more exact oracle changes (`0.33` versus `0.11`) and gives a weak positive result on `real_seed=1`, but it remains unsafe overall:

- mean Tesseract is still worse than no-op (`+0.11`);
- most oracle changes are missed;
- false/wrong non-nearest changes remain larger than exact useful changes;
- train-tuned margin is identical to argmax, so the learned margin is still not a reliable abstention signal.

The evidence suggests that simply bolting a small patch CNN onto the same top-4 objective is not enough. Future work should either create out-of-fold first-stage scores and a proper calibration set, or move to a stronger policy-level objective that explicitly penalizes false changes and hard-positive misses.

W&B offline run: `ly4wl7xd`

## Files

- Result JSON: `experiments/results/eval300_realseed_lro_top4_patch_selector_scoreseed2_2026_06_26.json`
- Script: `scripts/train_assignment_topk_patch_selector_split_diagnostic.py`

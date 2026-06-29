# Eval600 Source-OOF to Eval300 Top-4 Patch Selector Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N041/N042/N043 show that scalar score/rank features, reward losses, and score-only budget abstention do not recover the top-4 oracle. N040 tested a small patch-CNN selector under same-Eval300 leave-real-seed-out calibration. This diagnostic asks:

> Does adding local nearest/candidate/difference image patches help when training externally on Eval600 with source-OOF score features and evaluating Eval300 seed1?

## Setup

- Table: `eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl`
- Scores: N041 source-OOF mixed oracle-choice and safe-improve scores
- Candidate set: nearest/no-op plus union top-4 from both score files
- Patch input: `96x32` grayscale diff-crop, channels = nearest, candidate, absolute difference
- Model: small 3-layer CNN plus scalar selector features
- Split: train all non-Eval300-seed1 groups, validate Eval300 seed1
- Seeds: selector seeds `0,1,2`

## Results

Means are over 3 selector seeds.

| Policy | Changed groups | Tesseract delta | False changes | Wrong changes | Exact oracle changes | Missed oracle changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| shortlist oracle | 15.00 | -20.00 | 0.00 | 0.00 | 15.00 | 2.00 | 0.00 |
| patch argmax | 5.33 | +3.00 | 5.33 | 5.33 | 0.00 | 15.33 | 0.00 |
| patch train-tuned margin | 1.33 | +1.00 | 1.33 | 1.33 | 0.00 | 16.67 | 0.00 |

## Interpretation

This is a valid negative result for the small local patch selector under external Eval600-to-Eval300 calibration.

Compared with the scalar source-OOF selector, the patch model is more conservative, but it still selects only false/wrong non-nearest changes and recovers zero exact oracle changes. The margin suppresses most changes, but the remaining changes still worsen Tesseract.

The key conclusion is now stronger: small local image-diff patches are not enough as the richer evidence. The next representation should not be another scratch patch-CNN of this scale. It should either use a stronger fixed visual/latent encoder with recorded provenance, or change the local code-effect representation around MLIC++ latent/codebook structure more substantially.

## Next Action

Do not promote or actual-OSCR this selector. Do not capacity-sweep this same small patch-CNN family. The next useful branch should use materially richer evidence, such as:

- structured MLIC++ latent/codebook windows with a different calibration scheme;
- fixed pretrained visual embeddings after recording weight provenance and license risk;
- or an uncertainty/verifier model that combines multiple evidence families.

W&B offline run: `hdxd4tkj`.

## Files

- Script: `scripts/train_assignment_topk_patch_selector_external_diagnostic.py`
- Result: `experiments/results/eval600_source_oof_to_eval300_top4_patch_selector_2026_06_26.json`

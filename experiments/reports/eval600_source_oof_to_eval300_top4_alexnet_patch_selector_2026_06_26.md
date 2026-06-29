# Eval600 Source-OOF to Eval300 Top-4 AlexNet Patch Selector Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N041-N044 showed that score/rank features, reward-policy losses, strict score budgets and a small scratch diff-crop patch-CNN do not recover the top-4 oracle under Eval600-to-Eval300 transfer. This diagnostic asks:

> Does a fixed pretrained visual encoder over local nearest/candidate/difference patches provide enough additional candidate-local evidence to select the useful top-4 assignment changes safely?

## Setup

- Table: `eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl`
- Scores: N041 source-OOF mixed oracle-choice and safe-improve scores
- Candidate set: nearest/no-op plus union top-4 from both score files
- Patch input: `96x32` grayscale diff-crop, channels = nearest, candidate, absolute difference
- Visual encoder: fixed torchvision AlexNet ImageNet1K V1 feature extractor
- Selector: scalar features plus fixed AlexNet patch embedding, MLP head
- Split: train all non-Eval300-seed1 groups, validate Eval300 seed1
- Seeds: selector seeds `0,1,2`

The AlexNet checkpoint was preexisting in the local torch cache and is now recorded in `baselines/checkpoint_manifest.yaml`. It is used only as a non-promotion diagnostic because ImageNet-trained weight/data license terms still require manual publication review.

## Results

Means are over 3 selector seeds.

| Policy | Changed groups | Tesseract delta | False changes | Wrong changes | Exact oracle changes | Missed oracle changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| shortlist oracle | 15.00 | -20.00 | 0.00 | 0.00 | 15.00 | 2.00 | 0.00 |
| AlexNet argmax | 3.00 | +0.67 | 3.00 | 3.00 | 0.00 | 16.33 | 0.00 |
| AlexNet train-tuned margin | 3.00 | +0.67 | 3.00 | 3.00 | 0.00 | 16.33 | 0.00 |

## Interpretation

This is a valid negative result. Fixed AlexNet patch embeddings reduce the aggressiveness of the scratch patch selector in N044, but the selected changes are still all false/wrong under the Eval300 seed1 validation split. The model recovers zero exact oracle changes and produces a small Tesseract regression. PARSeq is unchanged, so the failure is specifically the Tesseract-preservation selection problem rather than a broad OCR instability.

The result narrows the next branch. Generic natural-image features are not enough as a lightweight drop-in verifier for this shortlist. Future work should either:

- use OSCAR-LIC-native local code-effect evidence around latent/codebook structure;
- use an OCR/text-specific fixed verifier with explicit license/provenance review;
- or change the candidate generation/assignment objective so the top-1 scorer is already policy-aligned instead of relying on a shallow second-stage selector.

## Next Action

Do not promote this selector and do not sweep AlexNet MLP capacity. Keep the top-4 shortlist oracle as a non-deployable upper-bound substrate, but move the deployable selector away from generic fixed visual features.

W&B offline run: `idwilglc`.

## Files

- Script: `scripts/train_assignment_topk_alexnet_patch_selector.py`
- Result: `experiments/results/eval600_source_oof_to_eval300_top4_alexnet_patch_selector_2026_06_26.json`
- Feature cache: `experiments/features/eval600_source_oof_to_eval300_top4_alexnet_diffcrop_features_2026_06_26.npz`
- Checkpoint manifest: `baselines/checkpoint_manifest.yaml`

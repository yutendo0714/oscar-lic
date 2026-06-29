# Eval600 Hard-Negative Balanced210 External Diff-Crop Candidate Scorer

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N032 rejected a score-only top-k verifier, and N033 rejected using reconstructed-image/code-effect evidence only as a group-level gate. This diagnostic asks whether putting deployable image evidence directly into the candidate-level scorer fixes the Eval600 -> Eval300 transfer failure.

The tested evidence is a local diff-crop: for each top-8 candidate group, the scorer sees the crop around pixels that differ across code variants, plus the existing deployable image-diff scalar features. OCR deltas are used only for labels, threshold selection diagnostics and evaluation.

## Training

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1 .venv-mlic/bin/python scripts/train_assignment_patch_ranker.py \
  --table experiments/results/eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl \
  --output experiments/results/eval600_hardneg_balanced210_external_diffcrop_candidate_scorer_2026_06_26.json \
  --score-output experiments/results/eval600_hardneg_balanced210_external_diffcrop_candidate_scorer_scores_2026_06_26.jsonl \
  --val-seed 1 \
  --model-seed 0 --model-seed 1 --model-seed 2 \
  --epochs 350 --hidden-dim 48 --dropout 0.15 \
  --loss-mode bce_listwise --listwise-weight 1.0 --pairwise-weight 0.5 \
  --target oracle_choice \
  --image-mode diff_crop --include-image-features \
  --width 96 --height 32
```

Input table:

- `548` groups, `4384` candidate rows;
- Eval600 train groups are source-balanced hard-negative-balanced groups;
- Eval300 seed1 clean-val rows are the fixed external validation target;
- each group has nearest plus top-8 K64 utility-weighted assignment candidates.

## Results

The classifier-looking metrics are strong:

- train AUC/AP: `0.9992 / 0.9945`;
- Eval300 candidate AUC/AP: `0.8791 / 0.6051`.

But the policy is harmful:

| Policy | Changed groups | Tesseract delta | PARSeq delta | Tesseract improve groups | Tesseract worsen groups |
|---|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| always best score | 7.33 | +1.67 | 0.00 | 0.00 | 1.33 |
| top changed 3 | 3.00 | +1.00 | 0.00 | 0.00 | 0.67 |
| train-tuned conservative threshold | 5.00 | +0.67 | 0.00 | 0.00 | 0.67 |
| oracle | 17.00 | -24.00 | 0.00 | 17.00 | 0.00 |

All values are three-model-seed means over Eval300 seed1. Lower Tesseract delta is better.

## Interpretation

This is a sharper negative than N033:

- the local diff-crop scorer can identify positive candidate rows by aggregate AUC/AP;
- nevertheless, its highest-scored deployed choices do not select any Tesseract-improving Eval300 groups;
- all changed policy variants select only neutral or harmful changes, while the oracle remains large at `-24` Tesseract edit characters.

This reinforces that the remaining problem is not simple candidate recall, image localization or row classification. The selection objective must explicitly optimize policy-level safety: no-op precision, false/wrong-change suppression and hard-positive recall under external calibration. A candidate-local scorer can still be useful, but only if its training/evaluation is tied to deployable policy behavior rather than row AUC/AP.

W&B offline run: `xem245nk`

## Decision

Do not promote or rerun this diff-crop candidate scorer as another seed, threshold or capacity sweep.

Future candidate-local evidence should be paired with an explicit policy-level verifier or constrained selector over non-score features, and should be validated on source/domain-disjoint external splits before any counted `.oscr` promotion.

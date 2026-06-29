# Eval600 Hard-Negative Balanced210 External Diff-Crop Safe-Improve Candidate Scorer

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N034 showed that a diff-crop candidate scorer trained with `oracle_choice` labels has high Eval300 row AUC/AP but harmful policy behavior. This follow-up tests whether changing the target to the stricter deployable safety label fixes that gap:

`label_tesseract_parseq_safe_improves = 1` only when the candidate improves Tesseract while not worsening PARSeq.

The representation, split and model capacity are otherwise kept matched to N034 so the diagnostic isolates the label/objective change.

## Training

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1 .venv-mlic/bin/python scripts/train_assignment_patch_ranker.py \
  --table experiments/results/eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl \
  --output experiments/results/eval600_hardneg_balanced210_external_diffcrop_safeimprove_candidate_scorer_2026_06_26.json \
  --score-output experiments/results/eval600_hardneg_balanced210_external_diffcrop_safeimprove_candidate_scorer_scores_2026_06_26.jsonl \
  --val-seed 1 \
  --model-seed 0 --model-seed 1 --model-seed 2 \
  --epochs 350 --hidden-dim 48 --dropout 0.15 \
  --loss-mode bce_listwise --listwise-weight 1.0 --pairwise-weight 0.5 \
  --target safe_improve \
  --image-mode diff_crop --include-image-features \
  --width 96 --height 32
```

## Results

The safe-improve label is not externally learnable with this scorer:

- train AUC/AP: `0.9977 / 0.9699`;
- Eval300 candidate AUC/AP: `0.5186 / 0.0676`.

Policy behavior is worse than N034:

| Policy | Changed groups | Tesseract delta | PARSeq delta | Tesseract improve groups | Tesseract worsen groups |
|---|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| always best score | 23.00 | +3.33 | 0.00 | 1.67 | 4.00 |
| top changed 3 | 3.00 | +1.00 | 0.00 | 0.00 | 1.00 |
| train-tuned conservative threshold | 21.00 | +3.00 | 0.00 | 1.33 | 3.33 |
| oracle | 17.00 | -24.00 | 0.00 | 17.00 | 0.00 |

All values are three-model-seed means over Eval300 seed1. Lower Tesseract delta is better.

## Interpretation

This is a valid negative result for simply swapping the diff-crop candidate scorer target from `oracle_choice` to a stricter safe-improve label.

The model fits Eval600 train labels almost perfectly but does not transfer the rare safe-improvement boundary to Eval300. It also becomes more aggressive: always-best changes 23/75 groups on average, improving a few groups but selecting enough harmful rows to net worsen Tesseract.

This suggests the next attempt should not be another label swap on the same scorer. The useful direction is a group-level policy model that can abstain under uncertainty, or a source/domain-disjoint calibration protocol with explicit false/wrong-change constraints.

W&B offline run: `k5xbgn7o`

## Decision

Do not promote or rerun the diff-crop safe-improve candidate scorer as a seed, threshold, label or capacity sweep.

Future safe-improve supervision should be used only inside a policy-level selector or uncertainty-aware verifier that is explicitly optimized for high-precision no-op/change decisions.

# Eval600 Hard-Negative IAM Supplement20 Assignment Transfer

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

The score calibration audit showed an Eval600-to-Eval300 transfer gap: Eval600 train positives are almost perfectly separated, while Eval300 positive scores collapse toward the no-change distribution. IAM-heavy Eval300 rows are a visible failure mode. This diagnostic asks whether adding a small source-balanced IAM supplement to the hard-negative top-8 assignment substrate improves external calibration for the current group/no-op latent-window head.

This is a table-level assignment diagnostic. No actual `.oscr` stream was produced because the policy did not beat the promoted actual-bitstream baseline.

## New Data

Selection script: `scripts/select_assignment_hard_noheadroom_candidates.py`  
Selection script SHA256: `18a905905aef2b09603bab28a15b1b7d0a497a1777b472f7b022a622698e4f12`

Selection command used the Eval600 no-Eval300 utility table, excluded the prior oracle-safe123 and hard-noheadroom120 selections, required PARSeq base exactness, required no safe source under the prior table, limited Tesseract base distance to 5, and added `--include-source-name iam_words`.

Selection output:

- rows: 20
- source counts: `iam_words=20`
- eligible sources before cap: 34
- excluded sources: 243
- selected Tesseract base distance range: 0-4

Top-8 candidate reconstructions were generated with the frozen corrected MLIC++ checkpoint and the K64 utility-weighted codebook. The generated split has 160 rows, 20 groups and 8 candidates per group.

OCR checks on those 160 candidate reconstructions:

| OCR | W&B run | Samples | CER unicode | Exact unicode | Notes |
|---|---|---:|---:|---:|---|
| PARSeq | `m9hf0uo6` | 160 | 0.0000 | 1.0000 | all candidates are PARSeq exact |
| Tesseract PSM8 | `8eej9xt7` | 160 | 0.6377 | 0.0500 | 0 failed samples |

The resulting assignment utility table has 20 groups. It is not a pure no-headroom set: 5/20 groups have a PARSeq-safe Tesseract-improving assignment change, while 15/20 remain nearest/no-op. Oracle rank counts are `{0: 15, 1: 2, 2: 2, 4: 1}`, with oracle Tesseract delta sum `-6` and PARSeq delta sum `0`.

## Feature Artifact

The IAM supplement was exported as a structured latent-window artifact with local windows shaped `[20, 8, 32, 12, 12]`. A first export using latent window radius 5 produced incompatible 44x44 windows and was immediately overwritten with the radius-1 export; the checked artifact below is the compatible radius-1 version.

The supplement was merged with the existing Eval600 oracle-safe123 + hard-noheadroom120 + Eval300 seed1 val top-8 artifact:

- merged groups: 338
- previous groups: 318
- IAM supplement groups: 20
- merged latent window shapes: `[338, 8, 32, 12, 12]`

## Training

Trainer: `scripts/train_assignment_group_noop_window.py`  
Trainer SHA256: `799685b914aba922c2db976c7727d535174b1e8ecc34e3aab446506c30c0fe77`  
W&B offline run: `iikeyxg2`

Command:

```bash
.venv-mlic/bin/python scripts/train_assignment_group_noop_window.py \
  --features experiments/features/eval600_oraclesafe123_hardnoheadroom120_iamsupp20_plus_eval300_seed1val_top8_windows_2026_06_26.npz \
  --output experiments/results/eval600_hardneg_iamsupp20_to_eval300_assignment_group_noop_window_oraclechoice_2026_06_26.json \
  --score-output experiments/results/eval600_hardneg_iamsupp20_to_eval300_assignment_group_noop_window_oraclechoice_scores_2026_06_26.jsonl \
  --model-seed 0 --model-seed 1 --model-seed 2 \
  --epochs 900 --hidden-dim 96 --dropout 0.2 \
  --row-target oracle_choice \
  --group-change-weight 0.5 \
  --false-change-margin-weight 1.0 \
  --parseq-harm-weight 0.2 \
  --tesseract-nonimprove-weight 0.2 \
  --group-pos-weight --parseq-safe \
  --train-false-change-max 0 \
  --train-wrong-change-max 0 \
  --train-tesseract-worsen-max 0 \
  --val-seed 1 --cuda-required
```

## Results

Eval300 seed1 oracle remains `17` changed groups, PARSeq delta `0`, Tesseract edit delta `-24`.

| Policy | Changed groups mean | Tesseract delta mean | False changes mean | Wrong changes mean | Worsen groups mean | Exact oracle mean |
|---|---:|---:|---:|---:|---:|---:|
| row argmax | 11.33 | -0.33 | 7.00 | 2.33 | 2.00 | 2.00 |
| group threshold 0.5 | 14.00 | 0.00 | 9.67 | 2.67 | 2.00 | 1.67 |
| train-tuned group threshold | 19.00 | -1.00 | 11.67 | 4.00 | 2.67 | 3.33 |

Validation row metrics remain high, but they still do not translate into a safe deployed policy:

- Val AUC/AP mean: `0.8966 / 0.6642`
- Val group AUC mean: `0.7055`
- Train AUC/AP mean: `0.99999 / 0.99996`

Compared with the prior hard-negative oracle-choice run, the IAM supplement does not improve policy quality. The best train-tuned policy reaches only `-1.0` Tesseract edit character on average with `11.67` false changes, below both the N027 hard-negative run and the promoted actual-bitstream baseline (`-8` Tesseract edits with one false change on the comparable compact assignment path).

## Conclusion

This is a valid negative calibration result. A tiny source-balanced IAM supplement adds relevant hard data and contains 5 useful assignment-change groups, but it does not fix the external Eval300 no-op boundary for the current single-score group/no-op latent-window head.

Next work should not seed-sweep this setup. The useful implication is that calibration data must be larger and source/domain-balanced, or the scorer must receive richer deployable candidate-local code-effect evidence. Simply appending 20 IAM groups to the same head and objective is insufficient for promotion.

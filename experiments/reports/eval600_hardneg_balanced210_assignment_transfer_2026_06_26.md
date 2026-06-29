# Eval600 Hard-Negative Balanced210 Assignment Transfer

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N030 showed that a tiny IAM-only supplement is not enough to repair Eval600-to-Eval300 assignment calibration. This follow-up tests a larger source-balanced calibration append: use remaining Eval600 no-safe-source groups across all six sources, export the same deployable latent-window/code-effect evidence, and train the same group/no-op assignment head under the same objective.

The purpose is to decide whether the current architecture mainly needs more source-balanced labels, or whether it still lacks the deployable evidence/no-op structure needed for high-precision assignment.

No actual `.oscr` stream was produced because the table-level policy did not beat the promoted actual-bitstream baseline.

## Selection And Labeling

Selection script: `scripts/select_assignment_hard_noheadroom_candidates.py`  
Selection script SHA256: `18a905905aef2b09603bab28a15b1b7d0a497a1777b472f7b022a622698e4f12`

Selection constraints:

- start from the Eval600 no-Eval300 single-candidate utility table;
- exclude the prior oracle-safe123, hard-noheadroom120 and IAM supplement20 selections;
- require no safe single-candidate source;
- require PARSeq base exactness;
- require Tesseract base distance <= 6;
- cap each source name at 40 groups.

Selection result:

| Source | Groups |
|---|---:|
| iam_words | 14 |
| icdar2013 | 40 |
| icdar2015 | 40 |
| icdar_mlt2019_words | 40 |
| synthtext_words | 36 |
| textocr | 40 |
| total | 210 |

Top-8 K64 assignment candidate generation produced 1680 reconstructions. OCR results:

| OCR | W&B run | Samples | CER unicode | Exact unicode | Notes |
|---|---|---:|---:|---:|---|
| PARSeq | `cucaosnh` | 1680 | 0.0000 | 1.0000 | all candidates remain PARSeq exact |
| Tesseract PSM8 | `lkvvfxzo` | 1680 | 0.2672 | 0.4851 | 0 failed samples |

The assignment utility table has 210 groups and 1680 rows. Although all selected sources were no-safe under the single-candidate table, top-8 code assignment creates nontrivial headroom:

- safe Tesseract/PARSeq assignment-improving rows: 76
- assignment oracle changed groups: 19/210
- oracle Tesseract gains are all PARSeq-safe
- oracle rank counts: `{0: 191, 1: 9, 2: 8, 3: 2}`

## Feature Artifact

The balanced supplement was exported with the same radius-1 structured latent windows as N030:

- supplement groups: 210
- supplement rows: 1680
- latent window shape: `[210, 8, 32, 12, 12]`
- oracle missing groups: 0

It was merged into the previous 338-group artifact, producing a 548-group artifact:

- previous groups: 338
- balanced supplement groups: 210
- merged latent window shape: `[548, 8, 32, 12, 12]`

## Training

Trainer: `scripts/train_assignment_group_noop_window.py`  
Trainer SHA256: `799685b914aba922c2db976c7727d535174b1e8ecc34e3aab446506c30c0fe77`  
W&B offline summary run: `5q05gb1z`

The training command matches N030 except for the merged feature artifact:

```bash
.venv-mlic/bin/python scripts/train_assignment_group_noop_window.py \
  --features experiments/features/eval600_oraclesafe123_hardnoheadroom120_iamsupp20_balanced210_plus_eval300_seed1val_top8_windows_2026_06_26.npz \
  --output experiments/results/eval600_hardneg_iamsupp20_balanced210_to_eval300_assignment_group_noop_window_oraclechoice_2026_06_26.json \
  --score-output experiments/results/eval600_hardneg_iamsupp20_balanced210_to_eval300_assignment_group_noop_window_oraclechoice_scores_2026_06_26.jsonl \
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
| row argmax | 8.67 | +0.67 | 6.67 | 1.33 | 2.33 | 0.67 |
| group threshold 0.5 | 10.33 | +0.33 | 7.67 | 1.67 | 2.33 | 1.00 |
| train-tuned group threshold | 14.33 | -1.33 | 10.00 | 2.00 | 2.67 | 2.33 |

Row metrics remain high but still do not translate into a deployable policy:

- Val AUC/AP mean: `0.8969 / 0.6807`
- Val group AUC mean: `0.6893`
- Train AUC/AP mean: `0.9999 / 0.9994`

The best individual model seed under train-tuned thresholds reaches `-5` Tesseract edits, but with 10 false changes and one Tesseract-worsen group. The three-seed mean remains far below the promoted actual-bitstream baseline (`-8` Tesseract edits with one false change) and far below the Eval300 oracle (`-24`).

## Top-M Diagnostic

I also ran a non-deployable upper control: if the true oracle code is within the scorer top-M, select the oracle; otherwise keep nearest. This asks whether the scorer can at least form a useful shortlist.

| Model seed | Top-1 recovered | Top-2 recovered | Top-4 recovered | Top-4 Tesseract delta |
|---:|---:|---:|---:|---:|
| 0 | 0/17 | 5/17 | 11/17 | -16 |
| 1 | 1/17 | 6/17 | 9/17 | -12 |
| 2 | 1/17 | 4/17 | 10/17 | -13 |

This is the key positive diagnostic inside the negative result: top-1/no-op calibration fails, but a top-4 shortlist still carries 9-11 of the 17 useful Eval300 oracle changes. The next model should therefore be a second-stage verifier or high-precision no-op rejector over the shortlist, not another first-stage score threshold.

## Conclusion

This is a valid negative result for "more balanced labels plus the same single-score group/no-op head". The larger supplement is useful infrastructure: it adds 210 source-balanced no-safe groups and discovers 19 additional top-8 assignment positives. However, the current head still cannot transfer a high-precision no-op/change boundary to Eval300.

The next assignment attempt should change the model information or decision structure, not only the amount of similar calibration data. In particular, prioritize a separate high-precision no-op rejector, a top-k shortlist plus second-stage verifier, or richer candidate-local code-effect evidence that distinguishes useful rank-1/2/3 changes from false changes.

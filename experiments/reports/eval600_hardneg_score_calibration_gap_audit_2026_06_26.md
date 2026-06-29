# Eval600 to Eval300 Assignment Score Calibration Gap Audit

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

After N028 and N029, the remaining question is whether the current Eval600-trained assignment scores fail because the final policy knob is weak, or because the score/no-op boundary itself does not transfer to Eval300. This audit joins score JSONL files with the source/difficulty metadata in the assignment tables and summarizes score separation by partition, source and OCR difficulty.

Script: `scripts/analyze_assignment_score_calibration_gap.py`  
Script SHA256: `c821ff818567bb029d5f3f40d37c359d05acc6e3c585be7b67221ac63f6451c1`  
W&B offline run: `c5jzanwn`

## Inputs

Scores:

- hard-negative oracle-choice baseline scores
- matched pairwise group+row scores
- matched pairwise row-heavy scores

Tables:

- `experiments/results/eval600_assignment_oraclesafe123_k64_utilityw_a8_top8_table_2026_06_25.jsonl`
- `experiments/results/eval600_assignment_hard_noheadroom120_k64_utilityw_a8_top8_table_2026_06_26.jsonl`
- `experiments/results/eval300_assignment_top8_trainval_table_seed0_1_2_codegeom_imgdiff_2026_06_25.jsonl`

The metadata join had zero missing groups.

## Core Finding

The score boundary is externally miscalibrated.

For the oracle-choice scorer:

| Partition | Group type | Groups | Best-change score mean | q10 | q50 | q90 |
|---|---|---:|---:|---:|---:|---:|
| Eval600 train | no-change | 173 | 0.0081 | 0.00017 | 0.00074 | 0.02468 |
| Eval600 train | oracle-change | 70 | 0.6785 | 0.60149 | 0.68638 | 0.77932 |
| Eval300 val | no-change | 58 | 0.0578 | 0.00112 | 0.00616 | 0.13867 |
| Eval300 val | oracle-change | 17 | 0.1889 | 0.01557 | 0.06743 | 0.60112 |

For the pairwise group+row scorer:

| Partition | Group type | Groups | Best-change score mean | q10 | q50 | q90 |
|---|---|---:|---:|---:|---:|---:|
| Eval600 train | no-change | 173 | 0.00034 | 0.000036 | 0.000104 | 0.000782 |
| Eval600 train | oracle-change | 70 | 0.9734 | 0.93695 | 0.99371 | 0.99933 |
| Eval300 val | no-change | 58 | 0.0307 | 0.000126 | 0.000542 | 0.09023 |
| Eval300 val | oracle-change | 17 | 0.1050 | 0.000875 | 0.00947 | 0.33938 |

The training split is almost perfectly separable. Eval300 positives collapse toward the no-change distribution, so thresholds learned on Eval600 either select nothing or select many false changes.

## Source And Difficulty Signals

Eval300 always-best selection is not uniformly bad. It concentrates failures in specific sources:

- pairwise group+row, Eval300 IAM: 36 groups, 10 oracle-change groups, Tesseract delta `+2`, 26 false changes, 7 wrong changes.
- pairwise group+row, Eval300 ICDAR2015: 9 groups, 3 oracle-change groups, Tesseract delta `-4`, 6 false changes, 1 wrong change.
- no-oracle-change sources still get selected: TextOCR has 9/9 false changes under always-best.

OCR difficulty also matters:

- Tesseract nearest distance `0`: 16 groups, 0 oracle-change groups, all 16 become false changes under always-best.
- Tesseract nearest distance `3-5`: 26 groups, 9 oracle-change groups, pairwise group+row gives Tesseract delta `-5` but still 17 false and 5 wrong changes.
- Tesseract nearest distance `6+`: 9 groups, 4 oracle-change groups, but pairwise group+row worsens by `+3`, indicating high base OCR error alone is not a safe headroom proxy.

## Implication

This supports ADR-016: post-hoc policy budgets do not replace deployable evidence. The next useful work should improve the score representation or calibration data:

- add candidate-local code-effect evidence that separates Eval300 oracle-changes from no-change groups, especially IAM-like hard positives;
- add source/domain-disjoint calibration data rather than tuning thresholds on the same Eval600-selected substrate;
- keep no-op as an explicit decision, but do not expect score thresholds alone to solve it.

No actual bitrate claim is made from this audit.

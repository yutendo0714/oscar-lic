# Eval600/Eval300 STRHub Train-Teacher Loss Transfer Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream assignment diagnostic; no counted `.oscr` promotion stream

## Question

N055 showed that CRNN/ABINet confidence is dense but unsafe. This follow-up asks whether a more principled train-teacher signal works better:

> Does per-candidate teacher loss against ground-truth text provide a deployable proxy for OCR-useful assignment changes, without using held-out OCR labels for selection?

Held-out Tesseract is used only for evaluation and upper-bound rows. It is not used as a feature, training target, threshold signal or selector input.

## Setup

- Assignment table: `experiments/results/eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl`.
- Train-like split: 473 Eval600 groups.
- Validation split: fixed Eval300 seed1 val75 groups.
- Teacher losses:
  - CRNN: per-sample CTC loss against STRHub charset-adapted ground-truth text.
  - ABINet: per-sample cross-entropy loss against STRHub charset-adapted ground-truth text.
- OCR predictions and metrics still use the original reference string; charset adaptation is only for the internal teacher-loss target.
- W&B runs:
  - CRNN loss extraction: `e5bqv22c`
  - ABINet loss extraction: `9pk49iuy`
  - Aggregate analysis: `s4xivdqz`

## Loss Signal Density

| Split | Groups | CRNN loss-improve groups | ABINet loss-improve groups | Either loss-improve groups | Both teacher loss no-worse groups | Tesseract oracle-change groups |
|---|---:|---:|---:|---:|---:|---:|
| Eval600 train-like | 473 | 424 | 431 | 465 | 354 | 95 |
| Eval300 seed1 val75 | 75 | 67 | 66 | 75 | 51 | 17 |
| All | 548 | 491 | 497 | 540 | 405 | 112 |

Loss decrease is far denser than true Tesseract utility. It is therefore a dangerous primary selector signal unless paired with a much stronger abstention/verifier objective.

## Validation Results

Policies below are applied to Eval300 seed1 val75. Tesseract deltas are edit-character deltas versus nearest; lower is better.

| Policy | Changed | Exact oracle | False changes | Tesseract delta | Tesseract worsen groups | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|
| nearest | 0 | 0 | 0 | 0 | 0 | 0 |
| held-out Tesseract oracle | 17 | 17 | 0 | -24 | 0 | 0 |
| CRNN loss mean argmax | 67 | 3 | 52 | +3 | 8 | 0 |
| ABINet loss mean argmax | 66 | 5 | 50 | 0 | 9 | 0 |
| mean loss mean argmax | 67 | 5 | 51 | -2 | 8 | 0 |
| mean loss sum argmax | 65 | 5 | 49 | -1 | 8 | 0 |
| train-teacher loss no-worse held-out oracle | 5 | 3 | 0 | -6 | 0 | 0 |

The last row is non-deployable because it still uses held-out Tesseract oracle choice, then filters it by CRNN/ABINet loss no-worse. It is useful as an upper-bound diagnostic: even with oracle choice, the loss no-worse constraint preserves only `-6` edits on Eval300 seed1, below the current actual-bitstream learned baseline's `-8` aggregate gain across seed splits.

## Interpretation

Teacher loss is a better-defined signal than raw confidence, but it is still not a safe selector. It improves on the sparsity problem from N055, yet it is too permissive: nearly every Eval300 group has at least one candidate with lower CRNN or ABINet loss, while only `17/75` groups have held-out Tesseract oracle headroom. Direct argmax over loss decrease changes 65-67 groups and produces 49-52 false changes.

The no-worse upper bound is also a caution. If CRNN and ABINet loss no-worse is enforced as a hard constraint, it removes most of the held-out Tesseract oracle headroom on the Eval300 validation split. That means a future teacher-aware method should not be "choose lower teacher loss" or "hard-filter by both teacher losses." It needs a calibrated verifier, policy-level abstention objective, or richer code-effect representation that can learn when teacher-loss changes are semantically meaningful.

## Decision

This is N056, a valid negative result for direct CRNN/ABINet teacher-loss argmax and hard loss-no-worse filtering as assignment selection rules.

The positive artifact is the new per-sample teacher-loss runner, which can still support auxiliary losses or calibration diagnostics. Use it as an analysis substrate, not as a primary selection rule.

## Files

- Teacher-loss runner: `scripts/run_strhub_teacher_loss.py`
- Transfer analyzer: `scripts/analyze_strhub_teacher_loss_transfer.py`
- CRNN loss output: `experiments/results/crnn_loss_eval600_balanced210_plus_eval300_seed1_top8_candidates_2026_06_26/results.jsonl`
- ABINet loss output: `experiments/results/abinet_loss_eval600_balanced210_plus_eval300_seed1_top8_candidates_2026_06_26/results.jsonl`
- Aggregate result: `experiments/results/eval600_eval300_strhub_trainteacher_loss_transfer_2026_06_26.json`

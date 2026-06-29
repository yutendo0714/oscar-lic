# Eval300 Candidate-Local PARSeq Teacher Row Scorer

Date: 2026-06-25

## Purpose

This diagnostic tests whether PARSeq teacher evidence becomes useful when distilled at candidate-row level instead of group level. Inputs deliberately exclude direct OCR outputs and teacher features at inference:

- excluded: `img_teacher_*`, `parseq_*`, `tesseract_*`, `label_*`, `group_*`, `assignment_oracle_*`;
- included: deployable codebook/code relation features and non-teacher image-difference statistics;
- selection: nearest is the explicit no-op anchor, and non-nearest candidates are selected only by model score margin.

Promotion baseline remains the actual `.oscr` pooled image-diff gate: PARSeq unchanged, Tesseract `-8` edits, `5/75` changed groups, `1` false change.

## Runs

Shared settings:

- model seeds: `0,1,2`
- target split: `group_key[1] == 1` as the 75-group clean val shard
- feature dimension: `93`
- model: small row MLP, `hidden_dim=96`, `dropout=0.2`
- training: `200` CPU epochs, BCE against teacher-derived positive rows
- policy calibration: train-tuned margin with at most `1` train false change and `0` train PARSeq-worsen groups

Targets:

1. `combo`: max(PARSeq teacher edit gain, normalized edit gain) plus confidence gain.
2. `conf_gain`: positive PARSeq teacher confidence delta.
3. `norm_edit_gain`: normalized PARSeq teacher edit gain.

## Results

| Target | Val AUC/AP | Best non-oracle policy observation |
|---|---:|---|
| `combo` | `0.5867 / 0.4634` | train-tuned nearly no-op: `0` Tesseract edits, `0.33` changes; unconstrained score selection is harmful (`+2` edits, `58` false changes) |
| `conf_gain` | `0.5812 / 0.4609` | best fixed margin is still far below baseline: about `-2.33` Tesseract edits with `>54` false changes; train-tuned is no-op |
| `norm_edit_gain` | unavailable on val because val has no positives | train-tuned is effectively no-op; unconstrained selection worsens Tesseract by `+6.33` edits |

The clean-val oracle remains `-24` Tesseract edits with PARSeq unchanged and `17` exact oracle groups, so headroom remains high.

## Interpretation

Candidate-local teacher distillation is not sufficient with the current deployable scalar codebook/image-diff features. The models can fit the teacher target on train and show moderate val AP for confidence-style targets, but the score scale is not aligned with OCR-safe Tesseract changes. Unconstrained selection changes nearly every group and creates many false changes; strict train calibration collapses to no-op and misses all oracle-change groups.

This is a valid table-level negative diagnostic. It does not justify counted `.oscr` generation.

## Next

Do not retry this exact scalar row-scorer setup. The next useful step is to add stronger candidate-local evidence: real local image/code-effect patches or latent-window tensors tied to the teacher target, plus an explicit group-level no-op precision loss. Any new variant must first beat the promoted table precheck before actual bitstream evaluation.

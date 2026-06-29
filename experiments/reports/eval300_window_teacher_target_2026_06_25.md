# Eval300 Latent-Window PARSeq Teacher Target Diagnostic

Date: 2026-06-25

## Purpose

This diagnostic asks whether richer deployable local evidence rescues PARSeq teacher distillation. It augments the existing nearest-plus-top4 structured latent-window artifact with PARSeq teacher target arrays, then trains the tensor-backed code-effect reranker using `teacher_parseq_combo_gain` or `teacher_parseq_conf_gain` as the group target.

This is still a table-level precheck. It is not an actual `.oscr` promotion.

## Artifact

- Base feature artifact: `experiments/features/eval300_assignment_top4_codeeffect_latentctx_windows_seed0_1_2_2026_06_25.npz`
- Teacher-augmented artifact: `experiments/features/eval300_assignment_top4_codeeffect_latentctx_windows_parseqteacher_seed0_1_2_2026_06_25.npz`
- Valid rows matched to PARSeq table: `1213/1213`
- Teacher positives in top4 artifact:
  - edit gain: `2` rows, `1` group
  - confidence gain: `444` rows, `215` groups
  - combo gain: `446` rows, `216` groups

Important: the top4 artifact contains only `10` actual oracle-change val groups, whereas the full top8 clean-val table has `17`. This limits recoverable headroom before learning begins.

## Results

| Target array | Val AUC/AP vs actual oracle labels | `model_argmax` policy | Best margin precheck |
|---|---:|---|---|
| `teacher_parseq_combo_gain` | `0.3753 / 0.2252` | worsens Tesseract by `+9.33` edits, `56.33` false changes | best parseq-nonworse margin still `+2` to `+7` Tesseract edits depending seed/mean |
| `teacher_parseq_conf_gain` | `0.3888 / 0.2336` | worsens Tesseract by `+7.67` edits, `56.67` false changes | best parseq-nonworse margin still `+2` to `+5` Tesseract edits |

The top4 oracle itself is `-15` Tesseract edits with PARSeq unchanged and `0` false/wrong changes, so the artifact still has useful headroom. The teacher-trained model does not recover it.

## Interpretation

Richer latent-window evidence alone does not solve the teacher distillation problem. The dense confidence-gain teacher target is not aligned with held-out Tesseract-safe code assignment, and the sparse edit-gain target almost disappears after nearest-plus-top4 filtering. Training on confidence-style targets encourages broad unsafe changes; margin gating only suppresses damage toward no-op and never beats the promoted actual-bitstream gate.

This is a valid negative result. Do not run counted `.oscr` generation from these policies.

## Next

The next model should not use PARSeq confidence gain as a primary assignment target. More promising options are:

- train directly on actual safe assignment labels while using teacher features only as auxiliary regularization;
- improve candidate shortlist recall before teacher distillation;
- add an explicit false-change/no-op objective tied to policy precision rather than only cross-entropy against teacher-selected rows.

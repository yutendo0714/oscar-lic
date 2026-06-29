# Eval300 PARSeq Teacher Target Distillation Diagnostic

Date: 2026-06-25

## Purpose

This diagnostic tests whether PARSeq teacher evidence can be distilled into the deployable pooled image-diff group gate without using PARSeq inference-time features. It uses the PARSeq-augmented assignment table only for target construction, then excludes `img_teacher_*` features from the group gate inputs.

Promotion baseline remains the actual-bitstream pooled image-diff gate:

- Tesseract PSM8 delta: `-8` unicode edits and `-8` latin-alnum edits.
- PARSeq: unchanged.
- Policy shape: `5/75` changed groups, `4` exact oracle hits, `1` false change, `0` wrong changes.

## Inputs

- Table: `experiments/results/eval300_assignment_top8_trainval_table_seed0_1_2_codegeom_imgdiff_parseqfeat_2026_06_25.jsonl`
- Assignment scores:
  - `experiments/results/eval300_assignment_top8_patch_ranker_codegeom_oraclechoice_scores_seed0_model0_2026_06_25.jsonl`
  - `experiments/results/eval300_assignment_top8_patch_ranker_codegeom_oraclechoice_scores_seed1_model0_2026_06_25.jsonl`
  - `experiments/results/eval300_assignment_top8_patch_ranker_codegeom_oraclechoice_scores_seed2_model0_2026_06_25.jsonl`
- Trainer: `scripts/train_assignment_group_gate.py`

Shared settings:

- `--epochs 100`
- `--hidden-dim 32`
- `--dropout 0.2`
- `--lr 0.002`
- `--weight-decay 0.001`
- `--include-image-features`
- `--exclude-teacher-image-features`
- `--cpu`
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`

The earlier 500-epoch CPU attempts were interrupted because unconstrained CPU threading made them too slow. They are not used as evidence.

## Results

| Target | Val AUC/AP | Fixed 0.5 / 0.8 policy | Train-tuned policy |
|---|---:|---|---|
| `parseq_teacher_edit_improve` | unavailable | no-op: `0` Tesseract edits, `0` changes | `-3` Tesseract edits, `23` changes, `15` false changes, `4` wrong changes, `4` worsen groups |
| `parseq_teacher_norm_edit_improve` | unavailable | no-op: `0` Tesseract edits, `0` changes | `-3` Tesseract edits, `23` changes, `15` false changes, `4` wrong changes, `4` worsen groups |
| `parseq_teacher_conf_gain` | `0.7324 / 0.1221` mean | no-op: `0` Tesseract edits, `0` changes | `-3` Tesseract edits, `22.67` changes, `14.67` false changes, `4` wrong changes, `4` worsen groups |

For context, the oracle over the same clean val groups is still `-24` Tesseract edits with PARSeq unchanged, `17` changed groups and `0` false/wrong changes.

## Interpretation

The teacher target alone does not create a safe deployable selector. The confidence-gain target has moderate validation AUC, but the learned probabilities are poorly calibrated for policy use: fixed thresholds collapse to no-op, while train-tuned thresholds behave close to the unsafe assignment-always-best policy and produce many false/wrong changes.

This is a valid negative result, not an invalid run. It used clean val rows, excluded direct teacher image features at inference, and evaluated policy-level OCR deltas. No counted `.oscr` promotion should be run from these policies.

## Conclusion

Do not retry group-level PARSeq teacher targets with only scalar image-diff features and threshold tuning. The useful direction is a candidate-local teacher-distilled model that predicts OCR-safe code effects from deployable visual/latent evidence, with an explicit no-op precision objective and a clean table precheck before any actual stream generation.

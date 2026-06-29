# Eval300 PARSeq Teacher Feature Gate Diagnostic

Date: 2026-06-25

## Purpose

Add candidate-level PARSeq teacher confidence and edit-distance features to the assignment table, then retrain the pooled image-feature group gate. This tests whether OCR-teacher evidence helps recover more Tesseract held-out headroom without changing actual payload size.

This is a table-level diagnostic. It uses PARSeq teacher inference over candidate reconstructions, so it is not promoted as a deployable pure-image policy without a learned proxy or explicit encoder-cost accounting.

## Teacher Extraction

- OCR split: 2400 assignment candidate reconstructions.
- PARSeq checkpoint SHA256: `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`.
- W&B PARSeq run: `csp3watg`.
- Mean confidence: `0.902327`.
- Unicode CER over candidate reconstructions vs references: `0.053785`.

Added features include confidence, token count, prediction length, edit distance/NED to reference, normalized edit distance/NED, and deltas against the nearest-code reconstruction.

## Result

The result is a useful tie diagnostic, not a promotion.

- threshold `0.5`:
  - mean Tesseract delta `-7`;
  - best model seeds tie the promoted `-8`;
  - but the tied seeds use 7 changed groups with 3 false changes, worse than the promoted image-diff gate's 5 changed groups and 1 false change.
- threshold `0.8`: mean `-4.33`, weaker.
- train-tuned threshold: mean `-5.33`, weaker with more false/wrong changes.

PARSeq teacher features therefore move closer to the promoted result than scalar imgstroke features, but they do not improve the actual-bitstream baseline and reduce policy precision.

## Conclusion

Do not promote or run counted `.oscr` streams for this PARSeq-feature gate.

The result does support the broader OSCAR direction: OCR-teacher evidence is more relevant than generic stroke/projection scalars. The next step should distill teacher evidence into a deployable candidate-local predictor and evaluate it under clean split rules, rather than directly using PARSeq outputs as inference-time features.

W&B runs:

- teacher inference: `csp3watg` (`parseq_eval300_assignment_top8_imgdiff_teacher_2026_06_25`);
- gate diagnostic: `fcd0lfon` (`eval300_parseq_teacher_feature_gate_tie_negative_2026_06_25`).

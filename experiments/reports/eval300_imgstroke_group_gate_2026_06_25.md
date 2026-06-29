# Eval300 Imgstroke Group-Gate Diagnostic

Date: 2026-06-25

## Purpose

Add lightweight stroke-oriented reconstructed-image features to the assignment table and test whether they improve the pooled image-diff group gate.

New features include dark/ink fractions, ink centers, row/column intensity projections, dark-pixel projections, and dark/edge-region difference statistics. The assignment scorer and split are kept fixed relative to the promoted pooled image-diff gate.

## Result

The diagnostic is negative.

Compared with the promoted pooled image-diff gate (`-8` Tesseract edits, PARSeq unchanged, 5 changed groups), the imgstroke gate is weaker:

- fixed threshold `0.5`: mean `-5` Tesseract edits, PARSeq unchanged, `8.67` changed groups, `3.67` false changes, `1.67` wrong changes;
- fixed threshold `0.8`: same aggregate as threshold `0.5`;
- train-tuned threshold: mean `-5` Tesseract edits, `13.33` changed groups, `7.33` false changes, `2.0` wrong changes.

Validation AP rises to `0.432` mean, but policy-level OCR and precision worsen. This is another instance where a seemingly better classifier signal does not transfer to the actual selection objective.

## Conclusion

Do not promote the imgstroke group gate and do not run counted `.oscr` streams for it.

The added stroke/projection features may remain useful as diagnostic covariates, but adding more scalar image statistics to the current gate is not enough. The next model needs candidate-local evidence that directly separates safe extra oracle-change groups from false changes.

W&B offline run: `t4m1mhub` (`eval300_imgstroke_group_gate_negative_2026_06_25`).

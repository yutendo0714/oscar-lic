# Eval300 Top8 Group/No-Op Window Diagnostic

Date: 2026-06-25

## Purpose

This run applies the existing explicit group/no-op assignment head to the new nearest-plus-top8 structured latent-window artifact. The aim is to test whether fixing candidate recall plus a learned change/no-change head beats the promoted actual-bitstream image-diff gate.

## Configuration

- Feature artifact: `experiments/features/eval300_assignment_top8_codeeffect_latentctx_windows_parseqteacher_seed0_1_2_2026_06_25.npz`
- Model seeds: `0,1,2`
- Epochs: `400`
- Hidden dim: `96`
- Row target: actual oracle choice
- Group target: any oracle change
- Loss weights:
  - listwise: `1.0`
  - group change: `0.5`
  - false-change margin: `1.0`
  - PARSeq harm: `0.2`
  - Tesseract non-improve: `0.2`

## Results

- Row val AUC/AP: `0.8921 / 0.5752`
- Group val AUC: `0.6897`
- Default group threshold `0.5`: `-4` Tesseract edits, `11.33` false changes, `2.33` wrong changes.
- Train-tuned threshold: `-2.67` Tesseract edits, `13.67` false changes, `3.0` wrong changes.
- Oracle: `-24` Tesseract edits, PARSeq unchanged, `0` false/wrong changes.

The model perfectly fits the train-side oracle-change groups under the selected threshold, but this does not transfer to the clean val shard.

## Decision

This is a valid negative precheck. Do not generate counted `.oscr` streams from this policy.

The failure mode is now clearer: top8 candidate recall is fixed, but the group/no-op head overfits and still ranks false changes too high on clean val. The next attempt needs either stricter cross-fit calibration, an objective that directly penalizes validation-like false changes, or more data before more architecture tweaks.

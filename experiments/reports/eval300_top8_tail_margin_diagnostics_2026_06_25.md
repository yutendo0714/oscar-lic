# Eval300 Top8 Tail-Margin Diagnostics

Date: 2026-06-25

## Purpose

The previous top8 policy-error analysis showed two policy tails:

- rank-0/no-headroom groups still create many false changes;
- oracle-change groups, including close-rank positives, are often missed.

This diagnostic keeps the same top8 structured latent-window substrate and adds only targeted loss terms to the existing group/no-op trainer:

- `rank0_row_margin_weight`: force the nearest row above the best non-nearest row for no-headroom groups;
- `hard_positive_row_margin_weight`: force the oracle row above nearest for Tesseract-improving oracle-change groups;
- `hard_positive_group_margin_weight`: push the group-change logit positive for those hard positives.

This is a clean table precheck. No counted `.oscr` stream is promoted.

## Runs

| Variant | Window | Hidden/dropout/wd | Tail weights | Val AUC/AP | Group AUC | Best useful policy |
|---|---|---|---|---:|---:|---|
| `r1_hp1` | yes | `96 / 0.2 / 0.001` | rank0 `1.0`, row hard-positive `1.0`, group hard-positive `0.5` | `0.8861 / 0.5733` | `0.7067` | threshold 0.5: `-4.33` Tesseract edits, `12.0` false, `3.33` wrong |
| `reg_rank0heavy` | yes | `64 / 0.4 / 0.01` | rank0 `3.0`, row hard-positive `0.5`, group hard-positive `0.2` | `0.8889 / 0.5774` | `0.7204` | threshold 0.5: `-3.67` Tesseract edits, `12.0` false, `4.33` wrong |
| `codeonly` | no | `64 / 0.4 / 0.01` | rank0 `3.0`, row hard-positive `1.0`, group hard-positive `0.5` | `0.8878 / 0.5952` | `0.6785` | threshold 0.5: `-2.0` Tesseract edits, `10.67` false, `4.0` wrong |

The promoted actual-bitstream pooled image-diff baseline remains stronger: `-8` Tesseract unicode / `-8` latin-alnum edits, PARSeq unchanged, `5/75` changed groups, `1` false change and `0` wrong changes.

## Interpretation

The targeted tail-margin losses do not fix the policy tails. They preserve or slightly improve row AUC/AP, but the model still creates 10-14 false changes and 3-4 wrong changes on clean val, while recovering only about 3 exact oracle groups.

The code-only variant reduces false changes slightly, but loses useful Tesseract gain. The window variants retain more gain but keep too many bad changes. This is the same row-metric/policy-metric gap seen in earlier diagnostics.

## Decision

Do not run counted `.oscr` streams for these tail-margin variants.

The loss-level patch is useful infrastructure, but the result is a valid negative control. The next attempt should not simply increase these margin weights. More promising routes are:

- more labeled groups or cross-fit calibration that better estimates clean-val false-change risk;
- a separate no-headroom rejector trained/evaluated as a high-precision classifier before row selection;
- source/difficulty-aware calibration, especially for IAM and high Tesseract-distance samples;
- a richer candidate-local representation, but only if it is judged by policy errors rather than row AUC/AP.

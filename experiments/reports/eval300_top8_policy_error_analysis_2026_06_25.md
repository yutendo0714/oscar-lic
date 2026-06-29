# Eval300 Top8 Policy Error Analysis

Date: 2026-06-25

## Purpose

This analysis compares two failed top8 assignment policies on the same clean Eval300 val75 groups:

- row-only latent-window code-effect reranker trained on actual oracle labels;
- explicit group/no-op window head trained on the same top8 structured latent-window substrate.

The goal is not promotion. The goal is to identify whether the remaining failure is mostly row ranking, group no-op calibration, source-domain drift, or OCR-difficulty stratification.

## Inputs

- Group/no-op policy errors: `experiments/results/eval300_group_noop_window_top8_default_seed0_policy_errors_2026_06_25.json`
- Row-only code-effect policy errors: `experiments/results/eval300_codeeffect_window_top8_oraclelabels_seed0_policy_errors_2026_06_25.json`
- Analyzer: `scripts/analyze_assignment_policy_errors.py`

Both policies use model seed `0` score outputs and the clean Eval300 val75-compatible rows. These are table-level policy diagnostics only; no counted `.oscr` stream is promoted.

## Aggregate Results

| Policy | Groups | Oracle-change | Predicted-change | Tesseract edit delta | PARSeq delta | Missed oracle | False change | Wrong change | Exact oracle | Correct nearest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| group/no-op top8 | 75 | 17 | 17 | -4 | 0 | 12 | 12 | 2 | 3 | 46 |
| row-only top8 | 75 | 17 | 16 | -4 | 0 | 12 | 11 | 2 | 3 | 47 |

The two models fail in almost the same way. The explicit group head removes only one extra false change relative to row-only argmax and does not improve oracle recall.

## Strata

For the group/no-op model:

- Error counts: 46 correct-nearest, 12 false-change, 12 missed-oracle, 3 exact-oracle, 2 wrong-change.
- Missed oracle by source: IAM Words 7, ICDAR2015 2, ICDAR2013 2, SynthText words 1.
- False change by source: IAM Words 6, ICDAR2015 4, ICDAR-MLT2019 words 1, TextOCR 1.
- Wrong changes: IAM Words 2.

OCR-difficulty buckets show the useful-but-hard region:

- nearest Tesseract distance `3+`: 35 groups, 13 oracle-change groups, 9 missed oracle groups, 5 false changes, 2 wrong changes, only 2 exact oracle hits.
- nearest distance `2`: 10 groups, 3 oracle-change groups, 2 missed oracle groups, 3 false changes, 1 exact oracle hit.
- nearest distance `0` or `1`: no or little real headroom, but still 4 false changes.

Oracle-rank buckets are the clearest failure signal:

- rank `0`: 58 no-headroom groups; the group/no-op model still changes 12 of them, adding `+2` Tesseract edit characters.
- rank `1-2`: 10 oracle-change groups; the model recovers only 3 and misses 7 despite the oracle code being close in the top8 list.
- rank `5-7`: 5 oracle-change groups; the model recovers none exactly and includes one wrong change with `+2` Tesseract edits.

The row-only model has the same structure: 47 correct-nearest, 11 false-change, 12 missed-oracle, 3 exact-oracle and 2 wrong-change groups.

## Interpretation

The bottleneck is no longer top8 candidate recall. It is also not solved by adding a group/no-op head on top of the same score distribution. The clean-val failures separate into two simultaneous problems:

1. no-headroom suppression: rank-0 groups dominate the table and still generate 11-12 false changes;
2. hard-positive recall: high Tesseract-distance IAM examples and even rank `1-2` oracle groups are often missed.

This explains why high row AUC/AP did not transfer to policy quality. A model can separate many easy nearest rows while still failing the two policy-critical tails.

## Decision

Do not run counted `.oscr` streams for either top8 policy.

The next assignment model should use the top8 artifact, but train against direct policy errors:

- explicit no-headroom/rank-0 suppression;
- hard-positive recall for high nearest-Tesseract-distance groups;
- cross-fit calibration before any actual stream generation;
- actual safe assignment labels as the primary target, with PARSeq teacher signals only as optional auxiliary regularization.

Promotion remains gated by the existing actual-bitstream baseline: actual bpp unchanged, PARSeq unchanged, Tesseract better than `-8` unicode / `-8` latin-alnum edits versus nearest, and false/wrong changes close to the current `1/75` precision floor.

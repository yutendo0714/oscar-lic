# Eval600 Hard-Negative Balanced210 Diff-Crop Budget Selector

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N037 showed that a learned top-4 MLP selector fails despite a strong shortlist oracle. This diagnostic asks whether a simpler high-precision budget rule over the N034/N035 diff-crop scores transfers better.

The selector tunes on Eval600 train groups with strict constraints:

- false changes: `0`;
- wrong changes: `0`;
- Tesseract-worsen groups: `0`;
- PARSeq-worsen groups: `0`.

It then transfers either the learned score threshold or the learned budget fraction to Eval300 seed1.

## Results

Train tuning looks excellent:

- selected changes: `76/473`;
- train Tesseract delta: `-100`;
- false/wrong/worsen groups: `0`;
- tuned rule: top-4, `score_mean`, threshold `0.8394`.

Eval300 transfer fails:

| Policy | Changed groups | Tesseract delta | Exact oracle | Missed oracle | False changes | Worsen groups |
|---|---:|---:|---:|---:|---:|---:|
| nearest | 0 | 0 | 0 | 17 | 0 | 0 |
| oracle | 17 | -24 | 17 | 0 | 0 | 0 |
| threshold transfer | 2 | 0 | 0 | 17 | 2 | 0 |
| fraction transfer | 12 | +1 | 0 | 17 | 12 | 1 |

## Interpretation

This is a valid negative result for simple high-precision budget/threshold rules over the diff-crop score files.

The failure mode matches N029 and N037:

- train-side score thresholds can identify useful Eval600 changes under zero false/wrong/worsen constraints;
- the same score boundary selects no Eval300 oracle changes;
- the transferred changes are false, neutral or harmful.

The useful conclusion is sharper now: the strong `-19.33` shortlist oracle from N037 is real, but any selector using in-sample Eval600 score calibration is unreliable. The next credible selector needs out-of-fold/cross-fit score features or a source/domain-disjoint calibration split, not another post-hoc threshold over these score files.

W&B offline run: `qay9jfvo`

## Decision

Do not promote or rerun budget/threshold rules over the current diff-crop score files.

If score-based rules are used again, generate out-of-fold Eval600 scores first and tune only on calibration scores that are not produced by models trained on the same groups.

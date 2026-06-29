# TrOCR Eval300 Actual OSCR Current-vs-Shortlist Evaluation

Date: 2026-06-26 JST  
Stage: held-out OCR diagnostic  
Track: A pure visual bitstream evaluation-only diagnostic  
Evaluator role: held-out TrOCR-small-printed, not used for selection, training, utility labels, or hyperparameter tuning

## Purpose

The current best learned actual `.oscr` stream and the non-deployable top-4 shortlist oracle were already evaluated with PARSeq and Tesseract PSM8. This run adds a third OCR family, TrOCR-small-printed, only as an evaluation diagnostic.

This is not a promotion run and does not change the selector. It checks whether the actual decoded images that help Tesseract harm a separate neural OCR family.

## Inputs

Three actual `.oscr` reconstruction conditions were evaluated for Eval300 seed0/1/2 val75:

- nearest K64 utility-weighted assignment;
- current best learned stream: pooled image-diff group gate, model seed 1, threshold `0.5`;
- top-4 shortlist oracle: score-seed-2 oracle choice inside the deployable shortlist.

All three conditions use the existing actual decoded reconstructions. No rate estimate is introduced here.

## Aggregate Metrics

Per-condition mean over the three seed splits:

| Condition | Samples | Latin CER mean | Latin exact mean | Unicode CER mean | Unicode exact mean |
|---|---:|---:|---:|---:|---:|
| nearest K64 | 225 | 0.2968 | 0.4844 | 0.7636 | 0.1289 |
| current best | 225 | 0.2968 | 0.4844 | 0.7636 | 0.1289 |
| top-4 shortlist oracle | 225 | 0.2950 | 0.4844 | 0.7636 | 0.1289 |

Row-wise comparisons:

| Comparison | Profile | Delta char errors | Improved | Worsened | Exact delta | CER delta |
|---|---|---:|---:|---:|---:|---:|
| nearest -> current best | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| nearest -> current best | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| nearest -> top-4 shortlist oracle | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| nearest -> top-4 shortlist oracle | latin_alnum_ci_v1 | -2 | 1 | 0 | 0 | -0.001799 |
| current best -> top-4 shortlist oracle | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| current best -> top-4 shortlist oracle | latin_alnum_ci_v1 | -2 | 1 | 0 | 0 | -0.001799 |

## Interpretation

TrOCR-small-printed is weak on these scene-text word crops, especially under Unicode-strict normalization, so this should remain a supplementary diagnostic rather than a main metric. Still, it is useful as an independent neural OCR family:

- the current best learned actual stream is exactly neutral versus nearest for TrOCR across all `225` samples;
- the shortlist oracle does not worsen TrOCR on any sample;
- the shortlist oracle gives a tiny Latin-normalized improvement (`-2` character errors, `1` improved sample, `0` worsened).

This supports the existing PARSeq/Tesseract conclusion in a limited way: the top-4 oracle Tesseract gain does not appear to be a TrOCR-breaking artifact on this split. It also confirms TrOCR is too insensitive here to serve as the main selector target.

## W&B

Individual TrOCR runs:

| Condition | seed0 | seed1 | seed2 |
|---|---|---|---|
| nearest | `g2mp4ndc` | `djuqd4pt` | `ktwoghgx` |
| current best | `223we4qt` | `n8jghx3v` | `5h1tecro` |
| top-4 shortlist oracle | `5g8so5fb` | `e5fvyy6h` | `aemlnfsd` |

Aggregate offline W&B run: `eszdu4x5`.

## Files

- Aggregate JSON: `experiments/results/trocr_eval300_actual_oscr_current_vs_shortlist_2026_06_26.json`
- Nearest-vs-current comparison: `experiments/results/compare_trocr_eval300_val75_nearest_vs_pooled_imgdiff_model1_threshold05_q005_2026_06_26.json`
- Nearest-vs-shortlist comparison: `experiments/results/compare_trocr_eval300_val75_nearest_vs_shortlist_oracle_top4_scoreseed2_q005_2026_06_26.json`
- Current-vs-shortlist comparison: `experiments/results/compare_trocr_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top4_scoreseed2_q005_2026_06_26.json`

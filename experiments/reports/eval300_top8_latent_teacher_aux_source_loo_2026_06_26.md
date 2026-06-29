# eval300_top8_latent_teacher_aux_source_loo_2026_06_26

Source leave-out diagnostic for the latent/code teacher-auxiliary ranker.
This is not a deployable selector and does not export counted `.oscr` streams.

## Aggregate

| teacher weight | exact changed | wrong | Tesseract | PARSeq | rank<=1 | rank<=4 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 24 | 56 | -41 | -1 | 24 | 59 |
| 0.200 | 23 | 57 | -46 | 0 | 23 | 57 |

## By Held-Out Source

| source | teacher weight | oracle changes | exact changed | wrong | Tesseract | rank<=1 | rank<=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `iam_words` | 0.000 | 40 | 13 | 27 | -15 | 13 | 29 |
| `iam_words` | 0.200 | 40 | 13 | 27 | -21 | 13 | 27 |
| `icdar2013` | 0.000 | 6 | 1 | 5 | -4 | 1 | 4 |
| `icdar2013` | 0.200 | 6 | 2 | 4 | -4 | 2 | 5 |
| `icdar2015` | 0.000 | 14 | 4 | 10 | -10 | 4 | 12 |
| `icdar2015` | 0.200 | 14 | 3 | 11 | -9 | 3 | 11 |
| `icdar_mlt2019_words` | 0.000 | 5 | 1 | 4 | -4 | 1 | 2 |
| `icdar_mlt2019_words` | 0.200 | 5 | 1 | 4 | -5 | 1 | 3 |
| `synthtext_words` | 0.000 | 10 | 5 | 5 | -6 | 5 | 8 |
| `synthtext_words` | 0.200 | 10 | 3 | 7 | -4 | 3 | 7 |
| `textocr` | 0.000 | 5 | 0 | 5 | -2 | 0 | 4 |
| `textocr` | 0.200 | 5 | 1 | 4 | -3 | 1 | 4 |

## Interpretation

This checks whether the N102 teacher-loss auxiliary signal is source-robust when each source domain is held out. It is a diagnostic split over the existing Eval300 groups and must not be used as a publication selector or threshold.

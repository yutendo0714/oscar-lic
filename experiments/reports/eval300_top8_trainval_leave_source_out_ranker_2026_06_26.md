# eval300_top8_trainval_leave_source_out_ranker_2026_06_26

Leave-source-out candidate-ranking diagnostic for Eval300 top-8 assignment.
This is not a deployment selector and does not export counted `.oscr` streams.

## Source Results

| held-out source | groups | oracle changes | rank<=1 | rank<=4 | exact changed | wrong | Tesseract delta | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `iam_words` | 127 | 40 | 9 | 26 | 9 | 31 | -12 | 0 |
| `icdar2013` | 28 | 6 | 1 | 4 | 1 | 5 | -1 | 0 |
| `icdar2015` | 43 | 14 | 4 | 10 | 4 | 10 | -10 | 0 |
| `icdar_mlt2019_words` | 27 | 5 | 1 | 4 | 1 | 4 | -5 | 0 |
| `synthtext_words` | 36 | 10 | 4 | 7 | 4 | 6 | -4 | 0 |
| `textocr` | 39 | 5 | 1 | 3 | 1 | 4 | -1 | 0 |

## Aggregate

- held-out oracle-change groups: `80`
- learned score rank<=1: `20`
- learned score rank<=4: `54`
- oracle-change-only Tesseract delta: `-33`
- oracle-change-only PARSeq delta: `0`

## Interpretation

Leave-source-out ranking tests whether the N091 trainval signal is source-local. If rank<=1/rank<=4 collapse relative to N091, the next useful work is source/domain-balanced calibration or richer representation; if ranking remains comparable, no-op acceptor evidence is the main bottleneck.

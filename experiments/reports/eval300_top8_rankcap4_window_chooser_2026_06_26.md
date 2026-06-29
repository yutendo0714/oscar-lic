# eval300_top8_rankcap4_window_chooser_2026_06_26

Rankcap local latent-window candidate chooser. Diagnostic only; no `.oscr` selector is promoted.

| seed | rank1 | rank<=2 | rank<=4 | covered-only policy | all-change policy |
|---:|---:|---:|---:|---|---|
| 20260626 | 7 | 9 | 12 | 7/17 exact, false 0, wrong 5, miss 5, T -7, P 0 | 7/17 exact, false 0, wrong 10, miss 0, T -5, P 0 |
| 20260627 | 5 | 9 | 12 | 5/17 exact, false 0, wrong 7, miss 5, T -6, P 0 | 5/17 exact, false 0, wrong 12, miss 0, T -2, P 0 |
| 20260628 | 6 | 9 | 12 | 6/17 exact, false 0, wrong 6, miss 5, T -7, P 0 | 6/17 exact, false 0, wrong 11, miss 0, T -3, P 0 |

## Interpretation

This is the first clean rankcap4 chooser that uses local latent-window code-effect evidence. It should be compared with N115's shallow tabular chooser and N114's rankcap upper bound. No actual `.oscr` promotion is justified unless wrong/missed counts fall near the current precision floor and table-level Tesseract beats `-8` with PARSeq unchanged.

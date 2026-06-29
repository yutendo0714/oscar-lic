# eval300_top8_oof_source_oof_risk_calibration_2026_06_26

Source-OOF calibrated risk thresholds over deterministic N117 proposal anchors. Diagnostic only; no `.oscr` stream is exported.

W&B: `whucl72w`

| anchor | all proposals | best budget0 | best budget1 | rank1/rank<=4 |
|---|---|---|---|---:|
| topk_rank_anchor | 6/17 exact, false 58, wrong 11, miss 0, T 3, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 6/12 |
| oof_rank_anchor | 5/17 exact, false 58, wrong 12, miss 0, T 1, P 0 | 1/17 exact, false 1, wrong 1, miss 15, T -1, P 0 | 1/17 exact, false 1, wrong 1, miss 15, T -1, P 0 | 5/14 |
| z_fused_anchor | 5/17 exact, false 58, wrong 12, miss 0, T 2, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 5/12 |

## Interpretation

This diagnostic tests whether train-in-sample risk thresholding caused N124's false/wrong leakage. It is not promotable unless source-OOF thresholds beat the current actual-bitstream floor before counted `.oscr` export.

# eval300_top8_oof_shortlist_policy_selector_2026_06_26

No-op-inclusive selector over the N117 source-modulo OOF shortlist. Diagnostic only; no `.oscr` stream is exported.

W&B offline run: `ehwksv4h`.

## Validation

| model | argmax | budget0 margin | budget1 margin | rank1/rank<=4 |
|---|---|---|---|---:|
| logistic_l2_c02 | 0/17 exact, false 1, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/12 |
| hist_gradient_l2 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 1, wrong 1, miss 16, T 2, P 0 | 0/17 exact, false 1, wrong 1, miss 16, T 2, P 0 | 0/12 |
| random_forest_d5 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 1/17 exact, false 2, wrong 1, miss 15, T 2, P 0 | 0/11 |

## Interpretation

The selector is diagnostic-only. A useful result must beat the current actual-bitstream floor at table level before any counted `.oscr` stream export. The N117 shortlist improves candidate coverage, but this experiment tests whether ordinary train/val no-op supervision can use that headroom safely.

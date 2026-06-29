# eval300_top8_verifier_supervision_gap_2026_06_26

W&B: `gma1wetq`

Diagnostic-only supervision-density audit for the N117 OOF top-8 shortlist. It uses OCR deltas only as labels/evaluation and does not train or promote a selector.

## Summary

| partition | label inventory | min exact to beat current floor | best covered-oracle Tesseract |
|---|---|---:|---:|
| train | 60/63 covered, noheadroom 162, cand +/- 60/1075 | 3 | -81 |
| val | 15/17 covered, noheadroom 58, cand +/- 15/372 | 3 | -22 |
| all | 75/80 covered, noheadroom 220, cand +/- 75/1447 | 3 | -103 |

## Class Balance

| partition | candidate negative / positive | group no-change / change-positive |
|---|---:|---:|
| train | 17.92 | 2.75 |
| val | 24.80 | 4.00 |
| all | 19.29 | 3.00 |

## Validation Source Breakdown

| source | groups | oracle | covered | no-headroom | candidate +/- |
|---|---:|---:|---:|---:|---:|
| iam_words | 36 | 10 | 9 | 26 | 9/175 |
| icdar2013 | 8 | 3 | 2 | 5 | 2/41 |
| icdar2015 | 9 | 3 | 3 | 6 | 3/46 |
| icdar_mlt2019_words | 3 | 0 | 0 | 3 | 0/16 |
| synthtext_words | 10 | 1 | 1 | 9 | 1/53 |
| textocr | 9 | 0 | 0 | 9 | 0/41 |

## Validation Modulo Breakdown

| source_mod | groups | oracle | covered | no-headroom | candidate +/- |
|---|---:|---:|---:|---:|---:|
| r0 | 15 | 2 | 1 | 13 | 1/73 |
| r1 | 16 | 2 | 2 | 14 | 2/83 |
| r2 | 10 | 2 | 2 | 8 | 2/49 |
| r3 | 13 | 5 | 5 | 8 | 5/66 |
| r4 | 21 | 6 | 5 | 15 | 5/101 |

## Best Exact Frontier

This is a non-deployable ordering of covered oracle changes. It estimates how many exact useful changes a future verifier must accept to beat the current actual-bitstream floor without bad changes.

| k | cumulative Tesseract | source | reference | topk | delta |
|---:|---:|---|---|---:|---:|
| 1 | -3 | icdar2015 | heart | 1 | -3 |
| 2 | -6 | iam_words | Mauro's | 2 | -3 |
| 3 | -9 | iam_words | Lawrence | 6 | -3 |
| 4 | -11 | icdar2013 | SLUSH | 1 | -2 |
| 5 | -12 | iam_words | own | 1 | -1 |
| 6 | -13 | iam_words | predetermined | 1 | -1 |
| 7 | -14 | iam_words | understand | 1 | -1 |
| 8 | -15 | synthtext_words | vista" | 1 | -1 |
| 9 | -16 | iam_words | predetermined | 2 | -1 |
| 10 | -17 | icdar2015 | Accessories | 2 | -1 |
| 11 | -18 | iam_words | text | 2 | -1 |
| 12 | -19 | iam_words | own | 3 | -1 |

## Interpretation

N117 shortlist supervision remains label-sparse and no-headroom dominated: validation covers 15/17 oracle-change groups, but has 58 no-headroom groups and 24.8 shortlist candidate negatives per positive. A verifier must accept at least 3 exact useful changes with near-zero bad moves to beat the current actual-bitstream Tesseract floor (-8). This supports adding clean no-headroom/hard-positive supervision or changing the verification objective before any further N117/N120 score-risk tuning.

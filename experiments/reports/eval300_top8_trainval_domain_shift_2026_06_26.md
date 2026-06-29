# eval300_top8_trainval_domain_shift_2026_06_26

Train/val shift diagnostic for the Eval300 top-8 tabular assignment path.

## Oracle-Change Distribution

### Source

- `train/iam_words`: `30`
- `train/icdar2013`: `3`
- `train/icdar2015`: `11`
- `train/icdar_mlt2019_words`: `5`
- `train/synthtext_words`: `9`
- `train/textocr`: `5`
- `val/iam_words`: `10`
- `val/icdar2013`: `3`
- `val/icdar2015`: `3`
- `val/synthtext_words`: `1`

### Top-K Bucket

- `train/rank_gt4`: `8`
- `train/rank_le1`: `30`
- `train/rank_le2`: `15`
- `train/rank_le4`: `10`
- `val/rank_gt4`: `5`
- `val/rank_le1`: `6`
- `val/rank_le2`: `4`
- `val/rank_le4`: `2`

## N091 Val Status

### By Source

- `iam_words/exact`: `4`
- `iam_words/wrong_change`: `6`
- `icdar2013/exact`: `2`
- `icdar2013/wrong_change`: `1`
- `icdar2015/exact`: `1`
- `icdar2015/wrong_change`: `2`
- `synthtext_words/wrong_change`: `1`

### By Oracle Top-K Bucket

- `rank_gt4/wrong_change`: `5`
- `rank_le1/exact`: `4`
- `rank_le1/wrong_change`: `2`
- `rank_le2/exact`: `2`
- `rank_le2/wrong_change`: `2`
- `rank_le4/exact`: `1`
- `rank_le4/wrong_change`: `1`

## Reference Overlap

- train oracle-change unique references: `42`
- val oracle-change unique references: `14`
- overlap count: `6`

## Interpretation

- The diagnostic is descriptive: it explains N091-N093 behavior but is not a selector.
- If failures cluster by source or deep top-k rank, next evidence should target that stratum directly rather than sweeping thresholds.

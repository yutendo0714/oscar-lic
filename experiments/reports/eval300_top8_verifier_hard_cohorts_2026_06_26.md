# eval300_top8_verifier_hard_cohorts_2026_06_26

W&B: `zmzewu8d`

Train-only hard cohort selection for the next N117 verifier objective. No validation rows are written to the cohort file.

## Summary

- Rows: `222`
- Positives: `60`
- Negatives: `162`
- Negative/positive: `2.7`

## Source Counts

| source | rows |
|---|---:|
| iam_words | 88 |
| icdar2013 | 20 |
| icdar2015 | 34 |
| icdar_mlt2019_words | 24 |
| synthtext_words | 26 |
| textocr | 30 |

## Interpretation

Prepared a train-only hard cohort with 60 covered-oracle positives and 162 matched no-headroom negatives (2.7 negatives per positive). This is a supervision manifest for a future contrastive/selective verifier, not a selector policy and not validation-tuned.

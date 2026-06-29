# eval300_top8_trainval_tabular_listwise_ranker_2026_06_26

Train-partition tabular listwise ranker for Eval300 top-8 assignment.
This is a candidate-ranking diagnostic, not an actual `.oscr` promotion.

## Split

- train listwise groups: `63`
- validation groups: `75`
- validation oracle-change groups: `17`
- feature count: `218`

## Oracle-Change-Only Policy

| changed | exact | false | wrong | missed | Tesseract delta | PARSeq delta |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 65 | 0 | 10 | 0 | -12 | 0 |

## Rank Audit

| metric | <=1 | <=2 | <=4 | <=8 |
|---|---:|---:|---:|---:|
| learned score rank | 7 | 7 | 10 | 17 |
| first-stage topk rank | 6 | 10 | 12 | 17 |

## Interpretation

- If score rank1 is still low, broader train-split scalar supervision is not enough.
- Promotion requires a separate no-op acceptor and counted `.oscr` evaluation; neither is attempted here.

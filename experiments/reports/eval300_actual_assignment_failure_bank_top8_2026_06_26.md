# eval300_actual_assignment_failure_bank_top8_2026_06_26

This is a diagnostic case bank for already-evaluated actual `.oscr` assignment policies.
Held-out OCR outcomes are used only to categorize failures and should not become deployable features.

## Summary

- Cases: `75`
- Current Tesseract delta vs nearest: `-8`
- Shortlist Tesseract delta vs nearest: `-24`
- Shortlist vs current delta: `-16`
- Current categories: `{'correct_nearest': 57, 'exact_oracle_change': 4, 'false_change': 1, 'missed_oracle': 13}`
- Shortlist categories: `{'correct_nearest': 58, 'exact_oracle_change': 17}`
- Next-model targets: `{'keep_current_good_change': 4, 'keep_noop': 57, 'recover_shortlist_oracle_change': 13, 'reject_current_bad_change': 1}`
- Oracle rank by target: `{'keep_current_good_change': {'1': 2, '2': 1, '3': 1}, 'keep_noop': {'0': 57}, 'recover_shortlist_oracle_change': {'1': 4, '2': 3, '3': 1, '6': 4, '7': 1}, 'reject_current_bad_change': {'0': 1}}`

## Priority Cases

### recover_shortlist_oracle_change

| key | source | ref | nearest -> current -> shortlist | deltas current/shortlist/vs-current | codes nearest/current/shortlist/oracle | rank oracle |
|---|---|---|---|---:|---|---:|
| 1/12/1 | iam_words | `Lawrence` | `Be aanenien cp:` -> `Be aanenien cp:` -> `sonst cps` | 0/-3/-3 | 2/2/49/49 | 6 |
| 1/9/0 | iam_words | `own` | `QOAUN` -> `QOAUN` -> `(SSA` | 0/-1/-1 | 62/62/31/31 | 1 |
| 1/32/0 | iam_words | `predetermined` | `gredelermin eof` -> `gredelermin eof` -> `gredelermin en` | 0/-1/-1 | 62/62/50/50 | 2 |
| 1/51/1 | iam_words | `THE` | `J Ap` -> `J Ap` -> `Jip` | 0/-1/-1 | 59/59/56/56 | 7 |
| 1/53/1 | icdar2015 | `Accessories` | `‘fccessoes` -> `‘fccessoes` -> `fccessoes` | 0/-1/-1 | 32/32/43/43 | 2 |
| 2/4/0 | iam_words | `own` | `(SES` -> `(SES` -> `QA` | 0/-1/-1 | 18/18/36/36 | 3 |
| 2/5/4 | iam_words | `Lawrence` | `acne int cps` -> `acne int cps` -> `eraser cps` | 0/-1/-1 | 34/34/2/2 | 6 |
| 2/34/0 | icdar2015 | `finest` | `pines` -> `pines` -> `ginest` | 0/-1/-1 | 18/18/1/1 | 6 |
| 2/39/1 | iam_words | `predetermined` | `gredeermin eof` -> `gredeermin eof` -> `gredelermin en` | 0/-1/-1 | 18/18/1/1 | 1 |
| 2/43/4 | iam_words | `understand` | `durch harrd` -> `durch harrd` -> `durch hand.` | 0/-1/-1 | 7/7/21/21 | 1 |
| 2/50/0 | iam_words | `text` | `le yt` -> `le yt` -> `le xt` | 0/-1/-1 | 18/18/46/46 | 2 |
| 2/53/1 | synthtext_words | `vista"` | `vista!` -> `vista!` -> `vista"` | 0/-1/-1 | 45/45/20/20 | 1 |
| 2/69/1 | icdar2013 | `COSTA` | `<Q. WA` -> `<Q. WA` -> `<q. A` | 0/-1/-1 | 1/1/25/25 | 6 |

### reject_current_bad_change

| key | source | ref | nearest -> current -> shortlist | deltas current/shortlist/vs-current | codes nearest/current/shortlist/oracle | rank oracle |
|---|---|---|---|---:|---|---:|
| 2/51/1 | iam_words | `from` | `qvorn` -> `qyvorn` -> `qvorn` | 1/0/-1 | 2/48/2/2 | 0 |

### improve_first_stage_shortlist_recall

| key | source | ref | nearest -> current -> shortlist | deltas current/shortlist/vs-current | codes nearest/current/shortlist/oracle | rank oracle |
|---|---|---|---|---:|---|---:|
| n/a | n/a | n/a | n/a | 0 | n/a | 0 |

### keep_current_good_change

| key | source | ref | nearest -> current -> shortlist | deltas current/shortlist/vs-current | codes nearest/current/shortlist/oracle | rank oracle |
|---|---|---|---|---:|---|---:|
| 1/63/2 | icdar2015 | `heart` | `ad` -> `heart:` -> `heart:` | -3/-3/0 | 53/19/19/19 | 1 |
| 2/6/5 | icdar2013 | `SLUSH` | `SLUS ~` -> `SLUSH` -> `SLUSH` | -2/-2/0 | 34/62/62/62 | 1 |
| 2/13/1 | iam_words | `Mauro's` | `Wasnen'e` -> `Ware's` -> `Ware's` | -3/-3/0 | 46/18/18/18 | 2 |
| 2/64/0 | icdar2013 | `JOHN` | `JOny` -> `JOHy` -> `JOHy` | -1/-1/0 | 18/54/54/54 | 3 |

## Interpretation

The immediate selector problem is not extra bitrate. It is a high-precision choice among existing code-index options.
The highest-value targets are the shortlist oracle recoveries, while the current false/wrong changes define the abstention floor.

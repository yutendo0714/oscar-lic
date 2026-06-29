# eval300_top8_evidence_cache_nonnearest_ranker_2026_06_26

Tiny patch/code-effect non-nearest ranker over the N088 evidence cache.
This isolates candidate ranking from no-op acceptance and is not deployable by itself.

## Oracle-Change-Only Policy

| val seed | changed | exact | false | wrong | missed | Tesseract delta | PARSeq delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 25 | 0 | 0 | 0 | 0 | 0 |
| 1 | 6 | 19 | 0 | 6 | 0 | -1 | 0 |
| 2 | 11 | 16 | 0 | 9 | 0 | -1 | 0 |

## Target Rank Audit

| target | groups | selected status | oracle nonnearest score <=1/2/4/8 | oracle topk <=1/2/4/8 | best-nonnearest Tesseract delta sum |
|---|---:|---|---|---|---:|
| keep_current_good_change | 4 | exact:1, wrong_change:3 | 1/3/3/4 | 2/3/4/4 | 0 |
| keep_noop | 57 | exact:57 | 0/0/0/0 | 57/57/57/57 | 7 |
| recover_shortlist_oracle_change | 13 | exact:1, wrong_change:12 | 1/2/5/13 | 4/7/8/13 | -2 |
| reject_current_bad_change | 1 | exact:1 | 0/0/0/0 | 1/1/1/1 | 0 |

## Interpretation

- Compare recover rank1 against the N087 glyphcode-only floor of `1/13`.
- If rank1 remains low, this cache/model is not enough for the next no-op acceptor.

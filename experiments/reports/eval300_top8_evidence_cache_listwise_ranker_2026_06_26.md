# eval300_top8_evidence_cache_listwise_ranker_2026_06_26

Listwise non-nearest candidate ranker over the N088 evidence cache.
This isolates candidate ranking from no-op acceptance and is not deployable by itself.

## Oracle-Change-Only Policy

| val seed | train groups | changed | exact | false | wrong | missed | Tesseract delta | PARSeq delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17 | 0 | 25 | 0 | 0 | 0 | 0 | 0 |
| 1 | 11 | 6 | 19 | 0 | 6 | 0 | -1 | 0 |
| 2 | 6 | 11 | 15 | 0 | 10 | 0 | -1 | 0 |

## Target Rank Audit

| target | groups | selected status | oracle nonnearest score <=1/2/4/8 | oracle topk <=1/2/4/8 | best-nonnearest Tesseract delta sum |
|---|---:|---|---|---|---:|
| keep_current_good_change | 4 | exact:1, wrong_change:3 | 1/3/4/4 | 2/3/4/4 | 0 |
| keep_noop | 57 | exact:57 | 0/0/0/0 | 57/57/57/57 | 7 |
| recover_shortlist_oracle_change | 13 | wrong_change:13 | 0/2/7/13 | 4/7/8/13 | -2 |
| reject_current_bad_change | 1 | exact:1 | 0/0/0/0 | 1/1/1/1 | 0 |

## Interpretation

- Compare recover rank1 against the N087/N089 floor of `1/13`.
- Because only oracle-change groups supervise the listwise loss, this is a candidate-ranker feasibility test, not a selector.

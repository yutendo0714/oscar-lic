# Exception Probe Expansion Comparison

This compares two actual compact-stream OCR label tables. Negative OCR deltas mean the expanded exception/drop stream has fewer OCR edit errors than the matched current stream.

## Summary

- Base table: `eval300_exception_counterfactual_probe_actual_ocr_v2_rel2_2026_06_26`
- Expanded table: `eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26`
- Common rows: `53`; added rows: `10`; removed rows: `0`.

| subset | rows | labels | tess unicode | tess latin | parseq unicode | rate bytes | current bpp | expanded bpp |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 53 | `{"beneficial_no_profile_harm": 16, "harmful_any_profile": 4, "neutral_all_profiles": 33}` | -15 | -11 | 0 | -1290 | 3.332659 | 3.248499 |
| expanded | 63 | `{"beneficial_no_profile_harm": 17, "harmful_any_profile": 5, "neutral_all_profiles": 41}` | -14 | -11 | 0 | -1720 | 3.340284 | 3.245654 |
| common | 53 | `{"beneficial_no_profile_harm": 16, "harmful_any_profile": 4, "neutral_all_profiles": 33}` | -15 | -11 | 0 | -1290 | 3.332659 | 3.248499 |
| added | 10 | `{"beneficial_no_profile_harm": 1, "harmful_any_profile": 1, "neutral_all_profiles": 8}` | 1 | 0 | 0 | -430 | 3.381320 | 3.230337 |
| removed | 0 | `{}` | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 |

## Added Rows

| seed | source_index | source | action | label | ref | tess current | tess expanded | unicode | latin | bytes | tags |
|---:|---:|---|---|---|---|---|---|---:|---:|---:|---|
| 0 | 54 | textocr | drop_current | neutral_all_profiles | government | government | government | 0 | 0 | -43 | train_close_current_only |
| 0 | 73 | icdar2013 | drop_current | beneficial_no_profile_harm | HORNBY | —CO ay | ~—COR iy | 0 | -1 | -43 | train_close_current_only |
| 0 | 80 | iam_words | drop_current | neutral_all_profiles | energetic |  |  | 0 | 0 | -43 | train_close_current_only,train_iam_current_only_near |
| 0 | 158 | iam_words | drop_current | harmful_any_profile | job | joe | jee | 1 | 1 | -43 | train_close_current_only,train_iam_current_only_near |
| 1 | 67 | iam_words | drop_current | neutral_all_profiles | her | Aer- | Aer- | 0 | 0 | -43 | train_close_current_only,train_iam_current_only_near |
| 1 | 101 | iam_words | drop_current | neutral_all_profiles | me | y= | y= | 0 | 0 | -43 | train_close_current_only,train_iam_current_only_near |
| 1 | 105 | textocr | drop_current | neutral_all_profiles | In |  |  | 0 | 0 | -43 | train_close_current_only |
| 1 | 149 | textocr | drop_current | neutral_all_profiles | Sony | Sony | Sony | 0 | 0 | -43 | train_close_current_only |
| 2 | 4 | synthtext_words | drop_current | neutral_all_profiles | I'm | Tx | To | 0 | 0 | -43 | train_close_current_only |
| 2 | 149 | textocr | drop_current | neutral_all_profiles | OZ | oe | oe | 0 | 0 | -43 | train_close_current_only |

## Conclusion

The expanded table adds useful evidence only if its added rows improve the safety/utility frontier. Here the added rows are mostly neutral, so the narrower base table remains the sharper frontier for the next verifier design.

## Next Action

Use this comparison to avoid threshold-widening loops. Add new labels only when they target missing hard-negative modes or materially different evidence.

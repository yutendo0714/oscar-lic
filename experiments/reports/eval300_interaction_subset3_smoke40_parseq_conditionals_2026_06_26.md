# eval300_interaction_subset3_smoke40_parseq_conditionals_2026_06_26

This report converts all-subset OCR outcomes into conditional add-one-candidate utility edges.
Negative delta means the added candidate reduced OCR edit distance.

## unicode_strict_v1

### Transition Summary

| transition | edge rows | delta chars | improved/worse/unchanged | added actual bpp | added enh bpp | added PSNR | gain/add-enh-bpp |
|---|---:|---:|---|---:|---:|---:|---:|
| 0_to_1 | 120 | -13 | 5/0/115 | 0.0000 | 2.1017 | 0.0641 | 0.0515 |
| 1_to_2 | 240 | 4 | 0/2/238 | 2.0165 | 2.0165 | 0.0648 | -0.0083 |
| 2_to_3 | 120 | -4 | 2/0/118 | 2.0151 | 2.0151 | 0.0615 | 0.0165 |

### Best Subset

Best-any cardinality counts: `{'1': 40, '2': 0, '3': 0}`
Best pair beats best single on `0/40` samples.
Triple beats best single on `0/40` samples.
Total regret from stopping after the best single instead of best-any subset: `0` edit errors.

## latin_alnum_ci_v1

### Transition Summary

| transition | edge rows | delta chars | improved/worse/unchanged | added actual bpp | added enh bpp | added PSNR | gain/add-enh-bpp |
|---|---:|---:|---|---:|---:|---:|---:|
| 0_to_1 | 120 | -4 | 2/0/118 | 0.0000 | 2.1017 | 0.0641 | 0.0159 |
| 1_to_2 | 240 | 4 | 0/2/238 | 2.0165 | 2.0165 | 0.0648 | -0.0083 |
| 2_to_3 | 120 | -4 | 2/0/118 | 2.0151 | 2.0151 | 0.0615 | 0.0165 |

### Best Subset

Best-any cardinality counts: `{'1': 40, '2': 0, '3': 0}`
Best pair beats best single on `0/40` samples.
Triple beats best single on `0/40` samples.
Total regret from stopping after the best single instead of best-any subset: `0` edit errors.

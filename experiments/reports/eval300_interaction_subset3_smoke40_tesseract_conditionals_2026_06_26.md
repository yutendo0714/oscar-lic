# eval300_interaction_subset3_smoke40_tesseract_conditionals_2026_06_26

This report converts all-subset OCR outcomes into conditional add-one-candidate utility edges.
Negative delta means the added candidate reduced OCR edit distance.

## unicode_strict_v1

### Transition Summary

| transition | edge rows | delta chars | improved/worse/unchanged | added actual bpp | added enh bpp | added PSNR | gain/add-enh-bpp |
|---|---:|---:|---|---:|---:|---:|---:|
| 0_to_1 | 120 | -179 | 116/0/4 | 0.0000 | 2.1017 | 0.0641 | 0.7098 |
| 1_to_2 | 240 | 178 | 11/121/108 | 2.0165 | 2.0165 | 0.0648 | -0.3678 |
| 2_to_3 | 120 | -6 | 26/31/63 | 2.0151 | 2.0151 | 0.0615 | 0.0248 |

### Best Subset

Best-any cardinality counts: `{'1': 36, '2': 4, '3': 0}`
Best pair beats best single on `4/40` samples.
Triple beats best single on `2/40` samples.
Total regret from stopping after the best single instead of best-any subset: `9` edit errors.

## latin_alnum_ci_v1

### Transition Summary

| transition | edge rows | delta chars | improved/worse/unchanged | added actual bpp | added enh bpp | added PSNR | gain/add-enh-bpp |
|---|---:|---:|---|---:|---:|---:|---:|
| 0_to_1 | 120 | -130 | 78/2/40 | 0.0000 | 2.1017 | 0.0641 | 0.5155 |
| 1_to_2 | 240 | 148 | 9/88/143 | 2.0165 | 2.0165 | 0.0648 | -0.3058 |
| 2_to_3 | 120 | -22 | 25/19/76 | 2.0151 | 2.0151 | 0.0615 | 0.0910 |

### Best Subset

Best-any cardinality counts: `{'1': 37, '2': 2, '3': 1}`
Best pair beats best single on `2/40` samples.
Triple beats best single on `3/40` samples.
Total regret from stopping after the best single instead of best-any subset: `7` edit errors.

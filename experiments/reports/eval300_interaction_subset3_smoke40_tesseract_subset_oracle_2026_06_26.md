# eval300_interaction_subset3_smoke40_tesseract_subset_oracle_2026_06_26

This is a diagnostic all-subset oracle over counted raw-residual `.oscr` streams.
It is not a deployable selector because it chooses subsets using OCR outcomes.

## unicode_strict_v1

| policy | candidates | delta chars | total bpp | enh bpp | PSNR delta | improved/worse/unchanged |
|---|---:|---:|---:|---:|---:|---|
| single_a | 1 | -63 | 6.0395 | 2.1017 | 0.0515 | 39/0/1 |
| single_b | 1 | -58 | 6.0395 | 2.1017 | 0.0604 | 38/0/2 |
| single_c | 1 | -58 | 6.0395 | 2.1017 | 0.0804 | 39/0/1 |
| pair_ac | 2 | -36 | 8.0560 | 4.1182 | 0.1322 | 24/0/16 |
| pair_bc | 2 | -32 | 8.0560 | 4.1182 | 0.1409 | 21/2/17 |
| subset3_abc | 3 | -32 | 10.0711 | 6.1332 | 0.1903 | 23/2/15 |
| pair_ab | 2 | -22 | 8.0560 | 4.1182 | 0.1135 | 22/4/14 |
| subset_oracle | mixed | -72 | 6.2417 | 2.3039 | 0.0655 | 39/0/1 |

Best fixed policy by cardinality:

| cardinality | policy | delta chars | total bpp |
|---:|---|---:|---:|
| 1 | single_a | -63 | 6.0395 |
| 2 | pair_ac | -36 | 8.0560 |
| 3 | subset3_abc | -32 | 10.0711 |

Triple vs additive singles: expected `-179`, observed `-32`, residual `147`, gain ratio `0.1788`.

Subset oracle policy counts: `{'single_a': 36, 'pair_ac': 2, 'pair_bc': 1, 'pair_ab': 1}`
Subset oracle cardinality counts: `{'1': 36, '2': 4}`

## latin_alnum_ci_v1

| policy | candidates | delta chars | total bpp | enh bpp | PSNR delta | improved/worse/unchanged |
|---|---:|---:|---:|---:|---:|---|
| single_a | 1 | -47 | 6.0395 | 2.1017 | 0.0515 | 26/0/14 |
| single_b | 1 | -42 | 6.0395 | 2.1017 | 0.0604 | 26/1/13 |
| single_c | 1 | -41 | 6.0395 | 2.1017 | 0.0804 | 26/1/13 |
| pair_ac | 2 | -27 | 8.0560 | 4.1182 | 0.1322 | 16/0/24 |
| subset3_abc | 3 | -26 | 10.0711 | 6.1332 | 0.1903 | 17/1/22 |
| pair_bc | 2 | -19 | 8.0560 | 4.1182 | 0.1409 | 14/4/22 |
| pair_ab | 2 | -10 | 8.0560 | 4.1182 | 0.1135 | 15/4/21 |
| subset_oracle | mixed | -54 | 6.3712 | 2.4334 | 0.0632 | 27/0/13 |

Best fixed policy by cardinality:

| cardinality | policy | delta chars | total bpp |
|---:|---|---:|---:|
| 1 | single_a | -47 | 6.0395 |
| 2 | pair_ac | -27 | 8.0560 |
| 3 | subset3_abc | -26 | 10.0711 |

Triple vs additive singles: expected `-130`, observed `-26`, residual `104`, gain ratio `0.2000`.

Subset oracle policy counts: `{'single_a': 37, 'pair_ac': 1, 'pair_bc': 1, 'subset3_abc': 1}`
Subset oracle cardinality counts: `{'1': 37, '2': 2, '3': 1}`

## Interpretation

Negative deltas mean fewer OCR edit errors than the frozen MLIC++ base. The all-subset oracle is an upper-control for a conditional allocator: it asks whether a policy should keep adding candidates or stop after one candidate.

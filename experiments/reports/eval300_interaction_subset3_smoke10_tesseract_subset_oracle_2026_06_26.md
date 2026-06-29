# eval300_interaction_subset3_smoke10_tesseract_subset_oracle_2026_06_26

This is a diagnostic all-subset oracle over counted raw-residual `.oscr` streams.
It is not a deployable selector because it chooses subsets using OCR outcomes.

## unicode_strict_v1

| policy | candidates | delta chars | total bpp | enh bpp | PSNR delta | improved/worse/unchanged |
|---|---:|---:|---:|---:|---:|---|
| single_a | 1 | -20 | 4.8440 | 1.5346 | 0.0574 | 9/0/1 |
| single_b | 1 | -19 | 4.8440 | 1.5346 | 0.0506 | 9/0/1 |
| single_c | 1 | -19 | 4.8440 | 1.5346 | 0.0559 | 9/0/1 |
| pair_ac | 2 | -14 | 6.3172 | 3.0078 | 0.1145 | 8/0/2 |
| subset3_abc | 3 | -9 | 7.7885 | 4.4791 | 0.1660 | 5/1/4 |
| pair_ab | 2 | -5 | 6.3172 | 3.0078 | 0.1092 | 5/1/4 |
| pair_bc | 2 | -4 | 6.3172 | 3.0078 | 0.1063 | 5/2/3 |
| subset_oracle | mixed | -23 | 5.1870 | 1.8776 | 0.0702 | 9/0/1 |

Best fixed policy by cardinality:

| cardinality | policy | delta chars | total bpp |
|---:|---|---:|---:|
| 1 | single_a | -20 | 4.8440 |
| 2 | pair_ac | -14 | 6.3172 |
| 3 | subset3_abc | -9 | 7.7885 |

Triple vs additive singles: expected `-58`, observed `-9`, residual `49`, gain ratio `0.1552`.

Subset oracle policy counts: `{'single_a': 8, 'pair_ac': 2}`
Subset oracle cardinality counts: `{'1': 8, '2': 2}`

## latin_alnum_ci_v1

| policy | candidates | delta chars | total bpp | enh bpp | PSNR delta | improved/worse/unchanged |
|---|---:|---:|---:|---:|---:|---|
| single_a | 1 | -14 | 4.8440 | 1.5346 | 0.0574 | 6/0/4 |
| single_b | 1 | -12 | 4.8440 | 1.5346 | 0.0506 | 6/1/3 |
| single_c | 1 | -12 | 4.8440 | 1.5346 | 0.0559 | 6/1/3 |
| pair_ac | 2 | -9 | 6.3172 | 3.0078 | 0.1145 | 5/0/5 |
| subset3_abc | 3 | -6 | 7.7885 | 4.4791 | 0.1660 | 4/1/5 |
| pair_ab | 2 | -2 | 6.3172 | 3.0078 | 0.1092 | 4/1/5 |
| pair_bc | 2 | 2 | 6.3172 | 3.0078 | 0.1063 | 2/3/5 |
| subset_oracle | mixed | -16 | 4.9585 | 1.6491 | 0.0591 | 6/0/4 |

Best fixed policy by cardinality:

| cardinality | policy | delta chars | total bpp |
|---:|---|---:|---:|
| 1 | single_a | -14 | 4.8440 |
| 2 | pair_ac | -9 | 6.3172 |
| 3 | subset3_abc | -6 | 7.7885 |

Triple vs additive singles: expected `-38`, observed `-6`, residual `32`, gain ratio `0.1579`.

Subset oracle policy counts: `{'single_a': 9, 'pair_ac': 1}`
Subset oracle cardinality counts: `{'1': 9, '2': 1}`

## Interpretation

Negative deltas mean fewer OCR edit errors than the frozen MLIC++ base. The all-subset oracle is an upper-control for a conditional allocator: it asks whether a policy should keep adding candidates or stop after one candidate.

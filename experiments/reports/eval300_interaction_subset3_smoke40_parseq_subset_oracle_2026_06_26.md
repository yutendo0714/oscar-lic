# eval300_interaction_subset3_smoke40_parseq_subset_oracle_2026_06_26

This is a diagnostic all-subset oracle over counted raw-residual `.oscr` streams.
It is not a deployable selector because it chooses subsets using OCR outcomes.

## unicode_strict_v1

| policy | candidates | delta chars | total bpp | enh bpp | PSNR delta | improved/worse/unchanged |
|---|---:|---:|---:|---:|---:|---|
| single_a | 1 | -5 | 6.0395 | 2.1017 | 0.0515 | 2/0/38 |
| single_b | 1 | -5 | 6.0395 | 2.1017 | 0.0604 | 2/0/38 |
| pair_ab | 2 | -5 | 8.0560 | 4.1182 | 0.1135 | 2/0/38 |
| subset3_abc | 3 | -5 | 10.0711 | 6.1332 | 0.1903 | 2/0/38 |
| single_c | 1 | -3 | 6.0395 | 2.1017 | 0.0804 | 1/0/39 |
| pair_ac | 2 | -3 | 8.0560 | 4.1182 | 0.1322 | 1/0/39 |
| pair_bc | 2 | -3 | 8.0560 | 4.1182 | 0.1409 | 1/0/39 |
| subset_oracle | mixed | -5 | 6.0395 | 2.1017 | 0.0515 | 2/0/38 |

Best fixed policy by cardinality:

| cardinality | policy | delta chars | total bpp |
|---:|---|---:|---:|
| 1 | single_a | -5 | 6.0395 |
| 2 | pair_ab | -5 | 8.0560 |
| 3 | subset3_abc | -5 | 10.0711 |

Triple vs additive singles: expected `-13`, observed `-5`, residual `8`, gain ratio `0.3846`.

Subset oracle policy counts: `{'single_a': 40}`
Subset oracle cardinality counts: `{'1': 40}`

## latin_alnum_ci_v1

| policy | candidates | delta chars | total bpp | enh bpp | PSNR delta | improved/worse/unchanged |
|---|---:|---:|---:|---:|---:|---|
| single_a | 1 | -2 | 6.0395 | 2.1017 | 0.0515 | 1/0/39 |
| single_b | 1 | -2 | 6.0395 | 2.1017 | 0.0604 | 1/0/39 |
| pair_ab | 2 | -2 | 8.0560 | 4.1182 | 0.1135 | 1/0/39 |
| subset3_abc | 3 | -2 | 10.0711 | 6.1332 | 0.1903 | 1/0/39 |
| single_c | 1 | 0 | 6.0395 | 2.1017 | 0.0804 | 0/0/40 |
| pair_ac | 2 | 0 | 8.0560 | 4.1182 | 0.1322 | 0/0/40 |
| pair_bc | 2 | 0 | 8.0560 | 4.1182 | 0.1409 | 0/0/40 |
| subset_oracle | mixed | -2 | 6.0395 | 2.1017 | 0.0515 | 1/0/39 |

Best fixed policy by cardinality:

| cardinality | policy | delta chars | total bpp |
|---:|---|---:|---:|
| 1 | single_a | -2 | 6.0395 |
| 2 | pair_ab | -2 | 8.0560 |
| 3 | subset3_abc | -2 | 10.0711 |

Triple vs additive singles: expected `-4`, observed `-2`, residual `2`, gain ratio `0.5000`.

Subset oracle policy counts: `{'single_a': 40}`
Subset oracle cardinality counts: `{'1': 40}`

## Interpretation

Negative deltas mean fewer OCR edit errors than the frozen MLIC++ base. The all-subset oracle is an upper-control for a conditional allocator: it asks whether a policy should keep adding candidates or stop after one candidate.

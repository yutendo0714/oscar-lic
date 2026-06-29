# eval300_interaction_subset3_smoke10_parseq_subset_oracle_2026_06_26

This is a diagnostic all-subset oracle over counted raw-residual `.oscr` streams.
It is not a deployable selector because it chooses subsets using OCR outcomes.

## unicode_strict_v1

| policy | candidates | delta chars | total bpp | enh bpp | PSNR delta | improved/worse/unchanged |
|---|---:|---:|---:|---:|---:|---|
| single_a | 1 | -3 | 4.8440 | 1.5346 | 0.0574 | 1/0/9 |
| single_b | 1 | -3 | 4.8440 | 1.5346 | 0.0506 | 1/0/9 |
| single_c | 1 | -3 | 4.8440 | 1.5346 | 0.0559 | 1/0/9 |
| pair_ab | 2 | -3 | 6.3172 | 3.0078 | 0.1092 | 1/0/9 |
| pair_ac | 2 | -3 | 6.3172 | 3.0078 | 0.1145 | 1/0/9 |
| pair_bc | 2 | -3 | 6.3172 | 3.0078 | 0.1063 | 1/0/9 |
| subset3_abc | 3 | -3 | 7.7885 | 4.4791 | 0.1660 | 1/0/9 |
| subset_oracle | mixed | -3 | 4.8440 | 1.5346 | 0.0574 | 1/0/9 |

Best fixed policy by cardinality:

| cardinality | policy | delta chars | total bpp |
|---:|---|---:|---:|
| 1 | single_a | -3 | 4.8440 |
| 2 | pair_ab | -3 | 6.3172 |
| 3 | subset3_abc | -3 | 7.7885 |

Triple vs additive singles: expected `-9`, observed `-3`, residual `6`, gain ratio `0.3333`.

Subset oracle policy counts: `{'single_a': 10}`
Subset oracle cardinality counts: `{'1': 10}`

## latin_alnum_ci_v1

| policy | candidates | delta chars | total bpp | enh bpp | PSNR delta | improved/worse/unchanged |
|---|---:|---:|---:|---:|---:|---|
| single_a | 1 | 0 | 4.8440 | 1.5346 | 0.0574 | 0/0/10 |
| single_b | 1 | 0 | 4.8440 | 1.5346 | 0.0506 | 0/0/10 |
| single_c | 1 | 0 | 4.8440 | 1.5346 | 0.0559 | 0/0/10 |
| pair_ab | 2 | 0 | 6.3172 | 3.0078 | 0.1092 | 0/0/10 |
| pair_ac | 2 | 0 | 6.3172 | 3.0078 | 0.1145 | 0/0/10 |
| pair_bc | 2 | 0 | 6.3172 | 3.0078 | 0.1063 | 0/0/10 |
| subset3_abc | 3 | 0 | 7.7885 | 4.4791 | 0.1660 | 0/0/10 |
| subset_oracle | mixed | 0 | 4.8440 | 1.5346 | 0.0574 | 0/0/10 |

Best fixed policy by cardinality:

| cardinality | policy | delta chars | total bpp |
|---:|---|---:|---:|
| 1 | single_a | 0 | 4.8440 |
| 2 | pair_ab | 0 | 6.3172 |
| 3 | subset3_abc | 0 | 7.7885 |

Triple vs additive singles: expected `0`, observed `0`, residual `0`, gain ratio `n/a`.

Subset oracle policy counts: `{'single_a': 10}`
Subset oracle cardinality counts: `{'1': 10}`

## Interpretation

Negative deltas mean fewer OCR edit errors than the frozen MLIC++ base. The all-subset oracle is an upper-control for a conditional allocator: it asks whether a policy should keep adding candidates or stop after one candidate.

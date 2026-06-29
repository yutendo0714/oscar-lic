# Center Utility Assignment Audit

This diagnostic asks whether train-side codebook-center utility alone can recover top-8 assignment oracle codes.
It does not export streams and does not use OCR strings as selector inputs.

## Summary

### Partition `all`

- Groups: `300`; oracle-change groups: `80`.
- Oracle center-utility rank1/rank<=4: `36` / `122`; oracle-change rank1/rank<=4: `8` / `30`.
- Oracle center utility above nearest: `35` overall, `35` on oracle-change groups.
- Oracle-change eligibility by relative error: `{'re101': 1, 're105': 5}`.

| policy | changed | exact | missed | false | wrong | Tess delta | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| center_top1_no_relative_guard | 264 | 8 | 8 | 192 | 64 | -16 | -1 |
| center_top1_re101 | 7 | 1 | 78 | 5 | 1 | -1 | 0 |
| center_top1_re105 | 23 | 2 | 75 | 18 | 3 | -1 | 0 |

### Partition `train`

- Groups: `225`; oracle-change groups: `63`.
- Oracle center-utility rank1/rank<=4: `29` / `102`; oracle-change rank1/rank<=4: `5` / `21`.
- Oracle center utility above nearest: `24` overall, `24` on oracle-change groups.
- Oracle-change eligibility by relative error: `{'re101': 0, 're105': 0}`.

| policy | changed | exact | missed | false | wrong | Tess delta | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| center_top1_no_relative_guard | 195 | 5 | 6 | 138 | 52 | -11 | -1 |
| center_top1_re101 | 0 | 0 | 63 | 0 | 0 | 0 | 0 |
| center_top1_re105 | 0 | 0 | 63 | 0 | 0 | 0 | 0 |

### Partition `val`

- Groups: `75`; oracle-change groups: `17`.
- Oracle center-utility rank1/rank<=4: `7` / `20`; oracle-change rank1/rank<=4: `3` / `9`.
- Oracle center utility above nearest: `11` overall, `11` on oracle-change groups.
- Oracle-change eligibility by relative error: `{'re101': 1, 're105': 5}`.

| policy | changed | exact | missed | false | wrong | Tess delta | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| center_top1_no_relative_guard | 69 | 3 | 2 | 54 | 12 | -5 | 0 |
| center_top1_re101 | 7 | 1 | 15 | 5 | 1 | -1 | 0 |
| center_top1_re105 | 23 | 2 | 12 | 18 | 3 | -1 | 0 |

## Conclusion

Center utility is not a sufficient candidate-conditioned assignment signal. On validation, oracle-change center-rank1 is 3 and relative-error eligibility is {'re101': 1, 're105': 5}; this explains why N136 changes few useful codes and remains below the current compact policy.

## Next Action

Use center utility only as an auxiliary feature. The next assignment path should add candidate-local code-effect evidence or a policy-level no-headroom verifier, then validate with compact actual `.oscr` bytes.

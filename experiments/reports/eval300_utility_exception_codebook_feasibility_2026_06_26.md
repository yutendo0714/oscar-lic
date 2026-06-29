# Utility Exception Codebook Feasibility

This audit appends train oracle-change residual exception centers to the K64 utility-weighted codebook and checks distance feasibility only.
It does not claim OCR improvement because the new centers have not been decoded and evaluated.

Train oracle-change vectors per seed: `{'0': 21, '1': 20, '2': 22}`.

| extra centers | val oracle exception close re101 | val oracle exception close re105 | val exception chosen on oracle | val exception close no-headroom re101 | val exception close no-headroom re105 | val aug rel median |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 16 | 16 | 31 | 31 | 0.9619 |
| 8 | 16 | 16 | 16 | 36 | 36 | 0.9619 |
| 16 | 9 | 9 | 9 | 13 | 13 | 1.0000 |
| 32 | 3 | 3 | 3 | 8 | 8 | 1.0000 |
| 64 | 3 | 3 | 3 | 8 | 8 | 1.0000 |

## Conclusion

Utility-positive exception centers test whether changing the symbol proposal distribution can reduce the relative-error feasibility bottleneck observed in N137. This is a feasibility diagnostic only; new centers have no OCR evaluation until actual candidate reconstructions are generated.

## Next Action

If exception centers improve validation oracle feasibility without creating many close no-headroom opportunities, regenerate a small actual compact stream/OCR smoke for the best low-K setting.

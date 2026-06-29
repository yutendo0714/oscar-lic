# K80 Utility-Exception Compact Stream Audit

This audit checks actual `.oscr` streams for the K80=K64+16 exception-center codebooks.
The key question is whether appended exception centers are selected by the existing nearest assignment path.

## Summary

- Seeds: `3`; rows: `225`.
- Total selected candidates: `75`.
- Total assignment changes: `0`.
- Mean actual total bpp: `3.774026`.
- Mean enhancement payload bpp: `0.003781`.
- All stream byte checks matched results rows: `True`.

| seed | rows | selected | changed | avg bpp | enh bpp | PSNR delta | stream bytes ok |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 75 | 25 | 0 | 3.674724 | 0.004054 | -0.002472 | True |
| 1 | 75 | 25 | 0 | 3.888970 | 0.003417 | -0.003209 | True |
| 2 | 75 | 25 | 0 | 3.758383 | 0.003873 | -0.002792 | True |

## Conclusion

Actual compact .oscr streams were generated for the K80=K64+16 utility-exception codebooks, but nearest assignment selected no non-nearest/exception code on the 3-seed validation smoke. Therefore appending exception centers alone is a no-op for the current selected candidates; OCR evaluation is not informative until an explicit exception selector or no-headroom verifier can select them.

## Next Action

Do not run more nearest-assignment K80 OCR sweeps. Build a small policy that can explicitly propose exception centers with a no-headroom guard, then rerun actual compact stream/OCR only if assignment changes.

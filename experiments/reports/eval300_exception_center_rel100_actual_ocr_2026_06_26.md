# Exception-Center Actual OCR Diagnostic

Selected rows: `22` (proxy oracle-change `9`, proxy no-headroom `13`).

## Rate

| policy | images | selected candidates | mean bpp | enh bpp | PSNR delta |
|---|---:|---:|---:|---:|---:|
| current | 150 | 50 | 3.823677 | 0.003645 | -0.003016 |
| exception | 150 | 22 | 3.793834 | 0.001563 | -0.001407 |

## OCR Aggregate

| model | profile | current errors | exception errors | delta |
|---|---|---:|---:|---:|
| tesseract | unicode_strict_v1 | 316 | 319 | 3 |
| tesseract | latin_alnum_ci_v1 | 263 | 269 | 6 |
| parseq | unicode_strict_v1 | 55 | 55 | 0 |
| parseq | latin_alnum_ci_v1 | 36 | 36 | 0 |

## Selected-Row Delta

| model | profile | selected delta | improved | worsened | equal |
|---|---|---:|---:|---:|---:|
| tesseract | unicode_strict_v1 | 0 | 1 | 1 | 20 |
| tesseract | latin_alnum_ci_v1 | 2 | 0 | 2 | 20 |
| parseq | unicode_strict_v1 | 0 | 0 | 0 | 22 |
| parseq | latin_alnum_ci_v1 | 0 | 0 | 0 | 22 |

## Conclusion

The rel<=1.0 explicit exception-center policy lowers enhancement rate by selecting fewer candidates, but it is not an OCR improvement: PARSeq is unchanged and Tesseract worsens versus the matched current compact seed1/2 subset.

## Next Action

Do not run OCR for more simple exception-distance thresholds. Add a stronger no-headroom verifier or candidate-local visual/code-effect features before another actual-stream exception-center smoke.

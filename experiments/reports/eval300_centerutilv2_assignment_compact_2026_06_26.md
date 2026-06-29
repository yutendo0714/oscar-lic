# eval300_centerutilv2_assignment_compact_2026_06_26

W&B: `i0aqw89l`

Center-utility-biased code assignment diagnostic using compact `TEXT_SYMBOLS` streams. Rate is counted from actual `.oscr` files; OCR is PARSeq and Tesseract PSM8 over decoded reconstructions.

## Result

| policy | changed codes / selected | mean bpp | enh bpp | Tesseract chars | vs nearest | vs current | PARSeq chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| re101 | 7/75 | 3.774026 | 0.003781 | 469 | -1 | +7 | 78 |
| re105 | 23/75 | 3.774026 | 0.003781 | 469 | -1 | +7 | 78 |

## Interpretation

- Both relative-error settings preserve PARSeq exactly versus nearest (`0` unicode char delta over 225 samples).
- Both settings improve Tesseract by only `-1` unicode character versus nearest, despite changing 7/75 and 23/75 selected codes for re1.01 and re1.05 respectively.
- Both settings are `+7` Tesseract unicode characters worse than the current pooled image-diff policy at the same selected-candidate compact payload shape.
- This closes scalar center-utility-biased assignment as a promotion route; center utility is useful diagnostic metadata, not sufficient deployable evidence.

## Invalid Precheck

Initial center-utility assignment attempt loaded old utility-weighted codebooks without center_utility arrays, making assignment equivalent to nearest. Those centerutil_assign compact seed0 outputs are not used for N136 claims.

## W&B Runs

- re101 PARSeq: `{'0': 'vkuyna2p', '1': '1qx6gjzi', '2': 'kka8lbiu'}`; Tesseract: `{'0': 'ja0z1fqm', '1': 'bj5gppbd', '2': 'id83zh9k'}`
- re105 PARSeq: `{'0': '8p3dk4d0', '1': '5alf9v7f', '2': '5rkrlxcr'}`; Tesseract: `{'0': '8d1452c8', '1': 'njl80eun', '2': '79yhvfky'}`

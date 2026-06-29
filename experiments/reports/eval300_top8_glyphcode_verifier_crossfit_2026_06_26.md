# eval300_top8_glyphcode_verifier_crossfit_2026_06_26

Cross-fit diagnostic verifier over top-8 glyph-code differential features.
This uses held-out OCR-derived oracle labels as diagnostic targets only; it is not a promoted selector.

## Aggregate Margin Policy

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| changed_groups | 1.0 | 0.0 | 1.0 | 1.0 |
| exact | 19.333333333333332 | 4.109609335312651 | 14.0 | 24.0 |
| false_change | 0.3333333333333333 | 0.4714045207910317 | 0.0 | 1.0 |
| wrong_change | 0.3333333333333333 | 0.4714045207910317 | 0.0 | 1.0 |
| missed_oracle | 5.0 | 4.08248290463863 | 0.0 | 10.0 |
| tesseract_delta_vs_nearest | -0.6666666666666666 | 1.699673171197595 | -3.0 | 1.0 |
| parseq_delta_vs_nearest | 0.0 | 0.0 | 0.0 | 0.0 |
| accuracy | 0.7733333333333334 | 0.164384373412506 | 0.56 | 0.96 |

## Per-Seed

| val seed | threshold | changed | exact | false | wrong | missed | Tesseract delta | PARSeq delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.21615513748858128 | 1 | 24 | 1 | 0 | 0 | 1 | 0 |
| 1 | -0.27543312552677884 | 1 | 20 | 0 | 0 | 5 | -3 | 0 |
| 2 | -0.4199162912802193 | 1 | 14 | 0 | 1 | 10 | 0 | 0 |

## Interpretation

- The useful question is whether glyphcode evidence can improve exact top8 recovery without losing the no-op precision floor.
- If false/wrong changes remain high or Tesseract delta fails to beat the current `-8` actual-bitstream policy, treat this as a representation diagnostic only.

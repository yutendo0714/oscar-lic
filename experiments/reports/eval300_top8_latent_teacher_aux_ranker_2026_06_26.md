# eval300_top8_latent_teacher_aux_ranker_2026_06_26

Top-8 latent/code listwise ranker with CRNN/ABINet teacher-loss auxiliary regularization.
Teacher outputs are train-only auxiliary targets, not selector inputs. This is not an actual `.oscr` promotion.

## Validation Summary

| teacher weight | seed | exact changed | wrong | Tesseract | PARSeq | rank<=1 | rank<=4 | teacher loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0 | 7 | 10 | -12 | 0 | 7 | 13 | 0.2466 |
| 0.000 | 1 | 6 | 11 | -10 | 0 | 6 | 13 | 0.2716 |
| 0.000 | 2 | 5 | 12 | -10 | 0 | 5 | 12 | 0.2717 |
| 0.050 | 0 | 7 | 10 | -12 | 0 | 7 | 13 | 0.1834 |
| 0.050 | 1 | 6 | 11 | -10 | 0 | 6 | 12 | 0.1791 |
| 0.050 | 2 | 5 | 12 | -10 | 0 | 5 | 11 | 0.1866 |
| 0.200 | 0 | 8 | 9 | -12 | 0 | 8 | 13 | 0.1415 |
| 0.200 | 1 | 4 | 13 | -8 | 0 | 4 | 12 | 0.1328 |
| 0.200 | 2 | 4 | 13 | -10 | 0 | 4 | 13 | 0.1429 |

## Interpretation

This tests teacher-output regularization as distillation into deployable latent/code evidence. It deliberately excludes reference length and OCR-distance core fields from the input core tensor. Any positive result would still require a separate no-op acceptor and counted `.oscr` evaluation.

## Best Run

- teacher weight: `0.2`
- seed: `0`
- validation Tesseract delta: `-12`
- validation exact changed groups: `8`

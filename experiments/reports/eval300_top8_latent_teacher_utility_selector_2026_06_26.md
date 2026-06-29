# eval300_top8_latent_teacher_utility_selector_2026_06_26

No-op-inclusive latent/code selector trained on CRNN/ABINet teacher-loss utility.
Teacher signals are train targets, not selector inputs; no counted `.oscr` streams are exported.

## Validation Policy

| min gain | seed | changed | exact changed | false | wrong | missed | Tesseract | PARSeq | T worse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0 | 75 | 2 | 58 | 15 | 0 | 2 | 0 | 10 |
| 0.000 | 1 | 72 | 2 | 56 | 14 | 1 | 3 | 0 | 9 |
| 0.000 | 2 | 69 | 3 | 54 | 12 | 2 | 2 | 0 | 9 |
| 0.050 | 0 | 16 | 0 | 12 | 4 | 13 | 1 | 0 | 1 |
| 0.050 | 1 | 12 | 0 | 9 | 3 | 14 | 1 | 0 | 1 |
| 0.050 | 2 | 9 | 0 | 7 | 2 | 15 | 2 | 0 | 2 |
| 0.100 | 0 | 4 | 0 | 2 | 2 | 15 | 1 | 0 | 1 |
| 0.100 | 1 | 4 | 0 | 3 | 1 | 16 | 0 | 0 | 1 |
| 0.100 | 2 | 4 | 0 | 3 | 1 | 16 | 0 | 0 | 1 |

## Interpretation

This is the teacher-separated counterpart to N104. It tests whether CRNN/ABINet loss utility can define a no-op-inclusive target that transfers to held-out Tesseract/PARSeq evaluation without using teacher signals as selector inputs.

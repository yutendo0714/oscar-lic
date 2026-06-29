# eval300_top8_latent_policy_utility_selector_2026_06_26

No-op-inclusive latent/code selector trained on train OCR utility labels.
This is non-promotable because train Tesseract/PARSeq deltas are used as teacher labels.

## Validation Policy

| seed | changed | exact changed | false | wrong | missed | Tesseract | PARSeq | T worse | P worse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7 | 2 | 4 | 1 | 14 | 0 | 0 | 2 | 0 |
| 1 | 6 | 0 | 4 | 2 | 15 | 1 | 0 | 1 | 0 |
| 2 | 5 | 0 | 4 | 1 | 16 | 2 | 0 | 2 | 0 |

## Interpretation

This tests whether no-op-inclusive policy utility labels change the failure mode. Because train Tesseract/PARSeq deltas define the target, this is diagnostic only; promotion would require held-out OCR separation and counted `.oscr` streams.

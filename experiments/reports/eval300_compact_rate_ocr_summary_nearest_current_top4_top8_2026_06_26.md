# eval300_compact_rate_ocr_summary_nearest_current_top4_top8_2026_06_26

W&B: `ruuwcyk6`

Compact actual-byte OCR summary built from existing OCR bootstrap JSONs and N134 `.oscr` section accounting.

## Compact Rates

| policy | streams | selected | mean total bpp | enh payload bpp | optional bpp | selected optional bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nearest | 225 | 75 | 3.774026 | 0.003781 | 0.054197 | 43.000000 |
| current | 225 | 75 | 3.774026 | 0.003781 | 0.054197 | 43.000000 |
| top4 | 225 | 75 | 3.774026 | 0.003781 | 0.054197 | 43.000000 |
| top8 | 225 | 75 | 3.774026 | 0.003781 | 0.054197 | 43.000000 |

## Unicode OCR vs Nearest

| OCR | policy | CER | char errors | delta chars | delta chars CI95 | exact rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| paddleocr | nearest | 0.098949 | 113.000000 | 0.000000 | [0.000000, 0.000000] | 0.768889 |
| paddleocr | current | 0.098949 | 113.000000 | 0.000000 | [0.000000, 0.000000] | 0.768889 |
| paddleocr | top4 | 0.098949 | 113.000000 | 0.000000 | [0.000000, 0.000000] | 0.768889 |
| paddleocr | top8 | 0.098949 | 113.000000 | 0.000000 | [0.000000, 0.000000] | 0.768889 |
| paddleocr_e2e | nearest | 0.156743 | 179.000000 | 0.000000 | [0.000000, 0.000000] | 0.760000 |
| paddleocr_e2e | current | 0.156743 | 179.000000 | 0.000000 | null | 0.760000 |
| paddleocr_e2e | top4 | 0.156743 | 179.000000 | 0.000000 | null | 0.760000 |
| paddleocr_e2e | top8 | 0.156743 | 179.000000 | 0.000000 | [0.000000, 0.000000] | 0.760000 |
| parseq | nearest | 0.068301 | 78.000000 | 0.000000 | [0.000000, 0.000000] | 0.835556 |
| parseq | current | 0.068301 | 78.000000 | 0.000000 | [0.000000, 0.000000] | 0.835556 |
| parseq | top4 | 0.068301 | 78.000000 | 0.000000 | [0.000000, 0.000000] | 0.835556 |
| parseq | top8 | 0.068301 | 78.000000 | 0.000000 | [0.000000, 0.000000] | 0.835556 |
| tesseract | nearest | 0.411559 | 470.000000 | 0.000000 | [0.000000, 0.000000] | 0.333333 |
| tesseract | current | 0.404553 | 462.000000 | -8.000000 | [-19.000000, 0.000000] | 0.337778 |
| tesseract | top4 | 0.394046 | 450.000000 | -20.000000 | [-32.000000, -10.000000] | 0.342222 |
| tesseract | top8 | 0.390543 | 446.000000 | -24.000000 | [-38.000000, -13.000000] | 0.342222 |
| trocr | nearest | 0.762697 | 871.000000 | 0.000000 | [0.000000, 0.000000] | 0.128889 |
| trocr | current | 0.762697 | 871.000000 | 0.000000 | [0.000000, 0.000000] | 0.128889 |
| trocr | top4 | 0.762697 | 871.000000 | 0.000000 | [0.000000, 0.000000] | 0.128889 |
| trocr | top8 | 0.762697 | 871.000000 | 0.000000 | [0.000000, 0.000000] | 0.128889 |

## Key Pairwise Unicode Deltas

| OCR | comparison | delta chars | delta chars CI95 | delta CER |
| --- | --- | ---: | ---: | ---: |
| paddleocr | current_vs_top4 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| paddleocr | current_vs_top8 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| paddleocr | top4_vs_top8 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| paddleocr_e2e | current_vs_top4 | null | null | null |
| paddleocr_e2e | current_vs_top8 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| paddleocr_e2e | top4_vs_top8 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| parseq | current_vs_top4 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| parseq | current_vs_top8 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| parseq | top4_vs_top8 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| tesseract | current_vs_top4 | -12.000000 | [-19.000000, -6.000000] | -0.010508 |
| tesseract | current_vs_top8 | -16.000000 | [-26.000000, -8.000000] | -0.014011 |
| tesseract | top4_vs_top8 | -4.000000 | [-11.000000, 0.000000] | -0.003503 |
| trocr | current_vs_top4 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| trocr | current_vs_top8 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| trocr | top4_vs_top8 | 0.000000 | [0.000000, 0.000000] | 0.000000 |

## Interpretation

All four policies use matched compact TEXT_SYMBOLS actual bpp; OCR differences in this table come from selector/assignment choices, not from extra transmitted enhancement budget.

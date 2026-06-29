# PaddleOCR Eval300 Actual OSCR Current-vs-Shortlist Evaluation

Date: 2026-06-26 JST
Stage: held-out OCR diagnostic
Track: A pure visual bitstream evaluation-only diagnostic
Evaluator role: held-out PaddleOCR text recognizer, not used for selection, training, utility labels, feature extraction, or hyperparameter tuning

## Purpose

This run adds a stronger scene-text OCR family to the existing PARSeq, Tesseract and TrOCR evidence for the actual decoded Eval300 `.oscr` streams.

The evaluated recognizer is PaddleOCR `TextRecognition` with `PP-OCRv6_medium_rec` on CPU. It is used only on already-decoded word-crop images. No selector, threshold, utility label, training loss, or candidate feature consumed PaddleOCR outputs.

## Environment And Model

- Python env: `.venv-paddleocr`
- Lock file: `environment/locks.paddleocr-py310-cpu-2026_06_26.txt`
- Lock SHA256: `420056e7f6b5983373e365dda1e6d9c89fffff5103b806943f1d20fdffa48f3f`
- Resolved environment: `environment/resolved_environment.paddleocr_py310_cpu_2026_06_26.json`
- Resolved environment SHA256: `d6208a461c77b7c96d086b83992558d8e30e186ac16bb5bc65ee664cf4d4a285`
- `paddlepaddle`: `3.3.1`
- `paddleocr`: `3.7.0`
- Model: `PP-OCRv6_medium_rec`
- Model directory: `/root/.paddlex/official_models/PP-OCRv6_medium_rec`

Model bundle files:

| File | Bytes | SHA256 |
|---|---:|---|
| `.gitattributes` | 1575 | `a3a43406012ee8c7b9b38dfab1b11610f33e8245cd864a9a9302d042616a859e` |
| `README.md` | 23474 | `75ff9c4853ed171f36224127805062f9bcd1cfbbcacae3d577344866df74f6d3` |
| `inference.json` | 221814 | `0b2e25e990bd072f1bf77d59d67d508bce6c4bd44af6624e0fb27d6da2cd00e8` |
| `inference.pdiparams` | 76465087 | `1b01c79a914587933f615569e75de54f2e638ebb5d3f3b3c1b38c24ede8c7319` |
| `inference.yml` | 150580 | `991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129` |

The installation choice follows the current upstream split between the `paddleocr` package and a separately installed PaddlePaddle inference engine. The local environment used CPU-only PaddlePaddle to avoid mixing this evaluator setup with CUDA training experiments.

## Inputs

Three actual `.oscr` reconstruction conditions were evaluated for Eval300 seed0/1/2 val75:

- nearest K64 utility-weighted assignment;
- current best learned stream: pooled image-diff group gate, model seed 1, threshold `0.5`;
- top-4 shortlist oracle: score-seed-2 oracle choice inside the deployable shortlist.

All images are actual decoded reconstructions from the counted `.oscr` streams. No estimated bitrate is introduced here.

## Aggregate Metrics

Per-condition mean over the three seed splits:

| Condition | Samples | Latin CER mean | Latin exact mean | Unicode CER mean | Unicode exact mean |
|---|---:|---:|---:|---:|---:|
| nearest K64 | 225 | 0.0801 | 0.8089 | 0.0990 | 0.7689 |
| current best | 225 | 0.0801 | 0.8089 | 0.0990 | 0.7689 |
| top-4 shortlist oracle | 225 | 0.0801 | 0.8089 | 0.0990 | 0.7689 |

Row-wise comparisons:

| Comparison | Profile | Delta char errors | Improved | Worsened | Exact delta | CER delta |
|---|---|---:|---:|---:|---:|---:|
| nearest -> current best | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| nearest -> current best | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| nearest -> top-4 shortlist oracle | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| nearest -> top-4 shortlist oracle | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| current best -> top-4 shortlist oracle | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| current best -> top-4 shortlist oracle | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |

## Interpretation

PaddleOCR is much stronger than TrOCR on these word crops (`unicode CER` around `0.099` versus TrOCR's much weaker crop performance), so this is useful held-out evidence.

The key result is stability: neither the current best actual stream nor the non-deployable top-4 shortlist oracle changes any PaddleOCR output relative to nearest across all `225` evaluated samples. This does not prove OCR improvement for PaddleOCR, but it is important negative-risk evidence: the Tesseract gain of the top-4 actual shortlist oracle is not achieved by breaking this stronger OCR family.

This evaluator should remain held-out. It is now locked for evaluation-only smoke, but its predictions must not be used for selector training, candidate labels, source filtering, confidence features, or hyperparameter selection.

## W&B

Aggregate offline W&B run: `xb5dwhsq`.

Per-condition PaddleOCR runs were executed with `--no-wandb`; the aggregate JSON and comparisons were logged as one offline artifact.

## Files

- Runner: `scripts/run_paddleocr_ocr_smoke.py`
- Aggregate JSON: `experiments/results/paddleocr_eval300_actual_oscr_current_vs_shortlist_2026_06_26.json`
- Nearest-vs-current comparison: `experiments/results/compare_paddleocr_eval300_val75_nearest_vs_pooled_imgdiff_model1_threshold05_q005_2026_06_26.json`
- Nearest-vs-shortlist comparison: `experiments/results/compare_paddleocr_eval300_val75_nearest_vs_shortlist_oracle_top4_scoreseed2_q005_2026_06_26.json`
- Current-vs-shortlist comparison: `experiments/results/compare_paddleocr_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top4_scoreseed2_q005_2026_06_26.json`

Note: an initial original Eval300 full-crop run was written under `experiments/results/paddleocr_eval300_original_smoke10_2026_06_26/`; the directory name is misleading, but the contained summary records `samples=300`. It is retained as an audit artifact and is not used in the current-vs-shortlist comparison.

# CompressAI Text-Crop Rate-OCR Sweep

**Date:** 2026-06-25 JST  
**Split:** `data/splits/text_crops_parseq_smoke_v1.jsonl`  
**Codec:** CompressAI `bmshj2018-hyperprior`, MSE, qualities 1-6  
**Rate:** complete OSCAR CompressAI smoke container bytes from actual `compress/decompress`
**Combined wandb offline run:** `km4obt7n`

## Aggregate Curve

MS-SSIM is omitted because the 32 px word-crop height is too small for the standard multi-scale metric.

| q | actual bpp | PSNR | PARSeq CER | PARSeq exact | PARSeq latin CER | Tesseract CER | Tesseract exact | Tesseract latin CER | PARSeq wandb | Tesseract wandb |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.6033376472 | 25.5183318257 | 0.1488095238 | 0.6250000000 | 0.1197604790 | 0.4761904762 | 0.1250000000 | 0.3652694611 | `tswigt58` | `dtrvqngs` |
| 2 | 0.7346418653 | 27.3449565470 | 0.1071428571 | 0.5833333333 | 0.0718562874 | 0.3809523810 | 0.1666666667 | 0.2754491018 | `4ygjpfx3` | `rvxb2p5c` |
| 3 | 0.9070354526 | 29.3803426127 | 0.0952380952 | 0.7083333333 | 0.0598802395 | 0.3928571429 | 0.2500000000 | 0.2335329341 | `561pgr9e` | `v0zhjb4s` |
| 4 | 1.1417957704 | 31.6770346959 | 0.0833333333 | 0.7500000000 | 0.0479041916 | 0.3690476190 | 0.2500000000 | 0.2335329341 | `yi5ada3a` | `2o7tplmd` |
| 5 | 1.4374058418 | 33.3313662807 | 0.0892857143 | 0.7083333333 | 0.0538922156 | 0.3928571429 | 0.2500000000 | 0.2095808383 | `2e3v8wm9` | `n6zrsj1n` |
| 6 | 1.7922247743 | 35.3800166895 | 0.0714285714 | 0.7500000000 | 0.0359281437 | 0.3571428571 | 0.2500000000 | 0.1976047904 | `nq4pv22n` | `8tvud9wd` |

Original upper bounds:

| Evaluator | unicode CER | unicode exact | latin CER | latin exact |
|---|---:|---:|---:|---:|
| PARSeq | 0.0714285714 | 0.7083333333 | 0.0359281437 | 0.7916666667 |
| Tesseract PSM8 | 0.3511904762 | 0.2916666667 | 0.2095808383 | 0.5000000000 |

## Reading

The PARSeq teacher curve reaches the original-crop CER at q6 and exceeds original exact rate at q4/q6. That does not mean compression improves text generally; it means the teacher has sample-level instability and sometimes benefits from smoothing or artifact changes. Any utility oracle must therefore report original-recognizable strata and per-sample deltas relative to original OCR output, not only ground-truth CER.

Tesseract is much noisier and weaker on these crops, but it responds differently from PARSeq. This makes it useful as a held-out warning signal, not yet as a final paper metric. A stronger recognizer family such as PaddleOCR or TrOCR is still required before claims about recognizer-invariant preservation.

For the first OSCAR-LIC oracle, the practical early target is to beat CompressAI q2-q4 at matched actual total bpp on PARSeq while not worsening Tesseract relative to the same-rate baseline.

## Original-Recognizable Strata

Strata files:

- PARSeq: `experiments/results/ocr_strata_parseq_text_crops_smoke_v1_2026_06_25.jsonl`, SHA256 `de89af3155daa6128f563b7f992467bfbec06564e8b9a24ea240e116b0114327`
- Tesseract: `experiments/results/ocr_strata_tesseract_text_crops_smoke_v1_2026_06_25.jsonl`, SHA256 `af5aaa9c8f920dbb6bc6d25a48d0b791e86b83abc2e416bd42ec999478fb311a`

`unicode_strict_v1` summary:

| Evaluator | Method | Original-exact samples | Worsened original-exact | Fixed original-error | Exact on original-exact | CER on original-exact |
|---|---|---:|---:|---:|---:|---:|
| PARSeq | CompressAI q1 | 17 | 4 | 2 | 0.7647058824 | 0.0727272727 |
| PARSeq | CompressAI q2 | 17 | 4 | 1 | 0.7647058824 | 0.0454545455 |
| PARSeq | CompressAI q3 | 17 | 1 | 1 | 0.9411764706 | 0.0090909091 |
| PARSeq | CompressAI q4 | 17 | 0 | 1 | 1.0000000000 | 0.0000000000 |
| PARSeq | CompressAI q5 | 17 | 0 | 0 | 1.0000000000 | 0.0000000000 |
| PARSeq | CompressAI q6 | 17 | 0 | 1 | 1.0000000000 | 0.0000000000 |
| PARSeq | MLIC++ corrected | 17 | 0 | 0 | 1.0000000000 | 0.0000000000 |
| Tesseract | CompressAI q1 | 7 | 4 | 0 | 0.4285714286 | 0.1960784314 |
| Tesseract | CompressAI q2 | 7 | 3 | 0 | 0.5714285714 | 0.1176470588 |
| Tesseract | CompressAI q3 | 7 | 2 | 1 | 0.7142857143 | 0.0392156863 |
| Tesseract | CompressAI q4 | 7 | 1 | 0 | 0.8571428571 | 0.0196078431 |
| Tesseract | CompressAI q5 | 7 | 2 | 1 | 0.7142857143 | 0.0784313725 |
| Tesseract | CompressAI q6 | 7 | 1 | 0 | 0.8571428571 | 0.0196078431 |
| Tesseract | MLIC++ corrected | 7 | 3 | 0 | 0.5714285714 | 0.0980392157 |

This is the first concrete reason to keep OSCAR-LIC simple but recognizer-invariant: the teacher-recognizable subset can look solved while a held-out recognizer still loses exact words.

## Next Use

1. Construct an `original_recognizable` flag for each evaluator and profile.
2. Compute per-sample OCR deltas against original OCR output and ground truth.
3. Use q2-q4 as baseline budgets for the first greedy counterfactual allocation smoke.

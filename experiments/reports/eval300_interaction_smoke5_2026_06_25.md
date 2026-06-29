# Eval300 Raw Residual Interaction Smoke5

**Date:** 2026-06-25  
**W&B:** offline run `fhk5uy8l` (`eval300_interaction_smoke5_actual_2026_06_25`)  
**Scope:** sampled multi-candidate interaction smoke using counted raw latent residual `.oscr` streams.

## Purpose

The promoted codebook-assignment path currently sends at most one distinct enhancement candidate per source image, so within-image subset/Shapley interaction cannot be measured there. This smoke samples five images from the raw Eval300 candidate table, each with two distinct safe-improving latent candidates, and compares:

- `single_a`: candidate A only;
- `single_b`: candidate B only;
- `pair_ab`: candidates A+B together.

Rate is measured from actual `.oscr` files. This is a high-rate raw-residual diagnostic, not a promotion candidate.

## Actual Stream Summary

| policy | selected candidates/image | total bpp | enhancement bpp | PSNR delta |
|---|---:|---:|---:|---:|
| single_a | 1.0 | 5.9949 | 2.1475 | +0.0539 dB |
| single_b | 1.0 | 5.9949 | 2.1475 | +0.0655 dB |
| pair_ab | 2.0 | 8.0547 | 4.2074 | +0.1222 dB |

## OCR Interaction

Interaction residual is:

```text
observed_pair_delta - (single_a_delta + single_b_delta)
```

Positive means the pair underperforms an additive single-candidate utility model.

| OCR | profile | single A | single B | additive expected pair | observed pair | interaction residual |
|---|---|---:|---:|---:|---:|---:|
| Tesseract PSM8 | unicode strict | -4 | -4 | -8 | -1 | +7 |
| Tesseract PSM8 | latin alnum | -4 | -3 | -7 | -1 | +6 |
| PARSeq | unicode strict | -1 | -1 | -2 | -1 | +1 |
| PARSeq | latin alnum | -1 | -1 | -2 | -1 | +1 |

## Interpretation

This small smoke shows real non-additivity. Tesseract utility does not add across two individually useful latent residual candidates; applying both together recovers only one edit character instead of the eight predicted by adding the two single-candidate effects. PARSeq shows a milder saturation effect: either candidate fixes the one remaining PARSeq error, so the pair cannot improve twice.

For OSCAR-LIC this means a future multi-candidate allocator should not greedily sum marginal utilities without a diminishing-return or interaction model. A simple next step is to add per-image budgeted subset labels from raw candidates before training any multi-candidate policy.

## Artifacts

```text
06694fe79b9c066df6e5513120d95dba69d632e9a00533fdf4e2b887b36d4016  scripts/run_mlicpp_latent_residual_stub.py
1601a035ef680b2be37d53e0535db72e6bd2f09c7989cbbb083b925483c31139  scripts/build_interaction_smoke_policies.py
9e25242e9310d2f7dcf369fa771d69170f8f22a78907de088106153561d181ac  scripts/make_ocr_split_from_codec_results.py
1df8afd1b345316817804f2c1786ef24f01f39d8e3994499ec1ebcafa4150f64  experiments/results/eval300_interaction_smoke5_summary_2026_06_25.json
0cf54be3389317653355495d5a6af8c52e2eb04a6f1208d35ab3848c3b7721d0  experiments/results/compare_tesseract_eval300_interaction_smoke5_base_vs_single_pair_2026_06_25.json
ae45bc6f76ebba35a19e7216e7654b88676051a0ecc2c58fabc2a84536afc1a7  experiments/results/compare_parseq_eval300_interaction_smoke5_base_vs_single_pair_2026_06_25.json
```

# Eval300 Raw Residual Interaction Smoke20

Date: 2026-06-25

W&B:

- Summary artifact run: `if3r0rhv`
- PARSeq OCR runs: base `1reo2hjr`, single A `du2yx343`, single B `stv1s5gr`, pair AB `1inlpter`
- Tesseract PSM8 OCR runs: base `atot8zwy`, single A `y5trznws`, single B `p3xyl5h9`, pair AB `j5yzoewc`

## Purpose

The prior 5-image interaction smoke showed that two individually useful raw latent residual candidates do not combine additively. This expands the same counted `.oscr` raw-residual diagnostic to 20 Eval300 images, each with two distinct safe-improving candidates in the single-candidate table.

This is high-rate raw residual analysis, not a promotion candidate.

## Actual Stream Summary

| policy | selected candidates/image | total bpp | enhancement bpp | PSNR delta |
|---|---:|---:|---:|---:|
| single_a | 1.0 | 5.2794 | 1.7686 | +0.0549 dB |
| single_b | 1.0 | 5.2794 | 1.7686 | +0.0605 dB |
| pair_ab | 2.0 | 6.9769 | 3.4662 | +0.1169 dB |

Rate is measured from actual `.oscr` files containing `BASE_MAIN`, `TEXT_GATE` and `TEXT_MAIN`.

## OCR Interaction

Interaction residual is:

```text
observed_pair_delta - (single_a_delta + single_b_delta)
```

Positive means the pair underperforms an additive single-candidate utility model.

| OCR | profile | single A | single B | additive expected pair | observed pair | interaction residual |
|---|---|---:|---:|---:|---:|---:|
| Tesseract PSM8 | unicode strict | -31 | -28 | -59 | -10 | +49 |
| Tesseract PSM8 | latin alnum | -23 | -17 | -40 | -6 | +34 |
| PARSeq | unicode strict | -4 | -4 | -8 | -4 | +4 |
| PARSeq | latin alnum | -1 | -1 | -2 | -1 | +1 |

## Interpretation

The 20-image result confirms and strengthens the 5-image smoke: OCR utility is highly non-additive for multi-candidate residual enhancement. For Tesseract, applying both individually useful candidates recovers only `10` unicode edit characters, while an additive marginal model would predict `59`. PARSeq also saturates: the pair does no better than either single candidate.

For OSCAR-LIC, this means multi-candidate allocation should not sum independent marginal utilities without a diminishing-return or subset-aware component. The current center hypothesis should be refined from "marginal OCR utility per bit" to "conditional marginal OCR utility per bit given already selected enhancement units" for any multi-candidate stream.

## Artifacts

- Summary JSON SHA256: `215cf3652f9de32bc69ae3d677745b68b1e0c763f0f5dfbe7d5d3e68887c7dc2`
- Tesseract compare SHA256: `0522e6fb51ed0e7edcf1c3a6388622c7abe53dd6e3b16779cb9f0f13ba3f2ddc`
- PARSeq compare SHA256: `519cc8a2a697864d8b35fb9cd3b8c6283b6eaa08141ad647aa7bc06bab18533e`
- Split SHA256: `cacb6cf6041618b3b1dda0b263ff8e454c1deb1b6e50166a3c6c6df130d70d2b`

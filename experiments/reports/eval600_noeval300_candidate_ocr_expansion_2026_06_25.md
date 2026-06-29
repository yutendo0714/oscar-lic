# Eval600 No-Overlap Candidate/OCR Expansion

Date: 2026-06-25

## Purpose

The Eval300 top8 assignment diagnostics exposed poor clean-val and source-disjoint generalization. This run expands the true single-candidate latent residual label substrate to a larger, Eval300-disjoint crop split before training another selector.

This is infrastructure, not a promoted compression result. Candidate bpp values below are measured from the serialized single-candidate residual `.oscr` files and are used to build utility labels; they are not a proposed operating point.

## Inputs

- Source manifest: `/dpl/veritext_crops/veritext_text_fidelity_eval_balanced_v1/manifest.jsonl`
- Source manifest SHA256: `a9ec12ce6e206bbe71b173b519646b093e839376f0260f9c9c7bf69349cfd68b`
- Excluded Eval300 split: `data/splits/text_crops_eval300_v1.jsonl`
- Excluded Eval300 split SHA256: `c4af53ebe08dd8ba088382344a776ba2f86087bb1db9d1b3c9fc9dbcc68c6f55`
- Corrected MLIC++ checkpoint: `experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar`
- Checkpoint SHA256: `67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559`

## Split

Created `data/splits/text_crops_eval600_noeval300_v1.jsonl`.

- Rows: 600
- Eval300 key overlap: 0
- Seed: `20260626`
- Filters: width >= 24, height >= 24, text length 2 to 24, alnum required
- Source balance: 100 each from `iam_words`, `icdar2013`, `icdar2015`, `icdar_mlt2019_words`, `synthtext_words`, `textocr`
- Split SHA256: `1f234a536ef980467a21c779631b53f3bb4a9acb6839d8ce87cdfd0957cd25b8`
- Metadata SHA256: `1febcf42c3f3768e600fa9d01009761c4818fbc997de946646cb3b4880102c03`

## Candidate Reconstructions

Generated one-candidate MLIC++ latent residual reconstructions with tile `4`, residual scale `0.25`, and quant step `0.05`.

- Candidate rows: 9600
- Images: 600
- Candidates per image: min 10, max 60, mean 16.0
- Candidate row source counts: `iam_words` 1730, `icdar2013` 2050, `icdar2015` 1410, `icdar_mlt2019_words` 1700, `synthtext_words` 1350, `textocr` 1360
- Actual serialized single-candidate bpp: min 2.3524305556, max 14.4791666667, mean 5.5093301305
- JSONL SHA256: `f5a242fd23b2bc4bd313b26f241e544df39be9ac3b8dd057812e7d6ecbd7d5ec`
- Metadata SHA256: `c8565149dd316386715c45c0e269e001067ca6d5f60cba4691f4cdb02ceb0829`
- Asset directory size: 68M

## OCR Runs

### PARSeq

- W&B run: `3lot3tud`
- Samples: 9600
- Device: CUDA on NVIDIA GeForce RTX 4070 Ti SUPER
- Checkpoint SHA256: `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`
- Results SHA256: `9369b88ce2b17093100d1f12645d9dbbacecce30db0c8aea036c60aa287d707f`
- Summary SHA256: `3e4534b2dd93e7f0f6bf3691bdf821fcff852a49351e9e1c9937d0f0873762ea`

Metrics:

| profile | CER | WER | exact |
|---|---:|---:|---:|
| unicode_strict_v1 | 0.0701797243 | 0.1882291667 | 0.8117708333 |
| latin_alnum_ci_v1 | 0.0507183908 | 0.1442708333 | 0.8557291667 |

### Tesseract PSM8

- W&B run: `vd6nv38v`
- Samples: 9600
- Failed samples: 0
- Tesseract: 5.5.0, `eng`, `--psm 8`, `--oem 1`
- `eng.traineddata` SHA256: `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`
- Results SHA256: `3ae0a13a40af12c50de354d9edb0ae1cb1d7235a0fedfc1298ff4b30c8898855`
- Summary SHA256: `678dbbb05c531fdac39808714718335e2acf3271913d3d3dc53ab2db26c1b360`

Metrics:

| profile | CER | WER | exact |
|---|---:|---:|---:|
| unicode_strict_v1 | 0.3669167685 | 0.7444791667 | 0.4177083333 |
| latin_alnum_ci_v1 | 0.3040589080 | 0.6383333333 | 0.4904166667 |

## Interpretation

Eval600 gives a larger, Eval300-disjoint pool for utility-table construction and source/difficulty-aware calibration. The immediate use is to build a PARSeq/Tesseract utility table from these OCR outputs, then train a high-precision no-headroom rejector or candidate-local code-effect predictor with stricter clean split checks before any counted `.oscr` promotion run.

Do not compare these raw single-candidate residual bpp values to the current promoted K64 assignment gate. The promoted baseline remains the pooled image-diff group gate with measured actual `.oscr` bpp and PARSeq/Tesseract row-wise OCR verification.


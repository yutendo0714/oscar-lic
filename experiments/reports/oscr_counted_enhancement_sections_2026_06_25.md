# OSCR Counted Enhancement Section Smoke

**Date:** 2026-06-25  
**Checkpoint:** `experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar`  
**Checkpoint SHA256:** `67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559`  
**GPU:** NVIDIA GeForce RTX 4070 Ti SUPER  

## Purpose

Verify the first counted frozen-base OSCAR enhancement container path:

1. serialize corrected MLIC++ output as a `BASE_MAIN` section;
2. append counted optional `TEXT_GATE` and `TEXT_MAIN` sections;
3. parse and CRC-check the complete `.oscr` file;
4. decode only the `BASE_MAIN` payload and confirm base reconstruction is unchanged;
5. report actual transmitted bpp from the complete file bytes.

The `TEXT_MAIN` payload in this smoke is deterministic dummy bytes, not a learned enhancement model. Its purpose is byte accounting and decoder-compatibility validation.

## Implementation

- `src/oscarlic/candidates.py`
  - added concrete `encode_gate_payload()` / `decode_gate_payload()`;
  - supports dense bitset and sparse sorted index-list gates;
  - keeps the gate header at 14 bytes.
- `scripts/run_mlicpp_oscr_smoke.py`
  - added `--with-dummy-text-sections`;
  - appends `TEXT_GATE` and `TEXT_MAIN`;
  - still decodes from `BASE_MAIN` only.
- `scripts/analyze_mlicpp_candidate_overheads.py`
  - estimates incremental candidate/gate/main section cost from MLIC++ latent shapes.

## Commands

```bash
.venv-mlic/bin/python scripts/run_mlicpp_oscr_smoke.py \
  --checkpoint experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar \
  --split data/splits/clic_valid_mlicpp_small_v1.txt \
  --output-dir experiments/results/mlicpp_oscr_dummy_sections_clic9_2026_06_25 \
  --cuda --with-dummy-text-sections \
  --candidate-tile 4 4 --selected-fraction 0.10 \
  --enhancement-global-bytes 40 --enhancement-bytes-per-selected 40

.venv-mlic/bin/python scripts/run_mlicpp_oscr_smoke.py \
  --checkpoint experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar \
  --split data/splits/text_crops_parseq_smoke_v1.jsonl \
  --output-dir experiments/results/mlicpp_oscr_dummy_sections_text24_2026_06_25 \
  --cuda --with-dummy-text-sections \
  --candidate-tile 4 4 --selected-fraction 0.10 \
  --enhancement-global-bytes 40 --enhancement-bytes-per-selected 40
```

## Results

| Split | Images | Total bpp | Base payload bpp | Optional payload bpp | Non-base bpp | Gate bytes | Enhancement bytes | Structural bytes | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIC9 | 9 | 0.822365 | 0.733431 | 0.086453 | 0.088933 | 265.89 | 8093.33 | 168.00 | 35.0449 | 0.987100 |
| Text24 | 24 | 3.559793 | 2.713753 | 0.367506 | 0.846040 | 17.00 | 120.00 | 168.00 | 35.1379 | n/a |

`non_base_bpp` includes all bytes except the MLIC++ base payload. `structural_bytes` is the complete OSCR header/table/file-CRC overhead for the three-section stream. The incremental cost over the one-section base adapter is approximately two extra section-table entries plus the optional payload.

## Candidate Overhead Estimates

With two optional sections, 40 global bytes and 40 bytes per selected candidate:

| Split | Layout | Selected | Mean candidates | Mean gate bytes | Mean incremental optional bpp | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| CLIC9 | slice x 4x4 latent tile | 10% | 2013.33 | 265.89 | 0.087634 | 0.082318 | 0.092378 |
| CLIC9 | slice x 8x8 latent tile | 10% | 537.78 | 81.33 | 0.024846 | 0.022460 | 0.028448 |
| CLIC9 | 8ch group x 4x4 latent tile | 10% | 8053.33 | 1020.67 | 0.344591 | 0.324010 | 0.365907 |
| Text24 | slice x 4x4 latent tile | 10% | 20.00 | 17.00 | 0.595379 | 0.366477 | 0.916667 |
| Eval100 | slice x 4x4 latent tile | 10% | 17.50 | 16.75 | 0.730064 | 0.350543 | 1.833333 |

## Artifacts

- `experiments/results/mlicpp_oscr_dummy_sections_clic9_2026_06_25/summary.json`
  - SHA256 `6a7d9f1fd91fd48bc281c205d6ae22610149605161cdf30f53a20d3008e5d1ec`
- `experiments/results/mlicpp_oscr_dummy_sections_clic9_2026_06_25/results.jsonl`
  - SHA256 `4a718a8a234777c9ed112b454483b682490f9add11c88c27669de67573225500`
- `experiments/results/mlicpp_oscr_dummy_sections_text24_2026_06_25/summary.json`
  - SHA256 `c59000d4fe4b55158e7816f3f3f09408bb2c52400e027a155206d0320fbf4f12`
- `experiments/results/mlicpp_oscr_dummy_sections_text24_2026_06_25/results.jsonl`
  - SHA256 `2aba05ffc3dc60f4a31c54531ad5044f53a54628ef2559dbc8ec7de7366aa330`
- `experiments/results/candidate_overhead_clic9_2026_06_25.json`
  - SHA256 `63b1c3cee3ecca5825e5c17d6059e4010f0c0e729e2b09ee9eb47cdec09e9808`
- `experiments/results/candidate_overhead_text24_2026_06_25.json`
  - SHA256 `c04b780bc2fa6caa77734917db3122c39c5a1d62679d725439c833af7736ac49`
- `experiments/results/candidate_overhead_eval100_2026_06_25.json`
  - SHA256 `3af72602ce1f99cee6b26d7d6393ddcbd7cd449e75263af64d4c5bb3ea21168d`

## Interpretation

The frozen-base adapter can now carry counted optional OSCAR sections without touching the corrected MLIC++ base stream. This keeps actual bpp accounting clean and supports the publication claim that base reconstruction remains independently decodable.

For natural-image resolutions, slice-level `8x8` or `4x4` latent candidates have plausible overhead. For isolated word crops, fixed section overhead dominates and makes fine candidate granularity misleadingly expensive. The next prototype should therefore either batch text-crop experiments at the image level or evaluate on full images with detected/latent text regions, while using word crops mainly for controlled OCR sensitivity studies.

## Next Step

Implement a real frozen-base enhancement payload stub that consumes selected MLIC++ latent candidate geometry and predicts a small residual/refiner input, while preserving the section accounting validated here. The first learned variant should stay simple: slice-level candidates, coarse `8x8` or `4x4` latent tiles, and a lightweight utility predictor trained against the multi-teacher OCR objective.

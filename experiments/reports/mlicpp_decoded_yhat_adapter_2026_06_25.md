# MLIC++ Decoded `y_hat` Adapter Smoke

**Date:** 2026-06-25  
**Checkpoint:** `experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar`  
**Checkpoint SHA256:** `67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559`  
**GPU:** NVIDIA GeForce RTX 4070 Ti SUPER  

## Purpose

Expose reconstructed MLIC++ latent `y_hat` without modifying the external MLIC++ repository. This is the first required adapter for true OSCAR latent/enhancement candidate units: the decoder must be able to reconstruct the frozen base latent, optionally apply a counted enhancement, and then run `g_s(y_hat)`.

## Implementation

`scripts/inspect_mlicpp_decoded_yhat.py` copies the upstream `decompress()` dependency order and returns both `x_hat` and `y_hat`. It then compares the copied decoder output against upstream `model.decompress()` on the same bitstream.

Candidate stats are computed over the reconstructed `y_hat` with MLIC++ slice-aware latent tiles:

- slice count: 10;
- channels per slice: 32;
- spatial tile: `4x4` latent cells in this smoke;
- score: candidate RMS and mean absolute activation.

## Results

| Split | Images | Max `x_hat` diff vs upstream | Avg candidates | Avg mean candidate RMS | Avg max candidate RMS |
|---|---:|---:|---:|---:|---:|
| CLIC2 | 2 | 0.0 | 660.0 | 1.5318 | 11.6398 |
| Text24 | 24 | 0.0 | 20.0 | 2.1860 | 12.4526 |
| Eval100 | 100 | 0.0 | 17.5 | 2.3438 | 12.4913 |

Top-RMS candidate slices:

| Split | Top-1 slice distribution | Top-8 slice distribution |
|---|---|---|
| CLIC2 | `{2: 2}` | `{2: 16}` |
| Text24 | `{2: 24}` | `{1: 2, 2: 48, 3: 4, 4: 17, 5: 36, 6: 31, 7: 31, 8: 18, 9: 5}` |
| Eval100 | `{2: 100}` | `{1: 12, 2: 175, 3: 38, 4: 102, 5: 127, 6: 112, 7: 127, 8: 73, 9: 34}` |

## Artifacts

- `experiments/results/mlicpp_decoded_yhat_clic2_2026_06_25/summary.json`
  - SHA256 `cbb2dddf4fe064b3c442eac8bf8d885ba56ddb165c5ad73622e36c3137b1d832`
- `experiments/results/mlicpp_decoded_yhat_clic2_2026_06_25/decoded_yhat_stats.jsonl`
  - SHA256 `3dfd77e5827ea4339e0ceb2f72af90efaea80fe0e222a3f5295c46755829170b`
- `experiments/results/mlicpp_decoded_yhat_text24_2026_06_25/summary.json`
  - SHA256 `e2a73646534c60bc7aec60a8875858a1d4c5ef12a7a5fa5b5c2a7b4545cb259c`
- `experiments/results/mlicpp_decoded_yhat_text24_2026_06_25/decoded_yhat_stats.jsonl`
  - SHA256 `b7cf47fd14a3e72ec1ec9c9ebccd76b6c2fc1b9965ad924451cc791d70d460c6`
- `experiments/results/mlicpp_decoded_yhat_eval100_2026_06_25/summary.json`
  - SHA256 `ce86830ca01ec3d90aa456c1a53e1e5c8a076b352e1ab5f7a88f14128148961c`
- `experiments/results/mlicpp_decoded_yhat_eval100_2026_06_25/decoded_yhat_stats.jsonl`
  - SHA256 `9b409c879011922746e99d25a79da1d37365435fb0d329c0bf2d5b7df10ce347`

## Interpretation

The copied decoder is numerically identical to upstream for the tested streams, so it is safe to use as a research adapter for frozen-base latent enhancement. The activation magnitude signal is not the same as the likelihood-derived bit signal: top-RMS candidates concentrate strongly in slice 2, while earlier likelihood diagnostics showed text-crop estimated bits concentrated more in mid/late slices. A useful utility predictor should therefore combine several signals rather than ranking candidates by activation magnitude alone.

## Next Step

Add a minimal enhancement application path:

1. decode `BASE_MAIN` to `y_hat`;
2. decode counted `TEXT_GATE`;
3. apply a small counted residual/refiner payload on selected candidates;
4. run `g_s(y_hat_enhanced)`;
5. verify base-only fallback and complete-file bpp.

The first learned variant should stay simple: frozen MLIC++ base, slice-level `4x4` or `8x8` candidate geometry, multi-teacher utility labels, and no modification to the upstream base ANS stream.

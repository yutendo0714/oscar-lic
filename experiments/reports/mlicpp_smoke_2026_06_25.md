# MLIC++ Corrected Checkpoint Smoke

**Date:** 2026-06-25 JST  
**Purpose:** verify corrected MLIC++ checkpoint acquisition, SHA256, load path, CUDA path, and actual compress/decompress bitstream accounting.

## Checkpoint

- ID: `mlicpp_mse_0025_corrected`
- Source: `https://drive.google.com/file/d/1FWPezuHLTQhDmEhShViI3XOSXA5u_Bya/view`
- Local path: `experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar`
- Size: `1004218690` bytes
- SHA256: `67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559`
- Architecture: corrected `LatentResidualPrediction` and `SynthesisTransform`, not `Old` classes
- Loaded class: `MLICPlusPlus`
- Checkpoint epoch: `612`
- Checkpoint loss: `1.0770463943481445`

## Environment

- Environment: `.venv-mlic`
- Python: `3.10.12`
- PyTorch: `2.0.1+cu118`
- torchvision: `0.15.2+cu118`
- CompressAI: `1.2.0b3`
- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- CUDA availability: true

## Command

```bash
.venv-mlic/bin/python scripts/run_mlicpp_smoke.py \
  --cuda \
  --checkpoint experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar \
  --output-dir experiments/results/mlicpp_smoke_2026_06_25
```

The script uses upstream MLIC++ `model.compress`, `model.decompress`, `write_uints`, `write_body`, `read_uints`, and `read_body`. Reported bpp is computed from the complete produced stream file:

```text
actual_total_bpp = 8 * stream_path.stat().st_size / (original_height * original_width)
```

## Results

Output directory: `experiments/results/mlicpp_smoke_2026_06_25`

| Image | Size | Stream bytes | Actual bpp | PSNR | MS-SSIM | Enc sec | Dec sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `/dpl/clic/professional/valid/schicka-307.png` | 512x384 | 4948 | 0.2013346354 | 38.8613539494 | 0.9864925742 | 0.1089286804 | 0.1598052979 |
| `/dpl/clic/professional/valid/todd-quackenbush-222.png` | 720x439 | 32004 | 0.8100227790 | 33.5918699524 | 0.9838483334 | 0.3057374954 | 0.1965157986 |

Summary:

- Images: 2
- Average actual total bpp: `0.5056787072`
- Average PSNR: `36.2266119509`
- Average MS-SSIM: `0.9851704538`

## Output Hashes

Streams:

- `experiments/results/mlicpp_smoke_2026_06_25/streams/000_schicka-307.mlicpp`: `c08b5bac1a567cce2e7fb3e911e7778bb9b84d87b50482f0dde39b56b96d8d4b`
- `experiments/results/mlicpp_smoke_2026_06_25/streams/001_todd-quackenbush-222.mlicpp`: `26aa6c725b2d8e4e5510643f4464d8765f6caa1e125cc4bf4324ab95667c19e5`

Reconstructions:

- `experiments/results/mlicpp_smoke_2026_06_25/reconstructions/000_schicka-307_rec.png`: `482a0212af91cdde72312ed4ccdd1f0e6c74e31a60e1f106d719ec1f91553de6`
- `experiments/results/mlicpp_smoke_2026_06_25/reconstructions/001_todd-quackenbush-222_rec.png`: `6ec659bf06a34bcf7e9496ee53a4ed7951c5989724ed669c08513ebdd7fabf6a`

## Warnings

- This is a smoke test, not a benchmark reproduction or publication result.
- The command emitted PyTorch warnings about future `torch.meshgrid(indexing=...)` behavior and a CuDNN workaround related to missing `nvrtc.so`; the run completed successfully.
- The two CLIC images were selected for small size and fast verification, not for representativeness.
- Full benchmark reproduction still requires operating-point grids, frozen splits, repeated runs, and official result comparison.

## Next Step

Run G1 baseline reproduction on a fixed small validation subset, then freeze expected regression values in `baselines/expected_results.yaml`.

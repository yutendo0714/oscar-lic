# DCAE Lambda 0.013 Checkpoint Smoke

## Purpose

Resolve the previously open DCAE checkpoint item for G1 baseline foundation: download one official checkpoint, record SHA256/size/source, and verify that actual compress/decompress produces complete transmitted files whose byte lengths can be used for bpp accounting.

## Source and Checkpoint

- Repository: `external/dcae`
- Commit: `e2525a00467cbc326045674c7e5e0f1d9964604b`
- Remote: `https://github.com/CVL-UESTC/DCAE.git`
- Code license: MIT
- Checkpoint: `experiments/checkpoints/dcae/dcae_mse_0013.pth.tar`
- Source: official README Google Drive link for lambda `0.013` MSE
- Size: `1434279774` bytes
- SHA256: `aec3992a15648195e44d5b6bc3d971a19b272335086878cf1b7c6de5b411a4c1`
- Metadata after load: epoch `97`, loss `0.8895649313926697`

The checkpoint is a PyTorch pickle archive. `torch.load(..., weights_only=True)` under PyTorch `2.0.1+cu118` failed on `collections.Counter`, so the smoke used normal `torch.load` only after recording source URL, size, and SHA256. Checkpoint/data terms still require manual publication review.

## Compatibility Note

The pinned DCAE code calls an older `CompressionModel.__init__()` signature. The installed `.venv-mlic` environment uses `compressai==1.2.0b3`, whose constructor requires `entropy_bottleneck_channels`. `scripts/run_dcae_smoke.py` injects a narrow old-API shim before importing DCAE, leaving `external/dcae` unchanged.

## Command

```bash
.venv-mlic/bin/python scripts/run_dcae_smoke.py \
  --checkpoint experiments/checkpoints/dcae/dcae_mse_0013.pth.tar \
  --output-dir experiments/results/dcae_mse0013_smoke_2026_06_26 \
  --max-images 2 --cuda --allow-unsafe-torch-load
```

## Actual Rate Accounting

The wrapper writes the upstream `.bin` layout:

```text
uint16 height
uint16 width
uint32 y_stream_length
y_stream_bytes
uint32 z_stream_length
z_stream_bytes
```

Main rate is computed from complete `.bin` file bytes:

```text
bpp = 8 * complete_bin_file_bytes / (original_height * original_width)
```

## Smoke Result

| image | bytes | actual bpp | PSNR |
|---|---:|---:|---:|
| `data/toy/toy_00.png` | 1188 | 0.1546875 | 39.75058931417517 |
| `data/toy/toy_01.png` | 1132 | 0.14739583333333334 | 38.75589647928591 |

Average actual bpp: `0.15104166666666669`

Average PSNR: `39.25324289673054 dB`

## Conclusion

DCAE lambda `0.013` MSE is now downloaded, hash-pinned, and actual compress/decompress smoke-tested. It should remain a P1 secondary dictionary-entropy baseline until it is run on the same small CLIC validation subset and then, if stable, promoted to a fuller baseline curve.

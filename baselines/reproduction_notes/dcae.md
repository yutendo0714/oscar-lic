# DCAE Reproduction Note

- Repository: `CVL-UESTC/DCAE`
- Commit: `e2525a00467cbc326045674c7e5e0f1d9964604b`
- Official README describes OpenImages training and Kodak/CLIC/TESTIMAGES evaluation.
- Actual file compression/decompression entry: `compress_and_decompress.py`.

## Important cautions

- The README's historical clone command points to a different owner path than the inspected repository; use the registered URL and pin.
- Dependencies are listed without versions (`compressai`, `tensorboard`, `thop`, `timm`). Build and freeze a working lock rather than installing into another experiment environment.
- Verify repository license before copying code; README presence is not a license grant.
- Use one checkpoint only after recording its source and SHA256.
- Confirm that evaluation and actual stream scripts use the same model path and preprocessing.
- With the current `.venv-mlic` environment (`compressai==1.2.0b3`), upstream DCAE's `CompressionModel.__init__()` call is incompatible with the installed CompressAI API. `scripts/run_dcae_smoke.py` injects a narrow old-API compatibility shim before importing DCAE, leaving the pinned external repository unchanged.
- The lambda 0.013 MSE checkpoint is a PyTorch pickle archive. PyTorch 2.0.1 `weights_only=True` fails on `collections.Counter`; normal `torch.load` was used only after recording the official README source URL, file size, and SHA256.
- The actual-bitstream smoke path writes the upstream `.bin` layout (`uint16 H/W`, `uint32 y length`, y bytes, `uint32 z length`, z bytes) and computes bpp from the complete file length over original `H*W`.

## Resolved checkpoint and smoke

- Checkpoint: `experiments/checkpoints/dcae/dcae_mse_0013.pth.tar`
- Source: official README Google Drive link for lambda 0.013 MSE (`1kXfvxsljdN3EfXDGqzknFc2Ecsgf8qgS`)
- Size: `1434279774` bytes
- SHA256: `aec3992a15648195e44d5b6bc3d971a19b272335086878cf1b7c6de5b411a4c1`
- Checkpoint metadata: epoch `97`, loss `0.8895649313926697`
- Smoke command:

```bash
.venv-mlic/bin/python scripts/run_dcae_smoke.py \
  --checkpoint experiments/checkpoints/dcae/dcae_mse_0013.pth.tar \
  --output-dir experiments/results/dcae_mse0013_smoke_2026_06_26 \
  --max-images 2 --cuda --allow-unsafe-torch-load
```

- Smoke result: two toy images, avg actual bpp `0.15104166666666669`, avg PSNR `39.25324289673054 dB`.

DCAE is a secondary base and dictionary-prior ablation. Do not let it delay G1 on MLIC++.

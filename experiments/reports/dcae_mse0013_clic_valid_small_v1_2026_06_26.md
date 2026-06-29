# DCAE CLIC Small Actual-Bitstream Smoke

## Purpose

Run the resolved DCAE lambda `0.013` MSE checkpoint on the same nine-image CLIC small regression split used for MLIC++ and CompressAI wiring checks.

This is a G1/P1 baseline-foundation smoke, not a publication benchmark.

## Setup

- Split: `data/splits/clic_valid_mlicpp_small_v1.txt`
- Checkpoint: `experiments/checkpoints/dcae/dcae_mse_0013.pth.tar`
- Checkpoint SHA256: `aec3992a15648195e44d5b6bc3d971a19b272335086878cf1b7c6de5b411a4c1`
- DCAE repo commit: `e2525a00467cbc326045674c7e5e0f1d9964604b`
- Runtime: `.venv-mlic`, `torch==2.0.1+cu118`, `compressai==1.2.0b3`
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER

The run uses `scripts/run_dcae_smoke.py`, which keeps `external/dcae` unchanged and injects a narrow compatibility shim for the older `CompressionModel` constructor expected by DCAE.

## Commands

```bash
.venv-mlic/bin/python scripts/run_dcae_smoke.py \
  --checkpoint experiments/checkpoints/dcae/dcae_mse_0013.pth.tar \
  --output-dir experiments/results/dcae_mse0013_clic_valid_small_v1_2026_06_26 \
  --cuda --allow-unsafe-torch-load \
  --images data/raw/clic/professional/valid/schicka-307.png \
           data/raw/clic/professional/valid/todd-quackenbush-222.png \
           data/raw/clic/professional/valid/paul-itkin-46101.png \
           data/raw/clic/professional/valid/martin-wessely-211.png \
           data/raw/clic/mobile/valid/IMG_0318.png \
           data/raw/clic/professional/valid/mullermarc-184.png \
           data/raw/clic/professional/valid/thong-vo-428.png \
           data/raw/clic/professional/valid/daniel-robert-405.png \
           data/raw/clic/professional/valid/philipp-reiner-207.png
```

```bash
.venv-mlic/bin/python scripts/summarize_codec_results.py \
  --results experiments/results/dcae_mse0013_clic_valid_small_v1_2026_06_26/results.jsonl \
  --summary experiments/results/dcae_mse0013_clic_valid_small_v1_2026_06_26/summary.json \
  --output experiments/results/dcae_mse0013_clic_valid_small_v1_2026_06_26/aggregate_metrics.json \
  --recompute-ms-ssim
```

## Rate Accounting

Rate is computed from the complete upstream-compatible DCAE `.bin` stream:

```text
uint16 height
uint16 width
uint32 y_stream_length
y_stream_bytes
uint32 z_stream_length
z_stream_bytes
```

```text
bpp = 8 * complete_bin_file_bytes / (original_height * original_width)
```

Padding affects coding but the denominator remains original image pixels.

## Aggregate Results

| Method | Images | Mean actual bpp | Median actual bpp | Mean PSNR | Median PSNR | Mean MS-SSIM | Median MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| DCAE lambda 0.013 MSE | 9 | 0.5267282015 | 0.5031565454 | 33.5033526809 | 33.1402927361 | 0.9813084536 | 0.9803685546 |
| MLIC++ corrected lambda 0.0250 | 9 | 0.7335496046 | 0.7127703894 | 35.0448878656 | 34.7814535144 | 0.9871003959 | 0.9865429997 |
| CompressAI hyperprior q5 | 9 | 0.7739301317 | 0.7909870213 | 33.2730825742 | 32.7984011173 | 0.9845182498 | 0.9837424755 |

DCAE is a useful lower-rate secondary point on this small split: versus CompressAI q5 it spends `-0.2472` mean bpp and gets `+0.2303 dB` mean PSNR, but its mean MS-SSIM is lower by `-0.00321`. It remains below MLIC++ corrected lambda `0.0250` by `-1.5415 dB` and `-0.00579` MS-SSIM, while spending `-0.2068` mean bpp.

## Individual DCAE Results

| Image | Size | Actual bpp | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|
| `schicka-307.png` | 512x384 | 0.1386718750 | 37.7440044575 | 0.9818184972 |
| `todd-quackenbush-222.png` | 720x439 | 0.5458871172 | 31.9551809011 | 0.9758625031 |
| `paul-itkin-46101.png` | 960x604 | 0.7891280353 | 32.9925631779 | 0.9925808907 |
| `martin-wessely-211.png` | 960x640 | 0.9468750000 | 31.1613455872 | 0.9803685546 |
| `IMG_0318.png` | 996x756 | 0.4478442872 | 33.6394836707 | 0.9814712405 |
| `mullermarc-184.png` | 1158x772 | 0.2905133919 | 35.2585309121 | 0.9843305945 |
| `thong-vo-428.png` | 1264x790 | 0.5031565454 | 33.3086779525 | 0.9802792668 |
| `daniel-robert-405.png` | 1500x844 | 0.4622306477 | 33.1402927361 | 0.9800786376 |
| `philipp-reiner-207.png` | 1440x900 | 0.6162469136 | 32.3300947332 | 0.9749858975 |

## Hashes

- DCAE `results.jsonl`: `947ae6dbd8f2f59ef44a951368bc740176502b6eebdbabfe46c727f5502b611a`
- DCAE `summary.json`: `69d415ec842613d31f2d967c0185d7b4320756a2946741cba237b50ef02fb078`
- DCAE `aggregate_metrics.json`: `80209c29e777d230cf8425ca19480454aa63803310e88853bb09925d0e944027`
- Summarizer script: `389e4181101f3a8d139f930f27883176d982dedc01a415710f39d62ae9de115d`

## Conclusion

DCAE lambda `0.013` should be kept as a P1 secondary dictionary-entropy baseline point. It should not displace MLIC++ as the primary base codec for OSCAR-LIC. If DCAE becomes important for the paper, the next useful step is one higher-rate DCAE point to bracket the MLIC++/CompressAI q5 region.

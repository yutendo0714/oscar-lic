# Baseline Actual-Bitstream Regression: CLIC Small v1

**Date:** 2026-06-25 JST  
**Split:** `data/splits/clic_valid_mlicpp_small_v1.txt`  
**Purpose:** freeze a small actual-bitstream regression point for MLIC++ and CompressAI before OSCAR-LIC model changes.

## Split

The split contains 9 CLIC validation images with at most 1.3M pixels. It is not a publication benchmark; it is a fast regression set for checking codec wiring, stream accounting, and GPU execution.

## Methods

### MLIC++

- Checkpoint: `mlicpp_mse_0025_corrected`
- SHA256: `67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559`
- Script: `scripts/run_mlicpp_smoke.py`
- Result dir: `experiments/results/mlicpp_clic_valid_small_v1_2026_06_25`
- Rate: complete upstream MLIC++ stream file bytes

### CompressAI Hyperprior

- Model: `bmshj2018-hyperprior`
- Metric: `mse`
- Main regression quality: `5`
- Script: `scripts/run_compressai_smoke.py`
- Result dir: `experiments/results/compressai_bmshj2018_hyperprior_q5_clic_valid_small_v1_2026_06_25`
- Rate: complete OSCAR CompressAI smoke container bytes

Exploratory qualities `3` and `6` were also run. Quality 3 had severe failures on several larger images; quality 6 had one severe failure on `IMG_0318.png`. Direct `model.compress` to `model.decompress` reproduced those failures, so they are not caused by the lightweight serialization container. Quality 5 is the frozen CompressAI smoke regression point.

## Aggregate Results

| Method | Images | Mean actual bpp | Median actual bpp | Mean PSNR | Median PSNR | Mean MS-SSIM | Median MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLIC++ corrected lambda 0.0250 | 9 | 0.7335496046 | 0.7127703894 | 35.0448878656 | 34.7814535144 | 0.9871003959 | 0.9865429997 |
| CompressAI hyperprior q5 | 9 | 0.7739301317 | 0.7909870213 | 33.2730825742 | 32.7984011173 | 0.9845182498 | 0.9837424755 |

At similar actual bpp on this small regression split, MLIC++ is stronger by about `+1.77 dB` mean PSNR and `+0.00258` mean MS-SSIM. This is expected; the result is a wiring sanity check, not a paper claim.

## Individual MLIC++ Results

| Image | Size | Actual bpp | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|
| `schicka-307.png` | 512x384 | 0.2013346354 | 38.8614 | 0.986493 |
| `todd-quackenbush-222.png` | 720x439 | 0.8100227790 | 33.5919 | 0.983848 |
| `paul-itkin-46101.png` | 960x604 | 0.9966339660 | 34.7815 | 0.994543 |
| `martin-wessely-211.png` | 960x640 | 1.3160937500 | 33.1541 | 0.988046 |
| `IMG_0318.png` | 996x756 | 0.6227241731 | 35.1158 | 0.987273 |
| `mullermarc-184.png` | 1158x772 | 0.4115014750 | 36.6573 | 0.989488 |
| `thong-vo-428.png` | 1264x790 | 0.7127703894 | 34.8356 | 0.986543 |
| `daniel-robert-405.png` | 1500x844 | 0.6763728278 | 34.5988 | 0.985951 |
| `philipp-reiner-207.png` | 1440x900 | 0.8544943827 | 33.8077 | 0.981718 |

## Individual CompressAI Hyperprior q5 Results

| Image | Size | Actual bpp | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|
| `schicka-307.png` | 512x384 | 0.2948811849 | 37.3534 | 0.983742 |
| `todd-quackenbush-222.png` | 720x439 | 0.9140097456 | 32.4304 | 0.981568 |
| `paul-itkin-46101.png` | 960x604 | 0.9122865894 | 31.0114 | 0.993478 |
| `martin-wessely-211.png` | 960x640 | 1.2647786458 | 31.4558 | 0.985162 |
| `IMG_0318.png` | 996x756 | 0.7026054347 | 34.2030 | 0.985786 |
| `mullermarc-184.png` | 1158x772 | 0.4880134557 | 35.7117 | 0.988010 |
| `thong-vo-428.png` | 1264x790 | 0.7909870213 | 32.7984 | 0.982680 |
| `daniel-robert-405.png` | 1500x844 | 0.7686161137 | 33.2782 | 0.983544 |
| `philipp-reiner-207.png` | 1440x900 | 0.8291929945 | 31.2156 | 0.976692 |

## Regression Hashes

- MLIC++ `results.jsonl`: `8ebe3c6d16f57ae81123bc36c047b86e25cab2668237a3c923b91e1f7554d23c`
- MLIC++ `summary.json`: `1db6d3cfd0e32e2d6e1e45bf5cc7d9a1a1c0a630831796770939b9c120dfe345`
- CompressAI q5 `results.jsonl`: `d7277e417dd16e26ecf76e3bc7ce90b7d62d6fc9d47bb47df5924ba427de366f`
- CompressAI q5 `summary.json`: `104225cb2232d403c847474ced72fa327fd29d89b455c8c21f38bce0a6247f6e`

## Interpretation

This completes the first small version of `B001` and `B010` as actual-bitstream smoke/regression checks. The next step is not to train OSCAR-LIC yet; it is to lock OCR evaluation and create text-rich crops so that the utility oracle can be tested against detector-mask and random controls.

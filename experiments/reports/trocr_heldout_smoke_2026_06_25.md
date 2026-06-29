# TrOCR Held-Out Smoke

**Date:** 2026-06-25 JST  
**Evaluator:** `microsoft/trocr-small-printed`  
**Resolved revision:** `04e994ab854b0089d4929f48c2b4dbe2ce78a340`  
**Role:** second held-out OCR family for smoke diagnostics, not used for training gradients or oracle labels unless explicitly running a multi-teacher diagnostic.

## Model Lock

| File | Size | SHA256 |
|---|---:|---|
| `model.safetensors` | 245839136 | `49350a39968df83e5a1adc90fc0ede02ff247671aed70b842af350fd4a7103f3` |
| `config.json` | 4211 | `2b6e06edec319984eca99c932d940bfa1f308db3f74d6279440b906c373bf727` |
| `preprocessor_config.json` | 272 | `eec0c686d2e2ea6d5887f87e1c6a34ef5bf1ba5b2ac98b8675e131da646fa1d2` |
| `sentencepiece.bpe.model` | 1356293 | `6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4` |

## Smoke Results

TrOCR-small-printed is weak under `unicode_strict_v1` on these scene-text crops, mainly due to formatting/casing behavior. It is still useful as an independent OCR family under `latin_alnum_ci_v1`, where it responds monotonically to most quality changes.

| Input | wandb | Unicode CER | Unicode exact | Latin CER | Latin exact |
|---|---|---:|---:|---:|---:|
| original | `5cnfl35p` | 0.8095238095 | 0.1250000000 | 0.1556886228 | 0.6666666667 |
| CompressAI q1 | `xwm9gzua` | 0.8214285714 | 0.0833333333 | 0.3772455090 | 0.4166666667 |
| CompressAI q2 | `lso4o14d` | 0.8392857143 | 0.0833333333 | 0.2574850299 | 0.5000000000 |
| CompressAI q3 | `5rryl7ya` | 0.8214285714 | 0.1250000000 | 0.2275449102 | 0.4583333333 |
| CompressAI q4 | `lgswkxg3` | 0.7857142857 | 0.1250000000 | 0.1976047904 | 0.5833333333 |
| CompressAI q5 | `06l69sx1` | 0.7857142857 | 0.1250000000 | 0.1616766467 | 0.5416666667 |
| CompressAI q6 | `1ljbg5ks` | 0.7857142857 | 0.1250000000 | 0.1317365269 | 0.6666666667 |
| MLIC++ corrected | `4xm47o5u` | 0.7976190476 | 0.1250000000 | 0.1736526946 | 0.5833333333 |

## Interpretation

TrOCR-small-printed should not be the main held-out metric for this crop split, but it is a useful third recognizer family. It confirms that low-rate q1/q2 reconstructions damage recognizer behavior, and that high-rate q6 largely recovers original Latin-normalized performance. Future stronger held-out work should still try PaddleOCR or a scene-text-specific transformer recognizer.

## Frozen Artifacts

- Original summary SHA256: `d546bfb14ad6952d0af14dd6aff6da9d780d75c5afbae9766ab6f6d48a991e6f`
- Original results SHA256: `a675c77db5fe3d3e9f34a6e30e4096bcdc4cc10de4f73b46eba46e8c00075ae9`
- q5 summary SHA256: `aa6d381d6c9b7621cca6b66db2b05079c56acf3945b676f42a0793d7ac7bf4b4`
- q6 summary SHA256: `c52c981b569cecad5e6add38c2ecb2a7961253188acbb64fa2cabbe17e5656fc`
- MLIC++ summary SHA256: `d21ecc49aea5fddc65c7bf1d673a008f43fac3d38f68d0d0a72ee2da19bd6cb6`
- Script: `scripts/run_trocr_ocr_smoke.py`

# Eval100 Quality Oracle And Controls

**Date:** 2026-06-25 JST  
**Split:** `data/splits/text_crops_eval100_v1.jsonl`  
**Source manifest:** `/dpl/veritext_crops/veritext_text_fidelity_eval_balanced_v1/manifest.jsonl`  
**Seed:** `20260625`  
**Rate:** complete serialized codec bytes; oracle/control bpp is corpus-level and includes simulated selection overhead.

## Split

The split contains 100 crops balanced over six sources:

| Source | Samples |
|---|---:|
| iam_words | 17 |
| icdar2013 | 17 |
| icdar2015 | 17 |
| icdar_mlt2019_words | 17 |
| synthtext_words | 16 |
| textocr | 16 |

Split SHA256: `a464a0bc6ec15551b389373291576f93b75d9ebaf282135dc7c4c0f7baf9ab73`  
Meta SHA256: `4225da24307615dcdc4e18900716b5f338cadfda8e44b9840495e70dd116b827`

## Original OCR Upper Bounds

| Evaluator | wandb | Unicode CER | Unicode exact | Latin CER | Latin exact |
|---|---|---:|---:|---:|---:|
| PARSeq | `7bj8lqzh` | 0.0696798493 | 0.8300000000 | 0.0410958904 | 0.8700000000 |
| Tesseract PSM8 | `cmjevapi` | 0.3992467043 | 0.3600000000 | 0.3170254403 | 0.4700000000 |
| TrOCR-small-printed | `va3hzyi0` | 0.8512241055 | 0.1900000000 | 0.4109589041 | 0.5200000000 |

Tesseract and TrOCR are weak on this mixed split, but they still provide useful disagreement signals. PARSeq remains the strongest teacher-style recognizer.

## CompressAI q1-q6 Ladder

| q | Mean actual bpp | PSNR | PARSeq CER / exact | Tesseract CER / exact |
|---:|---:|---:|---:|---:|
| 1 | 0.7370173500 | 25.0947 | 0.2128060264 / 0.60 | 0.4783427495 / 0.28 |
| 2 | 0.8892914274 | 27.0259 | 0.1337099812 / 0.74 | 0.4406779661 / 0.32 |
| 3 | 1.0730491780 | 28.9260 | 0.0960451977 / 0.80 | 0.4199623352 / 0.34 |
| 4 | 1.3063046809 | 31.1193 | 0.0847457627 / 0.79 | 0.4086629002 / 0.37 |
| 5 | 1.6103200440 | 32.7498 | 0.0809792844 / 0.83 | 0.3935969868 / 0.36 |
| 6 | 2.0032350619 | 34.8909 | 0.0659133710 / 0.82 | 0.4067796610 / 0.38 |

Tesseract is nonmonotonic, which reinforces the need for multi-teacher and strata-aware utility rather than pure distortion.

## Overhead-Aware Multi-Teacher Oracle

Objective: PARSeq edit distance + Tesseract edit distance, `unicode_strict_v1`.  
Overhead: 40 bytes per selected candidate plus one 40-byte global gate cost.  
Uniform baselines are ordinary full-quality codec streams and do not pay selection overhead.

| Selection | Corpus bpp | Overhead bytes | Quality counts | Weighted loss | PARSeq CER | Tesseract CER |
|---|---:|---:|---|---:|---:|---:|
| uniform q2 | 0.807866 | 0 | `{2: 100}` | 305 | 0.1337 | 0.4407 |
| oracle to q2 bytes | 0.807034 | 1880 | `{1: 54, 2: 27, 3: 15, 4: 3, 6: 1}` | 236 | 0.0772 | 0.3672 |
| uniform q3 | 0.983484 | 0 | `{3: 100}` | 274 | 0.0960 | 0.4200 |
| oracle to q3 bytes | 0.981820 | 2280 | `{1: 44, 2: 19, 3: 15, 4: 8, 5: 7, 6: 7}` | 202 | 0.0565 | 0.3239 |
| uniform q4 | 1.217087 | 0 | `{4: 100}` | 262 | 0.0847 | 0.4087 |
| oracle to q4 bytes | 1.015566 | 2360 | `{1: 42, 2: 19, 3: 15, 4: 8, 5: 8, 6: 8}` | 200 | 0.0565 | 0.3202 |
| uniform q6 | 1.891991 | 0 | `{6: 100}` | 251 | 0.0659 | 0.4068 |
| oracle to q6 bytes | 1.015566 | 2360 | `{1: 42, 2: 19, 3: 15, 4: 8, 5: 8, 6: 8}` | 200 | 0.0565 | 0.3202 |

The oracle exhausts positive multi-teacher improvements around 1.016 corpus bpp. Past that point, spending more bits in this quality ladder does not help the two-recognizer objective. This is exactly the kind of signal the learned OSCAR allocator should imitate: spend sharply where utility exists, then stop.

## Controls

At q3 budget, all non-oracle controls lose to the oracle:

| Policy | Corpus bpp | PARSeq CER | Tesseract CER |
|---|---:|---:|---:|
| oracle to q3 bytes | 0.981820 | 0.0565 | 0.3239 |
| random | 0.982414 | 0.1055 | 0.4350 |
| low-confidence | 0.983484 | 0.1356 | 0.4595 |
| high-confidence | 0.982771 | 0.1149 | 0.4501 |
| uniform-cost | 0.982890 | 0.1168 | 0.4407 |

The result is a useful negative control: simple confidence or random allocation does not explain the gain.

## Paired Bootstrap

`scripts/bootstrap_selection_comparison.py` compares overhead-aware oracle q3 selection against uniform q3 with 1000 paired bootstrap resamples.

| Evaluator | Delta CER oracle - uniform | 95% CI | Delta exact | 95% CI |
|---|---:|---:|---:|---:|
| PARSeq | -0.0395480226 | [-0.0703434846, -0.0135123760] | +0.0400000000 | [0.0000000000, 0.0900000000] |
| Tesseract | -0.0960451977 | [-0.1308631808, -0.0623677608] | +0.0700000000 | [0.0200000000, 0.1200000000] |

The CER improvements are positive under paired resampling for both recognizers. The bpp delta is tiny and centered near zero because the q3 oracle was constrained to matched q3 bytes.

## Source Strata

`scripts/analyze_selection_strata.py` compares overhead-aware oracle q3 selection against uniform q3 by source dataset.

| Source | n | PARSeq uniform CER | PARSeq oracle CER | PARSeq delta | Tesseract uniform CER | Tesseract oracle CER | Tesseract delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| iam_words | 17 | 0.1071 | 0.0595 | -0.0476 | 0.8452 | 0.5357 | -0.3095 |
| icdar2013 | 17 | 0.0000 | 0.0000 | 0.0000 | 0.1386 | 0.0693 | -0.0693 |
| icdar2015 | 17 | 0.1190 | 0.0714 | -0.0476 | 0.5595 | 0.5000 | -0.0595 |
| icdar_mlt2019_words | 17 | 0.0808 | 0.0606 | -0.0202 | 0.2525 | 0.2222 | -0.0303 |
| synthtext_words | 16 | 0.2785 | 0.1392 | -0.1392 | 0.4177 | 0.3671 | -0.0506 |
| textocr | 16 | 0.0238 | 0.0238 | 0.0000 | 0.3929 | 0.3214 | -0.0714 |

The oracle improves Tesseract CER in every source stratum and does not increase PARSeq CER in any stratum. PARSeq exact rate drops by one sample in `icdar2015`, so exact-match behavior still needs per-word inspection, but the source-level CER evidence is not confined to a single synthetic domain.

## Frozen Artifacts

- Oracle summary: `experiments/results/greedy_quality_oracle_multiteacher_equal_overhead40_text_crops_eval100_v1_2026_06_25/summary.jsonl`
- Oracle summary SHA256: `77b69be20c068c5e08016ef169f883be190f2bb12d7a49ba258e6c132f3b56c3`
- Controls summary: `experiments/results/quality_policy_controls_overhead40_text_crops_eval100_v1_2026_06_25/summary.jsonl`
- Controls summary SHA256: `9e66c55168ee7e6a472b9876ec36f6cd97ca7d8e616e4882ca5107b082b76aef`
- Bootstrap q3 oracle-vs-uniform summary SHA256: `2253eb606ce9cba364e1b0fa00b74e755802d0a8d5b1893da45a70d4fac5fa57`
- Source-strata q3 oracle-vs-uniform summary SHA256: `591fd25b9cf8caccb581d052ab133e54aeb5a63a356a3b536700117aee9af93b`
- PARSeq original summary SHA256: `4ebe43b2d32862371dbf804ac33db556eaa0903eb33f3b3a58233822234d05f2`
- Tesseract original summary SHA256: `deb6ac75382699fcdf46a26cf12768cccbcffbeecb2e08ecc639671b68034fe3`

## Decision

G4 is not fully complete because true MLIC++ latent/enhancement candidates are still missing, but the larger quality-ladder evidence strongly supports continuing to a frozen-base OSCAR enhancement prototype. The next implementation should learn or approximate the multi-teacher utility ranking and include gate/packet overhead in training-time budget simulation.

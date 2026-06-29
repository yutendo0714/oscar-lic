# Eval600/Eval300 Candidate Utility Strata

Date: 2026-06-25

## Purpose

The Eval600->Eval300 external patch-gate transfer preserved PARSeq but recovered little Tesseract oracle headroom. This diagnostic compares the Eval600 and Eval300 one-candidate utility tables by source and OCR-difficulty strata to check whether the failure is mainly distribution shift, lack of headroom, or unsafe candidate density.

This is a candidate-table analysis, not a compression result.

## Overall Comparison

| table | rows | images | safe Tesseract/PARSeq image rate | multi-teacher image rate | harmful image rate | safe candidate rows | harmful candidate rows | safe delta sum | mean Tesseract base distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Eval600 | 9600 | 600 | 0.205 | 0.222 | 0.242 | 799 | 741 | -180 | 2.127 |
| Eval300 | 4760 | 300 | 0.233 | 0.267 | 0.257 | 392 | 395 | -98 | 2.160 |

The two tables are similar enough that simple distribution shift is unlikely to be the only reason the patch gate transfers weakly. Eval300 actually has slightly higher safe-image rate and similar harmful-image rate.

## Source Strata

| table | source | images | safe image rate | harmful image rate | safe delta sum | mean Tesseract base distance |
|---|---|---:|---:|---:|---:|---:|
| Eval600 | iam_words | 100 | 0.45 | 0.43 | -61 | 3.84 |
| Eval600 | icdar2013 | 100 | 0.12 | 0.14 | -20 | 0.75 |
| Eval600 | icdar2015 | 100 | 0.20 | 0.19 | -33 | 2.36 |
| Eval600 | icdar_mlt2019_words | 100 | 0.09 | 0.15 | -11 | 1.19 |
| Eval600 | synthtext_words | 100 | 0.19 | 0.30 | -29 | 2.14 |
| Eval600 | textocr | 100 | 0.18 | 0.24 | -26 | 2.48 |
| Eval300 | iam_words | 50 | 0.44 | 0.38 | -34 | 3.54 |
| Eval300 | icdar2013 | 50 | 0.16 | 0.22 | -10 | 1.14 |
| Eval300 | icdar2015 | 50 | 0.20 | 0.20 | -16 | 2.88 |
| Eval300 | icdar_mlt2019_words | 50 | 0.22 | 0.22 | -12 | 1.44 |
| Eval300 | synthtext_words | 50 | 0.16 | 0.34 | -11 | 1.84 |
| Eval300 | textocr | 50 | 0.22 | 0.18 | -15 | 2.12 |

IAM has the most headroom but also high harm density. ICDAR2013 has little headroom because base Tesseract is already strong. This suggests a safe selector needs a no-headroom/hard-positive distinction, not just a generic source prior.

## Base Tesseract Difficulty

| table | base Tesseract bucket | images | safe image rate | harmful image rate | safe delta sum |
|---|---|---:|---:|---:|---:|
| Eval600 | 0 exact | 233 | 0.000 | 0.073 | 0 |
| Eval600 | 1 char | 67 | 0.149 | 0.269 | -10 |
| Eval600 | 2-3 chars | 146 | 0.274 | 0.342 | -48 |
| Eval600 | 4+ chars | 154 | 0.474 | 0.390 | -122 |
| Eval300 | 0 exact | 101 | 0.000 | 0.149 | 0 |
| Eval300 | 1 char | 39 | 0.205 | 0.179 | -8 |
| Eval300 | 2-3 chars | 87 | 0.299 | 0.333 | -31 |
| Eval300 | 4+ chars | 73 | 0.493 | 0.356 | -59 |

The strongest pattern is no-headroom: when base Tesseract is exact, no candidate gives a Tesseract-improving/PARSeq-safe gain, but harmful candidates still exist. This is not directly deployable as a feature because Tesseract is held out, but it identifies the mechanism the selector must approximate.

## Base PARSeq Difficulty

PARSeq base distance is a weaker proxy. In Eval300, PARSeq-exact images still have `0.232` safe-image rate because Tesseract often has room to improve even when PARSeq is already correct. Therefore, a PARSeq-only no-headroom rejector would discard too much useful Tesseract headroom.

## Interpretation

The external patch-gate failure is not mainly because Eval600 and Eval300 have radically different source/headroom distributions. The core issue is conditional calibration: high-headroom and no-headroom groups are interleaved, and the current features do not reliably separate them without access to the held-out Tesseract state.

This points to two next directions:

- learn a deployable no-headroom proxy from image/base-reconstruction features and optionally teacher PARSeq confidence as an auxiliary signal, but do not use held-out Tesseract at inference;
- return to the top8 compact assignment path and train explicit false-change/no-headroom suppression with more source-disjoint evidence.

## Artifacts

- Result JSON SHA256: `924dff17966e9681a17836155e4b428bb893ea20a3f98e2ef8fd9fc716f930c1`
- Analyzer SHA256: `41e6a705bde319cf72e94a7ec3f8c6f61f4a523407f77735b9775271d1bb3d2f`
- W&B offline run: `wandb/offline-run-20260625_224406-3u0m7sba`

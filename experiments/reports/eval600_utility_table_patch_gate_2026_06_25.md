# Eval600 Utility Table and Patch-Gate Diagnostic

Date: 2026-06-25

## Purpose

The Eval300 top8 assignment models repeatedly failed because high row AUC did not translate into high-precision policy choices. The source-disjoint diagnostic suggested that the label set is too small and correlated. This run uses the new Eval300-disjoint Eval600 candidate/OCR substrate to build utility labels and test whether the existing patch-listwise gate improves when more groups are available.

This is a diagnostic, not a promoted OSCAR-LIC result. Policy bpp is computed from measured single-candidate `.oscr` rows, but no combined counted promotion stream was generated.

## Base and Utility Labels

MLIC++ base-only `.oscr` reconstructions were generated for all 600 Eval600 crops.

- Base rows: 600
- Mean actual base bpp: `3.7149301854`
- Mean PSNR: `34.90844663`
- PARSeq base unicode CER: `0.0773263434`
- Tesseract PSM8 base unicode CER: `0.4180865007`
- PARSeq base W&B: `w5yp0186`
- Tesseract base W&B: `5iff5liz`

Single-candidate utilities were computed from candidate OCR minus base OCR under `unicode_strict_v1`.

| evaluator | improved candidates | worsened candidates | same candidates | images with any improving candidate | best oracle CER |
|---|---:|---:|---:|---:|---:|
| PARSeq | 22 | 24 | 9554 | 12 | 0.0711009174 |
| Tesseract PSM8 | 801 | 719 | 8080 | 124 | 0.3591087811 |

The merged utility table has 9600 rows:

- `any_improves`: 818
- `pareto_improves`: 813
- `multi_teacher_improves`: 815
- `worsens_any`: 741

This confirms that Eval600 has useful candidate signal, especially for Tesseract, but also nearly as many harmful candidates as useful ones.

## Patch-Listwise Gate

Configuration:

- Table: `candidate_utility_table_eval600_noeval300_tile4_scale025_q005_with_image_features_2026_06_25.jsonl`
- Target: `tesseract_parseq_safe`
- Model: existing patch-listwise gate, hidden 32, dropout 0.2, 400 epochs
- Seeds: 0, 1, 2
- Split: internal group split, 25% validation per seed
- W&B aggregate run: `x4brl6zw`

Aggregate validation policies:

| policy | selected images | actual bpp | multi-teacher delta | PARSeq CER | Tesseract CER |
|---|---:|---:|---:|---:|---:|
| base | 0.00 | 3.793716 | 0.00 | 0.056712 | 0.416168 |
| oracle single candidate | 32.33 | 4.335617 | -44.00 | 0.053156 | 0.360474 |
| listwise train-tuned | 24.67 | 4.155729 | -1.67 | 0.056712 | 0.413933 |
| constrained train-tuned | 24.67 | 4.155729 | -1.67 | 0.056712 | 0.413933 |
| threshold 0.5 | 5.33 | 3.835567 | +0.33 | 0.056712 | 0.416629 |
| listwise top20 | 20.00 | 4.047576 | -1.00 | 0.056712 | 0.414809 |
| residual RMS top20 | 20.00 | 4.154893 | +3.00 | 0.057149 | 0.419732 |

## Interpretation

Eval600 helps enough that the learned patch gate can avoid the residual-RMS failure mode and preserve PARSeq, but it still recovers only a tiny fraction of the Tesseract oracle. The train-tuned/constrained gate improves Tesseract by about `0.0022` CER at a mean bpp increase of about `0.36`, while the oracle improves by about `0.0557` CER.

The result is therefore a valid weak-positive diagnostic but not a promotion. The next useful direction is not another small seed sweep of this same gate. The useful next changes are:

- train/evaluate with a fixed external split, for example train on Eval600 and evaluate on Eval300, rather than random internal splits;
- add a high-precision no-headroom rejector or source/difficulty-aware calibration;
- move from one-candidate policy simulation toward top8 assignment or compact-symbol training where the payload matches the promoted OSCAR path.

## Artifacts

- Utility table SHA256: `0ae51a2732aebcdcd933aaefcbd7ae8884bb0248e0c8fd5d589cd13a1f37f784`
- Image-feature table SHA256: `b4478e97b4ee949fa43b17d36ebff339d48de467861e908b2a4ff9f28607a021`
- Patch-gate result SHA256: `f0ffc9e603e00a1126b24bd91a0e7de1d9433c3c30c950c5d6a503c01a75e775`


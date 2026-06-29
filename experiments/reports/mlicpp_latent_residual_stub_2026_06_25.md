# MLIC++ Latent Residual Enhancement Stub

**Date:** 2026-06-25  
**Checkpoint:** `experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar`  
**Checkpoint SHA256:** `67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559`  
**Split:** `data/splits/text_crops_parseq_smoke_v1.jsonl`  

## Purpose

Move beyond dummy `TEXT_MAIN` bytes and verify a true counted frozen-base enhancement path:

1. encode corrected MLIC++ as `BASE_MAIN`;
2. decode base to reconstructed `y_hat`;
3. compute encoder-side latent residual `g_a(x) - y_hat`;
4. select latent candidates by residual magnitude;
5. send counted `TEXT_GATE` and quantized `TEXT_MAIN`;
6. apply residual to decoded `y_hat` and reconstruct through frozen `g_s`.

This is not the proposed learned model. It is a deliberately simple diagnostic stub to exercise the actual payload, decoder path and OCR loop.

## Implementation

`scripts/run_mlicpp_latent_residual_stub.py` writes a complete `.oscr` stream with:

- `BASE_MAIN`: upstream MLIC++ body, codec `1001`;
- `TEXT_GATE`: `OSCAR_GATE_V0`, codec `1`;
- `TEXT_MAIN`: `OSCAR_LATENT_RESIDUAL_V0`, codec `2`.

Residual payload format:

- header: `uint16 version`, `float32 quant_step`;
- body: selected candidate tensors serialized as signed int8 residuals;
- decoder-known candidate geometry determines tensor lengths.

## Codec Results

All rows use tile `4x4`, no channel subdivision, quant step `0.05`, residual scale `0.25`.

| Selection | Actual total bpp | Base payload bpp | Enhancement payload bpp | Avg selected | Gate bytes | Residual bytes | Base PSNR | Enhanced PSNR | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 4.804000 | 2.713753 | 1.611712 | 1.125 | 16.25 | 582.00 | 35.2251 | 35.2656 | +0.0405 |
| 10% | 5.890206 | 2.713753 | 2.697918 | 2.000 | 17.00 | 1030.00 | 35.2251 | 35.2913 | +0.0662 |

For reference, residual scale `1.0` degraded PSNR by `-0.8312 dB`, and scale `0.5` degraded PSNR by `-0.0333 dB`. The latent residual is therefore sensitive to scale; directly adding `g_a(x)-y_hat` is too aggressive.

## Codebook Diagnostic

`scripts/run_mlicpp_latent_codebook_stub.py` replaces raw int8 tensors with decoder-known codebook indices. This diagnostic trains a 16-entry residual codebook on the same Text24 split, so it is not a valid generalization result. The goal is only to check whether compact selected-candidate payloads can travel through the same counted section path.

| Method | Actual total bpp | Enhancement payload bpp | Avg selected | Gate bytes | Main bytes | Base PSNR | Enhanced PSNR | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw residual 5% | 4.804000 | 1.611712 | 1.125 | 16.25 | 582.00 | 35.2251 | 35.2656 | +0.0405 |
| Codebook K16 5% | 3.258330 | 0.066043 | 1.125 | 16.25 | 7.13 | 35.2251 | 35.2357 | +0.0106 |

The codebook path cuts enhancement payload sharply, but its current residual-magnitude selection and unsupervised codebook do not improve OCR beyond the MLIC++ base. This is the cleanest direction for the next prototype, but the codebook must be trained for OCR utility, not residual MSE alone.

## Single-Candidate Utility Labels

`scripts/generate_mlicpp_single_candidate_recons.py` generated one reconstruction for every Text24 latent candidate using the same raw residual scale/quantization. This produced 480 candidate reconstructions (3.6 MB) and enables true candidate-level OCR utility labels.

| Evaluator | Candidate rows | Improved | Same | Worsened | Images with any improving candidate | Base CER/exact | Best single-candidate oracle CER/exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| PARSeq | 480 | 0 | 480 | 0 | 0/24 | 0.0774 / 0.7083 | 0.0774 / 0.7083 |
| Tesseract | 480 | 48 | 383 | 49 | 9/24 | 0.3690 / 0.1667 | 0.2917 / 0.2917 |

This is the first true latent-candidate utility signal. PARSeq is insensitive to these small single-candidate residuals, while Tesseract has both helpful and harmful candidates. The best Tesseract candidates reduce corpus CER substantially, but they are evaluator-specific. A useful learned allocator must therefore use multi-teacher labels or a robust objective, rather than optimizing only one OCR family.

The candidate analyses were merged into `experiments/results/candidate_utility_table_text24_tile4_scale025_q005_2026_06_25.jsonl`, a supervised table with geometry, actual bytes, PARSeq/Tesseract deltas and multi-teacher labels. It contains 480 rows: 48 multi-teacher-improving candidates, 49 candidates that worsen at least one evaluator, and 383 neutral candidates. Slice alone is weakly predictive; the best positive rates are only 14.6% in slices 4 and 8. Source/domain is more informative, with `icdar2013` showing an 18.1% positive rate versus 5.0% for `icdar2015`. The learned selector therefore needs image/latent content features, not just slice or tile coordinates.

`scripts/augment_candidate_table_with_mlicpp_features.py` adds decoded `y_hat`, encoder `y_target`, and latent residual statistics to the supervised table. Single-feature diagnostics remain weak: latent residual RMS has AUC 0.5767 for multi-teacher improvement, while enhanced-image PSNR has AUC 0.3790. Top-48 by residual RMS contains 8 improving and 4 worsening candidates. Residual magnitude is therefore a useful weak prior, but not a reliable selector.

`scripts/train_candidate_utility_baseline.py` trains a tiny linear selector with an image-group validation split. It overfits the small table: train AUC is 0.7564, but validation AUC is 0.0552. A residual-RMS prior on the same validation split is stronger, with AUC 0.6337 and top-20 precision 0.25. The current feature table is useful for plumbing, but it is too small and too domain-dependent for a learned selector. The next table should be generated on Eval100/full-image crops and include richer content features.

## Eval100 Candidate Utility and Selector Diagnostics

The same single-candidate latent residual labeling pipeline was expanded to `data/splits/text_crops_eval100_v1.jsonl` with tile `4x4`, residual scale `0.25`, quant step `0.05`, and the top residual-magnitude candidates per image. This produced 1750 candidate reconstructions over 100 images.

| Evaluator | Candidate rows | Improved | Same | Worsened | Images with any improving candidate | Base CER/exact | Best single-candidate oracle CER/exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| PARSeq | 1750 | 10 | 1738 | 2 | 6/100 | 0.0791 / 0.8000 | 0.0640 / 0.8500 |
| Tesseract | 1750 | 127 | 1489 | 134 | 20/100 | 0.4030 / 0.3600 | 0.3427 / 0.3800 |

The merged Eval100 utility table contains 1750 rows: 137 candidates improve at least one evaluator, 135 are Pareto-improving, 136 improve the summed multi-teacher edit distance, and 136 worsen at least one evaluator. This is a much healthier label density than Text24 and confirms that true latent candidate utility exists beyond the small smoke split.

Adding the same MLIC++ latent statistics gives a stronger but still incomplete selector signal. On one image-group split, a tiny linear selector reaches validation AUC 0.7085 and AP 0.1875 for multi-teacher improvement, compared with residual-RMS AUC 0.5220 and AP 0.0631. Top-20 candidate-row precision is 0.30 for the linear selector versus 0.15 for residual RMS.

However, candidate-row classification does not yet translate cleanly into a transmitted one-candidate policy. On the same validation split, the true single-candidate oracle selects 5/25 images and improves PARSeq CER from 0.0687 to 0.0534 and Tesseract CER from 0.3740 to 0.3511 at 4.3443 actual bpp versus 3.6426 base bpp. A train-tuned linear threshold selects only 1/25 images and leaves both OCR metrics unchanged. Linear top-10 image selection also leaves OCR unchanged, while residual-RMS top-20 improves PARSeq but costs 5.5669 bpp and leaves Tesseract unchanged.

Twenty random image-group splits make the diagnosis stable:

| Policy | Selected images | Actual bpp | Multi-teacher delta | PARSeq CER/exact | Tesseract CER/exact |
|---|---:|---:|---:|---:|---:|
| Base | 0.00 | 3.4257 | 0.00 | 0.0846 / 0.7960 | 0.4111 / 0.3640 |
| Oracle single candidate | 6.75 | 4.0503 | -11.25 | 0.0658 / 0.8560 | 0.3444 / 0.3820 |
| Linear train-tuned | 9.00 | 4.2029 | -1.80 | 0.0846 / 0.7960 | 0.3980 / 0.3520 |
| Linear top-10 images | 10.00 | 4.2603 | -1.65 | 0.0846 / 0.7960 | 0.3991 / 0.3520 |
| Residual-RMS top-20 images | 20.00 | 5.2967 | -2.50 | 0.0801 / 0.8200 | 0.3965 / 0.3560 |

Interpretation: the residual candidate generator has headroom, but the current selector is not yet ranking the best candidate within each image. The next model should add a utility-aware gate/head trained on candidate counterfactual labels, richer content features, and a compact residual/codebook representation. Simply sending more high-residual latent patches is rate-inefficient and often not OCR-aligned.

`scripts/train_candidate_ranker_baseline.py` adds per-image normalized features and a small MLP with a pairwise within-image ranking loss. This is still a diagnostic baseline, not a final model. Across the same 20 splits, train AUC is high (mean 0.9777) while validation AUC is modest (mean 0.5921, AP 0.1214), so the model is over-capable for the available feature table. Still, actual one-candidate selection improves over the linear selector and residual-RMS at low budgets:

| Policy | Selected images | Actual bpp | Multi-teacher delta | PARSeq CER/exact | Tesseract CER/exact |
|---|---:|---:|---:|---:|---:|
| Ranker top-5 images | 5.00 | 3.7721 | -1.40 | 0.0846 / 0.7960 | 0.4005 / 0.3620 |
| Ranker top-10 images | 10.00 | 4.1919 | -2.00 | 0.0846 / 0.7960 | 0.3960 / 0.3660 |
| Ranker top-20 images | 20.00 | 5.1344 | -2.55 | 0.0802 / 0.8060 | 0.3959 / 0.3560 |
| Ranker train-tuned threshold | 11.85 | 4.3624 | -2.20 | 0.0842 / 0.7980 | 0.3948 / 0.3640 |

The ranker is the first non-oracle selector that consistently reduces Tesseract CER in the actual-bpp policy evaluation, but it captures only a small part of the oracle. This narrows the next research step: improve feature extraction and ranking supervision before spending effort on a heavier enhancement payload.

`scripts/augment_candidate_table_with_image_patch_features.py` adds approximate pixel-space patch features by mapping latent tiles back with a 16x scale and comparing the original crop against the MLIC++ base reconstruction. Features include candidate pixel location/area, original/base patch intensity and gradient statistics, local base reconstruction error and full-image base error. With these 180 ranker features, validation AUC improves from 0.5921 to 0.6709 and AP from 0.1214 to 0.1859, but the actual-bpp policy gain remains modest:

| Policy | Selected images | Actual bpp | Multi-teacher delta | PARSeq CER/exact | Tesseract CER/exact |
|---|---:|---:|---:|---:|---:|
| Image-feature ranker top-5 | 5.00 | 3.8023 | -1.90 | 0.0850 / 0.7960 | 0.3968 / 0.3720 |
| Image-feature ranker top-10 | 10.00 | 4.2745 | -2.40 | 0.0842 / 0.8000 | 0.3940 / 0.3720 |
| Image-feature ranker top-20 | 20.00 | 5.2648 | -2.40 | 0.0827 / 0.8040 | 0.3954 / 0.3640 |

A smaller, more regularized image-feature ranker (`hidden=32`, dropout `0.2`, 400 epochs) gives similar actual policy behavior and slightly higher validation AUC 0.6765/AP 0.1773. The conclusion is stable: simple statistics help but are not enough. The next step should expose structured local features, such as latent tokens, shallow CNN patch embeddings, or OCR-aware teacher features, while keeping the transmitted payload compact.

`scripts/augment_candidate_table_with_latent_vector_features.py` adds per-channel latent vector features from `y_hat`, `g_a(x)`, and residual patches. This creates a much wider table (1780 ranker features). The regularized ranker still overfits almost perfectly on train and validation AUC drops to 0.6552/AP 0.1826. Actual one-candidate policies are also worse than the image-statistics ranker: top-10 Tesseract CER is 0.4038 and top-5 is 0.4060. This is a useful negative result: flattening many latent channel features into a small MLP is not the right path at Eval100 scale. The next selector should use a shared local encoder or dimensionality-reduced latent tokens, not raw wide tabular vectors.

`scripts/train_candidate_patch_ranker.py` implements that shared local encoder idea in a minimal form: it crops original/base/difference patches for each candidate, resizes them to `32x32`, processes them with a tiny CNN, and joins the patch embedding with the compact tabular features. The binary + pairwise-loss version reaches much better candidate validation statistics than the tabular rankers (20-split mean validation AUC 0.7470/AP 0.3127), but the actual transmitted-policy gains remain limited:

| Policy | Selected images | Actual bpp | Multi-teacher delta | PARSeq CER/exact | Tesseract CER/exact |
|---|---:|---:|---:|---:|---:|
| Patch ranker top-5 | 5.00 | 3.8398 | -1.80 | 0.0838 / 0.8000 | 0.3986 / 0.3600 |
| Patch ranker top-10 | 10.00 | 4.2821 | -2.20 | 0.0827 / 0.8020 | 0.3968 / 0.3600 |
| Patch ranker top-20 | 20.00 | 5.2344 | -2.20 | 0.0816 / 0.8080 | 0.3978 / 0.3580 |

Continuous `benefit` and `benefit_per_bpp` targets underperform the binary target (validation AUC about 0.599 and weaker policy metrics). The likely reason is label sparsity: most candidates are neutral, and the useful signal is first to identify reliable non-harmful improvements. For now, keep binary multi-teacher improvement plus pairwise ranking as the selector target, and improve data volume/teacher features before trying a more delicate benefit regression.

## Eval100 Codebook Payload Diagnostic

The decoder-known K16 latent codebook diagnostic was also run on Eval100 with tile `4x4`, selected fraction `5%`, and residual scale `0.25`. This is not a valid learned generalization result because the codebook is trained on the diagnostic split and treated as model state, but it is useful for payload accounting.

| Method | Actual bpp | Base payload bpp | Enhancement payload bpp | Avg selected | Gate bytes | Code bytes | Base PSNR | Enhanced PSNR | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Eval100 codebook K16 5% | 3.8495 | 3.1353 | 0.0865 | 1.12 | 16.24 | 7.12 | 35.0743 | 35.0510 | -0.0233 |

OCR results:

| Method | Total bpp | PARSeq CER / exact | Tesseract CER / exact |
|---|---:|---:|---:|
| MLIC++ base | 3.4641 | 0.0791 / 0.8000 | 0.4030 / 0.3600 |
| Codebook K16 5% | 3.8495 | 0.0791 / 0.8000 | 0.4030 / 0.3500 |

The compact payload path is healthy, but residual-MSE codebook symbols do not improve OCR. This reinforces the core design choice: the compact `TEXT_MAIN` representation should be trained for OCR/task utility and paired with a utility-aware gate, not learned only as a residual reconstruction codebook.

The same K16 codebook path was then run with the utility oracle gate: for each image, select the best candidate from the Eval100 multi-teacher utility table when its summed edit-distance delta is negative, otherwise omit the optional sections and transmit base-only. This selected 0.26 candidates per image on average and trains the diagnostic codebook only on the selected residual vectors.

| Method | Actual bpp | Enhancement payload bpp | Avg selected | PARSeq CER/exact | Tesseract CER/exact |
|---|---:|---:|---:|---:|---:|
| MLIC++ base | 3.4641 | 0.0000 | 0.00 | 0.0791 / 0.8000 | 0.4030 / 0.3600 |
| Residual-MSE K16 5% | 3.8495 | 0.0865 | 1.12 | 0.0791 / 0.8000 | 0.4030 / 0.3500 |
| Oracle-selected K16 | 3.5637 | 0.0223 | 0.26 | 0.0734 / 0.8200 | 0.3766 / 0.3600 |
| Oracle-selected K64 -> 26 centers | 3.5637 | 0.0223 | 0.26 | 0.0659 / 0.8400 | 0.3917 / 0.3600 |

This is the strongest post-G4 signal so far. It shows that a compact codebook-style `TEXT_MAIN` can preserve useful OCR improvements when the gate is utility-aware. The remaining gap to the raw single-candidate oracle is expected because the codebook is tiny and still trained by residual clustering, not by task loss. The next prototype should train the compact symbols and gate jointly or alternatingly against the multi-teacher objective.

Increasing the oracle-selected codebook capacity to 26 effective centers nearly reaches the PARSeq raw oracle, but it worsens Tesseract relative to K16. This is another sign that visual/residual fidelity alone is not the right target: compact symbols need a multi-teacher utility objective or a constraint that prevents one recognizer from consuming the gain at another recognizer's expense.

## Eval300 Scale-Up

The true single-candidate pipeline was scaled to `data/splits/text_crops_eval300_v1.jsonl`, a deterministic 300-crop split with 50 images from each of six sources. MLIC++ base was first evaluated through the counted OSCR adapter:

| Split | Images | Actual bpp | Base payload bpp | PSNR |
|---|---:|---:|---:|---:|
| Eval300 MLIC++ base | 300 | 3.6726 | 3.3033 | 34.9438 |

Base OCR:

| Method | PARSeq CER / exact | Tesseract CER / exact |
|---|---:|---:|
| Eval300 MLIC++ base | 0.0661 / 0.8467 | 0.4326 / 0.3367 |

The single-candidate latent residual sweep produced 4760 candidate reconstructions. The raw oracle remains strong:

| Evaluator | Candidate rows | Improved | Same | Worsened | Images with any improving candidate | Base CER/exact | Best single-candidate oracle CER/exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| PARSeq | 4760 | 18 | 4723 | 19 | 11/300 | 0.0661 / 0.8467 | 0.0561 / 0.8700 |
| Tesseract | 4760 | 396 | 3980 | 384 | 71/300 | 0.4326 / 0.3367 | 0.3665 / 0.3733 |

The merged Eval300 utility table has 409 multi-teacher-improving candidates and 395 candidates that worsen at least one evaluator. This is a useful scale-up: the improvement/worsening density remains balanced enough to train a selector, but the policy problem is hard because wrong candidates are almost as common as useful ones.

Twenty image-group splits show that the current learned selectors still leave most oracle gain unused:

| Policy | Selected images | Actual bpp | Multi-teacher delta | PARSeq CER | Tesseract CER |
|---|---:|---:|---:|---:|---:|
| Base | 0.00 | 3.6665 | 0.00 | 0.0641 | 0.4294 |
| Oracle single candidate | 19.30 | 4.3327 | -28.40 | 0.0547 | 0.3622 |
| Linear top-40 | 40.00 | 4.9183 | -1.95 | 0.0655 | 0.4227 |
| Image-feature MLP top-15 | 15.00 | 4.0608 | -0.75 | 0.0639 | 0.4276 |
| Patch-CNN top-10 | 10.00 | 3.9439 | -1.70 | 0.0634 | 0.4254 |
| Patch-CNN train-tuned | 19.90 | 4.2753 | -1.30 | 0.0636 | 0.4262 |
| Listwise patch gate top-10 | 10.00 | 3.9308 | -2.75 | 0.0643 | 0.4217 |
| Listwise patch gate top-25 | 25.00 | 4.4421 | -3.35 | 0.0647 | 0.4197 |
| Listwise harm-0.5 top-20 | 20.00 | 4.2775 | -3.25 | 0.0647 | 0.4199 |
| Listwise harm-0.5 top-30 | 30.00 | 4.6447 | -3.30 | 0.0649 | 0.4195 |
| Listwise Tesseract/PARSeq-safe top-20 | 20.00 | 4.2233 | -3.30 | 0.0645 | 0.4199 |
| Listwise Tesseract/PARSeq-safe top-25 | 25.00 | 4.3970 | -3.80 | 0.0645 | 0.4186 |
| Listwise Tesseract/PARSeq-safe top-30 | 30.00 | 4.5673 | -3.75 | 0.0648 | 0.4185 |
| Listwise Tesseract/PARSeq-safe constrained tuned | 24.95 | 4.4173 | -3.55 | 0.0643 | 0.4196 |

Row-wise validation metrics follow the same ordering through the first rankers: linear selector AUC/AP is 0.5572/0.0994, image-feature MLP is 0.5705/0.1153, and the patch-CNN ranker is 0.6170/0.1316. The new image-level listwise patch gate improves AUC/AP to 0.6315/0.1450 and gives the strongest non-oracle Tesseract policy so far, but it slightly worsens PARSeq versus base. A Pareto/no-worse target gives effectively identical results because Eval300 has 409 summed multi-teacher positives and 408 Pareto positives. A generic harm-logit penalty of 0.5 is more conservative at threshold 0, but lowers AUC/AP to 0.6243/0.1430 and still worsens PARSeq in the top-k policies. The evaluator-specific `tesseract_parseq_safe` target is a better direction: safe-label AUC/AP is 0.6295/0.1410, and top-25 improves Tesseract to 0.4186 at lower bpp than standard listwise top-25 while slightly reducing the PARSeq tradeoff. Train-split constrained thresholding enforces zero protected-distance violation on train and gives a safer validation tradeoff (0.0643/0.4196). It is still not strict no-worse behavior, so the next selector should use better-calibrated protected scores, OCR-family-specific losses, or separate weighted heads.

Eval300 oracle-selected compact codebook diagnostics were also run with the same multi-teacher utility table. Both K16 and K64 select 0.2667 candidates/image on average. Because the current payload writes one index per selected vector, K16 and K64 have the same measured bytes in this stub.

| Method | Actual bpp | Enhancement bpp | Avg selected | PSNR delta | PARSeq CER/exact | Tesseract CER/exact |
|---|---:|---:|---:|---:|---:|---:|
| MLIC++ base | 3.6726 | 0.0000 | 0.00 | n/a | 0.0661 / 0.8467 | 0.4326 / 0.3367 |
| Oracle-selected K16 | 3.7860 | 0.0253 | 0.27 | -0.0010 | 0.0634 / 0.8533 | 0.4166 / 0.3467 |
| Oracle-selected K64 | 3.7860 | 0.0253 | 0.27 | +0.0113 | 0.0567 / 0.8667 | 0.4079 / 0.3600 |
| Oracle-selected K80 | 3.7860 | 0.0253 | 0.27 | +0.0148 | 0.0567 / 0.8667 | 0.4052 / 0.3600 |
| Raw single-candidate oracle | 4.3327 | n/a | 19.30/75 val images | n/a | 0.0547 / 0.8713 | 0.3622 / 0.3700 |

K64/K80 are now the best compact-payload results: they almost preserve the raw oracle's PARSeq gain at much lower enhancement rate, but still recover only part of the Tesseract gain. K80 adds a little PSNR and a tiny Tesseract gain over K64, but the large gap to the raw Tesseract oracle remains. This makes the next technical target sharper: keep the frozen MLIC++ base and counted OSCR payload, but train the compact codebook/symbol assignments with a multi-teacher utility objective rather than residual MSE.

### Learned Gate to Compact Codebook Smoke

The first non-oracle compact-payload smoke uses held-out 75-image Eval300 validation splits. A `tesseract_parseq_safe` listwise gate was trained on the other 225 images and exported its top-25 validation selections. Those selections were then passed as explicit rows to the counted MLIC++ codebook `.oscr` path rather than using an oracle gate.

| Method | Split | Actual bpp | Enhancement bpp | Avg selected | PSNR delta | PARSeq CER/exact | Tesseract CER/exact |
|---|---|---:|---:|---:|---:|---:|---:|
| Seed0 MLIC++ base | val75 | 3.6166 | 0.0000 | 0.00 | n/a | 0.0622 / 0.8133 | 0.4243 / 0.3200 |
| Raw learned top-25 candidates | val75 | 4.4463 | n/a | 25/75 | n/a | 0.0622 / 0.8133 | 0.4189 / 0.3200 |
| Learned top-25 + K16 codebook | val75 | 3.7558 | 0.0311 | 0.33 | +0.0073 | 0.0622 / 0.8133 | 0.4189 / 0.3200 |
| Learned top-25 + K25 q0.05 codebook | val75 | 3.7558 | 0.0311 | 0.33 | +0.0114 | 0.0622 / 0.8133 | 0.4189 / 0.3200 |

The seed0 result is the first positive learned gate + compact actual-bitstream result. It preserves the raw top-25 Tesseract gain while replacing the raw residual payload with a small counted codebook/index payload. The first apparent Tesseract regression was traced to an OCR protocol mismatch: Eval300 base/candidate labels use Tesseract PSM8, while an intermediate codebook OCR run accidentally used PSM7. With matched PSM8, K16 and quantized K25 both match the raw candidate policy.

Two additional held-out splits show the signal is not isolated:

| Seed | Actual bpp | Enhancement bpp | PSNR delta | Base Tesseract CER | Raw top-25 Tesseract CER | K16 Tesseract CER | PARSeq CER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.7558 | 0.0311 | +0.0073 | 0.4243 | 0.4189 | 0.4189 | 0.0622 |
| 1 | 3.9573 | 0.0262 | +0.0084 | 0.4256 | 0.4099 | 0.4073 | 0.0548 |
| 2 | 3.8358 | 0.0297 | +0.0108 | 0.4036 | 0.3933 | 0.4010 | 0.0874 |
| Mean | 3.8496 | 0.0290 | +0.0088 | 0.4178 | 0.4074 | 0.4091 | 0.0681 |

Across these three splits, K16 compact payloads retain about 84% of the raw learned top-25 Tesseract gain and preserve PARSeq at the base level. Seed2 loses much of the raw gain after K16 quantization, which points to codebook capacity/objective as the next bottleneck rather than the counted bitstream path.

A stricter train-codebook/eval-selection control was then run. Codebooks were trained on each split's train225 learned top-75 selections and loaded for the corresponding val75 learned top-25 selections. This preserves the actual payload and selector but removes validation-vector codebook fitting:

| Method | Actual bpp | Enhancement bpp | PSNR delta | PARSeq CER/exact | Tesseract CER/exact |
|---|---:|---:|---:|---:|---:|
| Seed0 selected-val K16 codebook | 3.7558 | 0.0311 | +0.0073 | 0.0622 / 0.8133 | 0.4189 / 0.3200 |
| Seed0 train225 K16 codebook -> val75 | 3.7558 | 0.0311 | -0.0026 | 0.0622 / 0.8133 | 0.4243 / 0.3067 |
| Seed0 train225 K64 codebook -> val75 | 3.7558 | 0.0311 | -0.0024 | 0.0622 / 0.8133 | 0.4189 / 0.3200 |

K16 trained on other images is too small and loses the Tesseract gain on seed0. K64 recovers the raw top-25 Tesseract result on seed0, but this capacity fix is not uniformly stable across splits:

| Seed | Actual bpp | Enhancement bpp | PSNR delta | Base Tesseract | Raw top-25 Tesseract | Selected-val K16 Tesseract | Train-K64 MSE Tesseract | Train-K64 utility a4 Tesseract | Train-K64 utility a8 Tesseract | PARSeq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.7558 | 0.0311 | -0.0025 | 0.4243 | 0.4189 | 0.4189 | 0.4189 | 0.4216 | 0.4162 | 0.0622 |
| 1 | 3.9573 | 0.0262 | -0.0031 | 0.4256 | 0.4099 | 0.4073 | 0.4230 | 0.4204 | 0.4178 | 0.0548 |
| 2 | 3.8358 | 0.0297 | -0.0029 | 0.4036 | 0.3933 | 0.4010 | 0.3959 | 0.3959 | 0.4010 | 0.0874 |
| Mean | 3.8496 | 0.0290 | -0.0028 | 0.4178 | 0.4074 | 0.4091 | 0.4126 | 0.4126 | 0.4117 | 0.0681 |

This is a useful capacity and objective control. The counted `.oscr` path and learned selector are healthy, and K64 train-side codebooks preserve PARSeq while improving mean Tesseract over base. Residual-MSE K64 retains about 50% of the raw top-25 Tesseract gain; a first utility-weighted K-means variant with `alpha=8` raises this to about 59%, improving seeds 0 and 1 but hurting seed 2. A weaker `alpha=4` rescues seed2 but weakens seed0/1, returning the three-seed mean to residual-MSE K64 level. On seed2 only, `alpha=16` worsens Tesseract to 0.4036. Selected-validation K16 still retains about 84%, so the final compact representation should keep enough symbol capacity and train assignments/centers with a stronger multi-teacher utility objective, not only residual fidelity or a scalar weighted MSE proxy.

Pairwise Tesseract output comparison between residual-MSE K64 and utility-weighted K64 shows the effect is sparse: 6/225 samples improve, 6/225 worsen, and 213/225 are unchanged; total edit distance improves by only 1 character. Seed1 gains mainly on IAM words, while a few ICDAR/SynthText samples account for harm. This points toward assignment-level harm control or diversity-aware prototypes rather than only stronger scalar utility weights.

A hard non-harm train-filtered K64 control was then tested by removing train candidates that worsened Tesseract or PARSeq before residual-MSE K64 clustering, while keeping the same held-out val top-25 selections. This is a negative result: seed0 drops back to its base Tesseract CER, seed1 matches residual-MSE K64, seed2 stays at the degraded utility-a8 level, and the three-seed mean Tesseract CER is 0.4161. Retained raw top-25 gain falls to about 16.5%, so no-worse labels are useful but should drive assignment/routing losses rather than hard prototype deletion.

The first assignment-aware diagnostic also rejects a simple shortcut. I added a utility-biased assignment mode that keeps the same transmitted index payload but allows the encoder to choose a train-utility-high center among near-nearest centers. On seed2, relative-error margin 1.05 changes 11/25 selected candidate indices and worsens Tesseract from 0.4010 to 0.4036; margin 1.01 changes only 1/25 index and still worsens to 0.4036. The changed samples are sparse but harmful, showing that center-level utility priors are not enough. A future assignment model must look at the candidate/image context and be calibrated for no-worse decisions.

The corresponding top-8 assignment oracle shows that this negative result is not because assignment lacks headroom. I generated all top-8 nearest code alternatives for the 25 selected candidates per split and selected the best Tesseract option under a PARSeq no-worse constraint. This keeps selected candidates and code-index payload size fixed. Across seed0/1/2, mean Tesseract improves from nearest-a8 0.4117 to 0.3910 while PARSeq remains unchanged. Seed0 has no assignment headroom, but seed1 improves 0.4178 -> 0.3916 and seed2 improves 0.4010 -> 0.3650. This is the strongest compact-payload upper-bound signal so far, and it reframes the next step as candidate-conditioned assignment learning.

A train-side assignment label set was then generated for the same target. For each seed, the train225 top-75 learned selections were expanded into top-8 code alternatives under the train K64 utility-weighted a8 codebook and evaluated with PARSeq and Tesseract PSM8. This produced 1800 train variants over 225 selected candidates. The PARSeq-safe Tesseract oracle changes 63/225 groups and gains 87 Tesseract edit-distance characters while preserving PARSeq at the selected-subset level. The target is therefore not a val-only fluke.

Simple assignment policies still fail to recover the oracle. Fixed rank choices and residual-relative thresholds produce much smaller gains or regressions; many train groups also have near-zero nearest residual distance, so raw relative error can explode and is not a stable standalone feature. A first tiny patch-difference assignment ranker using `[nearest, variant, |diff|]` images plus clipped rank/error features overfits train labels. Across seed0/1/2 train225->val75 experiments with three model seeds each, train AUC/AP is 0.933/0.659, but validation AUC/AP drops to 0.410/0.084. The train-tuned threshold policy worsens Tesseract by +2.11 edit characters on average and captures none of the val oracle gain.

Adding codebook-center geometry and code one-hot features is a partial improvement. It raises train225->val75 validation AUC/AP to 0.551/0.131 and reduces threshold-policy harm to +1.22 Tesseract edit characters. Seed2 improves most: validation AUC reaches about 0.632 and the threshold policy averages 0.0 Tesseract delta, but seed1 still worsens by +2 and seed0 has no oracle headroom, so any unnecessary change hurts. This is the clearest signal so far that codebook geometry is necessary but not sufficient. The next assignment head should expose richer latent residual/codebook features and learn an explicit no-op calibration objective.

A simple zero-residual escape code was tested on the harmful seed2 split by appending an explicit zero vector to the utility-weighted K64 codebook. It did not change Tesseract CER versus utility-weighted K64 and slightly worsened PSNR, so nearest-neighbor no-op is not enough; the no-op decision likely needs a learned confidence/assignment margin rather than an unconditional zero center.

Residual-relative zero assignment thresholds were then tested with the same seed2 utility-weighted K64+zero codebook. Threshold 0.8 zeroes 0.2267 candidates/image, about 17 of the 25 selected candidates, and moves Tesseract to 0.4036 CER, effectively removing the utility-weighted gain. Threshold 1.0 zeroes 0.0667 candidates/image, 5 of 25 selected candidates, but leaves Tesseract at 0.4010 CER and PARSeq at 0.0874 CER, matching the unthresholded zero-code negative control. This rejects residual relative error as a standalone no-op criterion; the next assignment rule should be trained directly against OCR utility/no-worse labels.

This result is still diagnostic, not final: it covers three random validation splits for selected-val codebooks, three train-codebook/eval-selection K64 capacity controls, and one simple utility-weighted K64 variant. The next step is a real utility-trained codebook that is trained on train images and evaluated on held-out images, ideally with differentiable or discrete assignment learning rather than a post-hoc scalar weight.

## OCR Smoke

| Method | Total bpp | PARSeq CER / exact | Tesseract CER / exact |
|---|---:|---:|---:|
| Original crops | n/a | 0.0714 / 0.7083 | 0.3512 / 0.2917 |
| MLIC++ base | 2.9644 | 0.0774 / 0.7083 | 0.3690 / 0.1667 |
| Residual stub 5% | 4.8040 | 0.0774 / 0.7083 | 0.3512 / 0.1667 |
| Residual stub 10% | 5.8902 | 0.0774 / 0.7083 | 0.3690 / 0.1667 |
| Codebook K16 stub 5% | 3.2583 | 0.0774 / 0.7083 | 0.3690 / 0.1667 |

The 5% stub recovers Tesseract CER to original-crop level without changing PARSeq, but exact rate remains at the MLIC++ base level. The 10% stub gives higher PSNR but does not improve Tesseract CER, which suggests the residual-energy candidate heuristic is not aligned with OCR utility.

## Artifacts

- 5% codec summary SHA256: `d50ef9c88ca6b56282e79e07f6b2070b86e69a8d37bc7d187780fee63ce93273`
- 5% codec results SHA256: `d4dd866ef9911aec38f0f3620b506df82614efae405392606ee2810f38091c92`
- 10% codec summary SHA256: `15490cb2f7306d5afa03cbb6ce51ba15f32d37f88e2438d9e418224bc2c5ed26`
- 10% codec results SHA256: `5f200aafbce8b0e120b07633da0fd72dde74bffa736d6df991d11a84e2afd6ad`
- 5% PARSeq summary SHA256: `6913ce5633e76c60fea592aee7f8a1fa8634ded2a8d70c4b2d8faeacfb7f04f1`
- 5% Tesseract summary SHA256: `3858dccae93538f9f8dc3e3f73f2affc8d8a6ee9bf789d334036e0cd0ad87c3d`
- 10% PARSeq summary SHA256: `53c2ab9ae81c553cbe06570adb0169aac52c6afceb833ed2c7345f572205440d`
- 10% Tesseract summary SHA256: `95112a6566defe783901123805621b6ec8ac90dcd4ae5f978821b08e8ac7b15b`
- Codebook K16 codec summary SHA256: `a0ea155bfa0f1cf9ec10888e5c3fa1bf7be0c3dbf96cafc9b5dd83b714d4ef48`
- Codebook K16 codec results SHA256: `6ed4e320a2ee9d0614d9d8b7022468680e2f5f4a5de776962c40eec708d34fff`
- Codebook K16 decoder-known codebook SHA256: `4c919c6a4d15e7b2a0e549a39f0e0f043a91a035547937e07dd5273622614883`
- Codebook K16 PARSeq summary SHA256: `563bb0bc9899447e5e6bbda6d51380d3624ea0d67cd6b885aaba6220344056a1`
- Codebook K16 Tesseract summary SHA256: `510ac5975f24bebf1ed03ede901bdf452f0abe14feb6ed0fb0a0a120c497ddb3`
- Single-candidate reconstruction split SHA256: `b8473ba3b30549f62dde4a7a4b9d507715d0c7edfbd87192bd47c0921d7a4a79`
- Single-candidate split metadata SHA256: `242c905d69301edd75d79b6c2e68e0140bbadc3947ef161f0f03b439cb5fc6cb`
- Single-candidate PARSeq summary SHA256: `3142e64721a852885f88a836121bb73e0e1694c9c7688996b4032404b3815826`
- Single-candidate Tesseract summary SHA256: `e13cac25589ac7d17f93f3223b4b3692db484e181bc1349935915644269e7f87`
- Single-candidate PARSeq utility analysis SHA256: `9da64d09396a6c3bcb82b44616d21352fb6d78c56da934ea90a591c80587acba`
- Single-candidate Tesseract utility analysis SHA256: `85bef694ac12e1bf83469dbe390daa649f01a9d6b0a0d24cbefbb0bbda108535`
- Candidate utility table SHA256: `e0f7c48f18d773d926e0b50799f7b31a2a5c094370ed25eabfe589f3abbb4405`
- Candidate utility table summary SHA256: `aed80168cf2e4a8c4ca96d6fb1451fbbc54863f6a0b2ef899eb567968f305657`
- Candidate utility table with latent features SHA256: `bd1330ec9b96ddaff0d6a06552366a6e43eeddcad0b894f8c7f3df234b826b71`
- Candidate utility latent-feature summary SHA256: `1b86cffa27bc6b1dc96ba90d82b0e24189982dd08504bf0c0816736e3d0752d3`
- Candidate utility linear baseline SHA256: `15e94a5339ca7c5e0c7879065ca0cbe17cd595e2d4e6567530699b3b99ef463d`
- Eval100 candidate utility table SHA256: `b79d7834d708542d15356495dfb686078d6f5a58d2d253cfb6ba655d2efe6e5e`
- Eval100 candidate utility table summary SHA256: `727e3f6334995b793fe4509e00380b9c0fa3240fe431551cea76602132de2d40`
- Eval100 candidate utility table with latent features SHA256: `8c4d6d2eefbffd616860c68128e2b85da5adf53b78f734c73a4d328b81dda295`
- Eval100 candidate utility linear baseline SHA256: `0d9e2fe9c9135036e789ff0140d65839c88d2133dc2f4e6ebbaa8a5a73e99591`
- Eval100 candidate selector single-split SHA256: `f9e417397f82659c02472bb6c9fba166cd67d59000db2f3ee87a8d7b8b677a6d`
- Eval100 candidate selector 20-split sweep SHA256: `e951033dceb4cc3c44e8df8cd3b4f928d2f0ad4662bb40540bfc74176172d758`
- Eval100 candidate ranker seed7 SHA256: `1f2e7030bce020eb74cfec91f1cb52cee4b22a5d0f8349756f44954d81f12e7a`
- Eval100 candidate ranker 20-split sweep SHA256: `cf2f95569237c9a9054471f76c2698a01188c9e2bd0e06cc9baab84ba1bfaa20`
- Eval100 candidate utility table with image patch features SHA256: `152a4b204c0f6685ddd261161a13b283c70449d6f9aaba73ad23d8f7722489f9`
- Eval100 candidate utility image-feature summary SHA256: `f0ee396b02784d614a179dc50498d1cc68e630b34553c84452afb4147984482c`
- Eval100 image-feature ranker 20-split sweep SHA256: `7af3ef678e2642431c17c91f73b7b132c0b2ef127a15d2b99d19eb19f1f56000`
- Eval100 regularized image-feature ranker 20-split sweep SHA256: `3bca279e871363fe9991255ecb8993c7393cd660c20988115006a0802994368c`
- Eval100 latent image-vector feature table SHA256: `f28f1e76a3ebfaee1ab89ee46a147e099f5da96a48373f4de5b5968cfe07baf5`
- Eval100 latent image-vector feature summary SHA256: `5d60fc39805a04dfac9b1d4471b2b9aa7d811e23b29ff324a7c359870af52d6b`
- Eval100 latent image-vector ranker 20-split sweep SHA256: `b64b7e81dac609ce9a3f4df85f4f9a0b1e8c3f3a5688b02e237e45186a7690a6`
- Eval100 patch ranker binary 20-split sweep SHA256: `07bdf1d32f3d88816fc2067c9d9fa4f5fbf46f5f062e2f80d6757d8a5509daf8`
- Eval100 patch ranker benefit 20-split sweep SHA256: `a57983f4bb78e0760178699c45dc95893580862e26553f92227160ee7cb4d0d6`
- Eval100 patch ranker benefit-per-bpp 20-split sweep SHA256: `490c2acbd7fc6e0bab7f256dd4953f97610126b31fe7f73be6921fdf21322141`
- Eval100 codebook K16 codec summary SHA256: `feb87bf365b97e30b53d6c680f3f8161b903a7a8b48acf91bcd01969479a7b48`
- Eval100 codebook K16 codec results SHA256: `5a5cada348865911c358c70e52601d29b9266813311b01e51b25731906cd552b`
- Eval100 codebook K16 codebook SHA256: `9fbf441e3b6f5ae55958fa10cb163d96cf1dd7b7de194564e710e999d1ad04bb`
- Eval100 codebook K16 OCR split SHA256: `da897d347a6c22066d34352488bb75bf53e2527e34422b9ef2049e67e74e2adf`
- Eval100 codebook K16 PARSeq summary SHA256: `4acdf93927c5acae2e661eebe1a61138302cc49879833c7d1807a6ecdc244464`
- Eval100 codebook K16 Tesseract summary SHA256: `2528140e1b9adf15e605881a6a6d337b863ea3e55e3a8f224254cf0f9b742d37`
- Eval100 oracle-selected K16 codec summary SHA256: `922c3ae4a1d176f1ceec52ae0ea1f0ee942d0079ee4bdfb79d6e606833ff39db`
- Eval100 oracle-selected K16 codec results SHA256: `d4f00e043c9224c94a4b180cbc3614746c404bab2787228edbab042388e2c730`
- Eval100 oracle-selected K16 codebook SHA256: `9059e4444bdadbb179b44409655000a81e09f42e62291215a40b50ec80cfd58a`
- Eval100 oracle-selected K16 OCR split SHA256: `83881508a21ebe59a9f4b6e6280e6f55e2dea356691eb9fcbb2a9f32c1d3f3a3`
- Eval100 oracle-selected K16 PARSeq summary SHA256: `1558715e5372fe7fa85e751d7b83334dbe7722a05bcf630e3f1f775f981d8376`
- Eval100 oracle-selected K16 Tesseract summary SHA256: `d22c9bcaab632c055058f867debeb349496b3ba8041cab15a0d97ef6ac1df7cf`
- Eval100 oracle-selected K64 codec summary SHA256: `291a51bee9f7c1284b1ddc251468990a5bdb7e74b88fa1c3073f4ac9fe9967e0`
- Eval100 oracle-selected K64 codec results SHA256: `6a3f1d08cd94250ba1ed63c736776dbf4ba6835b4f3f54969f22e3dc11b6e5a4`
- Eval100 oracle-selected K64 codebook SHA256: `dc4e93e479cb74ae42635117e575093c1199da0954bd9c309b11bc6de72fbbd3`
- Eval100 oracle-selected K64 OCR split SHA256: `79e798849055cc0fc0d600e71734ad7221c71a8e36e0145475b8d4ac2f670b7e`
- Eval100 oracle-selected K64 PARSeq summary SHA256: `231233a7e35da7e4a0feea79507e6c4d5d2cf477ecca9ba70e96a7bc4cfbb80a`
- Eval100 oracle-selected K64 Tesseract summary SHA256: `dede5cff0514b4aab9cb95182306b7e9be256fd5cedbb29bee87c75548c52d95`
- Eval300 split SHA256: `c4af53ebe08dd8ba088382344a776ba2f86087bb1db9d1b3c9fc9dbcc68c6f55`
- Eval300 single-candidate split SHA256: `13912904a20583389f4c7e71313eeae8c0f6df1e36c289d7abd4fbc620d8e3e5`
- Eval300 single-candidate PARSeq utility analysis SHA256: `7d4d23dc029db8eb82794d8995f108cc8ddcdfeb03b54ef40c8f5a6f1958a30d`
- Eval300 single-candidate Tesseract utility analysis SHA256: `25a89d96184b8bdfe21bd804ded06a97de078ed94a73110b7a122dbdd3460324`
- Eval300 candidate utility table SHA256: `dd76134eb0d44fa2ae0ec171265965b2ea11f3f900b9440e26f3386a87978361`
- Eval300 candidate utility table with image features SHA256: `d49e11a0f56880b028a131ec01500154968e0550a84b06eae218b1da7b72943b`
- Eval300 linear selector 20-split sweep SHA256: `9ee03c277c49c8fae165842030117df4b0d3c75f8f873b6c999e12e1820a5c9e`
- Eval300 image-feature MLP ranker 20-split sweep SHA256: `9ab415408159e978fcf864a101626012e5da00fef2e60ce62cc9d35e103a6c3b`
- Eval300 patch-CNN ranker 20-split sweep SHA256: `1d841c639f8748a41decc81690acdfa31e6da79c8d7f4639088d735e31780056`
- Eval300 listwise patch gate script SHA256: `8cad47a1b9d102a3200b6d0eec00ba247d2050dfb7a87262881bd016ef68c793`
- Eval300 listwise patch gate 20-split sweep SHA256: `f9d028d97be82d74402e9f5256601026e5908e09736b46870953470184e33a64`
- Eval300 Pareto listwise patch gate 20-split sweep SHA256: `0d6463d15e9d22a5d9d5447474ab2cdd0085f6c0c359f650ad0493aef5fc0039`
- Eval300 listwise patch gate script with harm-penalty option SHA256: `f3127a0b57697bf9083b786981734cb22523d8b16383534a0d6ce41c87e503eb`
- Eval300 harm-0.5 listwise patch gate 20-split sweep SHA256: `7a7c6501f4fa2d946da089890462940035c4d7c0554f5055159368255e20b8a7`
- Eval300 listwise patch gate script with protected-target option SHA256: `ac167e61e040b959f1165c2998b70781d0cb000cc9a436107cca66b478466fdb`
- Eval300 Tesseract/PARSeq-safe listwise patch gate 20-split sweep SHA256: `4e558923d27b4dfec06cfc661c3b5183612c0d81d8979cc2296ea16446430ec9`
- Eval300 listwise patch gate script with constrained-threshold option SHA256: `10b88b8eb4d725936d165473910addc5f839b58cda2403a680fb0681cf9879f8`
- Eval300 Tesseract/PARSeq-safe constrained-threshold listwise patch gate 20-split sweep SHA256: `5a53de1b563a5403a797f0b645033bcd1b78831c77ed76ce7725954af2ae9428`
- Eval300 explicit-selection/load-codebook/utility-weighted/zero-code codebook script SHA256: `1082e766dab9fa66d5b0897d2f746a5d9b593cbdae211a2c02c9b30a390fa332`
- Eval300 zero-code threshold codebook script SHA256: `e1166c774cae4266730e4ef8e3b621fdde63cf5d52d59e82484d1e4fb1663932`
- Eval300 utility-biased assignment codebook script SHA256: `d21c27c1f841f7dfe74231a1cbfc76d0a71f0b995a899718ce9b7654521c4872`
- Eval300 listwise selection export script with train/val export-subset SHA256: `652797caf8bfec8e7e68218069db21332d849dcad5bbe7fb68d979a2bb1f7d30`
- Eval300 seed0 val75 split SHA256: `1def545f651c9e3a8c353d138a4827f80c9d30af351dc825bfb121b9450adb12`
- Eval300 seed0 val75 learned top-25 selection SHA256: `fdeec593dd0d829cf9a40686dc1fc3c187277304d2bb10fb28d3836cc5ebd708`
- Eval300 seed0 val75 learned top-25 selection summary SHA256: `b83f829144270467d20c5f443434791a99dbcde8b05ed5e6bed5ef9eb320d794`
- Eval300 seed0 val75 learned top-25 K16 codec summary SHA256: `ffd51cd2b63ef0f569df507ada04e616d6f8c777cda665520279bc8aa1fd0632`
- Eval300 seed0 val75 learned top-25 K16 codec results SHA256: `157b9656888bf4d8af1a8ab8482c540be3fddb4b69fdadce42e408d79371c797`
- Eval300 seed0 val75 learned top-25 K16 codebook SHA256: `3b7ce6265f253f7eeeec629f9d8d0d87e45db34d5ea40edd689f1a8b746d2a7e`
- Eval300 seed0 val75 learned top-25 K16 OCR split SHA256: `947079c203311a99f99280699d63dc468491ad9c9e904e6afb0631dde51d3c75`
- Eval300 seed0 val75 learned top-25 K16 PARSeq summary SHA256: `8a19de1254eb46e917807624860381e509dc6232ac3c134d314a876c987d9e09`
- Eval300 seed0 val75 learned top-25 K16 Tesseract PSM8 summary SHA256: `8f451ac5ec69205f15ede45d68bcbf128ffbada518c2c55ec942b0736329b569`
- Eval300 seed0 val75 learned top-25 K25 q0.05 codec summary SHA256: `0e70739e00ed624b90da950d048b0a04813a68dc499a5a400e362ba30b1227fc`
- Eval300 seed0 val75 learned top-25 K25 q0.05 codec results SHA256: `710e0b18dce9263282f32252fd4dc31e9283260cf21946df0638f3a80fb489b5`
- Eval300 seed0 val75 learned top-25 K25 q0.05 codebook SHA256: `700d4f72a4285d79565fe0d10e930540eb6d7fbcc820d0d4b02514e476e2a81b`
- Eval300 seed0 val75 learned top-25 K25 q0.05 OCR split SHA256: `0f0cb135c2ce403e06de743bf5d4aa62fd5cbccdc6999c336e6dba3b5cb94807`
- Eval300 seed0 val75 learned top-25 K25 q0.05 PARSeq summary SHA256: `60c40a7a95a627fe2e2624a27c96166438760655c7cf3ed03b108ae9de625324`
- Eval300 seed0 val75 learned top-25 K25 q0.05 Tesseract PSM8 summary SHA256: `9c2eae96ea9bbc77a659f98b0d4e93406f5222f9217e01bb4e748d9173d652a8`
- Eval300 seed1 val75 split SHA256: `daecca6ded30c394f855f2cf685bf0e211c5601af33f7f10c98074014c1caa63`
- Eval300 seed1 val75 learned top-25 selection SHA256: `3ab1726efb4ddbdc4bef5414ade4abd7cfbe993bc0446e055ce637a31ff6d34c`
- Eval300 seed1 val75 learned top-25 selection summary SHA256: `9ffab8830987a76bc6ea1d97b359655f5516f262483a405e2dffa472c46ae87d`
- Eval300 seed1 val75 learned top-25 K16 codec summary SHA256: `036cb13015ce49e4180891cc34038dcce3c7d23eacb1b982495c74dfdb56fbec`
- Eval300 seed1 val75 learned top-25 K16 codec results SHA256: `2c802d48a9b69b019874d7da17f4653f0eda23d81982e8574a4034a6bc2ccfb4`
- Eval300 seed1 val75 learned top-25 K16 codebook SHA256: `33da7fddd140fd5dd943f5a750308b03f95979d16535fef0cf3223bc57b23c46`
- Eval300 seed1 val75 learned top-25 K16 OCR split SHA256: `5f03cb3b852af6dfedc701d5ab27491e799024e2b12b8a23c67f7512e3ad6473`
- Eval300 seed1 val75 learned top-25 K16 PARSeq summary SHA256: `deceaab8b0137f2042d088f593acb117e9b24efa1544d49b4d45180f34fb4e0f`
- Eval300 seed1 val75 learned top-25 K16 Tesseract PSM8 summary SHA256: `a370a6b3726cdbf2d812e1007594409134744e29ce3ee0589a866b1de3eb71df`
- Eval300 seed2 val75 split SHA256: `93ab2f5d0290dd641425497a1cbf74162c3cf9b3a16dd26eff1195401b19093f`
- Eval300 seed2 val75 learned top-25 selection SHA256: `2470f44b563b15114e3cabe3e3cf8216dcb657b2e4363688d9f1be09359006a9`
- Eval300 seed2 val75 learned top-25 selection summary SHA256: `f908c87995d39807d24300deb9e5f238be85f47c734c64c2bdf08c57d8203964`
- Eval300 seed2 val75 learned top-25 K16 codec summary SHA256: `3c92bdf075ccc23a3d1bf135676ef68eb6d05490fb84c38ddf08df827f4d8d52`
- Eval300 seed2 val75 learned top-25 K16 codec results SHA256: `a4ce61cc77e11e8240d42f0fac8237f383fdb98e7e8c3d0b4f9ddaeb45ef176b`
- Eval300 seed2 val75 learned top-25 K16 codebook SHA256: `d4692eab6446ddfca78cdc62511f6ae3d367dd6a98fd19fec715eaea3094c465`
- Eval300 seed2 val75 learned top-25 K16 OCR split SHA256: `4c9b1361d104a4bd6706459723f0e817e6fcd3fc742dc9cf312a2a5c2a3482c9`
- Eval300 seed2 val75 learned top-25 K16 PARSeq summary SHA256: `30156e066bd511c947a436963e4695c5870b51782dceee9ec13db8caf712b0e0`
- Eval300 seed2 val75 learned top-25 K16 Tesseract PSM8 summary SHA256: `44dee208a6ea0a7a4d1fa04744b773b2099e8f747265af78815c4eba2ad90fba`
- Eval300 seed0 train225 split SHA256: `9410a218c0b1c1a866f35064b986a2d8ce74dfb5c5a0e113589cdaec34caa1ff`
- Eval300 seed0 train225 learned top-75 selection SHA256: `1fc883005d55c706a490294a88fdf9257304664baf7efc2866d846f33bb6c991`
- Eval300 seed0 train225 learned top-75 selection summary SHA256: `21efa5b772816572ffe1ee7ea208f0aac87af207a6077ef5dd69bf98d62282bb`
- Eval300 seed1 train225 split SHA256: `c2913cc80244883f45f02be0c1627ee43566d50607a36d89ee29fada2c72c5bd`
- Eval300 seed1 train225 learned top-75 selection SHA256: `2ee6e3abd02d5be602e0089147b107686d7e9d7c5664c9fc02ba152138e7d13c`
- Eval300 seed1 train225 learned top-75 selection summary SHA256: `482da8752ee707eebd7724406d4cf0e9a59f43f12014554e057b4f5b419689b0`
- Eval300 seed2 train225 split SHA256: `f36e2f11e75f082ed860c5e7398d08a5da45a0881c613a6db3ab121d6ed7b65f`
- Eval300 seed2 train225 learned top-75 selection SHA256: `9708a6c8d0d88199bdda78ecf3fff473326adc42f72e0278141e04c51e5f222d`
- Eval300 seed2 train225 learned top-75 selection summary SHA256: `309e8c3e04ec9a6c0a53a7926bc3b8afa06d930868339c629ec1cd2b5ed5e89b`
- Eval300 seed0 train225 learned top-75 K16 codec summary SHA256: `b0d05d27e44d12c3b031e46a0697b2ed77497259348a6b1929c312577d19209d`
- Eval300 seed0 train225 learned top-75 K16 codec results SHA256: `671ef682e7e00480f1f69194a08be6c2c00a4247b7c0ab6695cf8a5cc3abd3e7`
- Eval300 seed0 train225 learned top-75 K16 codebook SHA256: `be5e72c17d785b74321cab9dcd7e3b080faa85cfb76fb017494d0e311caffa24`
- Eval300 seed0 train225 learned top-75 K64 codec summary SHA256: `f6b7dbd75b310e4569e5cb9febcbf97580d4bf31bea13e5a0de25ef8b309c8af`
- Eval300 seed0 train225 learned top-75 K64 codec results SHA256: `b527ab4aed1c4837707aead1386261d0598295cde77145b974fb52a1454e89a2`
- Eval300 seed0 train225 learned top-75 K64 codebook SHA256: `c2d9ce128de3e9cd6a7720327b0d58f1f3cd99cfacf1835df470aea6d7671e43`
- Eval300 seed1 train225 learned top-75 K64 codec summary SHA256: `429bac642f41a10dedad4906468b86542ac00794b74bdfaefff61036fa332e03`
- Eval300 seed1 train225 learned top-75 K64 codec results SHA256: `26684689deb7270ae41dfaa9ead69bf66fbb14679a48be737a94ead8313f8381`
- Eval300 seed1 train225 learned top-75 K64 codebook SHA256: `c4d83a1f45e0f0ef93e52fa0491bb8b044e4ed75e51ecf92b788b0859aed5330`
- Eval300 seed2 train225 learned top-75 K64 codec summary SHA256: `32bf0fcb9649f518a43b335b55c926cc892fbd72e9051a0f92b3ce3531f3372e`
- Eval300 seed2 train225 learned top-75 K64 codec results SHA256: `7c87528733cc1b30e6d512310f42ba9c7b9f48118f110c63e44448be0ae79299`
- Eval300 seed2 train225 learned top-75 K64 codebook SHA256: `5724ac5a0d37b524c1280d3a59e0288039400dc105aafa599b72d5e8ddefc4c4`
- Eval300 seed0 val75 learned top-25 train-codebook K16 codec summary SHA256: `48ea29f7a0e650bc7d1c8d105e40722fb165f9082af6349e18a060855ca79790`
- Eval300 seed0 val75 learned top-25 train-codebook K16 codec results SHA256: `5de2ce773f8262a6d70818bf26aedae89c286392cbacecb897fbfe5dde72305c`
- Eval300 seed0 val75 learned top-25 train-codebook K16 OCR split SHA256: `79c83999966b1fb73580d483e92b02d97de7a8c79cccfca845fd168544fd2231`
- Eval300 seed0 val75 learned top-25 train-codebook K16 PARSeq summary SHA256: `59d3afe08dac7d2a1cee97ca99afb357124d204fc03ed1304a73877a70517699`
- Eval300 seed0 val75 learned top-25 train-codebook K16 Tesseract PSM8 summary SHA256: `37169c1c12ebcf009243d5bd9f4fabf091b044c914b71281e000a5e302b9f4f8`
- Eval300 seed0 val75 learned top-25 train-codebook K64 codec summary SHA256: `df6eadb1c9a923cf4dc1de3cf769d89662805241a98b00c098ed03f936d9d842`
- Eval300 seed0 val75 learned top-25 train-codebook K64 codec results SHA256: `e2e772764a6b9e8e2116ec41776fca13f056071d63308233b8cf2bb851a4e1a5`
- Eval300 seed0 val75 learned top-25 train-codebook K64 codebook SHA256: `c2d9ce128de3e9cd6a7720327b0d58f1f3cd99cfacf1835df470aea6d7671e43`
- Eval300 seed0 val75 learned top-25 train-codebook K64 OCR split SHA256: `9a9123cc9d54382db95cb583fc9a2fecfcb7d574e3c06eb1caee2071904b98fb`
- Eval300 seed0 val75 learned top-25 train-codebook K64 PARSeq summary SHA256: `156efeb9afeebde6dfa61b0103da006d744a02ae9fe61ee6c4ee6857ccd23e73`
- Eval300 seed0 val75 learned top-25 train-codebook K64 Tesseract PSM8 summary SHA256: `377dbb864f4ce0ffa985446ddab85e9d06211e4bb6c0e80733455bdefb95170c`
- Eval300 seed1 val75 learned top-25 train-codebook K64 codec summary SHA256: `6a0d6d4f1133740ac4ec05a4bdfe4514f8254e2d473c8e149e761ef3c5e4dfd0`
- Eval300 seed1 val75 learned top-25 train-codebook K64 codec results SHA256: `98503c26a52428549a8ed1ff764e39e30c2deaa7a2dc5cae747c298f571c906f`
- Eval300 seed1 val75 learned top-25 train-codebook K64 codebook SHA256: `c4d83a1f45e0f0ef93e52fa0491bb8b044e4ed75e51ecf92b788b0859aed5330`
- Eval300 seed1 val75 learned top-25 train-codebook K64 OCR split SHA256: `3971d6a669082315c4a6009b131e16e659b11ccc3cf7a6fe01207d3de517c3bd`
- Eval300 seed1 val75 learned top-25 train-codebook K64 PARSeq summary SHA256: `193b0abe84e62abdb4e1c4a3023b0742234c27204dad619eb50c40d52d1576b4`
- Eval300 seed1 val75 learned top-25 train-codebook K64 Tesseract PSM8 summary SHA256: `10041e7fb4edafeca4c900e041edf4c188432b8b8d32bc22753bd013c1db395e`
- Eval300 seed2 val75 learned top-25 train-codebook K64 codec summary SHA256: `3a580c254adaf3b92e1fa8b46c345b38f0ce55edf13df286dbebdcb1398e75a2`
- Eval300 seed2 val75 learned top-25 train-codebook K64 codec results SHA256: `e9c0ffde7c2c5d29b9f1730464fcd404cc44065454404a94721dbf3098162374`
- Eval300 seed2 val75 learned top-25 train-codebook K64 codebook SHA256: `5724ac5a0d37b524c1280d3a59e0288039400dc105aafa599b72d5e8ddefc4c4`
- Eval300 seed2 val75 learned top-25 train-codebook K64 OCR split SHA256: `ea24ca86b6045c9875701f0bd59d9b7e23071af10c95c1d40cd750de027a0700`
- Eval300 seed2 val75 learned top-25 train-codebook K64 PARSeq summary SHA256: `3c9be22a9e2a861e5f0258535538121d3f6bfb02f909def2fce00da78231425b`
- Eval300 seed2 val75 learned top-25 train-codebook K64 Tesseract PSM8 summary SHA256: `b1ab7ebbb4ca4930b9b178125bc79ff8c4b501c08eada64f0d1781fea6d0c5aa`
- Eval300 train-codebook K64 three-seed aggregate SHA256: `0be2a1bea903bfcf0b2314c31ef39bd4e9674db8067ca6037da045cca4654bc6`
- Eval300 seed0 train225 learned top-75 K64 utility-weighted a8 codec summary SHA256: `c389e56fd12c1eb1f09e1ae030faec3bfae5d0fbaf7abc372f21ea4e1d92250f`
- Eval300 seed0 train225 learned top-75 K64 utility-weighted a8 codec results SHA256: `9ec9a1188dbf827b1510abc82cce9db70dbc9a8e7d0282eb08f96c4dce1c4e56`
- Eval300 seed0 train225 learned top-75 K64 utility-weighted a8 codebook SHA256: `9327e2af789bc25dc8f489e088bc232090f699654dd93cb1e1ecec30f47a5de2`
- Eval300 seed0 val75 learned top-25 train-codebook K64 utility-weighted a8 codec summary SHA256: `90185943e2721b7ef4d8bb303aa03924b7562ad48487483b342ccb7f35dad4e2`
- Eval300 seed0 val75 learned top-25 train-codebook K64 utility-weighted a8 codec results SHA256: `dd805f4f143f49e76c6ee1cd81f0895d64d25f3f8616cbda13bd899fd430a0bd`
- Eval300 seed0 val75 learned top-25 train-codebook K64 utility-weighted a8 OCR split SHA256: `dd266d1dfa1a15106151781aae50f99baaff149aa9642c751c5b55c0ab698f32`
- Eval300 seed0 val75 learned top-25 train-codebook K64 utility-weighted a8 PARSeq summary SHA256: `8b8349420353f605d835b02f30ea293db45527be0a03626bbc22a1f95fef5f5e`
- Eval300 seed0 val75 learned top-25 train-codebook K64 utility-weighted a8 Tesseract PSM8 summary SHA256: `f7ec6577ffc067c9ab1b74d53620c1bc7d9dfff6b949e6036090abe29853bd76`
- Eval300 seed1 train225 learned top-75 K64 utility-weighted a8 codec summary SHA256: `545218eecd565c69750ec055cf89cc1b8ab6c17d37747a78e694dec6659f12d5`
- Eval300 seed1 train225 learned top-75 K64 utility-weighted a8 codec results SHA256: `b26f785ab2538d2627a9880c8315feb9f0998437569a1a9cebe4d32262f56f0f`
- Eval300 seed1 train225 learned top-75 K64 utility-weighted a8 codebook SHA256: `be80b162a0e06a4b9f3405c2a536d5fcd51f445e925ce0d1e358832b486582ec`
- Eval300 seed1 val75 learned top-25 train-codebook K64 utility-weighted a8 codec summary SHA256: `edc115ce605c975c490c99e52a641fda6f69983e82fabed047bb5504aedeb3bf`
- Eval300 seed1 val75 learned top-25 train-codebook K64 utility-weighted a8 codec results SHA256: `f912368bca8bdd83c39f5e28aea8c27ffa94ea8b91b7678f417d77b6fb0088e4`
- Eval300 seed1 val75 learned top-25 train-codebook K64 utility-weighted a8 OCR split SHA256: `5f3985f0f9df95fbd7a00ac24d5ebda833834732af6fcf40fb623c407e06c7f6`
- Eval300 seed1 val75 learned top-25 train-codebook K64 utility-weighted a8 PARSeq summary SHA256: `250560260948595d7a28b9cfc265e6169800304c4d26828185f7ea494e1d8024`
- Eval300 seed1 val75 learned top-25 train-codebook K64 utility-weighted a8 Tesseract PSM8 summary SHA256: `244d12d6c7d0d6b1298c88b8ac444795ac1f5c7b37b19afb399b6fa12901c973`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a8 codec summary SHA256: `33599d2d29c8778d9b4239cc6a761adb184480a89ecd20edec7b0a83c1c06395`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a8 codec results SHA256: `1381d92aeb83c9a208fe50c25417f023c6aeae3fd6eb1c4c9e35082d7622792c`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a8 codebook SHA256: `f4a7fe3cb47e344a3bb71bfd15a6aa44262087c22073c55947d050fb36d3b379`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a8 codec summary SHA256: `b786d5a2e714314eb33cceb970ba5091338b7ed50abbeb0efae97d0150fc2e04`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a8 codec results SHA256: `9f7c445b191bcde865a200fc74ca4e36dbb3a495790353b715faee9e38ad3316`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a8 OCR split SHA256: `8d3c815a32c64a03be6a50f52b39adde7c253a8133c503863d32b7ce88465851`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a8 PARSeq summary SHA256: `58b2d56fddc8d378dc8be266f3b3b4ed5aa833b7f62cbf9794f1c07a49b460d6`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a8 Tesseract PSM8 summary SHA256: `16ee9881d28a7e8fd61d7655c974b2196e917ee5838a214ad8450c7927753e64`
- Eval300 train-codebook K64 utility-weighted a8 three-seed aggregate SHA256: `44d64a7bba531d805d4f23afdd167eacd3ff6f638ed703c7762c5476963c14e8`
- Eval300 seed0 train225 learned top-75 K64 utility-weighted a4 codec summary SHA256: `5d7182dd86026c18d39a25c3dc30b368093a96db8e7c9f48d08cf20ede4d5a0b`
- Eval300 seed0 train225 learned top-75 K64 utility-weighted a4 codec results SHA256: `c622f33227ad4d8277dc28588c97f6fc8cedbe82b2a5835e8b5f2d4e1f531b75`
- Eval300 seed0 train225 learned top-75 K64 utility-weighted a4 codebook SHA256: `657e0e536e713f23330f50db30482d434acc800ebecfb1565707f7880c523caf`
- Eval300 seed0 val75 learned top-25 train-codebook K64 utility-weighted a4 codec summary SHA256: `2a131fc592bbff4bf3872a77ddcfad77818db51f26ad3ff84cc2576042376c79`
- Eval300 seed0 val75 learned top-25 train-codebook K64 utility-weighted a4 codec results SHA256: `16c2e802b62ab02ac4228f8c1b5f1709d33a9f78a3f6206bdc70353a36473586`
- Eval300 seed0 val75 learned top-25 train-codebook K64 utility-weighted a4 OCR split SHA256: `534fad11aa39a9bc57df05b4f710b493c04971c5982099ecf7bbc3da5a6e0e97`
- Eval300 seed0 val75 learned top-25 train-codebook K64 utility-weighted a4 PARSeq summary SHA256: `ef600ddb15bc25abead839a6cc81ded683c08cbe47d330b677d1d99508634ea4`
- Eval300 seed0 val75 learned top-25 train-codebook K64 utility-weighted a4 Tesseract PSM8 summary SHA256: `c860980ecd0c56c15616c372568b117c0ed500f8d83c1d290909bbc3e6cfd5b7`
- Eval300 seed1 train225 learned top-75 K64 utility-weighted a4 codec summary SHA256: `851a85355174fbaf0f9e28cbbe1829b04c711e409da7848ff3de106c2537f00d`
- Eval300 seed1 train225 learned top-75 K64 utility-weighted a4 codec results SHA256: `1e55d9982d58157c6079452a494d85811ea84e618b48697ea313d4a2b54963b2`
- Eval300 seed1 train225 learned top-75 K64 utility-weighted a4 codebook SHA256: `23b522c9b1b719fd9c15ade73f49fc28530c2deb5ee5fb3bacdc360368a3552e`
- Eval300 seed1 val75 learned top-25 train-codebook K64 utility-weighted a4 codec summary SHA256: `c1d7c6aab890efd5e2f4d0616c9a5d3a699f3492f4362f7a1951cd282ea687c5`
- Eval300 seed1 val75 learned top-25 train-codebook K64 utility-weighted a4 codec results SHA256: `6e3c1902372a50c3e5e401cf40a708cb302e424b01314af501033c3103c3097d`
- Eval300 seed1 val75 learned top-25 train-codebook K64 utility-weighted a4 OCR split SHA256: `15ba90e27c35a9becf30e0eb22a839e791eece7ec092b05e521c65c8ad1a864b`
- Eval300 seed1 val75 learned top-25 train-codebook K64 utility-weighted a4 PARSeq summary SHA256: `3a2af0c3794216db46d3a68d4c46df5aafda9af402f1b8bd538434137cbb28e3`
- Eval300 seed1 val75 learned top-25 train-codebook K64 utility-weighted a4 Tesseract PSM8 summary SHA256: `a21f3490f7823353f1a97df5ee062ec27d3034c668098df2c01d66be0f78c37a`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a4 codec summary SHA256: `bb5e9b5db6451e275b98c8fea41f38c11143e3490a407fb7cc1936bb662119b5`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a4 codec results SHA256: `3c102f61dc68904e23836bdfe49e588c11df755cec79fb484b0cc45f084a42c1`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a4 codebook SHA256: `a2baa625c1d555dc7a73c82076b19c44ac7ba5fb7f1a6ecea2235b6c5df21f7d`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a4 codec summary SHA256: `7268c6b2da16ba3b7436ecd98d07e22b5b3269d0c0a94931b915f5fce4462c75`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a4 codec results SHA256: `bea431b4c052d537532ca31f1b957715592dd8e3c56bcae83c2b64996b37de7c`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a4 OCR split SHA256: `db4c4cb642a621e282f854aeadf73bdd9e860c3b7e3cde25520c428a8d2f829e`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a4 PARSeq summary SHA256: `87794d43a3abd26cf64d2b492d4dbe1e043fc93a31cc2e34b9a8713c97356d7b`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a4 Tesseract PSM8 summary SHA256: `1bf79a83de5e9a37fb0e679c98c518c8448a59981159c9d02babc0d235ff45a1`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a16 codec summary SHA256: `1d10e916625f4d8c6de417398c11e9f35be6426c49ade788df00a7aef7987760`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a16 codec results SHA256: `56dc77da74bd60abed7296c9e47e420499575d4bbc2d6c3acac68a43b36214f9`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a16 codebook SHA256: `aea0dcd27c7a3b15161d8591da299ee0a3670b28ea6d63c82ee346593898bf66`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a16 codec summary SHA256: `6474ed62ca29f22ad891013d2779fa82d9d43a47cf01f3058e9fce0e6e76f135`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a16 codec results SHA256: `268916235f70c028129f493ad8901ead0981d832b683c623f390b0c59dea9122`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a16 OCR split SHA256: `8416875298cf5c7b7e559b449f16f5a4c7dca383392f073c9b38e5122b02379e`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a16 PARSeq summary SHA256: `d38357aa6eb70cd3180e1b2d8a79a278c34fa6e8665d2cf17b5c49edcfa841ac`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a16 Tesseract PSM8 summary SHA256: `71ed8de1679c2a45fd30755c341a349fb2aab8d213a9da1661bdfc3a8a99becb`
- Eval300 train-codebook K64 utility-weighted a4 three-seed aggregate SHA256: `09d3a8f59d76864dd056154c9e5f4b11c2dcd9f6fd1a711c2b33ee3f7eedf36d`
- Eval300 seed2 train-codebook K64 utility-weight alpha sweep analysis SHA256: `71dd0e9cd08b0d0f726b07721a4300c5b4532efc38b58727580a53b66ef365ac`
- Eval300 seed0 train225 non-harm selection SHA256: `04fecae1f20a231a769f4e1c3804122165873362329bfeca9a11700f4b8a1d80`
- Eval300 seed0 train225 non-harm selection summary SHA256: `a75b057de663a8b6ff1a5eea5d18650aacfd998367d54fadd6ff7a5ec4c8c00d`
- Eval300 seed0 train225 K64 non-harm codec summary SHA256: `49af862310556135782c5fa88fe94404159f525a026d980e25a0b28b1ed9d15c`
- Eval300 seed0 train225 K64 non-harm codec results SHA256: `d47669f83112bac9de893ea11844b4a1a9f89ce932409a9bb7e4463ea6050f7c`
- Eval300 seed0 train225 K64 non-harm codebook SHA256: `73ff1d22d6f74096536236f1f3fb0c84327bc7862fb9a49af8f3360d17497cc1`
- Eval300 seed0 val75 K64 non-harm train-codebook codec summary SHA256: `669d85baf3c78673cb80171265ea2a73298813636f4f0962b80a49b43d61937d`
- Eval300 seed0 val75 K64 non-harm train-codebook codec results SHA256: `69ab4751234d33420922c05ac964183df0a8f07e9ec24cb973b4745bf149016a`
- Eval300 seed0 val75 K64 non-harm train-codebook OCR split SHA256: `9f65ca577784034fb57be777370bf58b1aea2838da242fca88d3bc4f3468b3e1`
- Eval300 seed0 val75 K64 non-harm train-codebook PARSeq summary SHA256: `f5b5a0895c2246306647ef2c1c2b436820d4e570efc468cef470303b100c4879`
- Eval300 seed0 val75 K64 non-harm train-codebook Tesseract PSM8 summary SHA256: `a97624bb10c6dc04b9da5a19db249a404bbcded3b5d58fb9734adac227fec820`
- Eval300 seed1 train225 non-harm selection SHA256: `2cc29da87420e3683abea7a3d1dc4bef9c143593e1029c42bfe5eba600444d4b`
- Eval300 seed1 train225 non-harm selection summary SHA256: `c5f2ff0ef59a2e298f03fa34daba8657c8cbdc498650372d2f4bda5655dc4834`
- Eval300 seed1 train225 K64 non-harm codec summary SHA256: `a3ce85857678a096dd0fcc0208b38fdf00dbaaa22ef62ae4baded174b0497224`
- Eval300 seed1 train225 K64 non-harm codec results SHA256: `217bb121c8e3b624f8f0254987c0c2fd9b255c4501bcf55477de13e770144b2c`
- Eval300 seed1 train225 K64 non-harm codebook SHA256: `4dcccc9bbc7c5cbc876ecfd0c080e5c9b3bddc9796e3a607454dc156eae4d135`
- Eval300 seed1 val75 K64 non-harm train-codebook codec summary SHA256: `84f68b86a20d52e50aa7ab28eb33627461eb70fd5494f01891e065cd7d0f59e8`
- Eval300 seed1 val75 K64 non-harm train-codebook codec results SHA256: `7774119a1a2ca0323afaa7236065a48998f1afa35f0dddbfe38521da9139b1e5`
- Eval300 seed1 val75 K64 non-harm train-codebook OCR split SHA256: `661016811bc54dbd8ab7aab976e58aa7cb8ff7c97db7132683b67cc7860ca833`
- Eval300 seed1 val75 K64 non-harm train-codebook PARSeq summary SHA256: `24805f8ab3b7cc03f09a605c8d6d2141249f7e090526301cb5b38b412df530f6`
- Eval300 seed1 val75 K64 non-harm train-codebook Tesseract PSM8 summary SHA256: `1d81322909afb162811b10e1233d7ec61b99e40479abf9840b9b8ecc4928b80c`
- Eval300 seed2 train225 non-harm selection SHA256: `2022842cdc5d9524ffd42d41bde0a0804f36cd1aa712a319c16c002e60fe78c2`
- Eval300 seed2 train225 non-harm selection summary SHA256: `23de935c29251eb64d22855630e849bb7e2785b9bf46f51ad105f916fcf11a23`
- Eval300 seed2 train225 K64 non-harm codec summary SHA256: `b67c1070ef6b81b14ae89aae7286687ec73dbcfe8eb936ad6e02bde06b9a213d`
- Eval300 seed2 train225 K64 non-harm codec results SHA256: `c7ba06d8c38f73db7ec5cb680eee9af00cb3ab25bf5182eac14b5c1d24af39fc`
- Eval300 seed2 train225 K64 non-harm codebook SHA256: `11e1bc92050977d4ac5da3fe5df1d77158b63eaf72e47284cc69f7a795d80cf3`
- Eval300 seed2 val75 K64 non-harm train-codebook codec summary SHA256: `fdc11167d2ee289ced12dd84839fb2614393a328f0c57e03b1044f53d4ed4d31`
- Eval300 seed2 val75 K64 non-harm train-codebook codec results SHA256: `be9a60b02fbe8099c9d882428d1a547691f4a7c7e7ea12ac0f0c8731c57d437d`
- Eval300 seed2 val75 K64 non-harm train-codebook OCR split SHA256: `0eee47b8a5ecd2bd9c820b1c7f55c4d44aba1e01a8a2d3e9d8588b8d129072da`
- Eval300 seed2 val75 K64 non-harm train-codebook PARSeq summary SHA256: `66734e85577fe3eac397715208c81cd21198ad58975905e987bb04400c9e09be`
- Eval300 seed2 val75 K64 non-harm train-codebook Tesseract PSM8 summary SHA256: `2b389680376e9174ea250bfb114e3b59ea230d8dff2840983422159b57ff579d`
- Eval300 train-codebook K64 non-harm three-seed aggregate SHA256: `7d768814d41c69e090c0594a1e39180a0c205f3fc93671375573b67389de9a22`
- Eval300 seed2 utility-biased assignment diagnostic analysis SHA256: `2e8e9327d7d057d336e68064d8ad20185de2083c8901b21a2c7d5f6fbab92574`
- Eval300 seed2 train225 K64 utility-weighted a8 center-utility codec summary SHA256: `7e1ab0f1659302e84a7a0d68da321bf1c81d19e4cb4df4451c32c684e667c37b`
- Eval300 seed2 train225 K64 utility-weighted a8 center-utility codec results SHA256: `67e50f9d6168fb4b8487a08991b8e70937818be96474ca57f359ccabfe7defde`
- Eval300 seed2 train225 K64 utility-weighted a8 center-utility codebook SHA256: `32affaf13c4660e7c389dfff4e94fcae0fb4cdcd76664dd22b765533300c4f1a`
- Eval300 seed2 val75 K64 utility-weighted a8 center-utility assignment re1.05 codec summary SHA256: `fcc52d6e5bda383a8628d86b33118b8e5bf9d6c6cbabf91c776e9485d1520f98`
- Eval300 seed2 val75 K64 utility-weighted a8 center-utility assignment re1.05 codec results SHA256: `de6c7774a2c1710721e34a4ac16002b4eb35c8ff34991636167535c63d9b80a8`
- Eval300 seed2 val75 K64 utility-weighted a8 center-utility assignment re1.05 OCR split SHA256: `ca3fc5978c6b0593765d3d537ea58138c293155561bb419502b002d0c19999c7`
- Eval300 seed2 val75 K64 utility-weighted a8 center-utility assignment re1.05 PARSeq summary SHA256: `13749c830c3b2e6f5606d6cbe69417f9b715f7b4198ce6c1356c2660bf22fdfb`
- Eval300 seed2 val75 K64 utility-weighted a8 center-utility assignment re1.05 Tesseract PSM8 summary SHA256: `cc2417592ab50ed11848c9acbc7e8288e84404125f700ce0ed09c2606d0c6a64`
- Eval300 seed2 val75 K64 utility-weighted a8 center-utility assignment re1.01 codec summary SHA256: `409193d96778846a3f8f91bc4397247d0a181610f048255fb9fd3f36d1f4a139`
- Eval300 seed2 val75 K64 utility-weighted a8 center-utility assignment re1.01 codec results SHA256: `2ddd3784c6029a4897312a8655e7e3a8e6b40720009c5dc0bdbf0b55874c2e6d`
- Eval300 seed2 val75 K64 utility-weighted a8 center-utility assignment re1.01 OCR split SHA256: `6eaa9f2959eacc7e56f00152be1e098a3f6ef8bd5f8f68dfb60c8391b6e71b61`
- Eval300 seed2 val75 K64 utility-weighted a8 center-utility assignment re1.01 PARSeq summary SHA256: `ca3f99c45c28f11af0cb04a1ff3dbfe9708365017987c49f0c12c820ead3ac84`
- Eval300 seed2 val75 K64 utility-weighted a8 center-utility assignment re1.01 Tesseract PSM8 summary SHA256: `0453072bc1fc985a372e08e81e04d40a069c00ca87200ae89d7cd5981c8ba6fb`
- Eval300 assignment candidate generation script SHA256: `594ef874c0969bfef6f259d346153190a59a00d68d933b6625ae0c90480bf12f`
- Eval300 seed2 top-8 assignment candidate generation summary SHA256: `f6f6ad4fcbed44fd832b00c6085c796506104286894a42e2db92f7cdc33399b8`
- Eval300 seed2 top-8 assignment candidate OCR split SHA256: `b08f3c137d7832c540bafc4af3256a6e457655cce2abdc46b7f86aa516430e66`
- Eval300 seed2 top-8 assignment candidate PARSeq summary SHA256: `07fff7d0b3bfdbc3def8dd008d8e83867f875ad899ee1199b3073c89a1e6127d`
- Eval300 seed2 top-8 assignment candidate Tesseract PSM8 summary SHA256: `7b5a378fb59ec0a5ac76abbbedf0fb6a0593024f311a1536fab5706130d4158e`
- Eval300 seed2 top-8 assignment oracle selected-subset analysis SHA256: `ae13c19750efde0f80445f443e37d752570d5a5a730a6c41c003fe460f15cb05`
- Eval300 seed2 top-8 assignment oracle split summary SHA256: `91412e17e016914d1751d684cbf54def27337837a5baec5867d857002f23f968`
- Eval300 seed2 top-8 PARSeq-safe assignment oracle OCR split SHA256: `5822655bf2a3a92ccae50437eb857127a3c2988c51caf0c9b53d59c8b0b16586`
- Eval300 seed2 top-8 PARSeq-safe assignment oracle PARSeq summary SHA256: `9fb22c83fb14cb414b2f7fe39f831274a0acda854ab22f649e886953140531be`
- Eval300 seed2 top-8 PARSeq-safe assignment oracle Tesseract PSM8 summary SHA256: `2a582dbc3fad4ef7107842f926fe5283f621c0b528d7cc843bc9f9644e4b82c8`
- Eval300 seed2 top-8 PARSeq-safe assignment oracle full-val summary SHA256: `4b726d97cf90245f2456009bb209a0647fdce74ba2f511eb446793554badba17`
- Eval300 seed0 top-8 assignment candidate generation summary SHA256: `11015b716d362acee12d094e61f6a9c1ba651fc19025042d71dfc224f301f0c4`
- Eval300 seed0 top-8 assignment candidate OCR split SHA256: `ec850e2d631a15008f0a4be173e6b8866707a87ba635955e71eb2ff508fd0410`
- Eval300 seed0 top-8 assignment candidate PARSeq summary SHA256: `5df2c61b91733dc1280fa79d87e2f50cb8149251bc9554f52f0f7488fa555f0d`
- Eval300 seed0 top-8 assignment candidate Tesseract PSM8 summary SHA256: `948a7f9f47c3cf29981cf0351ed6ae29079031f581ecbbe0346c3ff79d3addd5`
- Eval300 seed0 top-8 assignment oracle selected-subset analysis SHA256: `3a5700aed851e17c3670bb46b0277fc518f49d5ab8f576ec3f1c6b327197e6ce`
- Eval300 seed0 top-8 assignment oracle split summary SHA256: `c7def9b87128cb2fd73ddf0c526b929ab9eb88fda8102511e96d42aa5641568c`
- Eval300 seed0 top-8 PARSeq-safe assignment oracle OCR split SHA256: `4a0b13bf37301c92f8c91e1e44a942a141e7d231fa3271b3cb19d3a4d4bcc0ee`
- Eval300 seed0 top-8 PARSeq-safe assignment oracle PARSeq summary SHA256: `b75fb61672ab7adaacd6598df27c54824bdbb9bfb523d39618f3d67cfca538f3`
- Eval300 seed0 top-8 PARSeq-safe assignment oracle Tesseract PSM8 summary SHA256: `f2a3f9bc006830b83d63ea729e7313357c670228c7a5dff083bbbd0a531c3fd1`
- Eval300 seed1 top-8 assignment candidate generation summary SHA256: `7618421dfd108a141713944a9dcdc788692cc97de62e52277b427d2a23fcca9e`
- Eval300 seed1 top-8 assignment candidate OCR split SHA256: `2a54efcc2cd41447ee3720786594d488321a7f880d02a064bed21ad2e3d55efb`
- Eval300 seed1 top-8 assignment candidate PARSeq summary SHA256: `6a5d29f64d3e22fafa239edd645c7ec496323e609ea2cf09334c6437062f8edf`
- Eval300 seed1 top-8 assignment candidate Tesseract PSM8 summary SHA256: `613da1559fad0a00eba48d245ad88617d2802ba90b2b37d91314c82b1c7a24a4`
- Eval300 seed1 top-8 assignment oracle selected-subset analysis SHA256: `c19f7235ee73ed52410b71aa553be0f8cf1844a07627cb44f8943ea4f84cf4e1`
- Eval300 seed1 top-8 assignment oracle split summary SHA256: `78841a30ab6f09773a3d7f9a4e98728adf8daa11f7807ccb8479cf0d706575e8`
- Eval300 seed1 top-8 PARSeq-safe assignment oracle OCR split SHA256: `b9907909c6b98b7983c981ceaf7d970a0bdd2ca6ecc7de3518ae3eaade293321`
- Eval300 seed1 top-8 PARSeq-safe assignment oracle PARSeq summary SHA256: `ede7ba475a8b29003ed6da89fbfb20be266ad8bfa59f735ca9e3d5c26f544676`
- Eval300 seed1 top-8 PARSeq-safe assignment oracle Tesseract PSM8 summary SHA256: `41c97356a1575bcfc61a93c506016d8060813b172abe552e3a74302866e797e1`
- Eval300 top-8 assignment oracle three-seed aggregate SHA256: `682c0131290982ed817fa182da7ef6ffaadfe93e11b633be0720de3b7c9595ee`
- Eval300 assignment utility table builder script SHA256: `a80a624b9d4abcfc0ec21d5985d3e30143c49df6418b37adedb4a4176e3d7c5b`
- Eval300 val75 assignment utility table SHA256: `0c79a4d7d36c22c7ec00533bca67ef029417c682502bb9c46cf4ca224bcbc2d3`
- Eval300 val75 assignment utility table summary SHA256: `148f6ac196589b3b1ed98c7113e447a23545a4ac9d074d1fa6a3b4c8e504408e`
- Eval300 assignment policy diagnostic script SHA256: `c006a5a7d63ff5c09c219cb20d1e5d47aa5eefda30e223c7ca5e79abd49a6e14`
- Eval300 val75 assignment policy diagnostics SHA256: `fe3dfdf233278fa87d76068ccf2da16c0805478d0196b3546a72849e3ab3aa58`
- Eval300 val75 assignment patch-ranker negative diagnostic SHA256: `36b1d85a3453372faac5e4edf299138b65a80aab4a5fff740f31adba4e231ce7`
- Eval300 train225 seed0/1/2 top-8 assignment candidate OCR splits SHA256: `d8a0704c1340cf2c9e5ce0717ec5dcac5d5711fcf683b88b1b5e68428845aa6a`, `ff1f497dde13e9d5498bb245eb99c57e930e2d2b43e3ad6c0376e44f989b3dae`, `fd95307bfef368ed89205de7a9177dad099a8023d38566ab2f2ffe24f49bc2ba`
- Eval300 train225 seed0/1/2 top-8 assignment candidate generation summaries SHA256: `dc15332544f5adce11ae810d8b458cea2724507902fad743ab7c9e80f21bf7bc`, `06bbca3d591ccfa68fa4f86ce771bc05fc61b0cc3d800531a48787b69b53d9f1`, `8cbff7656c79e77e782fc84ba14fe504f4f00f13a223e7797dfcca76c8be53e3`
- Eval300 train225 top-75 assignment utility table SHA256: `177f91b5e3d0fc0813d11e1a782828bffa98131b4f444f4ba62ef42ae80969b7`
- Eval300 train225 top-75 assignment utility table summary SHA256: `3d6b8113a2a1656f40b30d7122ba051f8d4e7d06e00bb39f90ca7cf9db82b048`
- Eval300 train225 top-75 assignment policy diagnostics SHA256: `6b3ac6d23f0f68be68580052f7642f6fe141aac76a6f7d17b0f4dde00f17266d`
- Eval300 assignment train/val table maker script SHA256: `0652ed3b7314ce6dff335b673c0c410df1543e26a1e55479c0c01dfea838f913`
- Eval300 assignment patch-ranker script SHA256: `0d747f1b49325de428ad6f3ab9049b2a2d71d6358b55e84e76c60bb9118f499c`
- Eval300 assignment ranker aggregate script SHA256: `92fa89d4b29e0b484d8881bf3d181752fac2c1072fff97fbb52f12a33407d94f`
- Eval300 seed0/1/2 assignment train->val tables SHA256: `3841052518128b69a5daa87e373bbda6b6cd3f090016563593da8847cf6805b0`, `9e9df2fd7d1f271b4552b2d96ad9b974d63f8d70b69bc1fafc9cd95705c829ae`, `5a60ab5aabaadd8a7ee9d41ec527b3e1ad1290c9beee671af0e27cf593a8f5bf`
- Eval300 seed0/1/2 assignment train225->val75 patch-ranker results SHA256: `623038642a1991befdad715fc2c9ddf7176adc42c87df8dc99533b6494cd8c2a`, `6ab8588bfe4beb2b63abd262f041781e775569671d982e59f9b11e8f87b72aa7`, `032a8fbe5de75a10ec51ee7ad1d36aea9e54df3d275b9304e840c63861240739`
- Eval300 assignment train225->val75 patch-ranker three-seed aggregate SHA256: `b457b485e9d970946e37616b16d3f76b3c98ddda510dd2e583f5e086932aa00e`
- Eval300 assignment codebook-feature augmentation script SHA256: `7b679782b3552360827f378ae7b00f36908c3ba36bac46918cc773931d0bcc40`
- Eval300 codebook-geometry assignment patch-ranker script SHA256: `9affe5777e05e74362c14c344745462edc47c1b09f7b8314d8fa171955c17958`
- Eval300 seed0/1/2 codebook-geometry train->val tables SHA256: `80a4bb4938a9f1591bbc5fb7de0d1cc03a1e4953168188750475f5842d958459`, `0c1c0fb712da9623e20309aeb845b073c4b624219f036eab248a37f6f1bd8c2e`, `0730edc59650865fffce0bcf5a78f1c08df4efb74ca5008d910fa4e2ea47030b`
- Eval300 seed0/1/2 codebook-geometry train225->val75 patch-ranker results SHA256: `41d99058091415243b68946cf99bf49ff65af64f2c0911ac7e53f444ded72125`, `7d5a3376eeb9a4af2c8a1f1abac5e450df7be3a4626ce4bc1039ec222b416f88`, `c7c7fa984e758225291f46ba73221e9263dfb5faa543362f3a308485fbdb5000`
- Eval300 codebook-geometry train225->val75 patch-ranker three-seed aggregate SHA256: `f57937c4c392882da70682d1e18e0cde2ba31298e95695630241226cb9d5ea3b`
- Eval300 utility-weighted a8 vs residual-MSE K64 Tesseract diff analysis SHA256: `6d1ebe4e626ab819bf77717cfe7a050e066ce84e7675b86c203601e382b76f4a`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a8 zero-code codec summary SHA256: `4ca3c1cb10903b033e7cfe2d0b2945135aceabfb6e6487b71eb4ead611623eca`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a8 zero-code codec results SHA256: `1a48e47569e6eaed8f7f142e20e0c36c082399e5fb1b00fb0cf444529c8bec46`
- Eval300 seed2 train225 learned top-75 K64 utility-weighted a8 zero-code codebook SHA256: `2abc6b08401934b57564d001c0e1e6e5348a2ab9a87931d25d99ec5f4529976e`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a8 zero-code codec summary SHA256: `6284bfe21757a3a0eda05dcfc8884936ed07cd2a9301d15ebceab488b81aad18`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a8 zero-code codec results SHA256: `8cb9422e013915135acbbf58700128426b64dca75bace6d0d74dbd156140e313`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a8 zero-code OCR split SHA256: `05ec9a226a7865fbdfe7d686176de0d0d45892b5bcaa21ea165e2cc470370b5b`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a8 zero-code PARSeq summary SHA256: `136eed5e703ac90b744054f7a96410bc07264697772aefa9a2c791494dfe7e7d`
- Eval300 seed2 val75 learned top-25 train-codebook K64 utility-weighted a8 zero-code Tesseract PSM8 summary SHA256: `a595e9c3833050e908be9255623a3c71ac34a87c730b2a9ea923875f187f4e8a`
- Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 0.8 codec summary SHA256: `40e6271708153f6afa2407765d710a8e22667b1d852cd96e135c2eb7b591224b`
- Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 0.8 codec results SHA256: `652d04178c7556978dfd5a74d25a3bebbb47174c017817e2370237a333559594`
- Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 0.8 OCR split SHA256: `fbdfd371ce6204defd74bc3a54209b7a173057d963da7e37d54f7c2f9b33cf92`
- Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 0.8 PARSeq summary SHA256: `a26e9a494077aa39c72c6ce909015c7e1ae4fffef1dcf4051aea8e7aabf1ee03`
- Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 0.8 Tesseract PSM8 summary SHA256: `2a195cae8f2381ef260f5c5b08c9e428f1ea2244121c0519d0384bba8ba12143`
- Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 1.0 codec summary SHA256: `67c04fe6a0f658bd85edbfe045a7711e25cd144140806280643d213a0b213e91`
- Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 1.0 codec results SHA256: `5a177835a6fb6137fd50d6ade7e4d5d223dd506c1e5008493b4388b9a0894a2b`
- Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 1.0 OCR split SHA256: `782ed835886abece63d2cbd57975a2a10efd598cf2c871fb1938e3ad904b47d2`
- Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 1.0 PARSeq summary SHA256: `5c207b8356bfe180c25bdaf52e729e037ab39cb2d187e699e98ac28d3a3959bf`
- Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 1.0 Tesseract PSM8 summary SHA256: `e251ab2195bf9e2c8a5df3782e7c958bb65f7c53c983203baeeb159d14de0b38`
- Eval300 oracle-selected K16 codec summary SHA256: `656f3eec13b9dec787eaebe77b66ad88f1ee5c435f46234885fdd52dbfdb94f8`
- Eval300 oracle-selected K16 codec results SHA256: `d8cc4d77e9bef1754b5b6ca8975652fdc2c499fbf02789f5f59cd05b7f403339`
- Eval300 oracle-selected K16 codebook SHA256: `cb0e5a030b2c2aae563e401c24f984a8300cf9863ea5ba0281a413e0e7007c31`
- Eval300 oracle-selected K16 OCR split SHA256: `b7568c2c340906714f4ac9def89dc79fc2972db53cc33937673868b9d1dfee38`
- Eval300 oracle-selected K16 PARSeq summary SHA256: `f10cdc217bd2885af4931cbd1ee5615affd57b7a821431fa6897d8ed03c0f21e`
- Eval300 oracle-selected K16 Tesseract summary SHA256: `0e767ed70b4b1f8c4b80eb9d75b2a141cec107d85d4809908d2f18c0b99c15eb`
- Eval300 oracle-selected K64 codec summary SHA256: `ed8b386eacf5067ae9e81bd90190f0ce8f833aaa3d798c2cc2e69fb7a49eb8d5`
- Eval300 oracle-selected K64 codec results SHA256: `54ce03a15382e8ac24dfeab0e208ea312c6892397259ff194cc2fd798d07c3ca`
- Eval300 oracle-selected K64 codebook SHA256: `5962eeea1ee26b33d2f3082fbfc12b31eab793bf465146dda2b5df9ce9bcf050`
- Eval300 oracle-selected K64 OCR split SHA256: `b5a3911f67d0fe0a27ceb80b07db3c017ff7bbf89c2498507a23812a73e1bfd2`
- Eval300 oracle-selected K64 PARSeq summary SHA256: `c1c859d658ccc16590d6b6a36e2f95c8ce44d81bbb146064804d7fb6aa1bf342`
- Eval300 oracle-selected K64 Tesseract summary SHA256: `356f141018c59829d1087c11f3467780fcdd3617d9598d209779cfb628c47eb0`
- Eval300 oracle-selected K80 codec summary SHA256: `150a59f1dd809b52df7e280d78b1640b3d913dcb80d7b832d18567ef74f09135`
- Eval300 oracle-selected K80 codec results SHA256: `9a6a1b4afe0ade626283068546c1fd4d1374681061ee80d3944358a3f466ea56`
- Eval300 oracle-selected K80 codebook SHA256: `68dd909e681cc537f6bea3a157b80ca8ac3b3a73b4e1ae1419476bfdf47ee051`
- Eval300 oracle-selected K80 OCR split SHA256: `39f3c41828629222dafa2e1c482ca75ee8aec1e983f9fe1c0ae9b5d4fff8a4f6`
- Eval300 oracle-selected K80 PARSeq summary SHA256: `edf08372594400eeb2352834d0bbb2eb0a82e1b4f2f27a40e21795997b731165`
- Eval300 oracle-selected K80 Tesseract summary SHA256: `890bcc1a4617919030bce17afd3462807fe0d0b0b22b6aa477cf1b1ce506ded1`
- Explicit code-index assignment stream script SHA256: `8fa2aba3f88a9ced08fa383ff71d46e2c28859031c46089258b3cd92d3f33ba5`
- Assignment-policy export script SHA256: `92d9d62075b83a2e8c87419ff863e700a50026796aaba165a3725ba7989392b5`
- OCR pair-comparison script SHA256: `37efd8593f1b7169efc7f63af3d934dfa5abaa68deb36ed83351b0718a15491a`
- Assignment ranker script SHA256: `d270c704112c27802bb8de530ec330b6cfe3adaa67e9873a348f06fa017efff5`
- Eval300 learned oracle-choice actual Tesseract comparison SHA256: `7f0da077b2cc2046aa659d97785ddbcddf775a7d570d5a37f4730d181633ec4e`
- Eval300 learned oracle-choice actual PARSeq comparison SHA256: `541026e84dd3c1b9fecccbdc59b2d8aef5a9dbb6e4f01f4d0257285e6fd35f4e`
- Eval300 top-8 assignment oracle actual Tesseract comparison SHA256: `734a6e4fec4b9d4d13a2914efbd922e4815e77323365caccad45afe165ded54e`
- Eval300 ensemble learned assignment actual Tesseract comparison SHA256: `3c466801e039f642338ae86e00926fd32f15725962bbe59c155a90f41fc5d750`
- Eval300 combined learned assignment actual Tesseract comparison SHA256: `1bd305f68372eaaf8c2f161c255055f564b6939fc9f35d62a2ddf0366c8b23d7`

### Actual Learned Assignment Through Counted `.oscr`

The oracle-choice assignment ranker was promoted from table-only policy sums to real counted `.oscr` streams. `scripts/run_mlicpp_latent_codebook_stub.py` now supports `--assignment-mode explicit`, so the exported policy writes the chosen code index for each already-selected candidate. This keeps candidate count and code-index byte length fixed relative to nearest assignment; measured actual bpp is therefore identical to the nearest K64 utility-weighted a8 baseline for each held-out split.

| Policy | Val samples | Actual bpp | Enhancement bpp | PARSeq delta | Tesseract unicode char delta | Tesseract latin-alnum char delta |
|---|---:|---:|---:|---:|---:|---:|
| Nearest K64 utility-weighted a8 | 225 | 3.8496 | 0.0290 | 0 | 0 | 0 |
| Learned oracle-choice model0 | 225 | 3.8496 | 0.0290 | 0 | -3 | -5 |
| Learned oracle-choice model-seed ensemble | 225 | 3.8496 | 0.0290 | 0 | -3 | -5 |
| Combined seed0/1/2 ensemble | 225 | 3.8496 | 0.0290 | 0 | -3 | -4 |
| Top-8 PARSeq-safe assignment oracle | 225 | 3.8496 | 0.0290 | 0 | -24 | -19 |

Per-split actual bpp for the learned model0 policy is `3.755801`, `3.957302`, and `3.835844` for seeds 0/1/2, exactly matching nearest assignment. PARSeq OCR is unchanged exactly in raw outputs. Tesseract improves modestly but consistently enough to treat this as the first actual-bitstream learned assignment gain at unchanged payload size.

The result also exposes the next bottleneck. The actual oracle has much larger headroom under the same payload size, while model-seed ensembling does not improve the model0 policy and pooled seed training slightly weakens it. The next assignment experiment should use a group/listwise no-op-aware objective and better calibration, not another rowwise BCE ensemble.

### Group/Listwise Assignment Loss Diagnostic

`scripts/train_assignment_patch_ranker.py` now includes a group/listwise assignment loss (`--loss-mode listwise` or `bce_listwise`) and a train-tuned `top_changed` budget. It fixes the grouped pairwise key to include `real_seed`, which matters for pooled seed tables. The goal was to move from rowwise BCE toward a direct "which code index should this selected candidate use?" objective.

Current performance does not justify actual-bitstream promotion:

| Diagnostic | Seed0 Tesseract delta | Seed1 Tesseract delta | Seed2 Tesseract delta | Note |
|---|---:|---:|---:|---|
| Pure listwise always-best | 0 | -2 | 0 | Ties/loses to existing BCE oracle-choice baseline |
| Pure listwise top-changed-3 | 0 | -3 | +1 | Seed1 improves, seed2 cancels it |
| BCE with train-tuned top-changed | 0 | -2 | 0 | Calibration does not beat always-best baseline |

The implementation is still worth keeping because it is the right interface for the next assignment model, but the current features/model are the bottleneck. Loss shaping alone is not enough; the next attempt should add stronger candidate-conditioned latent/codebook features or a genuinely group-structured scorer with no-worse calibration.

Result SHAs:

- Listwise seed0/1/2 model0: `c342de46406c8d10a69ec1571258106edbc103e31824e5345bd424cd75320aae`, `ca14d092605b0c54aea8fbfcf629c1e94ea31277e4056b50db94538c76bfe0cc`, `acbe4264f6395dcbbba5a61dfbf6642edb42a708cdbf1bf17916064e8dd932f6`
- BCE-calibrated seed0/1/2 model0: `33b8e46649df1a7e7e056fa9de863038c8026f91b16c4e42309912f8c4fe7019`, `8a46eef5f47fe6fa7aae9d77168cf9ae30f7219ec4c75ed5048157d8ab15d81e`, `ad14f8d56f167b6325a9fdd18e6816126275684cd29a88fb4b5829cca0ede31d`
- Updated assignment ranker script: `fd1bb87c06086a67dc14a6a6602a7230948211b7849ce22593906bb7a9f454d1`

### Assignment Headroom Strata

The top-8 assignment oracle headroom was stratified over the 75 selected-candidate groups in the three val splits. This analysis uses selected groups rather than all 225 images because assignment can only change candidates that the learned gate already selected.

| Stratum | Groups | Oracle-changed groups | Tesseract delta |
|---|---:|---:|---:|
| Overall selected groups | 75 | 17 | -24 |
| Real seed 0 | 25 | 0 | 0 |
| Real seed 1 | 25 | 6 | -10 |
| Real seed 2 | 25 | 11 | -14 |
| IAM words | 36 | 10 | -14 |
| ICDAR2013 | 8 | 3 | -4 |
| ICDAR2015 | 9 | 3 | -5 |
| TextOCR | 9 | 0 | 0 |
| Nearest Tesseract distance 0 | 16 | 0 | 0 |
| Nearest Tesseract distance 1 | 14 | 1 | -1 |
| Nearest Tesseract distance 2 | 10 | 3 | -4 |
| Nearest Tesseract distance >=3 | 35 | 13 | -19 |

The most actionable finding is that headroom is not uniform. It concentrates in already-difficult Tesseract cases and in IAM/ICDAR-style word crops; no selected group with correct nearest Tesseract output benefits from assignment. The next assignment model should therefore expose OCR-difficulty/proxy features or domain-conditioned calibration, while still preserving the no-op behavior for easy groups.

Strata result SHA: `37d849c14a18709e2c67eaccd4e0e04a0deec2ace51068db930a29e369e9287b`. Script SHA: `afd3e110ab18f9c7678cfe56625ce474a311983aba388772020670cff7519615`.

As a sanity check, source/reference-length/OCR-distance metadata features were added to the assignment ranker on the unstable seed2 split. This is a diagnostic-only upper-control because OCR-distance features come from the generated assignment table. It still fails: CNN+metadata gives Tesseract delta `+2`, and tabular-only metadata gives `+1`, versus nearest `0` and oracle `-14`. This suggests that difficulty/domain is where headroom lives, but simply telling the ranker "this is a hard word" is not enough. The model must distinguish which code center changes the local glyph in the right direction.

Metadata diagnostic SHAs: `f5c065c1bea258fca7eb314a01b8560e5fad275b3385f9176649c86c2a016af9`, `6ea942cca3e2f988150a33507abfc69fdeff6a4fe50b8a82394cb5359839de15`. Updated ranker SHA: `144c822ba8f51587a6fecbf9d0f25643240e46f35d99cb2863b36a6f912f599d`.

A local diff-crop input mode was also tested on seed2. Instead of resizing the whole word image, it crops each group to the union of code-variant pixel differences and feeds only that local nearest/variant/diff tensor to the CNN. This improves neither the policy nor the main conclusion: val AUC/AP is `0.816/0.445`, but always-best and train-tuned top-changed both give Tesseract delta `0`. Localizing the bitmap effect is not enough; the assignment head still needs a better representation of code-center direction and glyph effect.

Diff-crop result SHA: `235859fcd5114412011b50ab992c9aa7d99ce27b9542abdbffe11ebf9c56f3cf`. Updated ranker SHA: `c87097829845bf51d7d34ceca5b93f657fc2ca8597e5bcf27439f03330701e03`.

Combining diff-crop with the existing latent-vector statistics table also fails on seed2: val AUC/AP drops to `0.783/0.388`, and Tesseract delta remains `0`. Result SHA: `e4c9ce03c51acddc5b3e44779f9a6248af6f41919fecea04d732f8fba7adc8b7`.

### Group-Gated Actual Assignment Promotion

The weak-positive group-level gate over fixed model0 assignment scores was promoted from table diagnostics to real counted `.oscr` streams. The exported policy uses threshold `0.5`: if the group gate rejects a selected candidate, it keeps the nearest code index; if accepted, it uses the model0 oracle-choice assignment score. This keeps the same selected candidates and the same code-index payload length as nearest/model0 assignment, so any OCR change comes from the reconstructed image, not from a rate change.

| Policy | Val samples | Actual bpp | Enhancement bpp | PARSeq unicode delta | Tesseract unicode delta | Tesseract latin-alnum delta |
|---|---:|---:|---:|---:|---:|---:|
| Nearest K64 utility-weighted assignment | 225 | 3.849649 | 0.028989 | 0 | 0 | 0 |
| Previous learned model0 assignment | 225 | 3.849649 | 0.028989 | 0 | -3 | -5 |
| Group-gated model0, threshold 0.5 | 225 | 3.849649 | 0.028989 | 0 | -5 | -7 |
| Pooled image-diff group gate, model1 threshold 0.5 | 225 | 3.849649 | 0.028989 | 0 | -8 | -8 |
| Top-8 PARSeq-safe assignment oracle | 225 | 3.849649 | 0.028989 | 0 | -24 | -19 |

Per-split group-gated actual bpp is `3.755801`, `3.957302`, and `3.835844` for seed0/1/2, exactly matching the nearest and previous model0 assignment streams. The policy changes 1/25, 4/25 and 10/25 selected groups in those splits. PARSeq outputs are unchanged exactly for all 225 samples. Tesseract improves by 5 unicode edit characters and 7 latin-alnum edit characters versus nearest, and by an additional 2/2 characters versus the previous actual model0 baseline. The model0-vs-groupgate comparison has one improved and zero worsened Tesseract samples, making this the new best learned actual-bitstream assignment baseline.

This is still not a final assignment model. It recovers only about 20.8% of the unicode oracle headroom and 36.8% of the latin oracle headroom, and the fixed threshold should be cross-fitted before being used as a paper claim. But it is a clean engineering result: a deployable group-level no-op gate can improve code-index assignment under identical transmitted payload size.

A coarse threshold sweep was added after the actual promotion. Thresholds `0.1` through `0.9` all keep the same table-level aggregate Tesseract gain (`-5` unicode edit characters) and zero PARSeq delta, but threshold `0.8`/`0.9` reduces changed selected groups from 15 to 14 relative to threshold `0.5`. The only threshold-0.8 vs 0.5 policy difference is one seed0 selected group whose OCR delta is zero. Seed0 threshold-0.8 was therefore regenerated through actual `.oscr` compress/decompress and rerun with PARSeq/Tesseract; OCR is identical to threshold `0.5`, actual bpp remains `3.755801`, and assignment changed fraction drops from `0.04` to `0.0` on seed0. The three-split threshold-0.8 comparison reuses identical seed1/2 outputs and matches the threshold-0.5 OCR aggregate. This makes threshold `0.8` a slightly more conservative equivalent baseline, but it does not close any additional oracle headroom.

A latent-vector feature variant of the same group gate was also tested using the existing `codegeom_latentvec` assignment tables. This is negative. It overfits the train groups perfectly but lowers the usable validation/policy behavior: the best threshold family reaches only `-1` Tesseract edit character over three splits, and seed2 becomes harmful at most thresholds. This rules out simply adding hand-built latent-vector statistics as the deployable difficulty/no-op proxy. The next representation should be more structured, such as a small local latent-token encoder or joint scorer over nearest/variant/code-center direction, not wider tabular latent descriptors.

Score-only confidence gates were tested as an even simpler no-op mechanism. Two gates were evaluated from the assignment scorer outputs: best non-nearest score minus nearest score, and top-score margin over the runner-up. Both are negative. They can reduce the number of changed groups, but they do not improve the actual target: non-nearest margin either stays near the original always-best policy or removes useful changes, while top-margin at best ties the older always-best `-3` Tesseract edit-character behavior and remains below the promoted group-gated `-5`. The assignment scorer's own confidence is therefore not a sufficient no-op proxy.

A linear-only group gate was tested as a simplicity and overfit control. It lowers train overfit compared with wider overfit-prone variants, but the best threshold family reaches only `-3` Tesseract edit characters with zero PARSeq delta across the three val splits. This is below the promoted MLP group gate at `-5`, so the next model should keep a small nonlinear group head and move the simplicity pressure to better representation/calibration rather than collapsing the no-op decision to a linear score.

A pooled seed0/1/2 group gate was then trained on the combined train-side groups and evaluated on the combined val-side groups. This raises the mean validation AUC to `0.620`, but policy-level behavior does not improve: train-tuned and threshold-0.5 pooled gates stay at `-3` Tesseract edit characters with zero PARSeq delta, below the promoted split-specific group gate at `-5`. One model seed at threshold `0.1` ties `-5` on the table, but the aggregate is unstable and still has false changes. This is a useful calibration warning: assignment gates should be selected by policy-level OCR deltas and false-change counts, not row/group AUC alone.

The first positive follow-up is a lightweight reconstructed-image difference feature set (`img_*`) for the group gate. It adds source/nearest/variant grayscale statistics and variant-vs-nearest/source diff/bbox statistics to the assignment rows, then trains the same small group gate with `--include-image-features`. The split-specific image-diff gate overfits and is negative, but the pooled seed0/1/2 image-diff gate is positive. Model seed 1 at threshold `0.5` changes only 5/75 selected assignments on the table and gives `-8` Tesseract edit characters with zero PARSeq delta.

This policy was promoted through actual counted `.oscr` streams. The selected candidates and code-index payload length are unchanged, so actual bpp remains exactly the nearest/groupgate value on each split (`3.755801`, `3.957302`, `3.835844`; mean `3.849649`, enhancement mean `0.028989`). PARSeq outputs are unchanged exactly relative to both nearest and the previous groupgate baseline. Tesseract improves by `-8` unicode and `-8` latin-alnum edit characters versus nearest, and by `-3` unicode / `-1` latin-alnum versus the previous promoted groupgate with zero worsened samples in the groupgate comparison. This recovers about 33.3% of unicode and 42.1% of latin top-8 oracle headroom at unchanged transmitted payload size.

Explicit policy-error analysis explains why the image-diff gate is better. The previous promoted group gate changes 15/75 selected groups, hits the exact oracle in 4/17 oracle-change groups, but still makes 8 false changes and 3 wrong non-nearest changes, for `-5` Tesseract unicode edits. The pooled image-diff gate changes only 5/75 groups, still hits the same 4 oracle changes, and reduces the bad changes to 1 false change and 0 wrong changes, for `-8` Tesseract unicode edits. The gain is therefore not from discovering many new oracle cases yet; it is mostly from suppressing harmful/no-headroom code choices while preserving the useful changes. The remaining bottleneck is coverage: 13/17 oracle-change groups are still missed, dominated by hard IAM/high-distance cases.

A second pooled image-diff gate seed was promoted as a robustness check. Model seed 2 at threshold `0.8` changes 7/75 groups on the table and has the same table-level `-8` Tesseract / `0` PARSeq delta. Real counted `.oscr` regeneration and PARSeq/Tesseract OCR confirm that its aggregate OCR is exactly tied with the model1/threshold-0.5 promoted result: nearest-vs-candidate is again Tesseract `-8` unicode / `-8` latin-alnum and PARSeq `0`, while model1-vs-model2 comparisons are exactly `0` for both OCR families. The policy-error mix is slightly less clean than model1 (2 false changes and 1 wrong change instead of 1 false and 0 wrong), so model1/threshold `0.5` remains the preferred baseline, but model2/threshold `0.8` verifies that the image-diff result is not a single lucky gate seed. An overlap check shows model1's five non-nearest choices are a strict same-code subset of model2's seven non-nearest choices; model2 only adds two non-useful changes.

A scorer/gate responsibility analysis was added for the preferred model1/threshold-0.5 policy. Among the 17 oracle-change groups, the assignment scorer puts the oracle code at rank 1 for only 4 groups; those are exactly the 4 groups recovered by the promoted policy. All 13 missed-oracle groups are scorer-ranking failures, not cases where the group gate rejected an oracle-best code. On those missed groups, blindly taking the scorer's best code would sum to `+2` Tesseract edits, while the oracle headroom is `-15`, so the image-diff gate is doing the right thing by rejecting current bad scorer choices. The next performance jump must therefore improve the candidate-conditioned code-effect scorer/ranker, not just loosen the no-op gate.

As a direct scorer-side control, the existing assignment ranker was extended with `--include-image-features` so the same `img_*` statistics could be used as row features. Two lightweight tabular-only scorer variants were trained on the pooled image-diff table: listwise CE and BCE+listwise. Both are negative. They produce high validation AUC/AP (`0.895/0.686` for listwise and `0.908/0.706` for BCE+listwise), but policy-level Tesseract gains are only around `-1.33` and `-0.33` characters, far below the promoted image-diff gate. Score-coverage diagnostics show why: on the 17 oracle-change groups, these tabular scorers rank the oracle code first for only 1 group, worse than the original codegeom scorer's 4/17. Hand-built image-diff statistics are useful as a no-op/code-effect gate, but not sufficient as the assignment scorer representation.

A non-deployable scorer top-M oracle upper-control was added to decide whether this is a candidate-generation or final-reranking problem. With the original codegeom scorer, selecting the oracle whenever it lies within score top-1/top-2/top-4 recovers `-9`, `-11` and `-15` Tesseract characters respectively, compared with the full top-8 oracle at `-24`. The tabular image-diff scorer is bad at rank 1 (`-2`) but places enough oracle codes in top 4 to reach `-16` under the same upper-control. This suggests a two-stage assignment head is plausible: use a conservative scorer to form a top-4 shortlist, then apply a stronger local glyph/code-effect reranker plus the existing no-op gate.

A first top-4 shortlist reranker was tried by filtering the assignment table to the codegeom scorer's top-4 rows per group, forcing the nearest code to remain available, and training the existing diff-crop CNN with image features. This is also negative. The filtered table keeps 73/80 train+val oracle-change groups and the val shortlist oracle is `-15`, but the learned reranker reaches high val AUC/AP (`0.925/0.781`) while still failing policy-level selection: mean Tesseract delta is only `-0.33`, and score coverage shows 0/17 val oracle-change groups ranked at 1. The lesson is that the next reranker needs a different representation/objective, not just a smaller candidate set with the same diff-crop CNN.

Result SHAs:

- Explicit group-gate policy JSONL seed0/1/2: `e134916ed828f367887a8d5a1f9df79bcde0842a9a194bd5990a50aba10e6077`, `7bfe9eca72049b0776a3c2bc44570423b80f7b1cacfad39246f9d1dc09ae3e9a`, `880cd0cb7f5b7eccd25d35b276131184e8c101526ca704f6bc9480b34f3ea803`
- Policy summary seed0/1/2: `2735756965938000404055484f3f2c4efdf64c1a49de4b38fcaaf6e02478404a`, `9406f7ae0c38ad68f26fa618b7565bda52b39df4046bc237bfaf8987aa10e79f`, `bf77d87908babd8b2c949bf050b09843d2ac1294b55f686c1a98d36a1edaf91a`
- Codec summary seed0/1/2: `f1c4743f134ffed89696f81fb6cd72d932bf10232095205d2b4d9da20769c6c7`, `15e64e32dfa9aee722d5e0d7412784bf5faea595c777e7ceee7cd657312ede04`, `3ec4f1b5ee8083bf3dd7a92dc8f3fa8133246b2b29b2acbf38c5b3b10af315b6`
- Nearest-vs-groupgate comparison: Tesseract `e553f91b9c604bf3590a85306a9edfcb212f9d74ed384b5c4d27c27f9aea608c`, PARSeq `b2559f9149a0e2cd297976d255d367895b1675d3a2b36f40bdb85879f7aaf9b7`
- Previous-model0-vs-groupgate comparison: Tesseract `ae010f62ba026d3d0d6f5fb57d59f28efb80f1cf25cd78c81d499d01a9f2700a`, PARSeq `87b70f33802f0a6d7b25033b5bf042b414596e49408d8b192dc5683594e2a6c2`
- Threshold-sweep aggregate: `7674d7547367b3f4bef3a02d4817cf499e0cca1db14ef19fd66bee371e9e4cfb`
- Threshold-0.8 explicit policy JSONL seed0/1/2: `e696cd29460ae2eef681f5171c678aeb317a13c4e6e2a229fcc34852bd9e1b57`, `56d644aa354a95686905e4921deec913d3b4e5177d7594dc3a200e3623ade66d`, `6b8d025141ffb0e8f00b121aea2129a6f64d5b2a931b94c7d9ab729cdf2750ae`
- Seed0 threshold-0.8 codec summary: `63aa23001cc3c23272a5fe733694ab7e08dbea7527db171486fa74d5edbab5e3`
- Threshold-0.5-vs-0.8 seed0 comparison: Tesseract `c6bad766a207163e8d0e3fdb6712b80574686ded0bfa9ed808a26fced314244a`, PARSeq `5d6d3da97d31b48dcfa48e10eb3c21c4c61b291bc2ead852b412145ff6f36555`
- Nearest-vs-threshold-0.8 comparison: Tesseract `e76704f9f1022c24ecd0d93e50ce01f43db6788ef1a7b16b403d1e49679d66b1`, PARSeq `582aaa92cec23f2aca5a9e17d59e0d800709c3bb39e3ee1587d79821e9717392`
- Latent-vector group-gate aggregate: `7507400ff76adc9a522736ab21b54a120b778511ea4e1d370b0dfe646e776ff2`
- Score-margin gate results: non-nearest seed0/1/2 `d84435311bb55306245d9b7768a2847cc3bd8046e12f3a72d5ca7000c6952088`, `22de1fa9d13556f60a849633051ae4123f83fb6bc5effa77088e01b0cba36ea6`, `14a14d2aa259b5ed9a3e54224e325eb785a8fab9500b05948df950ccc2778e97`; top-margin seed0/1/2 `47cf8c121aa65b6387532fbe47179f07f47ee6a2ea45538fd861f2dc92e8cfa1`, `af62316d0c4c29bece45c40b1e089c27678b9e30df81cf1863bd8f839bc21400`, `dcf353f20b72f5d6a209a1a5e30f74378a4207f067d643f3fb287705b36cbd91`
- Linear-only group-gate aggregate: `81e98ee9424b6e72e5d304c3b1027de15615111dd5096d3a228b799918b30099`
- Pooled seed0/1/2 group-gate aggregate: `d93b5c1043a0ec070c1f57dcd9b14182d7aba6da677eb13eb0833c1cf3cee548`
- Pooled image-diff group-gate aggregate: `503fb566460ab511a0abcf3953323442aa118682a3b1bffc6c1b2beda17dcf0f`
- Pooled image-diff model1 threshold-0.5 explicit policy: `2e179afda620047856a5f070beee1089f103f08f31db1ed17bb4aa2ec4666c69`
- Pooled image-diff actual codec summaries seed0/1/2: `c060e5c703781d2f778b12c877f0284959bcba07ef09c7195fff74652d05d17b`, `4e16d57d72bbbe21726527b45d95b9271e83704d2128c80f3c1e973080f89dbb`, `c0a1455b0164568efb2555e7919a2cfd96f783acaeec0d271bbd4567bf20f150`
- Nearest-vs-pooled-image-diff comparison: Tesseract `959c620b80aadb5626fd0eac2ef94f4fca81617db42c7687c299ea1350e25038`, PARSeq `1a84cb25c824f5336136e377777b3ea49b9040583b288de19ad3838951b3f49a`
- Groupgate-vs-pooled-image-diff comparison: Tesseract `d76b35150806e38996e272bd92510d429319895d06ee474a3cdfb77f9d90f5df`, PARSeq `c980ade5d226f86f7353f95a77518c0ca2788d5ef2419986579c04c38e051d0e`
- Explicit policy-error analysis: previous groupgate `09dfa26cee4b67ca5ce9a9e509863cd7ce211e7c2676ca70354a76c621542277`, pooled image-diff `f20a04320d4d827068eebaf3280aa8c5b1f5c185cc6d026c56872f5cd4922a59`; analysis script `09fee43d6a6db1ef2525e5c400859471f63c77116be5fdcb8311b02fd924fdb1`
- Pooled image-diff model2/threshold-0.8 robustness check: regenerated gate `78156bcd0a3e674276b072b3b2749e6466de5ca695b2a901a442152d52c1ea2b`, explicit policy `11bbe2c6f6004dea6eba7e59d66715f2d3f42a30f14caa12e4c166ef7df1b16a`, policy-error analysis `2a0802502aba51dcb1fea70a9f33913340cb656ce57071bc64b374c3f6fef223`
- Pooled image-diff model2/threshold-0.8 actual codec summaries seed0/1/2: `7efa6205e31879cf448e4d27091b9c3c6c490234be13ca4772db81ab6e033947`, `25ff17cd830492f0164d00b890eb6678dee0f02d3d3191a2bfc008982393b433`, `579106e21366505db40e2069a37e1d703ebadd3c625ff93bc778ec12f67e0b99`
- Pooled image-diff model2/threshold-0.8 comparisons: nearest Tesseract `35f659657cc00e5ab67977977898c28c4dc62af341488c7f48375daa9620771d`, nearest PARSeq `97226675ab173dbb3929f19aa742e99d7bfd21056a98d6d2b6a0f837553e8b18`, model1-vs-model2 Tesseract `6410314d94cd3f59ae2bb32fcb226a596f142078aa0be773f422a70d49793ed1`, model1-vs-model2 PARSeq `ded3d2026394ddf6af66a8eb30cd76be0bc73ab5c4cb1bb1546aa66049fd0a71`
- Pooled image-diff model1/model2 policy overlap: `68627bfa7dcba1da80130ec6678919ae8611f0397ab7133bdf924124b82a1ec9`
- Pooled image-diff model1 score-coverage analysis: `7c1686e8925d400ca972da1b1b823dae6bf75255f9d553da46331b8ca89b41bb`; analysis script `4ddfe428fee0aadbd4b31eeefa140f5d2b181d5886472bac740b1bb98ccc1bf8`
- Tabular image-diff assignment scorer controls: listwise aggregate `c1beb924978f004e16303e414775897a5c6e93d44df1e93280a9e4501ca81e50`, BCE+listwise aggregate `cd7a5e4fd0d5219a9d0daac0df14fe716e2d0589a952d99e458a2999b0b97f89`, listwise score coverage model0/1/2 `e5eb82aa0368037958288c1636df1da4bd4b76fdb01361af11f7efba195fd679`, `afb5b8eec0b14588e78e535c3ae30e426e976da3a5776f4272c15d69d81f011e`, `b3d6aa7919354b684cb8dee33824a25a3fb7db0155ad8e5eca45d111b4b7e282`; updated assignment ranker script `e81bcb92a92c81c1b959d59ce01a72a4ddd08e5fc9d7131116af77e432455234`
- Assignment score top-M oracle upper-controls: codegeom scorer `d4115cddb66212ed8793b246c3443a9c8cde127181b0ed213988911ddd23928a`, tabular image-diff scorer `3f323a7b65c2672ba720ef2313561279961518b449366382869965227935ff11`; script `3c402d38d299c1a2d271ab59e3ddd2cbdba73297069e1bb18b649184096db7cf`
- Top-4 shortlist diff-crop reranker negative control: shortlist table `08b9074885cb2aef09791a32db823d3168eeccbd1250ca616b18709aca8be5a3`, shortlist summary `b26b43c185bae464b5d9e253474842236c0a7b5a46d7e9c03fa7f502fa2580a1`, reranker aggregate `8361cd06b4fa5c8843542853384077ea0d4f6a9eda071292e7002b0a1aec9200`, score coverage `a51caf802a4dd1ade38a20583c273a75bc6afc62a01fc9a8c104054b2c3d7a68`
- Image-diff augmentation script: `aa6b9039dc706abe3b9eebb7042b65defffbd1241909797f1425b8720f8943bb`

W&B offline runs:

- PARSeq 5%: `x2x424m3`
- Tesseract 5%: `wszeclf5`
- PARSeq 10%: `ggzh6xpq`
- Tesseract 10%: `j9tt88id`
- PARSeq codebook K16: `uys8te3m`
- Tesseract codebook K16: `dcjfgsux`
- PARSeq single-candidate sweep: `gtuk2k5f`
- Tesseract single-candidate sweep: `1u56gkya`
- PARSeq Eval100 single-candidate sweep: `7t0vshhl`
- Tesseract Eval100 single-candidate sweep: `0p9jy46p`
- PARSeq Eval100 MLIC++ base OCR: `9ktkzzpb`
- Tesseract Eval100 MLIC++ base OCR: `tiluy40a`
- Eval100 selector/ranker aggregate JSON artifacts: `g3nrm4ky`
- PARSeq Eval100 codebook K16 OCR: `1bahab05`
- Tesseract Eval100 codebook K16 OCR: `948fsnoh`
- PARSeq Eval100 oracle-selected codebook K16 OCR: `rpe8jk58`
- Tesseract Eval100 oracle-selected codebook K16 OCR: `i3u7mj7q`
- PARSeq Eval100 oracle-selected codebook K64 OCR: `8lit9etc`
- Tesseract Eval100 oracle-selected codebook K64 OCR: `4cldpnwz`
- Eval100 codebook payload summary artifacts: `pru2if0g`
- PARSeq Eval300 MLIC++ base OCR: `ptdbcoix`
- Tesseract Eval300 MLIC++ base OCR: `u66g8o0r`
- PARSeq Eval300 single-candidate sweep: `9quxd097`
- Tesseract Eval300 single-candidate sweep: `gob5kbxl`
- PARSeq Eval300 oracle-selected codebook K16 OCR: `5pydnp6s`
- Tesseract Eval300 oracle-selected codebook K16 OCR: `4ui3dogv`
- PARSeq Eval300 oracle-selected codebook K64 OCR: `6pba1tli`
- Tesseract Eval300 oracle-selected codebook K64 OCR: `83w56y6j`
- Eval300 selector/ranker/codebook aggregate JSON artifacts: `otsoepp5`
- Eval300 listwise patch gate aggregate JSON artifacts: `0x8nc7z8`
- Eval300 harm-0.5 listwise patch gate JSON artifacts: `i15aij2h`
- Eval300 Tesseract/PARSeq-safe listwise patch gate JSON artifacts: `qemhqs3c`
- Eval300 Tesseract/PARSeq-safe constrained-threshold listwise patch gate JSON artifacts: `tidcm0oe`
- Eval300 seed0 val75 learned-selection codebook summary artifacts: `b7ug66zw`
- PARSeq Eval300 seed0 val75 learned top-25 K16 OCR: `tlue337p`
- Tesseract Eval300 seed0 val75 learned top-25 K16 PSM8 OCR: `9hmqpe9g`
- PARSeq Eval300 seed0 val75 learned top-25 K25 q0.05 OCR: `lxjxmhnv`
- Tesseract Eval300 seed0 val75 learned top-25 K25 q0.05 PSM8 OCR: `3j0ix93k`
- Eval300 seed1 val75 learned-selection codebook summary artifacts: `aad6z0lh`
- PARSeq Eval300 seed1 val75 learned top-25 K16 OCR: `nmezbr40`
- Tesseract Eval300 seed1 val75 learned top-25 K16 PSM8 OCR: `7rfvyilf`
- Eval300 seed2 val75 learned-selection codebook summary artifacts: `qz39x1vg`
- PARSeq Eval300 seed2 val75 learned top-25 K16 OCR: `od29xmom`
- Tesseract Eval300 seed2 val75 learned top-25 K16 PSM8 OCR: `fydtjm3y`
- Eval300 seed0 train-codebook negative-control summary artifacts: `zxaln5um`
- PARSeq Eval300 seed0 val75 train-codebook K16 OCR: `yho98i0x`
- Tesseract Eval300 seed0 val75 train-codebook K16 PSM8 OCR: `f4teyea8`
- Eval300 seed0 train-codebook K64 capacity-control summary artifacts: `6t5e86gu`
- PARSeq Eval300 seed0 val75 train-codebook K64 OCR: `4sojfq44`
- Tesseract Eval300 seed0 val75 train-codebook K64 PSM8 OCR: `cmapix9w`
- Eval300 train-codebook K64 three-seed capacity-control summary artifacts: `ejiwhxqr`
- PARSeq Eval300 seed1/seed2 val75 train-codebook K64 OCR: `9h4lwuan`, `zpgivl43`
- Tesseract Eval300 seed1/seed2 val75 train-codebook K64 PSM8 OCR: `hnkkfxdf`, `nxdou21x`
- Eval300 train-codebook K64 utility-weighted a8 three-seed summary artifacts: `r0hrzmpg`
- PARSeq Eval300 seed0/seed1/seed2 val75 train-codebook K64 utility-weighted a8 OCR: `qt26nysa`, `dra9pf2b`, `x3pznjfj`
- Tesseract Eval300 seed0/seed1/seed2 val75 train-codebook K64 utility-weighted a8 PSM8 OCR: `mpgxblgk`, `sglnc5kl`, `aonyd3k6`
- Eval300 seed2 train-codebook K64 utility-weight alpha sweep artifacts: `zw2t1kcu`
- PARSeq/Tesseract Eval300 seed2 val75 train-codebook K64 utility-weighted a4 OCR: `gm8wrglo`, `960fpmu5`
- PARSeq/Tesseract Eval300 seed2 val75 train-codebook K64 utility-weighted a16 OCR: `3zad4wve`, `43wvgaho`
- Eval300 train-codebook K64 utility-weighted a4 three-seed summary artifacts: `8dp0ppqw`
- PARSeq Eval300 seed0/seed1/seed2 val75 train-codebook K64 utility-weighted a4 OCR: `0r7p19fd`, `ruraaykm`, `gm8wrglo`
- Tesseract Eval300 seed0/seed1/seed2 val75 train-codebook K64 utility-weighted a4 PSM8 OCR: `qu5qn4vd`, `c2hv8b2b`, `960fpmu5`
- Eval300 train-codebook K64 non-harm filtered negative-control artifacts: `94b9pe8k`
- PARSeq Eval300 seed0/seed1/seed2 val75 train-codebook K64 non-harm OCR: `y8dap30g`, `2ea6i74r`, `otf2lf9w`
- Tesseract Eval300 seed0/seed1/seed2 val75 train-codebook K64 non-harm PSM8 OCR: `6ab91i1q`, `d1xnep9q`, `z8tg2jtj`
- Eval300 seed2 utility-biased assignment negative-control artifacts: `68qqhvlk`
- PARSeq/Tesseract Eval300 seed2 val75 K64 utility-biased assignment re1.05 OCR: `6x22d3a5`, `o9endgb3`
- PARSeq/Tesseract Eval300 seed2 val75 K64 utility-biased assignment re1.01 OCR: `l17z7rsi`, `rtu8isvx`
- Eval300 seed2 top-8 assignment oracle positive artifacts: `ala262xm`
- PARSeq/Tesseract Eval300 seed2 val75 top-8 PARSeq-safe assignment oracle OCR: `4xixubq2`, `gbzcurf7`
- Eval300 top-8 assignment oracle three-seed positive artifacts: `z560kej2`
- PARSeq/Tesseract Eval300 seed0/seed1/seed2 val75 top-8 PARSeq-safe assignment oracle OCR: `ajb8rpvj`/`2bk4p73e`, `1lad5xnu`/`n3jjxwft`, `4xixubq2`/`gbzcurf7`
- Eval300 val75 assignment utility table artifacts: `5qcnclal`
- Eval300 val75 assignment policy diagnostic artifacts: `3mugx5nm`
- Eval300 val75 assignment patch-ranker negative artifacts: `rp2bq0a4`
- PARSeq Eval300 seed0/seed1/seed2 train225 top-75 assignment candidate OCR: `6aarbrol`, `x9jpgl9c`, `5zt0mzvt`
- Tesseract Eval300 seed0/seed1/seed2 train225 top-75 assignment candidate PSM8 OCR: `on102en5`, `ltrasey8`, `w4jlgvu2`
- Eval300 train225 top-75 assignment utility/policy diagnostic artifacts: `2igyn2oi`
- Eval300 train225->val75 assignment patch-ranker negative aggregate artifacts: `1dw5bh0f`
- Eval300 codebook-geometry train225->val75 assignment patch-ranker diagnostic artifacts: `3sfw6cyf`
- Eval300 assignment latent-vector/tabular/conservative ranker diagnostics: `9ht5mv0t`
- Eval300 assignment oracle-choice target positive diagnostics: `ezu71fuk`
- Eval300 seed2 K64 utility-weighted a8 zero-code negative-control summary artifacts: `q0zzarj2`
- PARSeq/Tesseract Eval300 seed2 val75 K64 utility-weighted a8 zero-code OCR: `r5lq23f8`, `q9rsqbiy`
- Eval300 seed2 K64 utility-weighted a8 zero-code threshold negative-control artifacts: `oojmubjf`
- PARSeq/Tesseract Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 0.8 OCR: `krt2wy64`, `20xh2r67`
- PARSeq/Tesseract Eval300 seed2 val75 K64 utility-weighted a8 zero-code threshold 1.0 OCR: `dqyo0rsw`, `cicehl71`
- PARSeq Eval300 oracle-selected codebook K80 OCR: `aqoo0wr5`
- Tesseract Eval300 oracle-selected codebook K80 OCR: `kxum0sif`
- Eval300 codebook K80 upper-bound summary artifacts: `issrh5dv`
- Eval300 learned oracle-choice explicit assignment Tesseract/PARSeq OCR: `q3waqsdt`/`nayhb2vf`, `trrtz0wb`/`osvclskc`, and `cubjq16j`/`ggqzilt3`
- Eval300 learned oracle-choice explicit assignment comparison metrics: `1pa9yoii`/`1ccqz4vl`
- Eval300 assignment oracle actual comparison metrics: `u7nmjcdt`/`cwbu2crh`
- Eval300 learned oracle-choice model-seed ensemble OCR: Tesseract `7rpowbvo`, `rqfhgac7`, `k4d9ip62`; PARSeq `hsbeuswm`, `erxj14eo`, `aza86t6x`
- Eval300 learned oracle-choice model-seed ensemble comparison metrics: `qevckwq8`/`n05o7jou`
- Eval300 combined-seed oracle-choice ensemble OCR: Tesseract `zwy07h9l`, `79a8icup`, `8epib1cc`; PARSeq `bj3nzbkj`, `1h7oliq0`, `26l3ti7n`
- Eval300 combined-seed oracle-choice ensemble comparison metrics: `iutywjja`/`ok7ji49s`
- Eval300 group/listwise assignment-loss and calibration diagnostics: `9act8ur2`
- Eval300 assignment headroom strata metrics: `7zpug5ir`
- Eval300 diagnostic metadata/difficulty assignment features: `fdaiboyd`
- Eval300 local diff-crop assignment diagnostic: `vd0av080`
- Eval300 local diff-crop plus latent-vector assignment diagnostic: `cxj534c6`
- Eval300 tabular-only codegeom oracle-choice assignment diagnostic: `dvnzf3it`
- Eval300 small-regularized tabular codegeom oracle-choice diagnostic: `hoe5exb3`
- Eval300 oracle-choice model0 assignment policy error strata: `4fu2tqn2`
- Eval300 oracle-choice model0 OCR-distance difficulty gate sweep: `58guzacf`
- Eval300 group-level no-op gate over model0 assignment scores: `1os2qee7`
- Eval300 codegeom group gate trained on OCR-distance proxy: `6pd27jde`
- Eval300 group-gated model0 actual assignment comparisons: `41vri07u`
- Eval300 group-gate threshold sweep and threshold-0.8 seed0 actual OCR smoke: `0ydny07x`; PARSeq/Tesseract OCR runs `eguugc6i`/`cgfbddx3`
- Eval300 latent-vector group-gate negative control: `1dwjnk3p`
- Eval300 assignment score-margin gate negative control: `js0mzdwp`
- Eval300 linear-only group-gate simplicity control: `x3bnvvex`
- Eval300 pooled seed0/1/2 group-gate calibration control: `crpalrhg`
- Eval300 pooled image-diff group-gate sweep and actual comparisons: `phaprxue`, `kqy1pmto`, `ndgl9l1x`, `1brl5gkb`, `y2f6wf6j`; PARSeq OCR runs `38couqiu`/`v6o5wk5s`/`7m6duyge`, Tesseract OCR runs `lopcr5uq`/`s2seziag`/`9irjw6qs`
- Eval300 groupgate-vs-image-diff explicit policy-error comparison: `48czjju9`
- Eval300 pooled image-diff model2/threshold-0.8 actual robustness check: `nqphf62f`; PARSeq OCR runs `rkx6noph`/`07wx64ol`/`c4p50eg8`, Tesseract OCR runs `abh4nu0z`/`bcafh136`/`mzpao0r3`
- Eval300 pooled image-diff model1/model2 policy overlap artifact: `djqn21a0`
- Eval300 pooled image-diff model1 assignment score-coverage diagnostic: `fdiuqglc`
- Eval300 tabular image-diff assignment scorer negative controls: `97u4bfga`
- Eval300 assignment score top-M oracle upper-controls: `y4ifr2yl`
- Eval300 top-4 shortlist diff-crop/image-feature reranker negative control: `r1jnb1sf`

## Interpretation

The true frozen-base enhancement path works end to end: counted gate, counted residual payload, base-only fallback, latent modification and OCR evaluation are all executable. However, the naive int8 residual tensor is far too expensive and only weakly useful. This is a helpful negative result: OSCAR-LIC should not become a raw latent patch retransmission method.

The assignment headroom experiment is now sharper: the top-8 PARSeq-safe oracle can improve held-out Tesseract at unchanged payload size. Reconstructed-image patch rankers, codebook-center geometry, residual-vector tabular features and conservative score/margin policies fail when trained with "safe-improve row" labels, because no-headroom groups never learn nearest/no-op as a positive action. Switching the target to `label_assignment_oracle_choice` fixes that failure mode and gives the first learned assignment gain: codegeom oracle-choice improves held-out Tesseract edit distance by about 0.89 characters on average with zero PARSeq delta. A tabular-only codegeom oracle-choice diagnostic slightly raises this table-level mean to 1.0 character, but still has seed2 worsen cases and is not enough to promote as a new actual-bitstream model. A smaller strongly regularized tabular head drops the gain to 0.33 characters, so plain capacity reduction is not the fix. Policy-error analysis of the promoted model0 shows 9 missed oracle changes, 15 false changes and 4 wrong changes across 75 val groups; IAM words dominate both misses and false changes. A non-deployable OCR-distance gate improves model0 from -3 to -5 Tesseract edit characters by rejecting easy nearest-distance cases, so a learned difficulty/no-op proxy is worth building. A first deployable-ish codegeom group gate gives a weak table improvement and, importantly, survives actual-bitstream promotion: at unchanged actual bpp it improves nearest assignment by -5 unicode / -7 latin-alnum Tesseract edit characters while preserving PARSeq exactly, and it improves the previous model0 actual baseline by a further -2/-2 characters. A threshold sweep shows the current score scale is coarse but stable: threshold 0.8 is a slightly more conservative equivalent to 0.5, with one fewer changed group and the same OCR aggregate. Latent-vector statistics, score-margin gates, a linear-only group gate and a pooled seed0/1/2 codegeom gate all fail to beat that promoted MLP group gate. The breakthrough is adding local reconstructed-image/code-effect evidence: the pooled image-diff gate improves the actual-bitstream result to -8 unicode / -8 latin-alnum Tesseract edit characters versus nearest, while preserving PARSeq and actual bpp exactly. It changes fewer groups than the old group gate but cuts false/wrong changes from 11 to 1, and an independent gate seed reproduces the same actual OCR aggregate. Score-coverage analysis shows the remaining missed oracle cases are not primarily gate-threshold misses: the current assignment scorer ranks the oracle code first for only 4/17 oracle-change groups. Directly feeding hand-built image-diff statistics into a tabular scorer is also negative despite high AUC/AP, collapsing oracle-change rank1 coverage to 1/17. Score top-M upper controls show that top-4 shortlists retain much more headroom (`-15` to `-16` Tesseract characters), but a first top-4 diff-crop CNN reranker still fails to realize it. The immediate research target is therefore a genuinely different compact local latent/glyph/code-effect encoder with group-aware reranking over a short candidate list, not wider tabular image statistics, the same diff-crop CNN, or looser gates. It must lift oracle codes from top-4 into rank 1 while preserving the current image-diff gate's no-op precision. Threshold calibration should still be cross-fitted and selected by policy-level OCR/false-change metrics rather than AUC alone.

The next model should compress selected-candidate information much more aggressively:

- predict a low-dimensional residual code rather than raw `32 x tile x tile` int8 tensors;
- entropy-code residual symbols or use a small learned codebook;
- train utility selection against multi-teacher OCR, not residual magnitude;
- keep residual scale bounded or learned with a stabilizing prior.
- build candidate labels from single-candidate or small-subset OCR counterfactuals, since Tesseract shows real positive and negative candidate utility while PARSeq is invariant on this smoke.

This supports the earlier intuition that a simple model can help, but only if the simplicity is in the learned enhancement representation, not in blindly sending raw latent deltas.

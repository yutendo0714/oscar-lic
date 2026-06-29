# Eval300 Top-8 Actual OSCR Held-Out OCR Check

Date: 2026-06-26

## Purpose

This evaluation checks whether the non-deployable top-8 actual-counted OSCR oracle that improves Tesseract also harms held-out OCR families that were not used for selection. TrOCR-small-printed, PaddleOCR PP-OCRv6 recognizer-only, and PaddleOCR detector+recognizer E2E are run on the actual decoded top-8 reconstructions and compared row-wise against nearest, current best, and top-4 actual oracle outputs.

## Row-Wise Comparisons

| Comparison | Profile | Delta chars | Improved | Worsened | Exact delta | CER delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| paddleocr_current_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| paddleocr_current_to_top8 | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| paddleocr_e2e_current_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| paddleocr_e2e_current_to_top8 | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| paddleocr_e2e_nearest_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| paddleocr_e2e_nearest_to_top8 | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| paddleocr_e2e_top4_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| paddleocr_e2e_top4_to_top8 | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| paddleocr_nearest_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| paddleocr_nearest_to_top8 | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| paddleocr_top4_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| paddleocr_top4_to_top8 | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| trocr_current_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| trocr_current_to_top8 | latin_alnum_ci_v1 | -2 | 1 | 0 | 0 | -0.001799 |
| trocr_nearest_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| trocr_nearest_to_top8 | latin_alnum_ci_v1 | -2 | 1 | 0 | 0 | -0.001799 |
| trocr_top4_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| trocr_top4_to_top8 | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |

## Top-8 Metrics

| OCR | Seed | Unicode CER | Unicode exact | Latin CER | Latin exact | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| paddleocr_detector_recognizer | seed0 | 0.167568 | 0.760000 | 0.149171 | 0.786667 | null |
| paddleocr_detector_recognizer | seed1 | 0.164491 | 0.760000 | 0.167116 | 0.760000 | null |
| paddleocr_detector_recognizer | seed2 | 0.138817 | 0.760000 | 0.118734 | 0.786667 | null |
| paddleocr_text_recognition | seed0 | 0.108108 | 0.746667 | 0.088398 | 0.800000 | null |
| paddleocr_text_recognition | seed1 | 0.083551 | 0.773333 | 0.075472 | 0.786667 | null |
| paddleocr_text_recognition | seed2 | 0.105398 | 0.786667 | 0.076517 | 0.840000 | null |
| trocr | seed0 | 0.827027 | 0.053333 | 0.364641 | 0.440000 | f96e1lnm |
| trocr | seed1 | 0.728460 | 0.146667 | 0.283019 | 0.453333 | krwyz56o |
| trocr | seed2 | 0.735219 | 0.186667 | 0.237467 | 0.560000 | j2wo0yjk |

## Interpretation

Top-8 adds no held-out OCR regression in this smoke. TrOCR is unchanged under unicode/raw and improves Latin by -2 characters versus nearest/current; it is identical to top-4. PaddleOCR recognizer-only and PaddleOCR E2E are exactly unchanged versus nearest, current best, and top-4 across all 225 samples. This supports the top-8 upper-bound result as a Tesseract improvement that does not break these held-out OCR families, while keeping the policy non-deployable because oracle labels selected the code.

## Files

- Summary JSON: `experiments/results/eval300_top8_actual_oscr_heldout_ocr_2026_06_26.json`
- summary_trocr_seed0: `experiments/results/trocr_eval300_seed0_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- summary_trocr_seed1: `experiments/results/trocr_eval300_seed1_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- summary_trocr_seed2: `experiments/results/trocr_eval300_seed2_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- summary_paddleocr_seed0: `experiments/results/paddleocr_eval300_seed0_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- summary_paddleocr_seed1: `experiments/results/paddleocr_eval300_seed1_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- summary_paddleocr_seed2: `experiments/results/paddleocr_eval300_seed2_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- summary_paddleocr_e2e_seed0: `experiments/results/paddleocr_e2e_eval300_seed0_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- summary_paddleocr_e2e_seed1: `experiments/results/paddleocr_e2e_eval300_seed1_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- summary_paddleocr_e2e_seed2: `experiments/results/paddleocr_e2e_eval300_seed2_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- compare_trocr_nearest_to_top8: `experiments/results/compare_trocr_eval300_val75_nearest_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json`
- compare_trocr_current_to_top8: `experiments/results/compare_trocr_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json`
- compare_trocr_top4_to_top8: `experiments/results/compare_trocr_eval300_val75_shortlist_oracle_top4_vs_top8_scoreseed2_q005_2026_06_26.json`
- compare_paddleocr_nearest_to_top8: `experiments/results/compare_paddleocr_eval300_val75_nearest_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json`
- compare_paddleocr_current_to_top8: `experiments/results/compare_paddleocr_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json`
- compare_paddleocr_top4_to_top8: `experiments/results/compare_paddleocr_eval300_val75_shortlist_oracle_top4_vs_top8_scoreseed2_q005_2026_06_26.json`
- compare_paddleocr_e2e_nearest_to_top8: `experiments/results/compare_paddleocr_e2e_eval300_val75_nearest_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json`
- compare_paddleocr_e2e_current_to_top8: `experiments/results/compare_paddleocr_e2e_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json`
- compare_paddleocr_e2e_top4_to_top8: `experiments/results/compare_paddleocr_e2e_eval300_val75_shortlist_oracle_top4_vs_top8_scoreseed2_q005_2026_06_26.json`
- top8_actual_summary: `experiments/results/eval300_shortlist_oracle_top8_scoreseed2_actual_oscr_2026_06_26.json`
- trocr_runner: `scripts/run_trocr_ocr_smoke.py`
- paddleocr_runner: `scripts/run_paddleocr_ocr_smoke.py`
- paddleocr_e2e_runner: `scripts/run_paddleocr_e2e_smoke.py`
- compare_script: `scripts/compare_ocr_result_pairs.py`

## Hashes

- compare_paddleocr_current_to_top8: `78aac6e2174a80ca8965c7395a66a8d1e0c8075f4f74ef14b593b604b512fb46`
- compare_paddleocr_e2e_current_to_top8: `55e4e3573a7fc13fe9ebdcf504c654855d3e708e0d6bab8babf6830770ee92c8`
- compare_paddleocr_e2e_nearest_to_top8: `7fc22208575cb58f2b0ca3e145bfc5de569f2e08b63225a653b95abbc8e1bbc7`
- compare_paddleocr_e2e_top4_to_top8: `aeb95e2ea95629f2a1919913f173f22dd4c66c11416033b88d30a9fa953bd260`
- compare_paddleocr_nearest_to_top8: `110e2f3392686f231853baa8962c7ef11687aef88fe3ad9691ccddbc4b76a5aa`
- compare_paddleocr_top4_to_top8: `b936c91c85ff536d82ed63e4f28963f50f3c2b3434511320a0c367908bd21be4`
- compare_script: `37efd8593f1b7169efc7f63af3d934dfa5abaa68deb36ed83351b0718a15491a`
- compare_trocr_current_to_top8: `44f8ed5c897a68cf66b3ab5e6d9a120e83301b17f6046019ba8d2ab7aaafbf0d`
- compare_trocr_nearest_to_top8: `0dffd04d73db7b1339bc235e81d7cbb26277a4f0b7326baad1b53227286449b0`
- compare_trocr_top4_to_top8: `2f54264ec0b632c11251d0374485ac57a3876fe2c28574e6fe0a57cbe1e1476b`
- paddleocr_e2e_runner: `0534547333cad70b48027df7d6b4b675ec62bfc5ed2f42d8aa901259dcf217f4`
- paddleocr_runner: `155ddb5ebae6365dacada7bd73ff924f73fb6ed23b7e4e9197dd75c697bf3f51`
- script: `dcef470653804c52bfe2c8bc920239175bc38bf708f49c8254495256934b79da`
- summary_paddleocr_e2e_seed0: `b515380a4f134f914373aa498cec741a5713fbcef5e561d76209ae6e23a3d0b1`
- summary_paddleocr_e2e_seed1: `47cd7ad3f97b7f581d3e37b3b62ca6bda7c4fa4a1d36aca99ab2b82e324f3c24`
- summary_paddleocr_e2e_seed2: `7c9b4b8b75900f0f21c92fd9da51da8eed235add35de6bef21caeb9febbb7c59`
- summary_paddleocr_seed0: `01b117a08de1c018bb4d1fbe59f1d42954dca3f941581b76ece0f911339d228b`
- summary_paddleocr_seed1: `b7869b2c36ebfbd855120b7cbc69168e6687fc005e5bb760a2d9ecdb7422e105`
- summary_paddleocr_seed2: `0823f8f25fbd4f42f4c6c310e8ba4b2083f463cb59d636935db6adaae6df6653`
- summary_trocr_seed0: `9177af72bc11d2909681e0dbe6149d9bd77a72d7a617366d6223988cf4e4ebef`
- summary_trocr_seed1: `d688019ac1364e3d9aa453f2ed334b61a8d81ef8685276e8e211b0ef9992aae9`
- summary_trocr_seed2: `9486d7cf932770d3919d63f108b3a530979f1f9de9296767eae0e7cb293d4b36`
- top8_actual_summary: `2d6955a857394fc2cdd013e32babd0113fe1969c3f6207dd413eefc8e0ee991f`
- trocr_runner: `02cc38e2b05ba6a57a9d2f5c85c35eb53c5dea872ff159cba55fd18ba6872b64`

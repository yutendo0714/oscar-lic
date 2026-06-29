# Eval300 OCR Comparison Bootstrap CI

- Bootstrap samples: `10000`
- Seed: `20260626`
- Primary profile: `unicode_strict_v1`
- Rate is read from codec `results.jsonl` rows backing each OCR reconstruction path.
- Negative OCR deltas mean the candidate has fewer edit errors.

| comparison                 | samples | dCER obs [95% CI]                | dChars obs [95% CI] | dBPP obs [95% CI]             |
| -------------------------- | ------- | -------------------------------- | ------------------- | ----------------------------- |
| tesseract_nearest_top8     | 225     | -0.021016 [-0.033036, -0.011197] | -24 [-38, -13]      | 0.000000 [0.000000, 0.000000] |
| tesseract_current_top8     | 225     | -0.014011 [-0.022318, -0.006981] | -16 [-26, -8]       | 0.000000 [0.000000, 0.000000] |
| tesseract_top4_top8        | 225     | -0.003503 [-0.009937, 0.000000]  | -4 [-11, 0]         | 0.000000 [0.000000, 0.000000] |
| parseq_nearest_top8        | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| parseq_current_top8        | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| parseq_top4_top8           | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| trocr_nearest_top8         | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| trocr_current_top8         | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| trocr_top4_top8            | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| paddleocr_nearest_top8     | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| paddleocr_current_top8     | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| paddleocr_top4_top8        | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| paddleocr_e2e_nearest_top8 | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| paddleocr_e2e_current_top8 | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| paddleocr_e2e_top4_top8    | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |

## Interpretation

- The top-8 shortlist oracle is a statistically stronger non-deployable upper bound than the current learned actual-bitstream policy: Tesseract unicode improves over nearest by `-24` chars with CI `[-38, -13]`, and over current by `-16` chars with CI `[-26, -8]`, at unchanged actual byte-counted bpp.
- The top-8 oracle also improves over the top-4 oracle in the observed Tesseract count (`-4` unicode chars), but the CI touches zero (`[-11, 0]`). Treat this as useful additional headroom and target evidence, not yet as a statistically separated claim over top-4 at this sample size.
- PARSeq, PaddleOCR recognizer-only, and PaddleOCR E2E are exactly unchanged under the primary unicode profile. TrOCR unicode/raw is unchanged; the Latin-normalized TrOCR improvement versus nearest/current is `-2` chars with CI touching zero.
- All rate deltas are computed from actual serialized `.oscr`-backed codec `results.jsonl` byte counts; no entropy-estimate bpp is used.

## Inputs

- `tesseract_nearest_top8`: `experiments/results/compare_tesseract_eval300_val75_nearest_vs_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26.json` SHA256 `6ec74766cc8ea3aea202284b535ba7c325689344f30e772c49db91e55415ca76`
- `tesseract_current_top8`: `experiments/results/compare_tesseract_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26.json` SHA256 `ab65beab0b20e02bc237a62139cffe1e45c627f230b5db611de5a3b2f278558e`
- `tesseract_top4_top8`: `experiments/results/compare_tesseract_eval300_val75_shortlist_oracle_top4_vs_top8_scoreseed2_explicit_q005_2026_06_26.json` SHA256 `67126fc709dfde68d7073c5274ece13b6f15a7ff0f537fa5535fbf0545cf5e0d`
- `parseq_nearest_top8`: `experiments/results/compare_parseq_eval300_val75_nearest_vs_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26.json` SHA256 `99a458df4d9fc776c23090433a2c7f69650aba01ec452264684366f60587ab67`
- `parseq_current_top8`: `experiments/results/compare_parseq_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26.json` SHA256 `612a2a7933272c973b78fd122f79d1a62e0d784b0d10e59edfd4bcadf1822fdf`
- `parseq_top4_top8`: `experiments/results/compare_parseq_eval300_val75_shortlist_oracle_top4_vs_top8_scoreseed2_explicit_q005_2026_06_26.json` SHA256 `21ffee858aa47fb60a2019f404444acf3aa19b221681f9dc9e8e56771b33624b`
- `trocr_nearest_top8`: `experiments/results/compare_trocr_eval300_val75_nearest_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json` SHA256 `0dffd04d73db7b1339bc235e81d7cbb26277a4f0b7326baad1b53227286449b0`
- `trocr_current_top8`: `experiments/results/compare_trocr_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json` SHA256 `44f8ed5c897a68cf66b3ab5e6d9a120e83301b17f6046019ba8d2ab7aaafbf0d`
- `trocr_top4_top8`: `experiments/results/compare_trocr_eval300_val75_shortlist_oracle_top4_vs_top8_scoreseed2_q005_2026_06_26.json` SHA256 `2f54264ec0b632c11251d0374485ac57a3876fe2c28574e6fe0a57cbe1e1476b`
- `paddleocr_nearest_top8`: `experiments/results/compare_paddleocr_eval300_val75_nearest_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json` SHA256 `110e2f3392686f231853baa8962c7ef11687aef88fe3ad9691ccddbc4b76a5aa`
- `paddleocr_current_top8`: `experiments/results/compare_paddleocr_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json` SHA256 `78aac6e2174a80ca8965c7395a66a8d1e0c8075f4f74ef14b593b604b512fb46`
- `paddleocr_top4_top8`: `experiments/results/compare_paddleocr_eval300_val75_shortlist_oracle_top4_vs_top8_scoreseed2_q005_2026_06_26.json` SHA256 `b936c91c85ff536d82ed63e4f28963f50f3c2b3434511320a0c367908bd21be4`
- `paddleocr_e2e_nearest_top8`: `experiments/results/compare_paddleocr_e2e_eval300_val75_nearest_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json` SHA256 `7fc22208575cb58f2b0ca3e145bfc5de569f2e08b63225a653b95abbc8e1bbc7`
- `paddleocr_e2e_current_top8`: `experiments/results/compare_paddleocr_e2e_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top8_scoreseed2_q005_2026_06_26.json` SHA256 `55e4e3573a7fc13fe9ebdcf504c654855d3e708e0d6bab8babf6830770ee92c8`
- `paddleocr_e2e_top4_top8`: `experiments/results/compare_paddleocr_e2e_eval300_val75_shortlist_oracle_top4_vs_top8_scoreseed2_q005_2026_06_26.json` SHA256 `aeb95e2ea95629f2a1919913f173f22dd4c66c11416033b88d30a9fa953bd260`

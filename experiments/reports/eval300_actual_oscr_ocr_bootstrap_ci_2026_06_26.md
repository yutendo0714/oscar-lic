# Eval300 OCR Comparison Bootstrap CI

- Bootstrap samples: `10000`
- Seed: `20260626`
- Primary profile: `unicode_strict_v1`
- Rate is read from codec `results.jsonl` rows backing each OCR reconstruction path.
- Negative OCR deltas mean the candidate has fewer edit errors.

| comparison                  | samples | dCER obs [95% CI]                | dChars obs [95% CI] | dBPP obs [95% CI]             |
| --------------------------- | ------- | -------------------------------- | ------------------- | ----------------------------- |
| tesseract_nearest_current   | 225     | -0.007005 [-0.016726, 0.000000]  | -8 [-19, 0]         | 0.000000 [0.000000, 0.000000] |
| tesseract_nearest_shortlist | 225     | -0.017513 [-0.028120, -0.008628] | -20 [-32, -10]      | 0.000000 [0.000000, 0.000000] |
| tesseract_current_shortlist | 225     | -0.010508 [-0.016667, -0.005217] | -12 [-19, -6]       | 0.000000 [0.000000, 0.000000] |
| parseq_nearest_current      | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| parseq_nearest_shortlist    | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| parseq_current_shortlist    | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| trocr_nearest_current       | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| trocr_nearest_shortlist     | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| trocr_current_shortlist     | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| paddleocr_nearest_current   | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| paddleocr_nearest_shortlist | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |
| paddleocr_current_shortlist | 225     | 0.000000 [0.000000, 0.000000]    | 0 [0, 0]            | 0.000000 [0.000000, 0.000000] |

## Inputs

- `tesseract_nearest_current`: `experiments/results/compare_tesseract_eval300_val75_nearest_vs_pooled_imgdiff_model1_threshold05_explicit_q005_2026_06_25.json` SHA256 `959c620b80aadb5626fd0eac2ef94f4fca81617db42c7687c299ea1350e25038`
- `tesseract_nearest_shortlist`: `experiments/results/compare_tesseract_eval300_val75_nearest_vs_shortlist_oracle_top4_scoreseed2_explicit_q005_2026_06_26.json` SHA256 `b3a56106e82731ed2225d7bcd3fdae99e3f4233bf2826b860669738f5f6124c9`
- `tesseract_current_shortlist`: `experiments/results/compare_tesseract_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top4_scoreseed2_explicit_q005_2026_06_26.json` SHA256 `fbe7e039731835c66bb6049740e47b46551985b7f3fb2698bd7f2db878710557`
- `parseq_nearest_current`: `experiments/results/compare_parseq_eval300_val75_nearest_vs_pooled_imgdiff_model1_threshold05_explicit_q005_2026_06_25.json` SHA256 `1a84cb25c824f5336136e377777b3ea49b9040583b288de19ad3838951b3f49a`
- `parseq_nearest_shortlist`: `experiments/results/compare_parseq_eval300_val75_nearest_vs_shortlist_oracle_top4_scoreseed2_explicit_q005_2026_06_26.json` SHA256 `ddf3571f0ee6182988137583c24f026407e26c58afba0e6447004d09200f3af1`
- `parseq_current_shortlist`: `experiments/results/compare_parseq_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top4_scoreseed2_explicit_q005_2026_06_26.json` SHA256 `9e7d96b721625d6fe32030390286eaf6383dfe52dbd852e98beeb427b06b3a57`
- `trocr_nearest_current`: `experiments/results/compare_trocr_eval300_val75_nearest_vs_pooled_imgdiff_model1_threshold05_q005_2026_06_26.json` SHA256 `a7a4403f6ca7d8ac4c1400fdec88e8b469a2b9b4b4898c2c486ef4eef2cd8eac`
- `trocr_nearest_shortlist`: `experiments/results/compare_trocr_eval300_val75_nearest_vs_shortlist_oracle_top4_scoreseed2_q005_2026_06_26.json` SHA256 `58ba8d69b7104121f00b9bfc2460f7c62f5f6400ed98ce67811a4fb437105bc8`
- `trocr_current_shortlist`: `experiments/results/compare_trocr_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top4_scoreseed2_q005_2026_06_26.json` SHA256 `516e0044b1c01fd0795058085ec72522f3dbb156b632b396cb0323f37bed4214`
- `paddleocr_nearest_current`: `experiments/results/compare_paddleocr_eval300_val75_nearest_vs_pooled_imgdiff_model1_threshold05_q005_2026_06_26.json` SHA256 `ed61f9d90f64d9dcf3b70362a7f5fd48a82c3d5bc2092563c6d61fdb4659c14b`
- `paddleocr_nearest_shortlist`: `experiments/results/compare_paddleocr_eval300_val75_nearest_vs_shortlist_oracle_top4_scoreseed2_q005_2026_06_26.json` SHA256 `dfbe2b8b9558ffd32d76eecc105ca488e3ea0003595f421a6254fae3d613744a`
- `paddleocr_current_shortlist`: `experiments/results/compare_paddleocr_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top4_scoreseed2_q005_2026_06_26.json` SHA256 `fe2563616ed8b171b5c68734e1f6cb6a2eaf4a74fc946061ae338a1a471ef781`

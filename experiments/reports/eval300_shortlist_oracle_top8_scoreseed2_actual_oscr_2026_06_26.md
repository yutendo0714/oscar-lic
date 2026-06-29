# Eval300 Top-8 Shortlist Oracle Actual OSCR Smoke

Date: 2026-06-26

## Purpose

This run verifies the non-deployable top-8 shortlist oracle through the real OSCAR container path. The policy keeps the same K64 utility-weighted train codebooks, selected candidate count, explicit code-index payload length, corrected MLIC++ checkpoint, and counted `.oscr` accounting as the current assignment path. OCR changes therefore come from reassignment within an already transmitted code-index budget, not from free side information or estimated entropy.

## Policy

| Split | Groups | Changed | Exact oracle | Missed oracle | Tess table delta | PARSeq table delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed0 | 25 | 0 | 0 | 0 | 0 | 0 |
| seed1 | 25 | 6 | 6 | 0 | -10 | 0 |
| seed2 | 25 | 11 | 11 | 0 | -14 | 0 |
| total | 75 | 17 | 17 | 0 | -24 | 0 |

Against the baseline shortlist policy, the current policy delta is: groups +0, changed_groups +2, exact_oracle_changes +2, missed_oracle_changes -2, false_changes +0, wrong_changes +0, tesseract_delta_sum_vs_nearest -4, parseq_delta_sum_vs_nearest +0, tesseract_improve_groups +2, tesseract_worsen_groups +0.

## Actual Bitstream

| Split | Actual bpp | Enhancement bpp | Assignment changed fraction | PSNR delta dB |
| --- | ---: | ---: | ---: | ---: |
| seed0 | 3.755801 | 0.031080 | 0.000000 | -0.002472 |
| seed1 | 3.957302 | 0.026194 | 0.240000 | -0.003439 |
| seed2 | 3.835844 | 0.029693 | 0.440000 | -0.003484 |
| mean | 3.849649 | 0.028989 | 0.226667 | -0.003132 |

Actual bpp is counted from serialized `.oscr` files, including base, enhancement, gate, code, metadata, and container bytes.

## OCR Results

| Comparison | Profile | Delta chars | Improved | Worsened | Exact delta | CER delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| parseq_current_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| parseq_current_to_top8 | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| parseq_nearest_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| parseq_nearest_to_top8 | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| parseq_top4_to_top8 | unicode_strict_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| parseq_top4_to_top8 | latin_alnum_ci_v1 | 0 | 0 | 0 | 0 | 0.000000 |
| tesseract_current_to_top8 | unicode_strict_v1 | -16 | 14 | 0 | 1 | -0.014011 |
| tesseract_current_to_top8 | latin_alnum_ci_v1 | -11 | 9 | 0 | 0 | -0.009892 |
| tesseract_nearest_to_top8 | unicode_strict_v1 | -24 | 17 | 0 | 2 | -0.021016 |
| tesseract_nearest_to_top8 | latin_alnum_ci_v1 | -19 | 12 | 0 | 2 | -0.017086 |
| tesseract_top4_to_top8 | unicode_strict_v1 | -4 | 2 | 0 | 0 | -0.003503 |
| tesseract_top4_to_top8 | latin_alnum_ci_v1 | -2 | 1 | 0 | 0 | -0.001799 |

## Interpretation

Top-8 shortlist oracle is a stronger actual-bitstream upper bound than top-4: it recovers the two oracle changes missed by top-4, keeps PARSeq unchanged, and gives additional Tesseract improvement at identical mean actual bpp. This remains non-deployable because held-out OCR oracle labels choose the code at selection time. The next deployable target is not another threshold sweep over existing score families; it is a selector or first-stage candidate generator that can recover deeper-rank oracle changes while preserving the explicit no-op anchor.

## Outputs

- Summary JSON: `experiments/results/eval300_shortlist_oracle_top8_scoreseed2_actual_oscr_2026_06_26.json`
- policy_summary: `experiments/results/eval300_assignment_shortlist_oracle_top8_scoreseed2_explicit_codes_2026_06_26.jsonl.summary.json`
- stream_summary_seed0: `experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/summary.json`
- stream_summary_seed1: `experiments/results/mlicpp_latent_codebook_seed1_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/summary.json`
- stream_summary_seed2: `experiments/results/mlicpp_latent_codebook_seed2_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/summary.json`
- ocr_summary_parseq_seed0: `experiments/results/parseq_eval300_seed0_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- ocr_summary_parseq_seed1: `experiments/results/parseq_eval300_seed1_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- ocr_summary_parseq_seed2: `experiments/results/parseq_eval300_seed2_val75_shortlist_oracle_top8_scoreseed2_q005_2026_06_26/summary.json`
- ocr_summary_tesseract_seed0: `experiments/results/tesseract_eval300_seed0_val75_shortlist_oracle_top8_scoreseed2_q005_psm8_2026_06_26/summary.json`
- ocr_summary_tesseract_seed1: `experiments/results/tesseract_eval300_seed1_val75_shortlist_oracle_top8_scoreseed2_q005_psm8_2026_06_26/summary.json`
- ocr_summary_tesseract_seed2: `experiments/results/tesseract_eval300_seed2_val75_shortlist_oracle_top8_scoreseed2_q005_psm8_2026_06_26/summary.json`
- compare_parseq_nearest_to_top8: `experiments/results/compare_parseq_eval300_val75_nearest_vs_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26.json`
- compare_parseq_current_to_top8: `experiments/results/compare_parseq_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26.json`
- compare_parseq_top4_to_top8: `experiments/results/compare_parseq_eval300_val75_shortlist_oracle_top4_vs_top8_scoreseed2_explicit_q005_2026_06_26.json`
- compare_tesseract_nearest_to_top8: `experiments/results/compare_tesseract_eval300_val75_nearest_vs_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26.json`
- compare_tesseract_current_to_top8: `experiments/results/compare_tesseract_eval300_val75_pooled_imgdiff_model1_threshold05_vs_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26.json`
- compare_tesseract_top4_to_top8: `experiments/results/compare_tesseract_eval300_val75_shortlist_oracle_top4_vs_top8_scoreseed2_explicit_q005_2026_06_26.json`
- policy_jsonl: `experiments/results/eval300_assignment_shortlist_oracle_top8_scoreseed2_explicit_codes_2026_06_26.jsonl`
- policy_realseed0: `experiments/results/eval300_assignment_shortlist_oracle_top8_scoreseed2_realseed0_explicit_codes_2026_06_26.jsonl`
- policy_realseed1: `experiments/results/eval300_assignment_shortlist_oracle_top8_scoreseed2_realseed1_explicit_codes_2026_06_26.jsonl`
- policy_realseed2: `experiments/results/eval300_assignment_shortlist_oracle_top8_scoreseed2_realseed2_explicit_codes_2026_06_26.jsonl`
- export_script: `scripts/export_assignment_shortlist_oracle_policy.py`
- compare_script: `scripts/compare_ocr_result_pairs.py`
- oscr_script: `scripts/run_mlicpp_latent_codebook_stub.py`
- baseline_policy_summary: `experiments/results/eval300_assignment_shortlist_oracle_top4_scoreseed2_explicit_codes_2026_06_26.jsonl.summary.json`

## Hashes

- baseline_policy_summary: `868fa9787f88e8276f6b10b6792841201dd32de64b970fb86e88a7aea0593687`
- compare_parseq_current_to_top8: `612a2a7933272c973b78fd122f79d1a62e0d784b0d10e59edfd4bcadf1822fdf`
- compare_parseq_nearest_to_top8: `99a458df4d9fc776c23090433a2c7f69650aba01ec452264684366f60587ab67`
- compare_parseq_top4_to_top8: `21ffee858aa47fb60a2019f404444acf3aa19b221681f9dc9e8e56771b33624b`
- compare_script: `37efd8593f1b7169efc7f63af3d934dfa5abaa68deb36ed83351b0718a15491a`
- compare_tesseract_current_to_top8: `ab65beab0b20e02bc237a62139cffe1e45c627f230b5db611de5a3b2f278558e`
- compare_tesseract_nearest_to_top8: `6ec74766cc8ea3aea202284b535ba7c325689344f30e772c49db91e55415ca76`
- compare_tesseract_top4_to_top8: `67126fc709dfde68d7073c5274ece13b6f15a7ff0f537fa5535fbf0545cf5e0d`
- export_script: `0475934003548734b59a9c43a4dfd5baf33f57c935f462a8c20e6e40946c2906`
- ocr_summary_parseq_seed0: `0cb70c9657e2597133b16c71599024fa89ea6b71bffaede1e485c7852c04f93b`
- ocr_summary_parseq_seed1: `04cf19023a35575a39f0fa31793eda2dc160bd8f985653561cfbef472a44f7ae`
- ocr_summary_parseq_seed2: `85f454f3da7940bc9ddff745c99c365864f00ffcd82885bbed708c930dece1ae`
- ocr_summary_tesseract_seed0: `b2edbd32754587a78d8a923f3fef83581ae94bc77781c077da91880d1c7d755d`
- ocr_summary_tesseract_seed1: `d172793b69258237f4bd9774436789e54a8aa92597779919b1532ab1bd8790c0`
- ocr_summary_tesseract_seed2: `6165ca6fe872abed52bcecd06a837af4e7291197976f45007eed06bf8fdd73a8`
- oscr_script: `8fa2aba3f88a9ced08fa383ff71d46e2c28859031c46089258b3cd92d3f33ba5`
- policy_jsonl: `1a4560e5aab9bc43ebbaf11c9d7186d5a3c5eda77b0b48243f257cdb13dfa938`
- policy_realseed0: `17f4da36c071cd2c5cb4de38b43cd5284f1e7f6f96185dffd727d87999edf6e6`
- policy_realseed1: `12723ce0b8fe4b684ee7d85c093fc5a2d10ce2147f624ec1b1f1d35f7babf1f7`
- policy_realseed2: `635068d125247ac5037978246792becc9293982738702b7d896232d0f8ba1b33`
- policy_summary: `e6bd7de58c3eb5dc6ec21695aa65e22e3976bb511a2e2d35bf233a0894414e00`
- script: `31a9763b6bb0102051f93cdb0004983c8b9d9cde33569c749b06cd5075aebeec`
- stream_summary_seed0: `76523e26f09797b499879ec80997d1d0e6f41172cc1851749bc977bc3b83b9e5`
- stream_summary_seed1: `dd991e9a1c887083d42a040f05e4dafd2f8499bda8b0ad715597088c341212ab`
- stream_summary_seed2: `dc0a4bb367734b970331763e740c5ef9f2b72c1baab56ad1e4052ab7fcfe5acc`

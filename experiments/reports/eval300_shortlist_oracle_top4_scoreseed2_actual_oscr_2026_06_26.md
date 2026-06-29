# Eval300 Top-4 Shortlist Oracle Actual OSCR Smoke

Date: 2026-06-26

## Purpose

This run checks whether the N037 top-4 shortlist headroom survives the real OSCAR container path. It is a non-deployable oracle upper bound: the policy chooses the true assignment-oracle code only when that code is present in the deployable score-defined shortlist, otherwise it falls back to nearest/no-op.

The test keeps the selected candidates, K64 utility-weighted train codebooks, code-index payload length, MLIC++ checkpoint, and `.oscr` accounting identical to the current assignment path. Therefore OCR changes come from explicit code-index reassignment, not from extra transmitted bytes.

## Inputs

- Assignment table: `experiments/results/eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl`
- Oracle-choice diff-crop scores: `experiments/results/eval600_hardneg_balanced210_external_diffcrop_candidate_scorer_scores_2026_06_26.jsonl`
- Safe-improve diff-crop scores: `experiments/results/eval600_hardneg_balanced210_external_diffcrop_safeimprove_candidate_scorer_scores_2026_06_26.jsonl`
- Exported policy: `experiments/results/eval300_assignment_shortlist_oracle_top4_scoreseed2_explicit_codes_2026_06_26.jsonl`
- Summary JSON: `experiments/results/eval300_shortlist_oracle_top4_scoreseed2_actual_oscr_2026_06_26.json`

## Policy

Score model seed 2 gives the strongest shortlist oracle in the N037 diagnostic.

| Split | Groups | Changed | Exact oracle | Missed oracle | Tesseract table delta | PARSeq table delta |
|---|---:|---:|---:|---:|---:|---:|
| seed0 | 25 | 0 | 0 | 0 | 0 | 0 |
| seed1 | 25 | 5 | 5 | 1 | -7 | 0 |
| seed2 | 25 | 10 | 10 | 1 | -13 | 0 |
| total | 75 | 15 | 15 | 2 | -20 | 0 |

There are no false changes, wrong changes, PARSeq-worsen groups, or Tesseract-worsen groups in the table policy.

## Actual Bitstream

| Split | Actual bpp | Enhancement bpp | Assignment changed fraction |
|---|---:|---:|---:|
| seed0 | 3.755801 | 0.031080 | 0.000 |
| seed1 | 3.957302 | 0.026194 | 0.200 |
| seed2 | 3.835844 | 0.029693 | 0.400 |
| mean | 3.849649 | 0.028989 | 0.200 over selected groups |

The actual bpp exactly matches the existing nearest/current-best K64 utility-weighted assignment streams for these splits.

## OCR Results

Against nearest K64 utility-weighted assignment:

| OCR | Profile | Delta char errors | Improved | Worsened | CER delta |
|---|---|---:|---:|---:|---:|
| PARSeq | unicode_strict_v1 | 0 | 0 | 0 | 0.000000 |
| PARSeq | latin_alnum_ci_v1 | 0 | 0 | 0 | 0.000000 |
| Tesseract PSM8 | unicode_strict_v1 | -20 | 15 | 0 | -0.017513 |
| Tesseract PSM8 | latin_alnum_ci_v1 | -17 | 11 | 0 | -0.015288 |

Against the current best pooled image-diff gate, model seed 1, threshold 0.5:

| OCR | Profile | Delta char errors | Improved | Worsened | CER delta |
|---|---|---:|---:|---:|---:|
| PARSeq | unicode_strict_v1 | 0 | 0 | 0 | 0.000000 |
| PARSeq | latin_alnum_ci_v1 | 0 | 0 | 0 | 0.000000 |
| Tesseract PSM8 | unicode_strict_v1 | -12 | 12 | 0 | -0.010508 |
| Tesseract PSM8 | latin_alnum_ci_v1 | -9 | 8 | 0 | -0.008094 |

W&B result artifact run: `ftpgkenc`. OCR runs: PARSeq `j46syya2`, `p3wsmk0f`, `xhkba819`; Tesseract `j0okdyhl`, `jjjem0b8`, `v68vq2gc`.

## Case Analysis

Case analysis is saved at `experiments/results/eval300_shortlist_oracle_top4_scoreseed2_case_analysis_2026_06_26.json`.

- Changed groups: 15/75.
- Missed oracle-change groups: 2/75.
- Changed-source distribution: IAM 9, ICDAR2015 3, ICDAR2013 2, SynthText 1.
- Changed nearest-Tesseract-distance distribution: distance 1: 1, 2: 3, 3: 1, 4: 4, 5: 3, 6: 1, 9: 2.
- Changed oracle rank distribution inside the original top8 table: rank 1: 6, rank 2: 4, rank 3: 2, rank 6: 2, rank 7: 1.

The selector target is therefore not simply "take top-1 when confident". Useful code choices include deeper rank-6/rank-7 variants, and the safe changes mostly appear in already difficult Tesseract cases. A deployable selector should combine explicit no-op anchoring with hard-positive recall over high-difficulty IAM/ICDAR-like cases, while still rejecting no-headroom groups.

## Interpretation

The top-4 shortlist is now verified as a real counted-bitstream upper-bound substrate. It improves over the current best learned actual stream at identical bpp, without PARSeq regression or Tesseract worsened samples.

This is not a promotable method because it uses assignment oracle labels at selection time. The correct next research target is a deployable selector/calibrator that preserves the explicit no-op anchor and recovers these shortlist changes without using held-out OCR labels. The failure mode remains calibration and harmful/no-headroom suppression, not rate accounting or bitstream implementation.

## Hashes

- Export script: `0475934003548734b59a9c43a4dfd5baf33f57c935f462a8c20e6e40946c2906`
- Result JSON: `0b7791c1829e53aee783ebab8506a4c5fbcb800967bb895c8448245d6c52bb7f`
- Policy JSONL: `cc0239203ca94987e860cebfe5628cff1c1c9b8c970a198121cd76f7f91a47fd`
- Policy summary: `868fa9787f88e8276f6b10b6792841201dd32de64b970fb86e88a7aea0593687`
- Tesseract nearest comparison: `b3a56106e82731ed2225d7bcd3fdae99e3f4233bf2826b860669738f5f6124c9`
- PARSeq nearest comparison: `ddf3571f0ee6182988137583c24f026407e26c58afba0e6447004d09200f3af1`
- Tesseract current-best comparison: `fbe7e039731835c66bb6049740e47b46551985b7f3fb2698bd7f2db878710557`
- PARSeq current-best comparison: `9e7d96b721625d6fe32030390286eaf6383dfe52dbd852e98beeb427b06b3a57`
- Case analysis JSON: `fc7e3019a608b01f7684a3df50ebd80c66756203e08979a664bed2c7e9d245a9`

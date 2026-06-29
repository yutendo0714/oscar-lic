# Eval600 Hard No-Headroom Top-8 Assignment Transfer

**Date:** 2026-06-26  
**Stage:** G5 diagnostic / top-8 compact-symbol assignment calibration  
**Track:** A, pure visual bitstream diagnostic  
**Conclusion:** partial positive data result, negative promotion result. Adding hard no-headroom negatives reduces Eval300 false changes, but the current group/no-op latent-window head still does not beat the promoted actual-bitstream assignment baseline.

## Purpose

The oracle-safe-only Eval600 top-8 transfer failed mainly through Eval300 false changes. This experiment adds hard no-headroom negatives from Eval600: sources with no PARSeq-safe Tesseract-improving single-candidate residual, PARSeq base exact, and Tesseract base edit distance at most 2. These are deliberately high-risk candidates where a model might be tempted to change the code assignment, but most should stay at nearest/no-op.

## Hard Negative Selection

Selection script: `scripts/select_assignment_hard_noheadroom_candidates.py`  
Script SHA256: `ef9986ce4b7466812742454063dcdedbbacf4b6e42dd4d7e5cc856de1b652a04`

Input utility table: `experiments/results/candidate_utility_table_eval600_noeval300_tile4_scale025_q005_with_image_features_2026_06_25.jsonl`

Selection output: `experiments/results/eval600_assignment_hard_noheadroom_candidate_selection_120_2026_06_26.jsonl`  
Selection SHA256: `ad0f0c7492a6ea974311695d3cef77c6a38313b911b8d9187d70e166e33bdeb7`

Selection summary:

- Eligible sources before cap: `332`
- Selected sources: `120`
- Constraints: no safe source, PARSeq base exact, Tesseract base distance `<= 2`
- Source counts: IAM `4`, ICDAR2013 `25`, ICDAR2015 `21`, ICDAR MLT2019 `25`, SynthText `25`, TextOCR `20`

## Top-8 Reconstructions And OCR

Generated split: `data/splits/text_crops_eval600_assignment_hard_noheadroom120_k64_utilityw_a8_top8_2026_06_26.jsonl`  
Split SHA256: `629e1d5057d2ed8d261176addd6cda1c5878b4be24e76ff5cef66248637ea1b3`  
Rows: `960` = 120 groups x nearest+top8 candidates

PARSeq OCR:

- Output: `experiments/results/ocr_eval600_assignment_hard_noheadroom120_parseq_2026_06_26/results.jsonl`
- SHA256: `88213c698c8cb8d1e4dc812751c811628dfcd84411e7598d6c3b8feae95b24ca`
- W&B run: `b4mr5j5a`
- Unicode strict CER: `0.0`, exact rate `1.0`

Tesseract PSM8 OCR:

- Output: `experiments/results/ocr_eval600_assignment_hard_noheadroom120_tesseract_psm8_2026_06_26/results.jsonl`
- SHA256: `ac7d8b92824b7730aaac283db0311b2e103c6e1e58d2e2e8754566977348b16a`
- W&B run: `64vj947g`
- Unicode strict CER: `0.01688`, exact rate `0.95417`, failed samples `0`

Assignment table: `experiments/results/eval600_assignment_hard_noheadroom120_k64_utilityw_a8_top8_table_2026_06_26.jsonl`  
Table SHA256: `0ef95394e68ecd0969d49cbe67a4c8bb9297b8135a97aed7a2e1a5f5c8f07ceb`

Table summary:

- Groups: `120`
- Rows: `960`
- Safe Tesseract gain groups: `4`
- Oracle changed groups: `4`
- Oracle nearest/no-change groups: `116`
- Oracle rank counts: rank 0 `116`, rank 1 `2`, rank 2 `1`, rank 4 `1`

This confirms the selected data are hard no-headroom negatives, not hidden positives.

## Tensor Artifacts

Hard negative tensor artifact: `experiments/features/eval600_assignment_hard_noheadroom120_k64_utilityw_a8_top8_latentctx_windows_2026_06_26.npz`  
SHA256: `f11c050a3ddb72b80c64ae57c1604412420a1b0773b4b0278e8036164ac91506`

Balanced merged artifact: `experiments/features/eval600_oraclesafe123_hardnoheadroom120_plus_eval300_seed1val_top8_windows_2026_06_26.npz`  
SHA256: `4818e593dcd76e70ef6bc131dabc4641c19971c6691351f620a4375e0e8270f2`

Merged groups:

- Eval600 oracle-safe groups: `123`
- Eval600 hard no-headroom groups: `120`
- Eval300 seed1 clean-val groups: `75`
- Total groups: `318`

## Transfer Results

Validation is fixed to Eval300 seed1 clean-val. Eval300 oracle headroom remains `-24` Tesseract edit characters with PARSeq unchanged and 17 changed groups.

### Oracle-choice target with hard negatives

Output: `experiments/results/eval600_hardneg_to_eval300_assignment_group_noop_window_oraclechoice_2026_06_26.json`  
SHA256: `7c2d50d5130b3663c81a8450c0bebf32173438d36323ead592790715662bd612`  
W&B run: `x94exiuz`

Aggregate over model seeds 0/1/2:

- Val row AUC/AP: `0.8932 / 0.6410`
- Val group AUC: `0.6775`
- Threshold `0.5`: changed groups `16.0`, Tesseract delta `-2.0`, false changes `9.33`, wrong changes `3.67`, Tesseract-worsen groups `3.33`, exact oracle groups `3.0`
- Best individual seed at threshold `0.5`: seed 0, Tesseract delta `-5`, false changes `11`, exact oracle groups `5`

Compared with oracle-safe-only transfer, false changes drop from `16.67` to `9.33` and row AP rises from `0.5416` to `0.6410`, but recovered OCR utility remains too low.

### Safe soft-utility target with hard negatives

Output: `experiments/results/eval600_hardneg_to_eval300_assignment_group_noop_window_softutility_2026_06_26.json`  
SHA256: `bed095a233f1ec203d57e906a626ca22c0f10a49c8383c36a1ad60d283098ecc`  
W&B run: `3vzcvcb3`

Aggregate over model seeds 0/1/2:

- Val row AUC/AP: `0.8461 / 0.4974`
- Val group AUC: `0.6839`
- Threshold `0.5`: changed groups `13.0`, Tesseract delta `-1.0`, false changes `8.67`, wrong changes `3.67`
- Train-tuned: collapses to no-op, Tesseract delta `0`

### Hard-positive margin target with hard negatives

Output: `experiments/results/eval600_hardneg_to_eval300_assignment_group_noop_window_oraclechoice_hardpos_2026_06_26.json`  
SHA256: `524013db72699daa4a1221f13937a4a372bfa3b9c62ab17cdf1f2b89cc059bc1`  
W&B run: `5mtvr2r1`

Aggregate over model seeds 0/1/2:

- Val row AUC/AP: `0.6595 / 0.4517`
- Threshold `0.5`: changed groups `33.0`, Tesseract delta `-2.33`, false changes `24.33`, wrong changes `5.67`
- Best individual seed at threshold `0.5`: seed 1, Tesseract delta `-6`, false changes `11`, exact oracle groups `4`

Hard-positive margins recover some positives, but destabilize calibration and do not beat the promoted actual-bitstream baseline.

### Consensus diagnostics

Oracle-choice consensus:

- Output SHA256: `36ddc3381658c771b1744f2435b5a96378b52ba9b30c1a99ae27296b08ac6a16`
- W&B run: `ljgxm9kc`
- Val policy: changed groups `21`, Tesseract delta `0`, false changes `12`, wrong changes `5`

Soft-utility consensus:

- Output SHA256: `0db57e220b5af42af9aba3691bd0aa5a466024ad944d4c6d28a07a3819b7c487`
- W&B run: `u5mrf5q3`
- Val policy: changed groups `29`, Tesseract delta `-3`, false changes `20`, wrong changes `6`

## Interpretation

Adding hard no-headroom negatives is the right data move: it improves validation AUC/AP and reduces false changes substantially. However, the current group/no-op latent-window head trades off false-change precision against hard-positive recall too sharply. The remaining bottleneck is not only no-headroom rejection; it is discriminating the 17 Eval300 oracle-positive groups without opening many no-change groups.

This suggests the next model should not simply add more scalar loss weights. Better directions:

1. source/difficulty-balanced calibration with explicit hard-positive strata;
2. pairwise contrast between oracle-positive Eval600 groups and hard no-headroom groups matched by source, text length and base OCR distance;
3. candidate-local code-effect evidence beyond the current small window head, but only after the balanced data substrate is fixed;
4. policy-level training or thresholding that optimizes exact-oracle hits under a false-change budget.

Do not run counted `.oscr` promotion for these policies. The best learned policy still underperforms the promoted actual-bitstream baseline (`-8` Tesseract edits, PARSeq unchanged, false-change floor near `1/75`).

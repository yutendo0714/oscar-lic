# Eval600 Source-OOF to Eval300 Top-4 Selector Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

ADR-025 rejected selectors calibrated on in-sample Eval600 score outputs. This diagnostic tests the next stricter setup:

> Replace Eval600 train-side first-stage scores with source-modulo out-of-fold scores, keep Eval300 scores external, and retrain the top-4 selector.

If the N037 failure were mainly caused by in-sample score calibration, OOF train scores should make the selector boundary more conservative and transferable.

## Setup

First-stage score generation:

- source split: Eval600 `source_index % 5`;
- train seed: `600`;
- model seed: `2`;
- patch ranker: N034/N035-style diff-crop CNN, `96x32`, `bce_listwise`, image scalar features included;
- targets:
  - `oracle_choice`;
  - `safe_improve`.

Mixed score files:

- Eval600 rows use source-modulo OOF scores;
- Eval300 seed1 rows keep the existing external N034/N035 score model seed 2 outputs.

Second-stage selector:

- same top-4 shortlist selector as N039;
- split mode `seed`: train on Eval600, validate on Eval300 seed1;
- selector seeds `0,1,2`.

## First-Stage OOF Scores

Oracle-choice OOF scorer:

- val AUC/AP: `0.9133 / 0.7409`;
- OOF always-best Tesseract delta: `+0.4`;
- oracle headroom across OOF folds: `-25.6` Tesseract edits, PARSeq `0`.

Safe-improve OOF scorer:

- val AUC/AP: `0.5503 / 0.1161`;
- OOF always-best Tesseract delta: `-0.2`;
- oracle headroom across OOF folds: `-25.6` Tesseract edits, PARSeq `0`.

OOF row metrics are therefore more honest than the in-sample scorer, but neither first-stage top-1 policy is safe enough by itself.

## Second-Stage Selector Results

Means are over 3 selector seeds on Eval300 seed1.

| Policy | Changed groups | Tesseract delta | False changes | Wrong changes | Exact oracle changes | Missed oracle changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| shortlist oracle | 15.00 | -20.00 | 0.00 | 0.00 | 15.00 | 2.00 | 0.00 |
| OOF selector argmax | 4.67 | +0.67 | 4.33 | 4.33 | 0.33 | 16.33 | 0.00 |
| OOF selector train-tuned margin | 1.00 | +0.33 | 1.00 | 1.00 | 0.00 | 17.00 | 0.00 |

## Interpretation

This is a valid negative result for OOF score calibration as the only change.

What improved:

- train-tuned margin is much more conservative than N037;
- false changes fall from `9.56` to `1.00`.

What failed:

- exact oracle recovery is still essentially absent;
- the conservative policy misses all `17` full-oracle changes;
- Tesseract remains worse than no-op;
- the shortlist oracle is still strong (`-20`), so the headroom is not gone.

The conclusion is that in-sample score calibration was a real flaw, but not the whole problem. The second-stage objective still does not learn a useful high-precision hard-positive detector from the available score/patch scalar evidence.

## Next Action

Do not promote or actual-OSCR this OOF selector. The next meaningful branch should train a policy-level objective that directly optimizes false-change suppression and hard-positive recall, or change the representation beyond this small diff-crop scorer. OOF scores remain useful infrastructure for any future selector calibration.

W&B offline runs:

- oracle-choice OOF scorer: `mew0ag4h`;
- safe-improve OOF scorer: `14y9ta7l`;
- top-4 OOF selector: `w74f4anb`.

## Files

- OOF oracle-choice result: `experiments/results/eval600_hardneg_balanced210_oraclechoice_diffcrop_source_oof_seed2_2026_06_26.json`
- OOF oracle-choice scores: `experiments/results/eval600_hardneg_balanced210_oraclechoice_diffcrop_source_oof_seed2_scores_2026_06_26.jsonl`
- OOF safe-improve result: `experiments/results/eval600_hardneg_balanced210_safeimprove_diffcrop_source_oof_seed2_2026_06_26.json`
- OOF safe-improve scores: `experiments/results/eval600_hardneg_balanced210_safeimprove_diffcrop_source_oof_seed2_scores_2026_06_26.jsonl`
- Mixed oracle-choice scores: `experiments/results/eval600_hardneg_balanced210_oraclechoice_diffcrop_source_oof_seed2_mixed_scores_2026_06_26.jsonl`
- Mixed safe-improve scores: `experiments/results/eval600_hardneg_balanced210_safeimprove_diffcrop_source_oof_seed2_mixed_scores_2026_06_26.jsonl`
- Selector result: `experiments/results/eval600_source_oof_to_eval300_top4_policy_selector_scoreseed2_2026_06_26.json`

# Eval300 Group/No-op Assignment Follow-up

**Date:** 2026-06-25  
**W&B:** offline runs `9z4trvnq`, `qwbu6asw`, `nay55x83`, `o882h3gb`  
**Scope:** table-level assignment diagnostics over the frozen nearest-plus-top4 structured-window artifact. These runs are not promoted to actual `.oscr` reconstruction because none beats the current promoted actual-bitstream pooled image-diff gate.

## Baseline To Beat

Current promoted learned baseline:

- policy: pooled image-diff group gate, model seed 1, threshold 0.5
- actual measured mean bpp: `3.849649`
- PARSeq edit delta versus nearest: `0`
- Tesseract edit deltas versus nearest: `-8` unicode / `-8` latin-alnum
- policy precision shape: 5 changed groups, 4 exact oracle hits, 1 false change, 0 wrong changes

## Follow-up Runs

| run | intent | key policy result | conclusion |
|---|---:|---:|---|
| `eval300_assignment_group_noop_window_v0_seed0_to_9_2026_06_25.json` | test whether the window group/no-op v0 has hidden strong model seeds | best `group_threshold_0.5` seed reaches only `-5` Tesseract edits; 10-seed mean `-3.6`, PARSeq `0`, false changes mean `12.5` | no hidden promotion seed |
| `eval300_assignment_group_noop_window_strongtarget_posw_seed0_1_2_2026_06_25.json` | train group head on strong Tesseract improvements (`<= -2`) with positive-class weighting | group policies miss almost all oracle changes and produce positive Tesseract deltas; row argmax mean only `-3.0` | strong target does not generalize |
| `eval300_assignment_group_noop_codeonly_rewardsoft_seed0_1_2_2026_06_25.json` | replace one-hot row CE with Tesseract reward-softmax listwise loss | row/group policies worsen Tesseract (`+5.7` to `+6.7` mean) with many wrong/false changes | reward-softmax overfits harmful Tesseract variation |
| `eval300_assignment_group_noop_*_candidate_bce_seed0_1_2_2026_06_25.json` | train row BCE only over non-nearest candidates with positive weighting | window variant reaches val AUC/AP `0.767/0.431`, but train-tuned policy gives only `-4.0` Tesseract edits with `10.33` false changes | candidate-only supervision does not fix policy precision |

## Diagnostics

The v0 10-seed run improves row validation AP relative to several earlier negative controls, but policy behavior remains dominated by false changes:

- `group_threshold_0.5`: mean changed groups `15.5`, exact oracle `2.9`, false changes `12.5`, Tesseract worsen groups `3.0`.
- `group_train_tuned`: mean changed groups `18.7`, exact oracle `3.2`, false changes `15.4`, Tesseract worsen groups `3.6`.
- `row_argmax`: mean changed groups `15.8`, exact oracle `3.2`, false changes `12.5`, Tesseract delta `-3.9`.

A small rank/error/OCR-difficulty heuristic sweep was also tried. Train-tuned safe heuristics did not transfer: the top strict train rules selected almost no train changes and produced many val false changes; val-safe upper heuristics reached only about `-1` Tesseract edit under false-change <= 1.

## Split Diagnostic

The trainer was extended with `--split-key-index` / `--val-key-value` so calibration can be checked against alternative `group_keys` splits. A code-only leave-real-seed-out diagnostic over `group_keys[:,0]` was run as an exploratory secondary check, not as a primary claim:

| holdout real seed | oracle Tesseract headroom | fixed threshold mean Tesseract delta | row argmax mean Tesseract delta | main failure |
|---:|---:|---:|---:|---|
| 0 | `-33` | `-3.67` | `-1.67` | many wrong changes (`9.0` mean at threshold 0.5) |
| 1 | `-29` | `+0.33` | `-0.67` | weak group AUC and false/wrong changes |
| 2 | `-40` | `-2.33` | `-3.0` | many missed oracle changes |

W&B run: `qwbu6asw`.

This supports the same conclusion: cross-split calibration alone is unlikely to rescue the current row scorer. The representation or candidate-conditioned evidence must improve before threshold calibration can be trusted.

## Tree Ranker Diagnostic

To test whether the failure was specific to neural MLP/CNN calibration, `scikit-learn==1.4.2` was added to `.venv-mlic` and a small tree-ranker diagnostic was implemented in `scripts/train_assignment_tree_ranker.py`. It uses tabular summaries of code centers, latent vectors and latent windows rather than raw image tensors.

| model | val AUC / AP | best policy behavior | conclusion |
|---|---:|---:|---|
| ExtraTrees | `0.942 / 0.883` | argmax selects no changes; train-tuned budget sends 20 groups but gives mean Tesseract `+0.67` with `16.67` false changes | high row separability still does not rank useful non-nearest changes |
| HistGradientBoosting | `0.942 / 0.872` | row argmax sends 7 groups for only `-1` Tesseract edit; train-tuned budget sends 20 groups for `-1` with 17 false changes | no promotion path |

W&B run: `nay55x83`. This reinforces the earlier warning: row AUC/AP is not an adequate proxy for actual OCR policy gain.

## Candidate-only BCE Diagnostic

The trainer was also extended with `--candidate-only-row-loss` and `--candidate-row-pos-weight` to test whether nearest/no-op rows were overwhelming the row objective. Two variants were run:

| variant | val AUC / AP | `group_threshold_0.5` | `group_train_tuned` | `row_argmax` | conclusion |
|---|---:|---:|---:|---:|---|
| code-only | `0.676 / 0.372` | Tesseract `-1.67`, false changes `9.33` | Tesseract `-3.67`, false changes `11.67` | Tesseract `-1.67`, false changes `31.0` | weaker than promoted gate |
| window | `0.767 / 0.431` | Tesseract `-3.0`, false changes `9.33` | Tesseract `-4.0`, false changes `10.33` | Tesseract `-2.33`, false changes `22.33` | better row metrics, still poor policy precision |

W&B run: `o882h3gb`.

This closes a useful loophole: the failure is not only because the no-op row dominates the row loss. Even when row supervision is restricted to non-nearest candidates, the learned policy still sends too many false changes and recovers too little of the oracle Tesseract headroom.

## Decision

Do not promote these group/no-op variants. The current bottleneck is not simply group-target imbalance or a scalar reward target. The next useful direction should either:

1. add a genuinely better local glyph/code-effect representation for row assignment, or
2. use cross-fitted calibration with a stronger held-out validation protocol before any actual `.oscr` promotion attempt.

For now, keep the pooled image-diff actual-bitstream gate as the learned baseline.

## Artifact Hashes

```text
afe88657dca65ea8f5c3285a8b3bcb24e854922fb9334ea70ed11e4793d42c88  experiments/results/eval300_assignment_group_noop_window_strongtarget_posw_seed0_1_2_2026_06_25.json
77ee62a31e33a461a2c6b6e08112ab42c314b5a4f1d4e47f1c3b42a352f21fa1  experiments/results/eval300_assignment_group_noop_codeonly_rewardsoft_seed0_1_2_2026_06_25.json
afd0057085d4e507e5e9d712320e8197069de873a9b5c26853c4cd7b28a2ca30  experiments/results/eval300_assignment_group_noop_window_v0_seed0_to_9_2026_06_25.json
75af1d5b760dcc7b5562feb17d3e780a010eaad0f702f454b28b98ef584bd2ac  experiments/results/eval300_assignment_group_noop_window_strongtarget_posw_scores_seed0_1_2_2026_06_25.jsonl
a1a86ad4149de5b78d6f2a816f32fe08c0b70492872d976bb95777970fab0c15  experiments/results/eval300_assignment_group_noop_codeonly_rewardsoft_scores_seed0_1_2_2026_06_25.jsonl
85b6c80eaf30d5506d8b63cedc319eb69b49d2b790ac5c1130ee3cf4634ff7ac  experiments/results/eval300_assignment_group_noop_window_v0_scores_seed0_to_9_2026_06_25.jsonl
1a137302521deeb55d8e7fa6a1c64b17974dba1d9868cb12eaa0c7909f971bfc  experiments/results/eval300_assignment_group_noop_codeonly_realseed0_seed0_1_2_2026_06_25.json
5987a8e1a852d4e979c935025c9fb4c595cc2ff19f554ec704dc833be21f3b90  experiments/results/eval300_assignment_group_noop_codeonly_realseed1_seed0_1_2_2026_06_25.json
c72633c8a50ab701ed9460c0c100de64d135695295122b00540753629751435e  experiments/results/eval300_assignment_group_noop_codeonly_realseed2_seed0_1_2_2026_06_25.json
d7b22ddca1e2cbeca8c8aaf15c4febd44bc147b2d31548d39290359882a1c039  experiments/results/eval300_assignment_tree_extra_trees_val1_seed0_1_2_2026_06_25.json
4ff0c63ccf10cbedb03c46f6fe0e4f2d202533aa16ab0d4aeead0b46bc589d6b  experiments/results/eval300_assignment_tree_histgbdt_val1_seed0_1_2_2026_06_25.json
6062fe89cd22949b3287a7926227fca7d5ff12528e16fe282a6b916ce80ec4cb  environment/locks.mlic_py310_cu118_plus_sklearn_2026_06_25.txt
454dba519d8cf1874e62512cda43e39d1a6f0839f1edec4f2c25efaa42f53dd8  scripts/train_assignment_group_noop_window.py
a0301511b3c136dc631856fc72fe9d04ad29ad9f608cff365d36ed354ae0ed51  experiments/results/eval300_assignment_group_noop_codeonly_candidate_bce_seed0_1_2_2026_06_25.json
a4e803d8220f501a8af5c89ef92ad4a696812058744a4bbaaf294f898e14b6d7  experiments/results/eval300_assignment_group_noop_codeonly_candidate_bce_scores_seed0_1_2_2026_06_25.jsonl
ba7ee8e6e750948d71ed8d80481aaa914c1131382ddc64546f66de3dd14f56fc  experiments/results/eval300_assignment_group_noop_window_candidate_bce_seed0_1_2_2026_06_25.json
8ab35c9e8098b092b5a3f0507dcaf44d0cf135a4b312a8510ccde2fc975b72c5  experiments/results/eval300_assignment_group_noop_window_candidate_bce_scores_seed0_1_2_2026_06_25.jsonl
```

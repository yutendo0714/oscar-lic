# Eval300 Top8 Latent-Window Artifact and Oracle-Label Precheck

Date: 2026-06-25

## Purpose

The previous nearest-plus-top4 latent-window artifact missed clean-val oracle groups: only `10/17` actual oracle-change groups were present. This run exports a nearest-plus-top8 structured latent-window artifact so assignment models can see the full candidate set used by the clean Eval300 table.

It then trains a tensor-backed latent-window code-effect reranker on the actual safe assignment labels. This is a table precheck only; no `.oscr` stream is promoted.

## Artifact Result

- Output: `experiments/features/eval300_assignment_top8_codeeffect_latentctx_windows_parseqfeat_seed0_1_2_2026_06_25.npz`
- Groups: `300`
- Rows: `2400`
- Shape: `mask [300,8]`, `latent_window_* [300,8,32,12,12]`
- Truncated groups: `0`
- Oracle-missing groups: `0`

This is useful infrastructure. It restores top8 oracle recall for the next assignment models.

## Model Precheck

Settings:

- target: `labels` (actual assignment oracle choice)
- model seeds: `0,1,2`
- latent-window branch enabled
- epochs: `300`
- hidden dim: `96`
- no-change weights: `0.5` and `2.0`
- PARSeq harm weight: `0.1`

| Variant | Val AUC/AP | Argmax policy | Best margin precheck |
|---|---:|---|---|
| `no_change_weight=0.5` | `0.8918 / 0.6124` | `-4` Tesseract edits, `11.67` false changes, `3` worsen groups | best seed reaches `-6` edits with `6` false changes |
| `no_change_weight=2.0` | not separately promoted | similar policy family | best seed still `-6`; stricter seed reaches only `-2` with lower false changes |

The top8 oracle is `-24` Tesseract edits with PARSeq unchanged, so the candidate set has headroom. The current latent-window classifier still cannot convert high row AUC/AP into a high-precision policy.

## Decision

Do not run counted `.oscr` for these policies. Keep the top8 latent-window artifact as the next training substrate, but change the objective/architecture before another promotion precheck.

Most likely next improvements:

- learn an explicit calibrated change/no-change head rather than using row argmax plus post-hoc margin;
- train with a policy-level false-change penalty or differentiable top-k/group objective;
- use top8 artifact for future models, since top4 recall was an avoidable ceiling.

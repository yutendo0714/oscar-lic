# Compute and Storage Budgeting

Exact cost depends on target hardware and must be measured, not guessed. This document defines the accounting method and stop rules.

## 1. Required pilot measurements

For each major stage, run at least 100 representative crops/images and record:

- mean, median and p90 wall time
- GPU seconds and peak memory
- CPU time and peak RSS
- bytes written for intermediate features/labels
- failure/retry rate

Extrapolate with confidence bounds before launching a full dataset.

## 2. Counterfactual oracle cost

With `N` candidates and greedy selection of `K` packets, naïve evaluation can require approximately:

\[
\sum_{k=0}^{K-1}(N-k)
\]

codec/OCR evaluations per image. This is the largest likely research bottleneck.

Cost controls, in order:

1. exhaustive oracle only on tiny crops/candidate sets;
2. cache base and candidate decoder features;
3. screen candidates using a cheap upper bound;
4. evaluate a shortlist per step;
5. sample states/coalitions for interaction estimates;
6. stop once the G4 conclusion is statistically stable.

Do not generate full-dataset oracle labels before recording the 100-crop estimate in `BLOCKERS.md`.

## 3. Storage formula

For `M` source samples, `N_l` labeled states per sample and `F` bytes per label record:

\[
Storage \approx M N_l F + feature\ caches + bitstreams + reconstructions.
\]

Store tensors in chunked, checksummed formats. Avoid one tiny file per candidate. Keep raw source data outside the repository and place only manifests/hashes here.

## 4. Checkpoint retention

Retain:

- best validation checkpoint per seed/operating point;
- last checkpoint for recovery;
- checkpoints used in any reported table;
- optimizer state only while training remains active.

Record deleted intermediate checkpoints and their selection criterion. Never delete a reported checkpoint before artifact release.

## 5. Run launch rule

A run requiring more than the locally approved GPU-hour or storage threshold needs a written estimate and decision entry. An AI agent must not infer unlimited compute from the research objective.

## 6. Current Eval300 Pilot Snapshot

Measured on 2026-06-25 in `/workspace/oscar-lic`:

- GPU visible: NVIDIA GeForce RTX 4070 Ti SUPER, 16GB.
- `experiments/features`: `7.8M`.
- `experiments/results`: `589M`.
- `experiments/reports`: `224K`.
- `experiments/manifests`: `16K`.
- `wandb`: `21M`.
- Main reproducibility manifest:
  - path: `experiments/manifests/oscar_lic_12h_repro_manifest_2026_06_25.yaml`;
  - size: `6583` bytes;
  - role: checkpoint/artifact/W&B/validation index for the 12h research pass.
- Structured local latent-window full train+val artifact:
  - path: `experiments/features/eval300_assignment_top4_codeeffect_latentctx_windows_seed0_1_2_2026_06_25.npz`;
  - size: `2.7M`;
  - groups/rows: `300` / `1213`;
  - arrays: `[300, 5, 32, 12, 12]` for each `latent_window_*` tensor.

This scale is safe for rapid assignment-head iteration. The next storage jump will come from larger OCR counterfactual tables and actual reconstructed `.oscr` outputs, not from the compact tensor/window artifacts.

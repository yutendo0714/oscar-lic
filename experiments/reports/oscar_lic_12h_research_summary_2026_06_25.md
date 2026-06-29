# OSCAR-LIC 12h Research Summary

Date: 2026-06-25

This report summarizes the 12h autonomous research pass over the OSCAR-LIC workspace. Detailed running state is in `STATUS.md`; the next-worker handoff is in `docs/HANDOFF.md`.

## Main Outcome

The strongest learned result remains a frozen-base, actual-byte-counted assignment improvement on top of corrected MLIC++:

- base codec: corrected MLIC++ lambda `0.025`;
- checkpoint SHA256: `67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559`;
- promoted policy: pooled image-diff group gate, model seed 1, threshold `0.5`;
- actual mean bpp unchanged versus nearest K64 assignment: `3.849649`;
- enhancement mean bpp: `0.028989`;
- PARSeq: unchanged exactly;
- Tesseract PSM8: `-8` unicode / `-8` latin-alnum edit characters versus nearest over seed0/1/2 val75;
- policy precision: `5/75` changed groups, `4` exact oracle hits, `1` false change, `0` wrong changes.

This is the current learned baseline. It should not be displaced by table-only metrics or estimated bitrate.

## Scientific Diagnosis

The central bottleneck is now code-effect ranking, not serialization or no-op gate recall.

- The top-8 assignment oracle at fixed selected candidates and fixed code-index payload has `-24` unicode / `-19` latin Tesseract edit-character headroom with PARSeq unchanged.
- The promoted image-diff gate recovers 4 oracle-change groups and misses 13. Score coverage shows all 13 misses are scorer-ranking failures, not gate rejection.
- Relaxing the current gate is unsafe because the current scorer's best choices on missed groups sum to `+2` Tesseract edits.
- AUC/AP is unreliable for promotion. Several high-AUC experiments worsen OCR policy or create many false changes.

## What Was Added Late In The Run

Tensor/latent diagnostics:

- Consolidated report: `experiments/reports/eval300_assignment_tensor_reranker_diagnostics_2026_06_25.md`.
- Machine-readable summary: `experiments/results/eval300_assignment_tensor_reranker_diagnostics_summary_2026_06_25.json`.
- Final diagnostics W&B run: `1s91vwj5`.

Structured local latent-window support:

- Exporter now supports `--include-latent-windows` and `--latent-window-radius`.
- Full train+val artifact: `experiments/features/eval300_assignment_top4_codeeffect_latentctx_windows_seed0_1_2_2026_06_25.npz`.
- Artifact SHA256: `9a1fab0995360ed5392a3b85567306f19573d6a8739a55198905bb66f93faa5a`.
- Arrays: `latent_window_yhat`, `latent_window_target`, `latent_window_residual`, each `[300, 5, 32, 12, 12]`.
- Trainer now supports `--use-latent-windows`.

Window-CNN smoke:

- h96 policy: mean Tesseract `-4.33`, false changes `14.67`;
- h64 regularized policy: mean Tesseract `-4.0`, false changes `14.33`;
- h96 with `no_change_weight=2.0`: mean Tesseract `-3.0`, false changes `14.0`;
- post-hoc margin gates over the h96 window scores: best seed0 Tesseract `-6`, with too many changes and false positives;
- conclusion: structured windows are a useful data substrate, but a standalone pooled CNN branch and scalar no-change pressure are not enough.

## Negative Results To Respect

Do not spend more time on these directions without a new hypothesis:

- flat 512-D latent concatenation;
- simple latent side branch;
- tiny pooled latent CNN branch;
- standalone structured-window CNN branch;
- direct scalar non-improve/harm logit penalties;
- direct group-change margin supervision over the same row logits;
- seed sweeps of the same code-effect MLP;
- post-hoc margin gates as a paper claim unless actual OCR and precision both beat the promoted baseline.

## Next Experiment

Build a small group-aware assignment model:

1. Input: nearest plus top4 candidates, code-center direction, core features, and structured latent windows.
2. Architecture: shared local code-effect/window encoder plus explicit group/no-op head.
3. Loss: listwise oracle-choice target with nearest as no-op anchor, plus group-level no-change calibration.
4. Selection: policy-level Tesseract/PARSeq deltas and false/wrong/missed oracle groups, not AUC/AP.
5. Promotion: actual `.oscr` reconstruction, complete-byte bpp, PARSeq/Tesseract row-wise comparison, and Tesseract better than `-8/-8` at unchanged bpp.

Starting config: `configs/training/assignment_group_noop_window_v0.yaml`, SHA `6660f3980cf7728913d51b28eff88920badec092f7ad94afd3e372f11037e20a`.

## Final Validation State

Latest lightweight validation passed:

- GPU visible: NVIDIA GeForce RTX 4070 Ti SUPER, driver `570.148.08`;
- Python compile passed for touched scripts;
- unit tests: `34 passed`;
- `scripts/validate_repo.py`: OK, 59 structured files, expected unresolved DCAE checkpoint warning only.
- `scripts/verify_repro_manifest.py`: OK, all referenced hashes matched.

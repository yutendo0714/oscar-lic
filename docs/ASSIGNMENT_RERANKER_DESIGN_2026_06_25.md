# Assignment Reranker Design

**Date:** 2026-06-25  
**Scope:** next OSCAR-LIC assignment head after the pooled image-diff group gate.

## Current Promotion Baseline

The best learned actual-bitstream assignment result is the pooled image-diff group gate over the simple codegeom oracle-choice scorer:

- actual bpp is unchanged versus nearest K64 assignment: mean `3.849649`, enhancement mean `0.028989`;
- PARSeq is unchanged exactly across seed0/1/2 val75;
- Tesseract improves by `-8` unicode and `-8` latin-alnum edit characters over nearest;
- only `5/75` groups change, with `4` exact oracle hits, `1` false change and `0` wrong changes.

This is the precision floor for the next scorer. A new model should not be promoted if it recovers more oracle cases by reintroducing broad false/wrong changes.

## Failure Diagnosis

The score-coverage and feature audits give a consistent diagnosis:

- the promoted gate is not the recall bottleneck: all `13` missed oracle-change groups are missed because the assignment scorer does not rank the oracle code first;
- blindly taking the current scorer best on those missed groups would add `+2` Tesseract edits, while the oracle has `-15` edits of headroom;
- top-M upper controls show shortlist potential: the current scorer can reach `-15` Tesseract characters if an oracle reranker can choose within score top4;
- a first top4 diff-crop/image-feature reranker is negative despite high AUC/AP: best-score policy is `+1` Tesseract edit, with `10` missed-oracle groups.
- a post-hoc score-margin gate over the first tensor code-effect scorer can tie the promoted `-8` aggregate Tesseract edit gain only when tuned on val; train-tuned thresholds fall back to `-7` and still keep many false changes, so this is a calibration diagnostic rather than a promotion.
- direct non-improve/harm logit penalties over the current tensor representation weaken policy-level Tesseract gains, so the false-change issue should be handled through representation and group-aware calibration rather than a scalar candidate suppression term alone.
- direct best-non-nearest-vs-nearest group-change supervision is also weak with the current row-logit representation, so the no-op decision likely needs its own group head and better context features.
- local latent context vectors (`y_hat`, `g_a(x)`, residual) are now available in a tensor artifact, but raw concatenation raises AP while weakening policy gain to about `-2` Tesseract edits. A simple separated latent branch only recovers to about `-5.33` and keeps too many false changes, a minimal pooled latent-CNN branch reaches only about `-5.0`, and margin gating does not rescue the branch. The exporter and trainer now also support structured local latent windows, with a full train+val artifact shaped `[300, 5, 32, 12, 12]`; the first standalone window-CNN branch is also negative (`-4.33`/`-4.0` mean Tesseract edits). The next latent model should use those local windows with joint no-op/group calibration instead of flattening context into the code-effect MLP, adding an uncalibrated side branch, or using a standalone pooled latent CNN.

Feature audit artifacts:

- `experiments/results/eval300_assignment_codegeom_imgdiff_feature_audit_model0_val75_2026_06_25.json`, SHA256 `05b6c57ecca516e340bfb085150272e981841d40efa55757cbfc54afc344b9cb`;
- `experiments/results/eval300_assignment_top4_imgfeat_reranker_feature_audit_model0_val75_2026_06_25.json`, SHA256 `19276c8a0346db991f4bdec3e880dafaede1757f4013d6f7639d2d1a504a68e3`;
- audit script `scripts/analyze_assignment_feature_audit.py`, SHA256 `35347dd4b51234366a05e60c70c4c0794a01290a008f28b076282ba462110f87`;
- W&B offline runs `2cr5oxi8` and `mes4amho`.

The audit also shows why AUC is misleading here. Core rank/error features have high univariate oriented AUC (`~0.91` max), and image/codebook features have high maxima but low family means. They are acting as weak filters, not as a candidate-specific glyph repair model.

## Next Model: Local Latent/Code-Effect Reranker

Keep the design small, but move the complexity to the representation that is currently missing.

Inputs per candidate code:

- nearest code index, candidate code index, top-k rank, relative residual error;
- trainable embeddings for nearest code, candidate code and candidate-minus-nearest direction;
- full K64 codebook center vectors or a learned projection of center vectors, not only scalar summaries/one-hot bits;
- local frozen-base latent tokens around the selected tile/channel group, including `y_hat`, residual direction and spatial position;
- local glyph crop from original/base reconstruction, but as context for the latent/code direction rather than as a standalone bitmap-diff classifier.

Architecture:

- code-effect encoder: small MLP over projected center-nearest direction, relative error and code embeddings;
- local context encoder: compact shared CNN or latent-token MLP over the selected tile neighborhood;
- group reranker: listwise scoring over nearest plus top4 candidates, with nearest as the explicit no-op anchor;
- auxiliary heads: Tesseract-gain, PARSeq-harm and no-worse logits, used for calibration and diagnostics;
- optional shared gate: predict whether any non-nearest code should be allowed, but never use it alone as the scorer.

Loss:

- primary listwise cross entropy to `label_assignment_oracle_choice`, with nearest fallback when the oracle is absent from the shortlist;
- pairwise margin terms for `oracle > nearest` and `oracle > current_score_best` in oracle-change groups;
- false-change suppression term for no-headroom groups, weighting non-nearest choices that have `tesseract_delta_vs_nearest >= 0`;
- PARSeq no-worse penalty for positive `parseq_delta_vs_nearest`;
- report row AUC/AP, but select by policy-level Tesseract delta, PARSeq delta, false/wrong changes and actual `.oscr` OCR.

Promotion criterion:

- same selected candidates and same code-index payload size as nearest;
- actual `.oscr` bpp identical to the promoted assignment baseline;
- PARSeq unchanged exactly or strictly no worse under row-wise comparison;
- Tesseract beats `-8` unicode / `-8` latin-alnum edits versus nearest;
- false+wrong changes stay near the current precision floor (`<= 1/75`) unless the recovered oracle gain is clearly larger and PARSeq remains unchanged.
- post-hoc margin gates are allowed as diagnostics, but a promotion should prefer a jointly calibrated scorer/no-op head unless a margin-gated policy beats the promoted baseline on both aggregate OCR and precision.

## Immediate Implementation Steps

1. Export compact tensor artifacts for top4 assignment groups:
   - code indices, ranks, relative errors;
   - full codebook center vectors for nearest/candidate/delta;
   - local latent-token windows around the selected candidate;
   - optional glyph crops already used by diff-crop, kept as context.
2. Extend the tensor-backed train script from the current standalone `latent_window_*` branch to a shared scorer plus explicit group/no-op head instead of widening the JSONL scalar table.
3. Train on seed0/1/2 cross-fits and evaluate all policies before any actual-bitstream run.
4. Run the feature audit and policy-error scripts on every candidate reranker.
5. Promote only after explicit policy export, counted `.oscr` reconstruction, PARSeq/Tesseract OCR and pairwise comparison.

# Eval600 Hard-Negative Balanced210 Top-k Score Verifier

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N031 showed that adding 210 source-balanced no-safe groups does not rescue the current single-score group/no-op assignment head. However, the first-stage scorer still has useful shortlist recall: the Eval300 seed1 top-4 upper recovers many oracle changes, and top-8 contains all 17 oracle changes.

This diagnostic asks whether a lightweight second-stage verifier can turn that shortlist into a safe deployed policy when it uses only deployable first-stage score features:

- score aggregates across model seeds;
- margins against nearest/no-op score;
- vote fraction across model seeds;
- rank/top-k position;
- group-change probability aggregates.

OCR deltas are used only as labels, train-threshold constraints and evaluation metrics. They are not used as inference features.

## Method

Added `scripts/train_assignment_topk_score_verifier.py`.

The verifier builds group-level and candidate-level logistic/MLP classifiers from the score JSONL emitted by the N031 group/no-op window head:

`experiments/results/eval600_hardneg_iamsupp20_balanced210_to_eval300_assignment_group_noop_window_oraclechoice_scores_2026_06_26.jsonl`

Train split:

- 473 Eval600 groups
- split key index 1 != Eval300 val seed

Validation split:

- 75 Eval300 seed1 groups
- 17 assignment-oracle change groups
- oracle top-8 Tesseract delta: `-24`
- oracle top-8 PARSeq delta: `0`

The policy tunes thresholds on train under strict zero tolerance for:

- false changes;
- wrong changes;
- Tesseract-worsen groups;
- PARSeq-worsen groups.

Then it transfers the learned threshold to Eval300 seed1.

## Results

| Diagnostic | Top-M oracle upper Tesseract delta | Val candidate AUC/AP | Val group AUC/AP | Val learned policy mean Tesseract delta | Val false changes mean | Notes |
|---|---:|---:|---:|---:|---:|---|
| top-2 MLP strict | -15 | 0.2858 / 0.0517 | 0.2651 / 0.0978 | +0.67 | 1.67 | no improving changes |
| top-4 MLP strict | -17 | 0.3212 / 0.0328 | 0.2875 / 0.1181 | +0.33 | 1.00 | best seed is no-op |
| top-4 linear strict | -17 | 0.2560 / 0.0270 | 0.2857 / 0.1191 | +0.67 | 1.33 | linear simplification is worse |
| top-8 MLP fast strict | -24 | 0.3464 / 0.0259 | 0.3071 / 0.1694 | +2.33 | 3.00 | all selected changes are harmful or neutral |

Key observation:

- The top-M upper remains strong, especially top-8 (`17/17` recovered, `-24` edits).
- The score-derived verifier is externally anti-calibrated: candidate and group AUCs on Eval300 are far below `0.5`.
- Train metrics are artificially perfect for these score features, but transfer policies miss all 17 useful Eval300 oracle changes.

The top-8 full-grid run was interrupted after writing partial policy files and is not used as evidence. The successful top-8 evidence is the `fast_strict` run listed above.

W&B offline run: `blrnzrcc`

## Interpretation

This is a valid negative result for "top-k verifier over current score features" as a promotion path.

The experiment changes the decision structure relative to N029 by training separate candidate and group verifiers, but because both verifiers only see first-stage score/rank/vote features, they inherit the same external calibration failure. This rules out a cheap score-only second stage.

The useful part is the upper-control gap:

- top-4/top-8 still contain enough oracle changes;
- the missing piece is not shortlist recall;
- the missing piece is deployable code-effect evidence that transfers across Eval600/Eval300 source and difficulty shifts.

## Decision

Do not continue with score-only top-k verifier, threshold, budget, consensus, or linear/MLP variants over the same score JSONL files.

Next assignment attempt should use richer candidate-local evidence before the policy layer, for example:

- local latent code-effect predictors trained directly on safe assignment labels;
- candidate-conditioned residual/codebook geometry beyond current score aggregates;
- source/domain-disjoint calibration labels;
- cross-fit policy objectives that optimize false-change suppression and hard-positive recall from deployable non-score features.

No counted `.oscr` promotion stream was run because table-level transfer is worse than no-op and far below the promoted actual-bitstream baseline.

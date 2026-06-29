# Conditional Utility Stop Model Plan

Date: 2026-06-26

## Motivation

N069-N075 reject independent additive multi-candidate utility and shallow conditional feature allocators for the current raw-residual enhancement setting.

Key evidence:

- N071 smoke40: Tesseract unicode best fixed single is `-63` edit errors at `6.0395` bpp, while best pair is `-36` at `8.0560` bpp and triple is `-32` at `10.0711` bpp despite higher PSNR.
- N072 conditionals: Tesseract unicode base-to-single edges improve by `-179`, while single-to-pair edges worsen by `+178`.
- N073 edge table: `0_to_1` has `116/120` Tesseract-improve/PARSeq-safe edges; `1_to_2` has only `11/240`.
- N074 stop baseline: `first_only` reaches precision `0.9667`, recall `0.7582`, F1 `0.8498` on the diagnostic edge label.
- N075 feature allocator: transition/candidate/rate/PSNR/source features do not beat `first_only`, and even a non-deployable OCR-state upper-control is worse than fixed `single_a` under greedy subset policy.

This means the next multi-candidate allocator must model allocation state. A candidate that is useful as the first enhancement unit is usually not useful after another unit has already been sent.

## Design Rule

Do not train a stage-agnostic additive utility score for multi-candidate allocation.

Instead, model:

1. first-candidate utility;
2. continue utility conditioned on the selected set;
3. an explicit stop/no-op decision;
4. policy-level false-change and missed-change costs.

## Minimal Model Family

Prefer a small model first.

Inputs should be deployable or decoder/encoder-available without held-out OCR:

- candidate identity features: tile, slice/channel group, code index, code rank, residual/code geometry;
- candidate local image/code-effect features from original/base/candidate recon crops;
- selected-set state: count, aggregate residual/code features, overlap indicators, current budget;
- candidate-vs-selected interaction: same tile/slice, neighboring tile, residual sign conflict, patch-diff cosine/L1;
- source/base quality proxies that do not use held-out OCR.

Outputs:

- `p_first_safe`: probability candidate is useful as the first sent unit;
- `p_continue_safe`: probability candidate is useful given current selected set;
- `p_stop`: probability additional candidates should be rejected;
- optional calibrated reward: expected Tesseract-like edit reduction with PARSeq-safe penalty, used only as a training label from oracle diagnostics.

## Label Policy

For diagnostic labels, start from:

```text
safe_continue_unicode =
  tesseract_delta_unicode < 0
  and parseq_delta_unicode <= 0
```

Use separate transition labels:

- `0_to_1`: first-unit label;
- `1_to_2`: second-unit continue label;
- `2_to_3`: third-unit continue label, diagnostic only until lower-rate compact symbols exist.

The N073 table is too small for promotion. It is only a sanity substrate for loss/selection design.

## Evaluation Floor

Any learned conditional allocator must beat all of the following before scaling labels or running actual `.oscr` promotion:

- N074 `first_only` row F1 `0.8498`;
- N071 best fixed single policy for Tesseract unicode (`single_a`, `-63` edit errors);
- N071 subset oracle regret of best-single stop (`9` Tesseract unicode edits) should be reduced without large false positives;
- PARSeq must remain non-worse.

Do not optimize row AUC alone. Report:

- row precision/recall/F1 by transition;
- selected edge count by transition;
- Tesseract/PARSeq delta sums;
- policy-level best-subset regret;
- actual bpp if converted into streams.

## N075 Outcome

The planned tiny N073 diagnostic has been run:

- transition-only exactly matches `first_only`;
- adding candidate identity, selected-set bits, enhancement bpp, PSNR gain and source/domain lowers edge F1;
- non-deployable OCR-state source distances improve edge Tesseract delta but fail coherent greedy subset selection;
- no learned rule beats the row-level `first_only` floor or the fixed one-candidate `single_a` policy.

Conclusion: do not scale multi-candidate labels for this feature family. Return to compact one-candidate symbol/assignment improvement, or introduce a materially richer selected-set/candidate-local code-effect representation before revisiting multi-candidate allocation.

## Next Experiment

Only revisit this line when at least one of the following changes is available:

1. compact symbols whose second/third units show conditional OCR gain under actual counted streams;
2. structured selected-set/candidate-local latent/code-token evidence beyond scalar rate/PSNR/source features;
3. a verifier or objective that beats fixed `single_a` on policy-level Tesseract/PARSeq without using held-out OCR at inference.

Promotion remains blocked until the allocator uses compact symbols and real counted `.oscr` streams.

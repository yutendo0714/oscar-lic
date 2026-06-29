# Eval600 to Eval300 No-Headroom Proxy Diagnostic

Date: 2026-06-25

## Purpose

The utility strata analysis showed a strong no-headroom structure: when base Tesseract is already exact, no candidate safely improves Tesseract while preserving PARSeq, but harmful candidates still exist. Because held-out Tesseract distance cannot be used as an inference feature, this diagnostic asks whether deployable image/base-reconstruction aggregate features can predict safe one-candidate headroom.

Target: group has at least one candidate with `tesseract_delta_distance < 0` and `parseq_delta_distance <= 0`.

This is a diagnostic classifier, not a compression result or promoted selector.

## Setup

- Train: Eval600 no-overlap candidate utility table
- Eval: fixed Eval300 candidate utility table
- Feature modes:
  - `image_only`: image geometry, candidate layout, actual candidate bytes, PSNR, and image/base patch error statistics aggregated per image
  - `image_parseq_base`: `image_only` plus PARSeq base distance/exact aggregates
  - `image_source`: `image_only` plus source one-hot, diagnostic only
  - `image_parseq_base_source`: all of the above, diagnostic only
- Models: logistic regression, ExtraTrees, HistGradientBoosting
- W&B run: `22stj9mv`

## Results

Eval300 safe-headroom prevalence is `0.233`.

| feature mode | model | AUC | AP | precision@20 | harm@20 | precision@40 | harm@40 | precision@80 | harm@80 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | logistic | 0.637 | 0.336 | 0.25 | 0.30 | 0.40 | 0.375 | 0.388 | 0.350 |
| image_only | extra_trees | 0.652 | 0.348 | 0.40 | 0.40 | 0.45 | 0.325 | 0.312 | 0.325 |
| image_only | hist_gbdt | 0.652 | 0.363 | 0.35 | 0.30 | 0.45 | 0.400 | 0.388 | 0.350 |
| image_parseq_base | extra_trees | 0.654 | 0.359 | 0.50 | 0.40 | 0.425 | 0.375 | 0.325 | 0.375 |
| image_source | extra_trees | 0.660 | 0.368 | 0.50 | 0.45 | 0.45 | 0.375 | 0.338 | 0.350 |
| image_parseq_base_source | extra_trees | 0.675 | 0.381 | 0.55 | 0.40 | 0.475 | 0.450 | 0.362 | 0.388 |
| image_parseq_base_source | hist_gbdt | 0.645 | 0.399 | 0.55 | 0.35 | 0.475 | 0.450 | 0.375 | 0.412 |

The proxy has nontrivial signal: `image_only` AP rises from the `0.233` prevalence to roughly `0.35-0.36`. However, high-ranked groups still have high harmful-candidate rates. Even the best diagnostic feature sets use source and/or PARSeq base signals and still show harm@40 around `0.45`.

## Interpretation

This supports the no-headroom hypothesis but does not produce a safe deployable rejector. The current aggregate features are too coarse to separate "safe hard positives" from "high-risk hard examples." Direct source features are diagnostic only, and PARSeq base features are not enough by themselves.

The next useful step is not to bolt this proxy onto the current patch gate. It should inform the next top8/compact-symbol model: add explicit false-change/no-headroom supervision, but train it with richer local candidate evidence and source-disjoint validation.

## Artifacts

- Result JSON SHA256: `423f9e3857b1b69abf0d056f03bf5b5161faf9db62c8a82f4a84175efbc83600`
- Proxy trainer SHA256: `0803de752701ed678d475fe19ec1152cdce58065c374ff7c9f8eef28426b075f`
- W&B offline run: `wandb/offline-run-20260625_225052-22stj9mv`

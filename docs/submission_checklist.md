# Submission Readiness Checklist

## Scientific validity

- [ ] Primary hypotheses and thresholds frozen before final test evaluation.
- [ ] Strong same-base OCR-loss and allocation controls included.
- [ ] At least two held-out OCR systems in the main result.
- [ ] Actual total bytes, not likelihood-only estimates, used for primary rate.
- [ ] Side-channel methods separated into Track B.
- [ ] Scene, screen and document domains represented or scope narrowed explicitly.
- [ ] Failure cases and non-text regressions reported.
- [ ] Generative hallucination and determinism checked if a refiner is used.

## Reproducibility

- [ ] code commit, environment, dataset manifests and checkpoint hashes frozen.
- [ ] one-command table generation from result JSON.
- [ ] clean-environment smoke reproduction completed.
- [ ] external licenses and notices bundled as permitted.
- [ ] no dataset/checkpoint bytes redistributed without permission.

## Statistics

- [ ] three seeds for learned project components.
- [ ] image-level paired 95% confidence intervals.
- [ ] common BD interval and interpolation reported.
- [ ] no test-driven hyperparameter selection.
- [ ] subgroup multiplicity marked confirmatory/exploratory.

## Paper claims

- [ ] every abstract/conclusion claim maps to `claim_evidence.yaml`.
- [ ] author-reported baseline numbers are not presented as local reproduction.
- [ ] limitations include OCR coverage, language scope, compute and privacy.
- [ ] claimed novelty is utility/invariance/scalability, not module count.

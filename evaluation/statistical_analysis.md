# Statistical Analysis Plan

## Confirmatory endpoints

1. Held-out micro CER at matched actual total bpp.
2. BD-Rate@CER over the preregistered common interval.
3. Worst-held-out-OCR CER.

Secondary endpoints include WER, exact accuracy, visual quality, latency and robustness.

## Units and pairing

- Primary sampling unit: source image/document.
- Predictions from multiple regions in one image are clustered within that image.
- Methods are evaluated on identical source images, OCR versions and corruption draws.

## Confidence intervals

Use paired image-level bootstrap with 10,000 resamples and a fixed analysis seed. For corpus-level CER, resample images and recompute aggregate edit counts rather than bootstrapping text regions independently.

Report 95% percentile intervals and paired difference intervals. Where the dataset is very small, add a permutation or exact paired analysis and state limitations.

## Multiple comparisons

The first confirmatory comparison is OSCAR-LIC versus the strongest validation-selected pure-image text-aware baseline. Other baselines and subgroup analyses are secondary. Apply Benjamini–Hochberg FDR within families of exploratory tests, or mark them explicitly exploratory.

## Seeds

- Development smoke tests: one seed.
- Gate decisions G4–G6: at least two seeds if training is involved.
- Final learned components: at least three seeds.
- Random-allocation controls: at least ten allocation seeds per trained model where cheap.

## Missing and failed cases

- Decode failure is not silently dropped.
- Primary OCR analysis counts a failed decode according to a preregistered conservative policy and separately reports failure rate.
- OCR-engine crash is retried once under the same input; persistent failures are recorded, not imputed per method.
- Dataset-level excluded annotations and reasons are fixed before method comparison.

## Practical significance

Statistical significance alone is insufficient. The minimum publishable target is a reproducible rate saving and cross-domain effect as defined in `PROJECT_SPEC.md`, while maintaining non-text quality and reasonable compute.

# Evaluation Protocol v1.0

**Frozen for the first OSCAR-LIC paper unless an ADR in `DECISIONS.md` changes it.**

## 1. Objective

This protocol prevents an OCR-aware codec from appearing superior because of hidden side information, a favored OCR model, oracle crops, inconsistent normalization, or estimated rather than serialized rate.

## 2. Evaluation tracks

### Track A — Pure visual bitstream (primary)

- Encoder input: image and model parameters only.
- Bitstream must not contain plaintext transcription, OCR tokens, per-word coordinates, prompts derived from OCR text, or external database identifiers.
- A detector or lightweight utility predictor may run at the encoder, but its output must either be decoder-derivable or be serialized and counted.
- Main claims and OSCAR-LIC comparisons use this track.

### Track B — Semantic-assisted coding

- Text strings, boxes, layout metadata, prompts, or symbolic tokens may be transmitted.
- Report `R_image`, `R_text`, `R_position`, `R_metadata`, protection overhead, encoder OCR latency, and privacy analysis separately.
- Do not merge Track A and Track B methods into an undifferentiated ranking.

### Track C — Machine-only / compressed-domain

- A machine task may consume latent features without RGB reconstruction.
- Report task-rate curves separately from human reconstruction curves.
- Include adapter parameters, model version, transmitted feature syntax, and whether the downstream model was fine-tuned.

## 3. Dataset roles

Each dataset manifest must declare exactly one role per experiment:

- `train`: may update weights.
- `validation`: may select checkpoints and hyperparameters.
- `test`: used only after configuration freeze.
- `calibration`: may fit thresholds or OCR confidence calibration but not network weights unless explicitly declared.

No test image may be used for utility oracle generation, qualitative model selection, early stopping, or prompt engineering. Dataset IDs and hashes come from `data/registry.yaml` and generated manifests.

## 4. Image handling

- Primary image representation: RGB, 8-bit integer, full resolution.
- Record decoder-required padding, but divide complete file bytes by the **original** `H × W`.
- Never silently resize benchmark images.
- Any colorspace conversion must record library, version, matrix, range, chroma subsampling, and rounding.
- Alpha channels are either rejected or handled by a separately counted stream.
- For tiled inference, tile size, overlap, blending, and duplicate side information are fixed in the config.

## 5. Rate

Primary rate is computed from the complete serialized file:

\[
\operatorname{bpp}=\frac{8\,\operatorname{filesize(bytes)}}{H W}.
\]

Count all of the following:

- base main and hyper streams
- enhancement main and hyper streams
- gate/index/ROI syntax
- dimensions, padding, model/profile IDs and headers
- CRC, packet tables, alignment and FEC
- text, boxes or prompts in Track B

Report entropy estimates only as diagnostics under `estimated_bpp`; never substitute them for `actual_total_bpp` in primary tables.

For third-party codecs whose container overhead is not image-independent, use their complete produced file. If a repository exposes only likelihood estimates, label that result `estimated_rate_only` and exclude it from the principal actual-bitstream BD-rate table.

## 6. Operating points

- Use at least four overlapping operating points; five or six are preferred.
- The target range should include low-rate points where OCR degrades, not only saturation points.
- Match methods by total bpp through interpolation; do not match only the enhancement payload.
- Do not extrapolate BD metrics outside the common quality interval.
- Record failed or nonmonotonic points; do not delete them silently.

Recommended initial full-image ranges:

- scene/document/screen Track A: approximately 0.03–0.40 total bpp, adjusted after pilot curves
- ultra-low generative auxiliary table: below 0.05 bpp
- base/enhancement budget fractions: 0%, 2.5%, 5%, 10%, 15%, 25% of total target

These are planning ranges, not guaranteed fair points; freeze dataset-specific grids after a pilot conducted only on validation data.

## 7. OCR systems

### Training teachers

The initial teacher pool is PARSeq, ABINet and CRNN from the pinned PARSeq model hub, subject to checkpoint and data-license verification.

### Held-out evaluators

At least two architecturally distinct held-out OCR systems are required. The intended initial suite is TrOCR, Tesseract and PaddleOCR, but exact model revisions and language packs must be locked in `baselines/ocr_models.yaml` before G2.

### Separation rules

- No held-out evaluator gradient, logits, confidence, pseudo-label, or hyperparameter feedback may enter training.
- If the same public pretraining corpus is likely shared across teachers/evaluators, record it as a leakage risk.
- A model update creates a new evaluator ID; results across revisions are not merged.

## 8. Recognition modes

### 8.1 Oracle-crop recognition

Ground-truth polygons/boxes define crops. This measures recognition preservation conditional on localization and is an upper-bound diagnostic, not the primary result.

### 8.2 Detector-frozen recognition

The same frozen detector is applied to original and decoded images. It isolates codec effects while retaining detection failure. Detector thresholds are calibrated on validation data and then frozen.

### 8.3 Native end-to-end OCR (primary deployment result)

Each OCR system performs its standard detector+recognizer pipeline. Match predictions to ground truth using the benchmark's official protocol where available. Otherwise use polygon IoU ≥0.5 followed by maximum-weight one-to-one matching, with transcription correctness computed under the declared normalization profile.

Report all three when feasible. Never describe oracle-crop results as end-to-end OCR.

## 9. Text normalization

Every OCR metric row includes a profile from `OCR_NORMALIZATION.md`. The primary multilingual profile is `unicode_strict_v1`; `latin_alnum_ci_v1` is a compatibility secondary metric. Raw exact string match is also reported for security-sensitive fields.

Do not tune normalization per method. Store both raw and normalized strings in protected result artifacts where dataset terms allow.

## 10. OCR metrics

Primary:

- micro CER
- micro WER where whitespace tokenization is linguistically meaningful
- normalized edit distance
- exact word/line accuracy
- end-to-end text spotting H-mean
- worst held-out OCR CER

Stratify by:

- text height: `≤8`, `9–16`, `17–32`, `>32` pixels
- script/language
- scene, screen and document domain
- contrast, blur and perspective bins where metadata is available
- numeric strings and security-sensitive tokens

For CJK or scripts without whitespace, character/grapheme error is primary. WER requires a frozen language-specific segmenter and a separately named profile.

## 11. Human/perceptual metrics

Report whole-image and region-stratified:

- PSNR and MS-SSIM
- LPIPS and/or DISTS
- text-region and non-text-region versions
- optional FID/KID only for distributional generative comparison

FID/KID cannot establish that a particular transcription is correct. A generated image with plausible but wrong text is a semantic failure.

## 12. Task metrics

Optional downstream evaluation:

- DocVQA ANLS
- KIE entity-level F1
- TextVQA accuracy
- table/document parsing score
- compressed-domain OCR CER

Task models are held out or explicitly categorized as training teachers; never mix the two roles.

## 13. Complexity

Measure after warm-up and with synchronized accelerators:

- encoder and decoder wall time
- entropy coding time separately
- OCR/detector or utility-predictor encoder time
- peak host and accelerator memory
- parameters, MACs/FLOPs with stated counting tool
- model/checkpoint bytes
- energy where instrumentation is available

Report hardware, precision, batch size, resolution and whether I/O is included. Median and p90 over images are preferred over a single mean.

## 14. Robustness

Evaluate clean and corrupted conditions:

- random bit error rates `1e-6, 1e-5, 1e-4, 1e-3`
- burst lengths `8, 32, 128, 256` bits
- packet loss `0.1%, 0.5%, 1%, 5%`
- common image corruption before encoding
- model INT8/FP16 conversion where supported
- rate-inflation and reconstruction adversarial attacks under the declared threat model

Record decode failure, silent corruption, base-only fallback, CER, total bpp and runtime. Invalid bitstreams are not assigned a visually convenient replacement without reporting the failure policy.

## 15. Statistics

- Final tables use at least three independent training seeds for learned project components.
- Use image-level paired bootstrap confidence intervals (default 10,000 resamples) for metric differences.
- For multiple primary comparisons, control false discovery rate or clearly mark exploratory analyses.
- Report per-image paired distributions, not only aggregate means.
- The unit of resampling is the source image/document, not individual text boxes from the same image.

## 16. BD-rate for OCR

For `BD-Rate@CER` or `BD-Rate@WER`:

1. Use total actual bpp.
2. Construct a Pareto envelope, retaining the lowest rate for each quality level.
3. Determine the common metric interval; no extrapolation.
4. Integrate log-rate over the common interval using the named interpolation method.
5. Report the interval, interpolation, and number of points.
6. Supplement with fixed-threshold rate savings because near-zero CER can make interpolation unstable.

Default implementation for the first paper: monotone PCHIP if SciPy is locked and curves support it; otherwise linear interpolation on the Pareto envelope. Do not mix interpolation methods within one table.

## 17. Qualitative examples

- Select examples by a preregistered criterion, e.g. median improvement, worst degradation, small-text quartile, numeric subset.
- Include failures and hallucinations.
- Display original, all methods at matched total bpp, OCR strings, edit operations and per-stream bit breakdown.
- Avoid cherry-picking only samples where the proposed method is visually striking.

## 18. Result validity labels

Every result is one of:

- `valid_primary`
- `valid_secondary`
- `estimated_rate_only`
- `author_reported`
- `invalid_metric_bug`
- `invalid_data_leakage`
- `invalid_wrong_checkpoint`
- `invalid_incomplete_rate`
- `failed_runtime`

Invalid runs remain in the registry but are excluded from claims.

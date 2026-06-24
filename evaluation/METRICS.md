# Metric Definitions

## 1. Sequence metrics

For reference sequence `r` and hypothesis `h`, let Levenshtein counts be substitutions `S`, deletions `D`, and insertions `I`.

### Character Error Rate

\[
CER=\frac{S+D+I}{N_c},
\]

where `N_c` is the number of reference character units. Primary aggregation is micro-averaged over the full test set. Also report image-level macro CER and bootstrap confidence intervals.

### Word Error Rate

\[
WER=\frac{S_w+D_w+I_w}{N_w}.
\]

Whitespace-token WER is reported only for appropriate scripts; see `OCR_NORMALIZATION.md`.

### Normalized Edit Distance

Per sample:

\[
NED=1-\frac{d(r,h)}{\max(|r|,|h|,1)}.
\]

Report mean NED and exact-match rate.

### Worst held-out OCR

\[
WorstOCR\mbox{-}CER=\max_{m\in\mathcal H} CER_m,
\]

where `H` is the preregistered held-out OCR set. Also report mean and each model separately; the maximum alone can be noisy.

## 2. Detection and spotting

Use official benchmark code where available. Otherwise:

- polygon IoU threshold: 0.5
- ignore regions respected before matching
- one-to-one maximum-weight matching
- detection precision/recall/H-mean
- end-to-end match requires both spatial match and transcription correctness under the frozen profile

Report case-sensitive and compatibility profile only when both were preregistered.

## 3. Rate

- `actual_total_bytes`: complete container bytes
- `actual_total_bpp = 8 * bytes / (H*W)`
- `estimated_bpp`: likelihood sum diagnostic
- `payload_bpp`: optional stream-only diagnostic, never primary
- `overhead_bpp`: header, gate, index, CRC, FEC and alignment

The result JSON contains a per-section byte map whose sum equals total file bytes.

## 4. Visual metrics

- RGB PSNR over `[0,255]` or `[0,1]`, with range declared
- MS-SSIM with package/version and channel handling
- LPIPS with backbone/version
- DISTS with implementation/version

Region metrics use a ground-truth or frozen detector mask fixed for all methods. Report mask coverage and dilation radius. Empty-region cases are counted and handled consistently.

## 5. OCR-aware BD-rate

`BD-Rate@CER` is the average percentage rate difference between test and anchor curves over their common CER interval, integrating log total bpp. Negative is better.

Required metadata:

```yaml
metric: cer_micro
lower_is_better: true
rate: actual_total_bpp
interpolation: pchip | linear
common_interval: [low, high]
anchor_points: 5
candidate_points: 5
pareto_filtered: true
extrapolation: false
```

Also report rate at fixed, interpretable thresholds, e.g. CER 0.05, 0.10 and 0.20 where curves overlap. Thresholds must be selected on validation data.

## 6. Utility prediction

For oracle utility `u` and prediction `û`:

- Spearman rank correlation
- Kendall tau
- top-k overlap at matched transmitted bytes
- regret relative to oracle:

\[
Regret(B)=L_{OCR}(S_{learned}(B))-L_{OCR}(S_{oracle}(B)).
\]

- calibration of predicted benefit and uncertainty
- actual downstream CER/WER, which remains decisive

A high correlation without downstream improvement is not sufficient.

## 7. Scalability

For progressive enhancement budgets:

- rate monotonicity violation count
- OCR monotonicity violation count and magnitude
- area under OCR-vs-rate curve
- base-only visual validity
- truncation/decode failure rate

OCR need not improve for every image at every packet, but aggregate regressions must be explained.

## 8. Robustness

- decode success rate
- silent corruption rate
- base fallback success
- `ΔCER`, `ΔWER`, `ΔLPIPS` relative to clean
- attack-induced bitrate ratio
- recovery latency

Use paired corruptions and fixed random seeds across methods.

## 9. Complexity

- median, mean and p90 latency
- peak allocated and reserved GPU memory
- CPU RSS where available
- model parameters and file size
- entropy coding sequential steps
- encoder OCR/utility overhead

Do not compare timings from different hardware in one ranking without qualification.

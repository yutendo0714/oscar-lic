# OSCAR-LIC Architecture Specification

## 1. Design objective

Create a two-layer learned image bitstream:

- `B0`: independently decodable base image stream
- `B1`: optional sparse text-preservation enhancement stream

No plaintext, OCR token sequence or lossless box coordinates are transmitted in Track A. All selection information required to decode `B1` is serialized and counted.

## 2. Reference pipeline

```text
x [B,3,H,W]
 ├─ BaseAnalysis g_a ─ y_b [B,C,H/16,W/16]
 │                    ├─ HyperAnalysis ─ z_b ─ entropy code ─ B0.hyper
 │                    └─ Base latent entropy code ─────────── B0.main
 │
 ├─ UtilityFeatureStem ─ q
 │       q + stopgrad(y_b,z_hat,x_base_proxy)
 │                 └─ UtilityPredictor ─ utility mean/logvar per candidate
 │                                       └─ BudgetAllocator ─ selected units m
 │
 └─ TextEnhancementAnalysis(x, base features)
                         └─ y_t candidates
                              ├─ gate/index entropy code ─ B1.index
                              └─ selected conditional latent code ─ B1.payload

Decode:
B0 → x_base, base features
B1.index + B1.payload + base features → delta_text
x_hat = clamp(x_base + alpha * delta_text)
optional ROI one-step refiner → x_final
```

## 3. Base codec adapter

Create `BaseCodecAdapter` rather than modifying upstream code throughout the repository. Required outputs:

```python
BaseEncodeOutput(
    y: Tensor,
    z: Tensor | None,
    likelihoods: dict[str, Tensor],
    analysis_features: dict[str, Tensor],
)

BaseDecodeOutput(
    x_hat: Tensor,
    synthesis_features: dict[str, Tensor],
)
```

Required methods:

```python
analysis(x) -> BaseEncodeOutput
quantize_and_estimate(output, mode) -> QuantizedBase
compress(x) -> list[bytes] + metadata
decompress(streams, metadata) -> BaseDecodeOutput
```

Upstream MLIC++ files should be vendored only through a clearly documented patch or imported from `external/mlic`. Never copy code without preserving license and provenance.

## 4. Candidate transmission unit

### Initial choice

A candidate is:

```text
latent spatial tile: 8 × 8 positions in y_t
channel group: 8 channels
candidate ID: (scale_id, tile_y, tile_x, group_id)
```

The exact `8×8×8` choice is a starting config, not a scientific constant. Ablations compare:

- spatial tiles: 4, 8, 16 latent positions
- channel groups: 4, 8, 16
- spatial-only versus channel-only versus joint units

Candidate boundaries should align with independently entropy-codable symbols where possible; otherwise “selecting a candidate” may still require coding unselected dependencies.

## 5. Utility feature inputs

Deployment predictor must not require a full OCR pass. Candidate inputs may include:

- base analysis feature at corresponding receptive field
- decoded hyperprior mean/scale
- base reconstruction uncertainty proxy
- low-cost edge/stroke features
- lightweight textness feature distilled from detector/OCR teachers
- candidate estimated rate
- global pooled context

Do not use ground-truth text, held-out OCR output or test annotation.

Recommended predictor:

```text
1×1 projection of base/hyper features
+ depthwise local block
+ 2–4 lightweight Transformer/RWKV blocks over candidate tokens
+ heads: utility_mean, utility_logvar, textness, rate_correction
```

Utility uncertainty can support conservative selection but must be ablated against mean-only allocation.

## 6. Budget allocator

For candidate `i`, predictor gives utility `u_i` and estimated total incremental bits `r_i`, including index overhead. Select under budget `B_t`:

\[
\max_m \sum_i m_i u_i,\quad \sum_i m_i r_i \le B_t,\quad m_i\in\{0,1\}.
\]

Primary deployment algorithms:

1. greedy `u_i / r_i` with exact serialized recheck
2. bucketed dynamic programming for small candidate counts
3. learned hard-concrete gates during training

After selection, serialize once, measure actual bits, and trim the lowest marginal selected candidates if the file exceeds the hard budget. Report how often correction is needed.

## 7. Enhancement encoder

Initial continuous residual variant:

```text
input: original x, detached or trainable base analysis features
stem: strided conv/GDN or TCM block to base latent resolution
fusion: cross-attention or gated residual with base y
output: y_t [B,C_t,H/16,W/16]
```

`C_t=64–128` candidate range. The enhancement must be conditional on `B0`; standalone visual quality is irrelevant.

Auxiliary topology targets can shape `y_t` without being transmitted separately:

- signed distance transform of text strokes
- edge orientation
- skeleton confidence
- local contrast/foreground probability

## 8. Gate and index coding

Preferred order:

1. Decoder predicts prior selection probability from base hyperlatent.
2. Encode binary residual gate under Bernoulli entropy model.
3. Within selected tiles, encode channel-group activity.
4. Encode payload conditioned on base features and earlier selected groups.

Alternatives such as run-length, quadtree or sorted candidate IDs must be compared by actual bytes. A dense binary map is not automatically worse at high text density.

## 9. Text payload entropy model

Initial model:

\[
p(\hat y_t\mid \hat y_b,\hat z_b,m)=\prod_j
\mathcal N(\hat y_{t,j};\mu_j,\sigma_j)*\mathcal U(-1/2,1/2).
\]

Inputs to parameter network:

- decoded base latent/hyper features
- selected gate
- previously decoded text channel groups
- optional local checkerboard context

Keep the number of serial slices small; OCR gains that require a prohibitively slow decoder are not practical wins.

## 10. Decoder and blending

The text decoder predicts a bounded residual and optional alpha/confidence map:

\[
\hat x=\operatorname{clip}(\hat x_b + a\odot \Delta x_t,0,1).
\]

Alpha is derived from decoded features and is not a free external mask. Penalize changes outside affected receptive fields and text neighborhoods to protect non-text content.

## 11. Optional one-step local refiner

Only after G5/G6:

- operate on decoded ROI tiles plus context margin
- condition on `x_base`, decoded text feature and topology feature
- one deterministic rectified-flow/distilled diffusion step
- fixed seed derived from bitstream checksum if stochastic operators remain
- blend with confidence and preserve untouched pixels exactly outside ROI

Main safety criterion: exact sequence must not vary across seeds/platform profiles beyond documented floating-point effects.

## 12. Compressed-domain OCR adapter

Optional Track C head:

```text
selected base/text latent → small transform neck → OCR encoder tokens → recognizer decoder
```

Train without changing held-out evaluators. This head demonstrates machine usability but must not replace reconstructed-image OCR in Track A.

## 13. Parameter-free and lightweight controls

Ablate against:

- Sobel energy
- Laplacian variance
- base reconstruction residual oracle
- detector textness
- OCR entropy/uncertainty where encoder OCR is allowed
- random ranking

A learned predictor is scientifically justified only if it beats these controls at matched compute/rate.

## 14. Recommended implementation sequence

1. adapter around frozen MLIC++
2. dense text residual without gating
3. fixed ground-truth/oracle mask to validate decoder capacity
4. actual gate serialization
5. oracle utility allocator
6. learned utility predictor
7. joint fine-tuning
8. optional refiner/robustness

## 15. Invariants checked by tests

- `B0` decodes when `B1` is absent
- stream lengths in header equal payload lengths
- total file size equals reported bytes
- same bitstream produces same shape and deterministic reconstruction profile
- unselected payload symbols are not serialized
- gate corruption is detected by CRC
- no OCR text string appears in Track A metadata

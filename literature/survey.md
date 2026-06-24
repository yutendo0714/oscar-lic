# Systematic Survey: Learned Image Compression and OCR-Aware Coding

**Evidence snapshot:** 2026-06-24  
**Scope:** general LIC, entropy modeling, Transformer/SSM/graph codecs, low-rate/generative/VQ, rate–perception, ICM, OCR-aware coding, RAW/ISP, robustness/security

## 1. How to read this survey

Reported BD-rate values are not globally comparable unless the reference codec/version, color space, test image handling, metric, number of operating points, actual/estimated rate and training corpus match. The registry therefore stores author-reported results as evidence, not as a universal ranking.

Evidence labels:

- `peer_reviewed`: accepted archival paper
- `standard`: normative/official standards source
- `preprint`: not peer-reviewed at snapshot
- `official code`: author or standards-body implementation
- `author reimplementation`: released by authors but explicitly not exact experimental code
- `not confirmed`: do not assume availability

## 2. Core LIC formulation

A transform codec maps image `x` to latent `y`, quantizes it, entropy-codes it under a learned probability model, and reconstructs `x_hat`:

\[
y=g_a(x),\quad z=h_a(y),\quad \hat y=Q(y),\quad \hat z=Q(z),\quad \hat x=g_s(\hat y).
\]

A conventional objective is

\[
L_{RD}=E[-\log_2 p(\hat z)-\log_2 p(\hat y\mid \hat z,c)] + \lambda d(x,\hat x),
\]

where `c` may include decoded spatial/channel context. The first three foundational steps were nonlinear transform coding with GDN, scale hyperprior, and joint autoregressive/hierarchical prior.

### 2.1 Entropy-model progression

1. factorized latent prior
2. hyperprior carrying image-adaptive scale/mean information
3. hyperprior plus pixel/latent autoregression
4. checkerboard and channel-group context to reduce serial steps
5. multi-reference channel/local/global context
6. external learned dictionaries and hierarchical dictionaries
7. content-adaptive graph or state-space context

The central tension is not merely rate versus distortion. A stronger context model often increases serial decoding, memory, kernel complexity or model fragility. A paper that gains 1% BD-rate but multiplies sequential entropy-decoding steps may be less useful for the proposed OCR edge/cloud scenario.

### 2.2 Current high-performance families

| Family | Representative methods | Strength | Main cost/risk |
|---|---|---|---|
| CNN + hyperprior/context | Cheng2020, ELIC | mature, efficient kernels | limited global adaptivity |
| Transformer/hybrid | WACNN, LIC-TCM, FTIC | long-range/frequency modeling | window design and memory |
| Multi-reference entropy | MLIC/MLIC++/MLICv2 | strong conditional entropy | slice dependencies, complex implementation |
| Dictionary prior | DCAE, HiDE | shared external priors cost no per-image payload | training-data bias, dictionary utilization |
| SSM/RWKV/Mamba | LALIC, MambaIC | linear long-range modeling | custom CUDA/reimplementation risk |
| Graph | GLIC | content-adaptive connectivity | graph construction/runtime and code maturity |
| Practical/mobile | block-based analytic overlap, practical NAS codec | deployment metrics | may not maximize standard PSNR BD-rate |

MLIC++ is a sensible first base for OSCAR-LIC because it exposes conditional entropy modeling where a text enhancement stream can be added. DCAE is strategically interesting because glyph/stroke prototypes are a natural extension of a shared dictionary. However, using DCAE as the only base could make novelty vulnerable to a “dictionary specialization” interpretation; utility-per-bit must remain the primary contribution.

## 3. Transformer, SSM and graph compression

Transformers initially improved nonlocal transform/context modeling but imposed quadratic or window-limited computation. FTIC introduces frequency-aware attention; LALIC and MambaIC use linear state-space/RWKV-like mechanisms; GLIC replaces fixed Euclidean neighborhoods with content-adaptive graph connectivity.

For text, long-range context has two distinct roles:

- **visual redundancy:** repeated colors, UI components and glyph strokes
- **linguistic correlation:** character sequences and repeated words

A visual codec should not silently use linguistic correction to hallucinate a plausible but wrong string. Therefore, long-range modules may support entropy prediction or visual feature allocation, while exact sequence evaluation remains ground-truth based.

## 4. Rate–distortion–perception and low-bitrate generative coding

The rate–distortion–perception theorem implies that improving distributional realism at fixed rate generally trades against distortion. GAN and diffusion codecs exploit a decoder prior to synthesize texture not transmitted in the bitstream.

### 4.1 Major lines

- HiFiC: adversarial high-fidelity decoder
- MRIC: one representation with adjustable realism
- CDC: conditional diffusion generates texture variables
- PerCo: VQ local code plus global semantic description at extreme rates
- GLC/RDVQ: VQ and entropy-constrained generative latent coding
- PICD: image stream plus explicitly encoded text/layout rendered by diffusion
- CoD: compression-oriented diffusion foundation model
- CADC: content-adaptive quantization and decoder-derived semantic conditioning
- AEIC: shallow sender, one-step diffusion receiver
- VLIC: VLM-based preference judge

### 4.2 Why text changes the evaluation

FID, KID and even LPIPS can improve when a sign becomes more plausible but says the wrong word. For OCR-aware compression, perceptual metrics are secondary to exact semantic integrity:

- CER/WER/exact-match
- numeric-field accuracy
- name/entity accuracy
- topology/confusion pairs such as `8↔3`, `O↔0`, `C↔G`, diacritics
- stochastic-seed consistency

Generative refinement should be local, deterministic in production, and optional for critical text. It should not receive plaintext in Track A.

## 4.3 Semantic text-guided compression is adjacent to, but not equivalent to, OCR preservation

A separate line of work uses a caption or high-level textual representation to improve visual compression. TGIC injects text into multiple codec components; TACO emphasizes text-adaptive encoding rather than a high-diversity text-conditioned decoder; LMM-driven image-text coding jointly produces/compresses captions and applies semantic-perceptual fine-tuning; SELIC derives BLIP/BERT semantic features at the encoder and fuses them into the entropy-coded latent.

These papers are highly relevant as architectural and training references, but they solve a different semantic problem from text-recognition-preserving compression:

- a caption such as “a street sign” does not identify the exact sign string;
- caption correctness can coexist with one-character OCR corruption;
- any caption, prompt, semantic tensor, box or layout payload must be counted in total serialized rate;
- running an LMM, BLIP, BERT or OCR at the sender changes encoder complexity and privacy exposure;
- semantic side-channel baselines belong in a separate comparison track unless plaintext/semantic metadata are permitted for all methods.

For OSCAR-LIC, these methods motivate semantic conditioning and multimodal loss design, but the primary Track A forbids transmitting plaintext. Exact ground-truth character preservation and held-out OCR transfer remain the acceptance criteria.

## 5. VQ-based compression

VQ creates discrete image tokens and aligns naturally with autoregressive/generative priors. Current problems include codebook collapse, rate mismatch, token granularity, slow autoregressive entropy coding, and discontinuous rate control. RDVQ attacks the rate mismatch by differentiable relaxation of codebook probabilities.

For OCR, a single dropped token can alter a full word. Candidate safeguards include:

- importance-ranked token transmission
- base/enhancement token hierarchy
- independent tile packets
- unequal error protection
- topology-aware codebook regularization
- exact symbol consistency loss

A VQ extension is valuable after the continuous-latent utility hypothesis is validated; starting with VQ would confound rate allocation with tokenizer quality.

## 6. Image Coding for Machines

ICM has three operational forms:

1. reconstruction-first: decode RGB then run task model
2. compressed-domain: feed codec latent directly to machine model
3. unified human/machine scalable stream

TransTIC adapts a pretrained codec through prompts. Rate–Distortion–Cognition work exposes controllable Pareto parameters. TTFT performs instance-specific low-rank adaptation. BridgingCompressionMLLM and TFGC move toward direct consumption of compressed latent/tokens by multimodal models. JPEG AI formalizes a compact representation for human and machine use and provides progressive, ROI and random-access features.

OCR differs from common detection/classification ICM in four ways:

- output is a variable-length exact sequence
- one local stroke can change semantics completely
- small text occupies few pixels but high semantic value
- language priors can hide visual information loss

Thus, OCR-aware ICM needs exact-sequence and cross-recognizer metrics, not only task accuracy on a fixed network.

## 7. OCR-aware and text-preserving compression

### 7.1 Existing approaches

| Work | Mechanism | Key evidence | Limitation relevant to OSCAR-LIC |
|---|---|---|---|
| Scene Text Quality Assessment (2023) | assess compressed text quality and iteratively adjust rate | objective and subjective text gains | iterative/non-joint; quality score is not marginal utility |
| Lai et al. (ICIP 2024) | OCR text-logit discrepancy loss | reported -32.64% BD-rate@CER and -28.03% @WER on two screenshot datasets | fixed OCR and screen focus |
| TFIC (2025 preprint) | frozen OCR supervises Transformer codec | large very-low-rate OCR gains in synthetic setup | single OCR, small synthetic test, code unavailable in survey |
| PICD (CVPR 2025) | losslessly encode recognized text/position and diffusion-render | strong screen-text perceptual results | solves a semantic side-channel problem, adds privacy/runtime/OCR-error dependence |

### 7.2 Critical interpretation

Compression can improve a particular OCR relative to the original image by denoising, contrast shaping or model-specific preprocessing. This is not proof that the original visual information was preserved. The decisive test is ground-truth transcription evaluated by held-out recognizers and humans.

Text masks are also an incomplete allocation signal. They answer “where is text?” rather than “where will the next bit reduce recognition error most?” A large high-contrast heading may need no enhancement, while a tiny digit or faint punctuation needs many bits.

### 7.3 Open technical problems

- differentiating sequence edit distance
- teacher/model invariance
- detector error and crop-oracle inflation
- side-information accounting
- multilingual/script transfer
- text-size imbalance
- exactness under generative priors
- encoder compute and edge deployment
- bitstream and model-version robustness
- privacy leakage from text-focused latent

## 8. Proposed research gap

No representative existing method in this snapshot jointly provides:

1. explicit marginal OCR utility per transmitted bit
2. cross-OCR invariance as a primary objective
3. plaintext-free scalable text enhancement
4. full gate/index/header rate accounting
5. robustness and privacy evaluation for the text layer

This combination defines OSCAR-LIC, but novelty must be demonstrated experimentally rather than asserted by module count.

## 9. RAW and ISP-aware compression

RAW-to-RGB processing and compression are usually optimized separately, although sensor noise, demosaicing and tone mapping determine text visibility. InvISP, RAWtoBit, compact RAW metadata, ρ-Vision, ParamISP and CoDISP show increasing integration of ISP, compression and machine vision.

A future OCR/RAW extension should target low-light receipts, signs, displays and motion blur, where RAW carries recoverable high-frequency evidence lost by a conventional ISP. It should be a second project because camera/sensor shift would substantially expand the first paper's scope.

## 10. Robustness and security

Threats are distinct:

- input attack that inflates bitrate
- input attack that increases distortion or task error
- backdoored codec/checkpoint
- bitstream collision or semantic ambiguity
- random bit flip/burst/packet loss
- decoder model drift
- privacy extraction from latent/side-channel

OCR makes small errors high-impact: a single bitstream fault can change an amount, identifier or name. The proposed text stream should use packet boundaries, CRC, resynchronization and optional unequal FEC. Robustness evaluation must use actual serialized streams.

## 11. Baseline strategy

### P0 required

- CompressAI hyperprior or joint AR-hyperprior
- MLIC++
- Lai-style single OCR loss reimplementation
- multi-teacher no-utility baseline
- uniform text mask allocation
- JPEG AI reference when locally available
- PARSeq/CRNN train teachers and held-out OCR

### P1 strong additions

- DCAE
- LIC-TCM or FTIC
- LALIC
- PICD author result or transparent reimplementation track
- one compressed-domain ICM baseline

### P2 optional

- MambaIC author reimplementation
- GLIC after complete code/checkpoints
- RDVQ/CoD/AEIC for generative extension
- RAW and adversarial defense methods

## 12. Evaluation hierarchy

1. correctness of serialized rate
2. ground-truth OCR semantics
3. held-out OCR transfer
4. human readability
5. image distortion/perception
6. complexity/memory
7. robustness/privacy

Reversing this hierarchy can produce visually impressive but semantically unsafe results.

## 13. Main research recommendation

Begin with the smallest falsifiable system:

```text
MLIC++
+ multi-teacher OCR-invariant loss
+ oracle utility allocation
+ entropy-coded sparse enhancement stream
```

Only after oracle and learned utility pass the held-out tests should a one-step generative refiner be added. This order isolates the scientific claim and prevents diffusion quality from masking a failed allocation hypothesis.

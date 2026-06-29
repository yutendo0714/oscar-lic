# LIC State Of The Art Map

**Updated:** 2026-06-26  
**Purpose:** concise strategic survey for OSCAR-LIC planning. The larger structured registry remains in `literature/paper_registry.csv`.

## Executive View

Modern LIC has split into four active fronts:

1. **Stronger entropy modeling:** hyperprior, autoregressive priors, channel/local/global contexts, dictionaries, graph neighborhoods, and state-space or linear attention modules.
2. **Perceptual and generative compression:** GAN/diffusion/VQ decoders trade exact distortion for realism, especially below 0.05 bpp.
3. **Machine/task-aware compression:** ICM and downstream-aware codecs optimize task accuracy, latent usability, or controllable cognition/distortion.
4. **Robustness/security:** learned codecs are vulnerable to adversarial bitrate inflation, RD degradation, backdoors, and bitstream collisions.

OSCAR-LIC sits at the intersection of entropy-modeled scalable LIC and OCR-aware task preservation, but must remain a pure visual bitstream method in its main track.

## Representative Works By Area

| Area | Representative works | Main idea | Datasets / metrics | Code status | Relevance to OSCAR-LIC |
|---|---|---|---|---|---|
| Foundational LIC | Balle 2017, Balle 2018 hyperprior, Minnen 2018 joint AR+hyperprior, Cheng 2020 GMM+attention | nonlinear transform coding, hyperprior side information, spatial autoregression, mixture likelihoods | Kodak, CLIC, Tecnick; PSNR, MS-SSIM, bpp | TensorFlow Compression / CompressAI references | baseline vocabulary and actual-bitstream sanity checks |
| Efficient entropy modeling | ELIC, MLIC, MLIC++, MLICv2 | channel grouping plus local/global/context references | Kodak, Tecnick, CLIC Pro; BD-rate vs VTM | MLIC official repo available | best first base for OSCAR-LIC because context and stream structure are strong |
| Transformer / long-range LIC | WACNN, LIC-TCM, FTIC, LALIC, MambaIC, GLIC | window attention, frequency-aware attention, RWKV/SSM, graph connectivity | Kodak, Tecnick, CLIC; BD-rate, latency, memory | mixed official/reimplementation | candidate baselines after MLIC++ reproduction; useful ideas for utility predictor tokens |
| Dictionary entropy | DCAE, HiDE, ARCHE | learn shared dictionaries/prototypes or efficient hierarchical priors to inform entropy modeling | OpenImages, Kodak, CLIC, TESTIMAGES | DCAE official; HiDE/ARCHE code unconfirmed | glyph/stroke dictionary extension is attractive but should be secondary |
| Generative compression | HiFiC, MRIC, CDC, CoD, CADC, AEIC, OSDiff, OneDC | adversarial or one-step diffusion prior synthesizes realistic detail | FID/KID/LPIPS/DISTS plus distortion metrics | HiFiC/CDC/CoD/AEIC/OSDiff available or advertised; CADC unconfirmed | powerful at low rate but dangerous for exact text unless local/deterministic and secondary |
| VQ generative compression | GLC, PerCo/PerCoV2, RDVQ | discrete tokens, diffusion/VQ priors, differentiable codebook-rate relaxation | ultra-low bpp; FID, LPIPS, DISTS | RDVQ official; PerCo open implementation | possible future enhancement tokenization, not the first path |
| OCR-aware compression | Scene Text Quality Assessment, Lai et al. 2024 text logit loss, TFIC, PICD | text quality assessor, OCR-logit loss, OCR-supervised text-focused codec, text/layout side-channel diffusion rendering | screenshot/scene-text/synthetic text; CER, WER, text accuracy, bpp | code often unavailable or side-channel-heavy | closest baselines and reviewer comparison set |
| Task-aware / ICM | TransTIC, Rate-Distortion-Cognition, TTFT, BridgingCompressionMLLM, TFGC, JPEG AI, PICM-Net | prompt/adapt a codec for downstream tasks or compressed-domain machine use | detection/classification/segmentation/LVLM/OCR tasks; task-rate curves | mixed; JPEG AI reference software official | informs evaluation discipline, but OCR exact sequence is stricter than generic task accuracy |
| Robustness / security | MALICE, backdoor attacks, SRDA/ARDA, LIC common corruptions, bitstream collisions | bitrate DoS, RD attack, poisoned checkpoints, corruption/semantic collision vulnerabilities | multiple LIC models; bpp inflation, PSNR drop, attack success, corruption RD | some code available, many preprints | OSCAR text stream needs CRC, packet boundaries, fallback, and actual serialized attack tests |

## Notes On Specific Recent Papers

### MLIC++ / MLICv2

MLIC++ partitions latent slices and models channel-wise, local spatial, and global spatial correlations with linear-complexity mechanisms. The official repository states that corrected MLIC++ checkpoints require `LatentResidualPrediction` and `SynthesisTransform`, not the old variants. This is the primary base codec target.

### DCAE

DCAE introduces a dictionary-based cross-attention entropy model, using learned dictionary priors from training data. It is strategically interesting because text/glyph/stroke prototypes could become specialized dictionary atoms, but using it too early would confound the main utility-per-bit contribution.

### ARCHE

ARCHE is a recent efficient RD codec that combines hierarchical, spatial and channel priors with residual refinement and excitation while avoiding recurrent or Transformer blocks. Its main strategic message for OSCAR-LIC is that simple convolutional priors can remain competitive; this supports keeping the first OSCAR enhancement model small and focused on code assignment/gating rather than adding a heavy global backbone.

### GLIC

GLIC uses content-adaptive dual-scale graphs and dynamically changes connectivity based on content complexity. This is conceptually aligned with spatially varying redundancy, but graph construction/runtime overhead and code maturity need checking before using it as a main baseline.

### CMIC / Scaling LIC

CMIC adapts Mamba-style state-space modeling with content-aware token reorganization and prompt dictionaries, while recent scaling-law work pushes an HPCM-style LIC model up to 1B parameters. Both suggest that global RD performance still benefits from better context capacity. For OSCAR-LIC, they are important frontier references but not the immediate path: the current Eval300 bottleneck is sparse candidate-conditioned assignment precision, not full-image transform capacity.

### RDVQ

RDVQ addresses the VQ rate mismatch by replacing hard assignments during training with differentiable codebook-distribution relaxation, allowing entropy loss to shape encoder token distributions. It is a strong VQ/generative baseline for ultra-low rates, but exact OCR text can fail catastrophically if token errors alter strokes.

### HVQ-CGIC / HyperVQ

HVQ-CGIC extends hyperprior-style entropy modeling to VQ-based controllable generative compression. This is directly relevant to OSCAR's compact code-index stream: future code assignment should be entropy/utility aware rather than nearest-residual only. The caution is that perceptual VQ gains still need exact text and held-out OCR validation.

### CoD / CADC / AEIC

CoD trains a compression-oriented diffusion foundation model rather than adapting a text-to-image foundation model. CADC adds uncertainty-guided adaptive quantization, information concentration, and decoder-derived textual conditioning. AEIC uses a shallow encoder and one-step diffusion receiver. These are important low-bitrate perceptual baselines, but OSCAR-LIC should not let stochastic generation become the main claim.

### CoD-Lite / Region-Adaptive Generative Compression

CoD-Lite shows the diffusion-compression trend moving toward real-time one-step convolutional decoders. Region-adaptive diffusion codecs bring non-uniform spatial bit allocation into generative compression. These works weaken latency objections to generative codecs, but they still do not solve exact text fidelity, hallucination, or side-map accounting. OSCAR-LIC should use them as comparison/extension context, not replace the deterministic text utility stream.

### OSDiff / OneDC / PerCoV2

OSDiff and OneDC represent the current one-step diffusion trend in perceptual compression: they reduce the traditional diffusion decoder latency problem and use compressed latents, semantic guidance, or pretrained tokenizer distillation. PerCoV2 extends perceptual compression around modern diffusion/VQ components and masked/autoregressive hyper-latent modeling. These are useful perceptual baselines, but they do not remove the exact-text risk. For OSCAR-LIC, their most relevant lesson is not to adopt a full generative decoder immediately; it is to make compact symbols and assignments trainable under rate and utility constraints.

### PICD

PICD explicitly encodes text and image separately, then diffusion-renders screen content. It is highly relevant as a strong perceptual screen-content baseline, but belongs to a semantic-assisted track because text/location side information, OCR errors, privacy, and runtime must all be counted.

### PICM-Net

PICM-Net studies progressive learned compression for machine perception using trit-plane coding and an adaptive decoding controller. It strengthens the framing of OSCAR-LIC optional sections: the text layer can be described as a task-conditioned enhancement level whose bytes are transmitted only when expected downstream utility justifies them.

### Lai et al. and TFIC

Lai et al. show that OCR text-logit loss can improve CER/WER BD-rate on screenshot datasets. TFIC trains a codec for text-specific OCR utility at low rate. Both motivate OCR-aware losses, but OSCAR-LIC must improve on their likely weaknesses: single-OCR dependence, limited cross-domain evidence, and lack of explicit marginal utility per actual bit.

### TextBoost

TextBoost is a 2026 OCR-guided ultra-low-bitrate scene-text codec that transmits filtered OCR text and geometry, renders that side information into a guidance map, and fuses it into the decoder. Author-reported results include up to 60.6% higher text-recognition F1 on TextOCR/ICDAR 2015 at comparable PSNR/bpp. This is an important reviewer-facing comparator because it directly targets scene-text fidelity, but it belongs in Track B: every OCR string, coordinate, filter decision, header, encoder OCR runtime and privacy risk must be counted and separated from the pure-image OSCAR Track A claim.

### EF-LIC

EF-LIC removes entropy coding and instead uses unconstrained vector quantization plus a context-conditioned autoregressive transform to reduce statistical and correlation redundancy before serialization. The official inference repository is now visible. Its relevance to OSCAR is practical rather than conceptual: it may become a low-latency baseline, but only after checkpoint SHA256, license status, output syntax and complete-file bpp are verified.

### FlowCodec

FlowCodec decouples latent compression from one-step latent transport through a pretrained large text-to-image prior with lightweight LoRA adaptation. It strengthens the low-bitrate generative-compression frontier and is simpler than many auxiliary-branch diffusion codecs. For OSCAR-LIC, it is still a comparison/extension candidate because exact text and identity preservation must be audited directly; perceptual metrics alone cannot certify OCR correctness.

### Control Your View

Control Your View studies high-resolution global semantic manipulation attacks on LIC. The key relevance is that robustness cannot be reduced to bit errors, PSNR loss, or bitrate inflation: text-bearing images need semantic integrity checks where small perturbations cannot rewrite names, amounts, identifiers, or OCR-critical strings after decoding.

### Robustness/Security Line

MALICE shows adversarial bitrate inflation can be severe. SRDA/ARDA analyze arbitrary-direction rate-distortion attacks and point to hyperprior/IGDN vulnerabilities. Backdoor work shows frequency triggers can poison learned codecs. Bitstream collision work raises semantic-integrity risks. For OSCAR-LIC, robustness must be measured on serialized streams because small text-layer faults can change names, amounts, or identifiers.

## Implications For OSCAR-LIC

1. Use MLIC++ as the first base, not because it is forever best, but because it is strong and exposes actual compression.
2. Treat DCAE/JPEG AI/RDVQ/CoD/PICD as comparison or extension tracks, not as the first implementation path.
3. Keep first OSCAR model simple: continuous conditional residual plus serialized sparse gate.
4. Make the oracle allocation result the central early evidence.
5. Every table should include actual total bpp, held-out OCR, and side-bit accounting.
6. Generative methods must be evaluated for exact text failure and hallucination, not only LPIPS/FID.
7. Robustness is not optional for text: base fallback, CRC, packet loss, and bit-flip behavior should be in the core evaluation plan once clean gains exist.

## Primary Sources Checked

- MLIC++ arXiv/GitHub: https://arxiv.org/abs/2307.15421, https://github.com/JiangWeibeta/MLIC
- DCAE arXiv/GitHub: https://arxiv.org/abs/2504.00496, https://github.com/CVL-UESTC/DCAE
- ARCHE arXiv: https://arxiv.org/abs/2603.10188
- LALIC arXiv/GitHub: https://arxiv.org/abs/2502.05741, https://github.com/sjtu-medialab/RwkvCompress
- MambaIC arXiv/GitHub: https://arxiv.org/abs/2503.12461, https://github.com/AuroraZengfh/MambaIC
- GLIC CVF/arXiv: https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Adaptive_Learned_Image_Compression_with_Graph_Neural_Networks_CVPR_2026_paper.html
- CMIC: https://arxiv.org/abs/2508.02192
- Scaling LIC to 1B: https://arxiv.org/abs/2508.09075
- JPEG AI: https://jpeg.org/jpegai/
- HiFiC: https://hific.github.io/, https://arxiv.org/abs/2006.09965
- CDC: https://openreview.net/forum?id=QIBpzaDCAv, https://github.com/buggyyang/CDC_compression
- PerCo: https://openreview.net/forum?id=ktdETU9JBg
- HVQ-CGIC / HyperVQ: https://arxiv.org/abs/2512.07192
- PICD: https://arxiv.org/abs/2505.05853
- RDVQ: https://arxiv.org/abs/2604.10546, https://github.com/CVL-UESTC/RDVQ
- CoD: https://arxiv.org/abs/2511.18706, https://github.com/microsoft/GenCodec
- CADC: https://arxiv.org/abs/2602.21591
- CoD-Lite: https://arxiv.org/abs/2604.12525
- Region-adaptive generative compression: https://arxiv.org/abs/2604.01122
- Diffusion-based generative compression survey: https://arxiv.org/abs/2601.18932
- OSDiff: https://arxiv.org/abs/2602.01570, https://github.com/cheesejiang/OSDiff
- OneDC: https://arxiv.org/abs/2505.16687, https://onedc-codec.github.io/
- PerCoV2: https://arxiv.org/abs/2503.09368
- Semantics-guided generative compression: https://arxiv.org/abs/2505.24015
- Lai et al. text quality enhancement: https://arxiv.org/abs/2402.08643
- TextBoost: https://arxiv.org/abs/2603.04115
- TFIC: https://arxiv.org/abs/2503.19495
- EF-LIC: https://arxiv.org/abs/2605.23323, https://github.com/SevenCTHU/EF-LIC
- FlowCodec: https://arxiv.org/abs/2606.21030
- Control Your View: https://arxiv.org/abs/2605.08727
- Scene text quality assessment: https://arxiv.org/abs/2305.11373
- PICM-Net: https://arxiv.org/abs/2512.20070
- TransTIC: https://arxiv.org/abs/2306.05085
- Rate-Distortion-Cognition: https://arxiv.org/abs/2407.11700
- MALICE: https://arxiv.org/abs/2205.13253
- SRDA/ARDA robustness analysis: https://arxiv.org/abs/2405.07717
- Bitstream collisions: https://arxiv.org/abs/2503.19817
- Diffusion bit-flip robustness: https://arxiv.org/abs/2604.05743
- Backdoor attacks: https://arxiv.org/abs/2302.14677

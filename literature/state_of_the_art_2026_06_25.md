# LIC State Of The Art Map

**Updated:** 2026-06-25  
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
| Dictionary entropy | DCAE, HiDE | learn shared dictionaries/prototypes to inform entropy model | OpenImages, Kodak, CLIC, TESTIMAGES | DCAE official; HiDE code unconfirmed | glyph/stroke dictionary extension is attractive but should be secondary |
| Generative compression | HiFiC, MRIC, CDC, CoD, CADC, AEIC | adversarial or diffusion prior synthesizes realistic detail | FID/KID/LPIPS/DISTS plus distortion metrics | HiFiC/CDC/CoD/AEIC code available; CADC unconfirmed | powerful at low rate but dangerous for exact text unless local/deterministic and secondary |
| VQ generative compression | GLC, PerCo, RDVQ | discrete tokens, diffusion/VQ priors, differentiable codebook-rate relaxation | ultra-low bpp; FID, LPIPS, DISTS | RDVQ official; PerCo open implementation | possible future enhancement tokenization, not the first path |
| OCR-aware compression | Scene Text Quality Assessment, Lai et al. 2024 text logit loss, TFIC, PICD | text quality assessor, OCR-logit loss, OCR-supervised text-focused codec, text/layout side-channel diffusion rendering | screenshot/scene-text/synthetic text; CER, WER, text accuracy, bpp | code often unavailable or side-channel-heavy | closest baselines and reviewer comparison set |
| Task-aware / ICM | TransTIC, Rate-Distortion-Cognition, TTFT, BridgingCompressionMLLM, TFGC, JPEG AI | prompt/adapt a codec for downstream tasks or compressed-domain machine use | detection/classification/segmentation/LVLM tasks; task-rate curves | mixed; JPEG AI reference software official | informs evaluation discipline, but OCR exact sequence is stricter than generic task accuracy |
| Robustness / security | MALICE, backdoor attacks, SRDA/ARDA, LIC common corruptions, bitstream collisions | bitrate DoS, RD attack, poisoned checkpoints, corruption/semantic collision vulnerabilities | multiple LIC models; bpp inflation, PSNR drop, attack success, corruption RD | some code available, many preprints | OSCAR text stream needs CRC, packet boundaries, fallback, and actual serialized attack tests |

## Notes On Specific Recent Papers

### MLIC++ / MLICv2

MLIC++ partitions latent slices and models channel-wise, local spatial, and global spatial correlations with linear-complexity mechanisms. The official repository states that corrected MLIC++ checkpoints require `LatentResidualPrediction` and `SynthesisTransform`, not the old variants. This is the primary base codec target.

### DCAE

DCAE introduces a dictionary-based cross-attention entropy model, using learned dictionary priors from training data. It is strategically interesting because text/glyph/stroke prototypes could become specialized dictionary atoms, but using it too early would confound the main utility-per-bit contribution.

### GLIC

GLIC uses content-adaptive dual-scale graphs and dynamically changes connectivity based on content complexity. This is conceptually aligned with spatially varying redundancy, but graph construction/runtime overhead and code maturity need checking before using it as a main baseline.

### RDVQ

RDVQ addresses the VQ rate mismatch by replacing hard assignments during training with differentiable codebook-distribution relaxation, allowing entropy loss to shape encoder token distributions. It is a strong VQ/generative baseline for ultra-low rates, but exact OCR text can fail catastrophically if token errors alter strokes.

### CoD / CADC / AEIC

CoD trains a compression-oriented diffusion foundation model rather than adapting a text-to-image foundation model. CADC adds uncertainty-guided adaptive quantization, information concentration, and decoder-derived textual conditioning. AEIC uses a shallow encoder and one-step diffusion receiver. These are important low-bitrate perceptual baselines, but OSCAR-LIC should not let stochastic generation become the main claim.

### PICD

PICD explicitly encodes text and image separately, then diffusion-renders screen content. It is highly relevant as a strong perceptual screen-content baseline, but belongs to a semantic-assisted track because text/location side information, OCR errors, privacy, and runtime must all be counted.

### Lai et al. and TFIC

Lai et al. show that OCR text-logit loss can improve CER/WER BD-rate on screenshot datasets. TFIC trains a codec for text-specific OCR utility at low rate. Both motivate OCR-aware losses, but OSCAR-LIC must improve on their likely weaknesses: single-OCR dependence, limited cross-domain evidence, and lack of explicit marginal utility per actual bit.

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
- LALIC arXiv/GitHub: https://arxiv.org/abs/2502.05741, https://github.com/sjtu-medialab/RwkvCompress
- MambaIC arXiv/GitHub: https://arxiv.org/abs/2503.12461, https://github.com/AuroraZengfh/MambaIC
- GLIC CVF/arXiv: https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Adaptive_Learned_Image_Compression_with_Graph_Neural_Networks_CVPR_2026_paper.html
- JPEG AI: https://jpeg.org/jpegai/
- HiFiC: https://hific.github.io/, https://arxiv.org/abs/2006.09965
- CDC: https://openreview.net/forum?id=QIBpzaDCAv, https://github.com/buggyyang/CDC_compression
- PerCo: https://openreview.net/forum?id=ktdETU9JBg
- PICD: https://arxiv.org/abs/2505.05853
- RDVQ: https://arxiv.org/abs/2604.10546, https://github.com/CVL-UESTC/RDVQ
- CoD: https://arxiv.org/abs/2511.18706, https://github.com/microsoft/GenCodec
- CADC: https://arxiv.org/abs/2602.21591
- Lai et al. text quality enhancement: https://arxiv.org/abs/2402.08643
- TFIC: https://arxiv.org/abs/2503.19495
- Scene text quality assessment: https://arxiv.org/abs/2305.11373
- TransTIC: https://arxiv.org/abs/2306.05085
- Rate-Distortion-Cognition: https://arxiv.org/abs/2407.11700
- MALICE: https://arxiv.org/abs/2205.13253
- SRDA/ARDA robustness analysis: https://arxiv.org/abs/2405.07717
- Bitstream collisions: https://arxiv.org/abs/2503.19817
- Backdoor attacks: https://arxiv.org/abs/2302.14677


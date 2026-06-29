# LIC Research Survey Notes

Date: 2026-06-25

Scope: learned image compression (LIC), low-bitrate/generative/VQ compression, entropy modeling, task-aware compression, and robustness/security. This is a working research note for OSCAR-LIC, not a final related-work section.

## High-Level Map

The current LIC landscape has split into three partly overlapping tracks:

1. Rate-distortion optimized neural codecs: hyperprior, autoregressive/channel/spatial context, transformer/CNN entropy models, and practical decoding complexity.
2. Rate-perception / generative codecs: GAN/VAE/VQ/diffusion priors for visually realistic low-bitrate reconstruction, often trading exact fidelity for perceptual quality.
3. Task/security-aware codecs: compression optimized for downstream recognizers, privacy, robustness, or adversarial/security constraints.

OSCAR-LIC fits most naturally in track 3 while borrowing compact-symbol ideas from tracks 1 and 2: keep a frozen high-quality base codec, transmit small actual-byte enhancement streams for text-sensitive latent regions, and optimize the enhancement for OCR utility/no-worse constraints rather than global PSNR alone.

## Core RD LIC And Entropy Modeling

Representative foundations:

- Ballé-style nonlinear transform coding and scale hyperpriors established the analysis/synthesis transform, additive quantization proxy, entropy bottleneck, and hyperlatent side information pattern.
- Minnen et al., "Joint Autoregressive and Hierarchical Priors for Learned Image Compression" (NeurIPS 2018), combined hyperprior side information with an autoregressive context model for stronger latent probability estimation. Source: https://papers.neurips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
- CompressAI remains the most practical reference implementation for factorized prior, hyperprior, joint autoregressive hierarchical prior, and Cheng-style attention codecs. Source: https://interdigitalinc.github.io/CompressAI/models.html

Modern entropy-modeling direction:

- ELIC, "Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding" (CVPR 2022), uses uneven channel grouping and space-channel context to improve RD while controlling speed. It also supports progressive/preview-style decoding, which is relevant to optional OSCAR enhancement layers. Paper/source: https://openaccess.thecvf.com/content/CVPR2022/papers/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.pdf and https://arxiv.org/abs/2203.10886
- MLIC / MLIC++ introduces multi-reference entropy modeling and a linear-complexity variant. The official repository is the base used here. Source: https://github.com/jiangweibeta/mlic and https://openreview.net/pdf/f165c1ebd7407a158d2dc517a15e852a486d614a.pdf
- LIC-TCM mixes CNN and Transformer blocks to capture local and non-local structure with practical complexity. Official implementation exists. Source: https://github.com/jmliu206/lic_tcm
- Dictionary-based entropy modeling (2025 arXiv) uses a learnable dictionary and cross-attention entropy model to inject dataset-level priors into probability estimation. Source: https://arxiv.org/html/2504.00496v1
- ARCHE (2026 arXiv) is another efficiency-oriented RD codec, combining hierarchical, spatial and channel priors with residual refinement and excitation while avoiding recurrent/Transformer blocks. It reports strong Kodak BD-rate reductions versus Ballé/Minnen baselines and VVC Intra, reinforcing that convolutional priors remain competitive when practical latency matters. Source: https://arxiv.org/html/2603.10188
- "What Matters in Practical Learned Image Compression" (CVPR 2026) emphasizes that practical codec quality depends on perceptual/runtime trade-offs, subjective testing and deployment latency rather than RD curves alone. This supports keeping OSCAR-LIC's main claim tied to actual transmitted bytes, decode compatibility and downstream OCR behavior rather than adding a heavy global model only to improve proxy metrics. Source: https://arxiv.org/abs/2605.05148

Implication for OSCAR-LIC:

- We should not compete by adding a heavier global entropy model to MLIC++. The more defensible axis is optional, actual-byte, task-aware enhancement that can be skipped by a base decoder.
- The observed Eval300 results reinforce this: compact K64 side symbols are enough to move OCR, but residual-MSE entropy/codebook objectives are not aligned with OCR utility.

## Transformer-Based Compression

Representative methods:

- TIC, "Transformer-based Image Compression" (DCC 2022), moved transformer blocks into learned compression and has a public CompressAI-based implementation. Source: https://github.com/lumingzzz/TIC
- LIC-TCM (CVPR 2023 highlight) uses mixed Transformer-CNN blocks, balancing non-local modeling and local inductive bias. Source: https://github.com/jmliu206/lic_tcm
- Linear attention modeling for LIC (2025 arXiv) continues the push toward cheaper long-range context and cites Contextformer-style spatio-channel attention. Source: https://arxiv.org/html/2502.05741v2

Implication:

- Transformers are useful for global context and entropy modeling, but OSCAR-LIC's current bottleneck is not global RD. It is sparse candidate selection and symbol assignment. A compact local latent/image patch encoder plus listwise ranking is better aligned than a large transformer in the main codec path.

## Low-Bitrate, VQ, And Generative Compression

GAN/perceptual foundations:

- HiFiC combines learned compression with GAN/perceptual losses and showed strong user-preferred reconstructions at lower rates than prior methods. Sources: https://hific.github.io/ and https://papers.neurips.cc/paper_files/paper/2020/file/8a50bae297807da9e97722a0b3fd8f27-Paper.pdf
- Earlier extreme generative compression work explored selective synthesis: preserve semantically important regions while synthesizing the rest. Source: https://data.vision.ee.ethz.ch/aeirikur/extremecompression/

Rate-distortion-realism:

- CRDR (WACV 2024) explicitly controls rate, distortion, and realism in a single model; official code is available. Source: https://github.com/iwa-shi/CRDR
- The rate-distortion-perception trade-off means a perceptually better reconstruction can be less faithful to the source. That is directly relevant to OCR: generated text-like details can improve naturalness while corrupting exact text.

Diffusion and ultra-low bitrate:

- Diffusion-based / foundation-prior codecs treat reconstruction as conditional generation from a compressed latent or token representation. The advantage is realism at <0.05 bpp; the drawback is slower decode and possible hallucination.
- DiffEIC uses latent feature guidance and diffusion priors for extreme compression. Source: https://github.com/huai-chang/DiffEIC
- RDEIC accelerates diffusion extreme compression using compressed feature initialization and residual diffusion. Sources: https://arxiv.org/html/2410.02640v1 and https://github.com/huai-chang/RDEIC
- DiffO / single-step diffusion targets ultra-low bitrate with VQ-residual training and rate-aware noise modulation, aiming to reduce diffusion decoding cost. Source: https://arxiv.org/html/2506.16572v1
- StableCodec (ICCV 2025) also pushes one-step diffusion for extreme image compression. Source: https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_StableCodec_Taming_One-Step_Diffusion_for_Extreme_Image_Compression_ICCV_2025_paper.pdf
- OSDiff / One-Step Diffusion for Perceptual Image Compression (2026 arXiv) targets diffusion decoding latency directly by using a single denoising step and a discriminator in Stable Diffusion feature space; code/models are advertised. Source: https://arxiv.org/html/2602.01570v1 and https://github.com/cheesejiang/OSDiff
- OneDC (2025 arXiv) argues multi-step sampling is unnecessary for generative compression, using a latent compression module, one-step diffusion generator, hyperprior semantic guidance and semantic distillation from a pretrained tokenizer. Source: https://arxiv.org/html/2505.16687v2 and https://onedc-codec.github.io/
- PerCoV2 (2025 arXiv) updates ultra-low-bitrate perceptual compression to the Stable Diffusion 3 ecosystem and explicitly models discrete hyper-latent distributions with masked/autoregressive methods. Source: https://arxiv.org/html/2503.09368v1

VQ/generative latent coding:

- GLC performs transform coding in a generative VQ-VAE latent space and reports ultra-low-bitrate realism/fidelity, including <0.04 bpp natural-image results and a public repository. Sources: https://arxiv.org/abs/2505.16177 and https://github.com/jzyustc/GLC
- RDVQ proposes differentiable VQ for rate-distortion optimization by replacing hard nearest-neighbor assignment with a differentiable distance-aware distribution, directly relevant to our codebook assignment problem. Source: https://arxiv.org/html/2604.10546v1
- Semantics-Guided Generative Image Compression (ICIP 2025) extends multimodal image semantic compression with semantic segmentation guidance and content-adaptive diffusion, reporting better PSNR/perceptual metrics and lower encode/decode time. Source: https://arxiv.org/abs/2505.24015

Implication:

- For OSCAR-LIC, diffusion/GAN full-image synthesis is risky because text must remain exact. However, VQ-residual and differentiable assignment ideas are very relevant. Our failed center-prior assignment diagnostic suggests that a future symbol learner should be candidate-conditioned, differentiable or listwise, and constrained by OCR no-worse labels.
- The fast one-step diffusion trend reduces the old "too slow" objection, but not the text-fidelity objection. PICD's separate text/image conditioning is a closer comparator for screen content, while OSCAR-LIC should stay deterministic and actual-byte counted for exact OCR claims.

## Task-Aware And Downstream-Aware Compression

Representative work:

- Recognition-aware LIC jointly optimizes rate-distortion and task losses for downstream recognition/classification. Source: https://arxiv.org/abs/2202.00198
- Task-Aware Quantization Network learns image-specific JPEG quantization tables for downstream task quality while preserving JPEG compatibility. Source: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650307.pdf
- AccelIR optimizes image compression for downstream restoration pipelines rather than standalone distortion. Source: https://openaccess.thecvf.com/content/CVPR2023/papers/Ye_AccelIR_Task-Aware_Image_Compression_for_Accelerating_Neural_Restoration_CVPR_2023_paper.pdf
- TAMC / taskonomy-aware multi-task compression asks which task representations should be compressed together, using task relationships rather than only pixel distortion. Source: https://openreview.net/forum?id=x33vSZUg0A
- PICD (CVPR 2025) targets both screen and natural images by separately encoding text and image content and rendering them with diffusion; it reports better text accuracy and perceptual quality than prior perceptual codecs. Source: https://openaccess.thecvf.com/content/CVPR2025/html/Xu_PICD_Versatile_Perceptual_Image_Compression_with_Diffusion_Rendering_CVPR_2025_paper.html
- PICM-Net (2025 arXiv) studies progressive learned compression for machine perception using trit-plane coding and an adaptive decoding controller to maintain downstream confidence. Source: https://arxiv.org/html/2512.20070v1

Implication:

- OSCAR-LIC's OCR-oriented design is well aligned with downstream-aware compression, but should be framed more specifically: preserve machine-readable text under actual transmitted bytes, not just improve a generic classifier.
- Our multi-teacher OCR evaluation is important. PARSeq and Tesseract disagree, and Tesseract is sensitive to sparse latent residuals while PARSeq is often invariant. A single teacher would produce brittle claims.
- Progressive machine-oriented work strengthens the case for optional sections: OSCAR's text layer can be presented as a task-conditioned enhancement level whose bytes are justified only when they improve downstream OCR under no-worse constraints.

## Robustness, Security, And Privacy

Adversarial robustness of LIC:

- "On the Adversarial Robustness of Learning-based Image Compression Against Rate-Distortion Attacks" proposes practical specific-ratio and agnostic-ratio RD attacks, analyzes entropy/hyperprior and IGDN sensitivity, and compares adversarial training with online updating. Source: https://arxiv.org/abs/2405.07717
- "Towards Robust Neural Image Compression" studies small adversarial perturbations that degrade learned compression robustness. Source: https://njuvision.github.io/RobustNIC/
- Manipulation attacks on LIC show imperceptible inputs can cause bitrate blow-up while preserving reconstruction quality, creating a potential denial-of-service vector. Source: https://bibbase.org/network/publication/liu-wu-wu-wang-feng-tan-garg-manipulationattacksonlearnedimagecompression-2023
- Bitstream collision attacks reveal cases where semantically different images can map to identical neural-compression bitstreams. Source: https://arxiv.org/abs/2503.19817
- Bitstream collision work is especially relevant to optional enhancement streams: an attacker may not need to destroy the base image if they can cause distinct text-sensitive inputs to share or confuse side information. We should test semantic integrity of `BASE_MAIN` plus `TEXT_*`, not only PSNR after perturbations.

Compression-resistant downstream attacks:

- Transferable LIC-resistant adversarial perturbations target classification systems that apply LIC before recognition; the BMVC 2024 work improves transfer across quality levels and codec architectures. Sources: https://arxiv.org/abs/2401.03115 and https://bmvc2024.org/proceedings/249/

Privacy-aware compression:

- Privacy-Shielded Image Compression (PSIC) uses learned compression to preserve perceptual quality while preventing exploitation by vision-language pretrained models, with optional decoding conditions to recover semantic information. Source: https://openreview.net/forum?id=olzs3zVsE7&noteId=ynl4Ife1Lg

Implication:

- OSCAR-LIC's bitstream must be evaluated not only for RD/OCR but also for robustness: truncation, bit errors in optional sections, malicious enhancement indices, and adversarial inputs that trigger excessive enhancement payload or OCR harm.
- The current `.oscr` base-plus-optional-section design is promising because base-only decode is preserved. Future tests should include corrupted `TEXT_GATE` / `TEXT_MAIN`, random index flips, and adversarial candidate selection stress.

## Datasets And Metrics Commonly Used

Common image compression datasets:

- Kodak: 24 images, legacy RD comparison.
- Tecnick: higher-resolution natural images.
- CLIC: common learned compression benchmark, often used for high-resolution and perceptual metrics.
- DIV2K/Flickr/COCO/OpenImages/ImageNet subsets: training or perceptual evaluation depending on paper.

Common metrics:

- Distortion: PSNR, MS-SSIM, sometimes LPIPS/DISTS.
- Perception/realism: FID, KID, human preference, sometimes CLIP-related metrics.
- Rate: actual bpp from arithmetic-coded bitstreams or serialized files; estimated likelihood bpp is insufficient for final claims.
- Task-aware: task accuracy, mAP, OCR CER/WER/exact match, restoration PSNR/LPIPS after downstream model.
- Robustness/security: bitrate blow-up, reconstruction degradation under bounded perturbation, bitstream collision rate, downstream attack success, corrupted-bitstream graceful failure.

OSCAR-LIC metric policy:

- Report actual complete-file bpp from `compress/decompress` and `.oscr` byte counts.
- Keep per-section byte breakdown: `BASE_MAIN`, `TEXT_GATE`, `TEXT_MAIN`, structural overhead.
- Report PARSeq and Tesseract PSM8 at minimum; add a stronger scene-text recognizer if dependency risk is acceptable.
- Use paired sample-level analysis, not only aggregate CER, because many diagnostics change only 1-12 samples out of 225.

## Current OSCAR-LIC Design Consequences

Evidence gathered on Eval300 so far:

- Learned `tesseract_parseq_safe` selection plus selected-val K16 codebook is positive: compact counted payload preserves PARSeq and improves Tesseract.
- Train-codebook/eval-selection K64 residual-MSE is positive but weak, retaining about half of raw top-25 Tesseract gain.
- Utility-weighted K64 with global `alpha=8` improves the mean to about 59% raw-gain retention but is unstable across seeds.
- `alpha=4`, hard non-harm filtering, zero-code thresholds, and center-utility-biased assignment do not solve the instability. The corrected N136 compact run is especially important: center-utility assignment preserves PARSeq at matched actual compact bpp, but yields only `-1` Tesseract char versus nearest and trails the current compact policy by `+7` chars. N137 shows this is not just a threshold choice: validation oracle-change codes are rarely center-utility rank1 and are often infeasible under tight relative-error guards. N138 shows train-positive exception centers can improve residual feasibility, N139 shows that appending K16 exception centers to K64 without an explicit selector is a real-stream no-op (`75` selected candidates, `0` assignment changes), N140 shows distance-only exception selectors are not source-robust, and N141 shows the conservative explicit exception policy lowers rate but worsens Tesseract versus current.

Design recommendation:

1. Keep frozen MLIC++ base and `.oscr` optional enhancement framing.
2. Keep the model simple: compact symbols and local patch/latent features rather than a large generative decoder.
3. Train a candidate-conditioned assignment/gating model with no-worse calibration. Center-level priors and global scalar weights are not enough.
4. Consider a small utility-positive exception-center bank as a compact symbol distribution change only if paired with an explicit exception selector and no-headroom gating based on richer candidate-local evidence; N138 shows useful feasibility improves while false opportunity also rises, N139 shows nearest assignment alone never selects the appended centers, N140 rejects distance-only selectors, and N141 shows sparse exception-only actual OCR is not enough.
5. Treat generative/diffusion ideas as optional perceptual priors only after text exactness is protected.
6. Add robustness tests early: malformed optional sections, bit flips, index corruption, adversarial candidate over-selection, and privacy/security analysis.

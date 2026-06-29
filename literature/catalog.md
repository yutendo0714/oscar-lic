# Literature Catalog

**Snapshot:** 2026-06-24  
**Source of truth:** `paper_registry.csv`

This catalog renders every registry entry. Performance numbers are author-reported unless a reproduction record explicitly says otherwise. Cross-paper numerical comparison requires matching reference codec, test protocol, training data and actual-rate accounting.

## General Learned Image Compression

### End-to-End Optimized Image Compression (2017, ICLR)

- **Evidence status:** peer_reviewed
- **Core idea:** Nonlinear transform coding with GDN and differentiable quantization surrogate.
- **Datasets:** Kodak; ImageNet-derived training
- **Metrics:** PSNR; MS-SSIM; rate
- **Reported result:** Foundational end-to-end learned transform codec.
- **Difference / caveat:** Historical baseline; implementation details and entropy coder differ from modern frameworks.
- **Implementation:** reference_framework; https://github.com/tensorflow/compression
- **Primary source:** https://openreview.net/forum?id=rJxdQ3jeg
- **Research priority:** P1

### Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules (2020, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Gaussian-mixture entropy model and attention-enhanced transforms.
- **Datasets:** Kodak
- **Metrics:** PSNR; MS-SSIM; bpp
- **Reported result:** Widely used strong convolutional baseline.
- **Difference / caveat:** CompressAI implementation may not be byte-identical to author code.
- **Implementation:** framework_reimplementation; https://github.com/InterDigitalInc/CompressAI
- **Primary source:** https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_Learned_Image_Compression_With_Discretized_Gaussian_Mixture_Likelihoods_and_Attention_CVPR_2020_paper.html
- **Research priority:** P1

### Scaling Learned Image Compression Models up to 1 Billion (2025, arXiv)

- **Evidence status:** preprint
- **Core idea:** Scale an HPCM-style LIC model from 68.5M parameters to 1B and fit power-law trends for model size and training compute.
- **Datasets:** LIC benchmark sets used by HPCM; verify exact splits
- **Metrics:** RD loss; BD-rate; scaling laws
- **Reported result:** Author reports state-of-the-art RD performance for HPCM-1B.
- **Difference / caveat:** Compute-heavy scaling study; not suitable as OSCAR first-path model but useful for upper-bound context.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2508.09075
- **Research priority:** P2

### Adaptive Learned Image Compression with Graph Neural Networks (2026, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Dual-scale content-adaptive graph and variable neighbor count.
- **Datasets:** Kodak; Tecnick; CLIC
- **Metrics:** PSNR; BD-rate
- **Reported result:** Author reports -19.29/-21.69/-18.71% BD-rate vs VTM-9.1.
- **Difference / caveat:** Code was announced; verify completeness/checkpoints at use time.
- **Implementation:** announced_verify_release; https://github.com/UnoC-727/GLIC
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Adaptive_Learned_Image_Compression_with_Graph_Neural_Networks_CVPR_2026_paper.html
- **Research priority:** P1

## Entropy Modeling

### Joint Autoregressive and Hierarchical Priors for Learned Image Compression (2018, NeurIPS)

- **Evidence status:** peer_reviewed
- **Core idea:** Combines hyperprior and autoregressive spatial context.
- **Datasets:** Kodak; web image training corpus
- **Metrics:** PSNR; MS-SSIM; rate
- **Reported result:** Improved entropy estimates over either prior alone.
- **Difference / caveat:** Sequential decoding cost is central.
- **Implementation:** reference_framework; https://github.com/tensorflow/compression
- **Primary source:** https://arxiv.org/abs/1809.02736
- **Research priority:** P0

### Variational Image Compression with a Scale Hyperprior (2018, ICLR)

- **Evidence status:** peer_reviewed
- **Core idea:** Side-information hyperprior predicts spatially varying latent scales.
- **Datasets:** Kodak; ImageNet-derived training
- **Metrics:** PSNR; MS-SSIM; rate
- **Reported result:** Established hyperprior as a standard LIC component.
- **Difference / caveat:** Compare actual bitstreams and color preprocessing carefully.
- **Implementation:** reference_framework; https://github.com/tensorflow/compression
- **Primary source:** https://openreview.net/forum?id=rkcQFMZRb
- **Research priority:** P0

### ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding (2022, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Uneven channel groups and space-channel context for efficiency and progressive decoding.
- **Datasets:** ImageNet; Kodak; CLIC; Tecnick
- **Metrics:** PSNR; MS-SSIM; bpp; runtime
- **Reported result:** Strong speed/RD trade-off and common baseline.
- **Difference / caveat:** Frequently used code is a reimplementation; label accordingly.
- **Implementation:** third_party_reimplementation; https://github.com/VincentChandelier/ELiC-ReImplemetation
- **Primary source:** https://arxiv.org/abs/2203.10886
- **Research priority:** P1

### MLIC: Multi-Reference Entropy Model for Learned Image Compression (2023, ACM Multimedia)

- **Evidence status:** peer_reviewed
- **Core idea:** Joint channel, local spatial and global references.
- **Datasets:** DIV2K; Flickr2K; CLIC; COCO; ImageNet; Kodak; Tecnick
- **Metrics:** PSNR; MS-SSIM; BD-rate
- **Reported result:** Author reports Kodak BD-rate -8.05% for MLIC and -11.39% for MLIC+ vs VTM-17.0.
- **Difference / caveat:** Use exact model generation and actual stream path.
- **Implementation:** official; https://github.com/JiangWeibeta/MLIC
- **Primary source:** https://arxiv.org/abs/2211.07273
- **Research priority:** P0

### Learned Image Compression with Dictionary-based Entropy Model (2025, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Learnable external dictionary supplies training-set priors to cross-attention entropy model.
- **Datasets:** OpenImages; Kodak; CLIC; TESTIMAGES
- **Metrics:** PSNR; MS-SSIM; bpp; latency
- **Reported result:** Reported state-of-the-art rate/latency balance.
- **Difference / caveat:** README does not fully pin dependency versions; create local lock.
- **Implementation:** official; https://github.com/CVL-UESTC/DCAE
- **Primary source:** https://arxiv.org/abs/2504.00496
- **Research priority:** P0

### MLIC++: Linear Complexity Multi-Reference Entropy Modeling for Learned Image Compression (2025, ACM TOMM)

- **Evidence status:** peer_reviewed
- **Core idea:** Linear-complexity local/global multi-reference entropy model.
- **Datasets:** MLIC-Train-100K; Kodak; Tecnick; CLIC Pro
- **Metrics:** PSNR; MS-SSIM; BD-rate; complexity
- **Reported result:** Author reports Kodak/Tecnick/CLIC Pro BD-rate -13.39/-17.59/-13.08% vs VTM-17.0.
- **Difference / caveat:** Repository notes architecture/checkpoint corrections; pin commit and model class.
- **Implementation:** official; https://github.com/JiangWeibeta/MLIC
- **Primary source:** https://arxiv.org/abs/2307.15421
- **Research priority:** P0

### MLICv2: Enhanced Multi-Reference Entropy Modeling for Learned Image Compression (2025, ACM TOMM)

- **Evidence status:** peer_reviewed
- **Core idea:** Token mixing, hyperprior-guided global correlation, channel reweighting and instance adaptation.
- **Datasets:** Kodak; Tecnick; CLIC Pro
- **Metrics:** PSNR; BD-rate
- **Reported result:** Author reports MLICv2+ BD-rate around -20.46/-24.35/-19.14% vs VTM-17.0.
- **Difference / caveat:** Confirm whether complete MLICv2 training/evaluation code is present in the selected snapshot.
- **Implementation:** paper_same_family_repo; https://github.com/JiangWeibeta/MLIC
- **Primary source:** https://arxiv.org/abs/2504.19119
- **Research priority:** P1

### HiDE: Hierarchical Dictionary-Based Entropy Modeling for Learned Image Compression (2026, arXiv)

- **Evidence status:** preprint
- **Core idea:** Hierarchical global/local dictionaries and multi-receptive-field parameter estimation.
- **Datasets:** Kodak; CLIC; Tecnick
- **Metrics:** PSNR; BD-rate
- **Reported result:** Preprint reports -18.50/-21.99/-24.01% BD-rate vs VTM-12.1.
- **Difference / caveat:** Not peer-reviewed at snapshot; code unconfirmed.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2603.06766
- **Research priority:** P2

## Transformer / SSM / Graph LIC

### The Devil Is in the Details: Window-based Attention for Image Compression (2022, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Window attention and entropy modeling tailored to LIC.
- **Datasets:** Kodak; Tecnick; CLIC
- **Metrics:** PSNR; MS-SSIM; bpp
- **Reported result:** Representative early Transformer LIC.
- **Difference / caveat:** Official implementation availability must be rechecked before use.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2022/html/Zou_The_Devil_Is_in_the_Details_Window-Based_Attention_for_Image_CVPR_2022_paper.html
- **Research priority:** P2

### Learned Image Compression with Mixed Transformer-CNN Architectures (2023, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Parallel Transformer-CNN mixture blocks and channel-wise entropy model.
- **Datasets:** Kodak; Tecnick; CLIC Pro
- **Metrics:** PSNR; MS-SSIM; bpp; complexity
- **Reported result:** Strong hybrid backbone baseline.
- **Difference / caveat:** Check training corpus and exact VTM reference.
- **Implementation:** official; https://github.com/jmliu206/LIC_TCM
- **Primary source:** https://arxiv.org/abs/2303.14978
- **Research priority:** P1

### Frequency-aware Transformer for Learned Image Compression (2024, ICLR)

- **Evidence status:** peer_reviewed
- **Core idea:** Frequency-decomposition window attention and frequency modulation.
- **Datasets:** OpenImages or author training corpus; Kodak; Tecnick; CLIC
- **Metrics:** PSNR; MS-SSIM; BD-rate
- **Reported result:** Author reports double-digit BD-rate gains over VTM on standard sets.
- **Difference / caveat:** Verify exact paper URL and training corpus from pinned README/paper before citation.
- **Implementation:** official; https://github.com/qingshi9974/ICLR2024-FTIC
- **Primary source:** https://openreview.net/
- **Research priority:** P1

### CMIC: Content-Adaptive Mamba for Learned Image Compression (2025, arXiv)

- **Evidence status:** preprint
- **Core idea:** Content-aware token reorganization and prompt dictionary adapt Mamba-style state-space modeling to image content.
- **Datasets:** Kodak; Tecnick; CLIC
- **Metrics:** PSNR; MS-SSIM; BD-rate; complexity
- **Reported result:** Author reports BD-rate reductions of 15.91%; 21.34%; 17.58% versus VTM-21.0 on Kodak; Tecnick; CLIC.
- **Difference / caveat:** Code/checkpoints were not confirmed at this snapshot; treat as preprint until venue/code status stabilizes.
- **Implementation:** announced_later; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2508.02192
- **Research priority:** P2

### Linear Attention Modeling for Learned Image Compression (2025, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Bi-RWKV transform and RWKV spatial-channel context.
- **Datasets:** Kodak; Tecnick; CLIC Pro
- **Metrics:** PSNR; MS-SSIM; BD-rate; memory
- **Reported result:** Author reports approximately -15% to -17% BD-rate vs VTM-9.1 depending on set/version.
- **Difference / caveat:** CUDA operator compatibility is a practical risk; pinned snapshot removed hard T_MAX.
- **Implementation:** official; https://github.com/sjtu-medialab/RwkvCompress
- **Primary source:** https://arxiv.org/abs/2502.05741
- **Research priority:** P1

### MambaIC: State Space Models for High-Performance Learned Image Compression (2025, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** State-space transform/context with local attention.
- **Datasets:** Flickr30k training; Kodak; Tecnick; CLIC
- **Metrics:** PSNR; MS-SSIM; bpp; complexity
- **Reported result:** High-resolution efficiency focus.
- **Difference / caveat:** Repository explicitly says it is not exact original code; training data differs from some competitors.
- **Implementation:** author_reimplementation; https://github.com/AuroraZengfh/MambaIC
- **Primary source:** https://arxiv.org/abs/2503.12461
- **Research priority:** P2

## Practical and Deployable LIC

### Block-based Learned Image Compression without Blocking Artifacts (2026, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Analytically derive minimum overlap for artifact-free tiled inference without retraining.
- **Datasets:** 4K test images; four CNN LIC models
- **Metrics:** RD equivalence; peak memory; peak compute
- **Reported result:** Reports average peak memory reduced to 13.94% encoder and 13.33% decoder of full-image processing.
- **Difference / caveat:** Primarily CNN-based; applicability to global attention/SSM must be tested.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Block-based_Learned_Image_Compression_without_Blocking_Artifacts_CVPR_2026_paper.html
- **Research priority:** P2

### Efficient Learned Image Compression without Entropy Coding (2026, arXiv / ICML verify)

- **Evidence status:** preprint
- **Core idea:** Entropy-coding-free multi-rate LIC using unconstrained vector quantization and context-conditioned autoregressive transform to reduce statistical/correlation redundancy.
- **Datasets:** Kodak and LIC perceptual benchmarks
- **Metrics:** LPIPS; bitrate; encode/decode latency
- **Reported result:** Author reports comparable performance to entropy-coded variants with faster encode/decode and up to 67.86% bitrate reduction over MS-ILLM on Kodak LPIPS.
- **Difference / caveat:** Official inference code exists; exact serialized index format/checkpoint hash/license and actual complete-file bpp must be audited before baseline use.
- **Implementation:** official_inference; https://github.com/SevenCTHU/EF-LIC
- **Primary source:** https://arxiv.org/abs/2605.23323
- **Research priority:** P2

### What Matters in Practical Learned Image Compression (2026, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Joint perceptual/runtime design and performance-aware NAS.
- **Datasets:** subjective study datasets; mobile tests
- **Metrics:** human preference; bitrate savings; mobile latency
- **Reported result:** Reports 2.3–3x bitrate savings in subjective tests and 230/150 ms 12MP encode/decode on iPhone 17 Pro Max.
- **Difference / caveat:** Subjective protocol and implementation access are essential for reproduction.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2026/html/Tatwawadi_What_Matters_in_Practical_Learned_Image_Compression_CVPR_2026_paper.html
- **Research priority:** P2

## Rate–Distortion–Perception

### Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff (2019, ICML)

- **Evidence status:** peer_reviewed
- **Core idea:** Formalizes incompatibility between perfect perception and minimal distortion at fixed rate.
- **Datasets:** theoretical and image experiments
- **Metrics:** rate; distortion; perception divergence
- **Reported result:** Theoretical basis for perceptual compression evaluation.
- **Difference / caveat:** Distributional realism does not guarantee instance-level text correctness.
- **Implementation:** theory; no verified public implementation
- **Primary source:** https://proceedings.mlr.press/v97/blau19a.html
- **Research priority:** P0

### VLIC: Vision-Language Models as Perceptual Judges for Human-Aligned Image Compression (2026, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Post-train diffusion codec with zero-shot VLM binary judgments.
- **Datasets:** human 2AFC and image compression sets
- **Metrics:** 2AFC agreement; user study; perceptual metrics
- **Reported result:** Reports competitive human-aligned performance.
- **Difference / caveat:** VLM judge bias and inability to enforce exact text semantics.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2026/html/Sargent_VLIC_Vision-Language_Models_As_Perceptual_Judges_for_Human-Aligned_Image_Compression_CVPR_2026_paper.html
- **Research priority:** P2

## Generative Compression

### High-Fidelity Generative Image Compression (2020, NeurIPS)

- **Evidence status:** peer_reviewed
- **Core idea:** Adversarial perceptual decoder with explicit rate control.
- **Datasets:** OpenImages; Kodak; CLIC
- **Metrics:** FID; LPIPS; PSNR; user study; bpp
- **Reported result:** Canonical GAN-based high-fidelity compression.
- **Difference / caveat:** GAN texture can alter text and exact symbols.
- **Implementation:** reference_or_related; https://github.com/tensorflow/compression/tree/master/models/hific
- **Primary source:** https://arxiv.org/abs/2006.09965
- **Research priority:** P1

### Lossy Image Compression with Conditional Diffusion Models (2023, NeurIPS)

- **Evidence status:** peer_reviewed
- **Core idea:** Separate transmitted content latent from generated texture variables.
- **Datasets:** multiple natural image datasets
- **Metrics:** FID; distortion metrics; bpp; steps
- **Reported result:** Strong perceptual quality with few-step conditional diffusion variants.
- **Difference / caveat:** Sampling latency and stochastic semantic changes.
- **Implementation:** official; https://github.com/buggyyang/CDC_compression
- **Primary source:** https://arxiv.org/abs/2209.06950
- **Research priority:** P1

### Multi-Realism Image Compression with a Conditional Generator (2023, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Single representation supports adjustable realism at decoder.
- **Datasets:** natural image benchmarks
- **Metrics:** rate; distortion; perceptual metrics; realism control
- **Reported result:** Demonstrates a continuum between distortion and realism without new bitstream.
- **Difference / caveat:** Decoder control may change text semantics.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2023/html/Agustsson_Multi-Realism_Image_Compression_With_a_Conditional_Generator_CVPR_2023_paper.html
- **Research priority:** P2

### CADC: Content Adaptive Diffusion-Based Generative Image Compression (2026, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Uncertainty-guided adaptive quantization, auxiliary information concentration and bitrate-free textual conditioning from reconstruction.
- **Datasets:** ultra-low-rate natural image benchmarks
- **Metrics:** perceptual metrics; bpp
- **Reported result:** Reports state-of-the-art perceptual quality at ultra-low rates.
- **Difference / caveat:** "Bitrate-free" conditioning is decoder-derived but its computation and possible semantic bias still matter.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2026/html/Sheng_CADC_Content_Adaptive_Diffusion-Based_Generative_Image_Compression_CVPR_2026_paper.html
- **Research priority:** P2

### CoD-Lite: Real-Time Diffusion-Based Generative Image Compression (2026, arXiv)

- **Evidence status:** preprint
- **Core idea:** One-step lightweight convolutional diffusion codec with compression-oriented pretraining; distillation and adversarial fine-tuning.
- **Datasets:** 1080p and generative compression benchmarks
- **Metrics:** FID; bpp; encode/decode FPS
- **Reported result:** Author reports 60 FPS encoding and 42 FPS decoding at 1080p on A100 and 85% bitrate reduction at comparable FID to MS-ILLM.
- **Difference / caveat:** Generative hallucination risk is acknowledged by authors; exact text/OCR preservation must be separately tested.
- **Implementation:** official_announced; https://github.com/microsoft/GenCodec/CoD_Lite
- **Primary source:** https://arxiv.org/abs/2604.12525
- **Research priority:** P2

### CoD: A Diffusion Foundation Model for Image Compression (2026, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Compression-oriented diffusion foundation model trained from scratch rather than adapting text-to-image models.
- **Datasets:** open image-only datasets; ultra-low-rate benchmarks
- **Metrics:** FID; LPIPS; DISTS; PSNR; bpp; training cost
- **Reported result:** Reports particularly strong results near 0.0039 bpp and much lower training cost than Stable Diffusion pretraining.
- **Difference / caveat:** Large decoder and generative semantic fidelity remain concerns for text.
- **Implementation:** official; https://github.com/microsoft/GenCodec/tree/main/CoD
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2026/html/Jia_CoD_A_Diffusion_Foundation_Model_for_Image_Compression_CVPR_2026_paper.html
- **Research priority:** P1

### FlowCodec: One-Step Flow Prior for Generative Image Compression (2026, arXiv)

- **Evidence status:** preprint
- **Core idea:** Decouples latent compression from one-step latent transport through pretrained flow/diffusion priors with lightweight LoRA adaptation and multi-bitrate control.
- **Datasets:** Kodak; Tecnick; DIV2K; CLIC 2020
- **Metrics:** PSNR; MS-SSIM; LPIPS; DISTS; user study; runtime
- **Reported result:** Author reports SOTA perceptual metrics for Qwen-image variant and favorable one-step encoding speed; supplementary reports text/identity semantic preservation metrics.
- **Difference / caveat:** Large pretrained generative priors may hallucinate or alter exact text; decoder cost and prompt/side-info policy must be counted.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2606.21030
- **Research priority:** P2

### Region-Adaptive Generative Compression with Spatially Varying Diffusion Models (2026, arXiv)

- **Evidence status:** preprint
- **Core idea:** Spatially varying diffusion denoising and importance maps support controllable non-uniform bit allocation.
- **Datasets:** ROI and full-image perceptual benchmarks
- **Metrics:** ROI-masked perceptual quality; full-image perceptual quality; bpp
- **Reported result:** Author reports state-of-the-art ROI-masked and full-image perceptual quality against spatially adaptive baselines.
- **Difference / caveat:** Importance maps and possible side information must be counted; exact text fidelity remains a risk.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2604.01122
- **Research priority:** P2

### Ultra-Low Bitrate Perceptual Image Compression with Shallow Encoder (2026, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Asymmetric shallow encoder with one-step diffusion decoder and dual-side distillation.
- **Datasets:** ultra-low-rate benchmarks; 1080p runtime tests
- **Metrics:** RDP; FPS; bpp
- **Reported result:** Reports 35.8 FPS encoding for 1080p images.
- **Difference / caveat:** Evaluate exact sequence fidelity before using for text.
- **Implementation:** official; https://github.com/LuizScarlet/AEIC
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Ultra-Low_Bitrate_Perceptual_Image_Compression_with_Shallow_Encoder_CVPR_2026_paper.html
- **Research priority:** P1

## Ultra-Low-Rate Generative Compression

### PerCo (SD): Open Perceptual Compression (2024, arXiv)

- **Evidence status:** preprint
- **Core idea:** Open Stable Diffusion implementation of PerCo-style compression.
- **Datasets:** MSCOCO-30k
- **Metrics:** FID; LPIPS; distortion; bpp
- **Reported result:** Open competitive alternative with fidelity/perception tradeoff.
- **Difference / caveat:** Verify current weight release and exact license.
- **Implementation:** announced_or_open_verify; https://github.com/Nikolai10/PerCo
- **Primary source:** https://arxiv.org/abs/2409.20255
- **Research priority:** P2

### Towards Image Compression with Perfect Realism at Ultra-Low Bitrates (2024, ICLR)

- **Evidence status:** peer_reviewed
- **Core idea:** VQ local representation plus global image description conditions a diffusion prior.
- **Datasets:** Kodak; natural image sets
- **Metrics:** FID; KID; bpp; distortion
- **Reported result:** Targets rates down to about 0.003 bpp.
- **Difference / caveat:** Original foundation model/code not fully open; not suitable as sole reproducibility baseline.
- **Implementation:** original_closed_or_unreleased; no verified public implementation
- **Primary source:** https://openreview.net/forum?id=ktdETU9JBg
- **Research priority:** P1

## VQ-Based Compression

### Generative Latent Coding for Ultra-Low Bitrate Image Compression (2024, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** VQ-VAE latent, categorical hyperprior and code prediction supervision.
- **Datasets:** CLIC2020; natural images; faces
- **Metrics:** FID; LPIPS; DISTS; bpp
- **Reported result:** Reports comparable FID to MS-ILLM with materially fewer bits on CLIC2020.
- **Difference / caveat:** Perceptual distribution metrics cannot substitute OCR correctness.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2024/html/Jia_Generative_Latent_Coding_for_Ultra-Low_Bitrate_Image_Compression_CVPR_2024_paper.html
- **Research priority:** P1

### HVQ-CGIC: Enabling Hyperprior Entropy Modeling for VQ-Based Controllable Generative Image Compression (2025, arXiv)

- **Evidence status:** preprint
- **Core idea:** Introduce a VQ hyperprior entropy model and RD-control loss for VQ-based controllable generative compression.
- **Datasets:** Kodak; generative compression comparisons
- **Metrics:** LPIPS; bits; RD control
- **Reported result:** Author reports 61.3% fewer bits at matched LPIPS versus Control-GIC; CDC; HiFiC on Kodak.
- **Difference / caveat:** Preprint timing/status and code availability need verification; exact OCR fidelity untested.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2512.07192
- **Research priority:** P2

### Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression (2026, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Differentiable codebook-distribution relaxation enables entropy-constrained VQ and test-time rate control.
- **Datasets:** DIV2K validation and perceptual benchmarks
- **Metrics:** DISTS; LPIPS; bitrate; parameters
- **Reported result:** Reports up to 75.71% bitrate reduction at matched DISTS and 37.63% at matched LPIPS vs RDEIC on DIV2K-val.
- **Difference / caveat:** Repository license and third-party tokenizer assets require review.
- **Implementation:** official; https://github.com/CVL-UESTC/RDVQ
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Differentiable_Vector_Quantization_for_Rate-Distortion_Optimization_of_Generative_Image_Compression_CVPR_2026_paper.html
- **Research priority:** P1

## Text-Explicit Generative Compression

### PICD: Versatile Perceptual Image Compression with Diffusion Rendering (2025, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Separately encode text and image, then diffusion-render screen content.
- **Datasets:** WebUI training; SCI1K; SIQAD; natural image sets
- **Metrics:** text Jaccard/accuracy; FID; LPIPS; DISTS; bpp; runtime
- **Reported result:** Strong text and perceptual quality for screen images.
- **Difference / caveat:** Text/position side-channel, OCR failure dependency, privacy and slow diffusion must be fully accounted.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2025/html/Xu_PICD_Versatile_Perceptual_Image_Compression_with_Diffusion_Rendering_CVPR_2025_paper.html
- **Research priority:** P0

## Semantic Text-Guided Compression

### Multi-Modality Deep Network for Extreme Learned Image Compression (2023, AAAI)

- **Evidence status:** peer_reviewed
- **Core idea:** Inject transmitted text semantics into encoder, entropy model and decoder through cross-modal attention/complement modules and semantic-consistency loss.
- **Datasets:** captioned image datasets; verify exact splits from paper
- **Metrics:** LPIPS; FID; KID; PSNR; MS-SSIM; user study; total bitrate
- **Reported result:** Author reports comparable or better perceptual results than prior methods operating at roughly 2x–4x the bitrate.
- **Difference / caveat:** Caption side information changes the coding problem; count compressed text and all metadata, and do not treat semantic plausibility as OCR preservation.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2304.13583
- **Research priority:** P1

### LMM-driven Semantic Image-Text Coding for Ultra Low-bitrate Learned Image Compression (2024, IEEE VCIP)

- **Evidence status:** peer_reviewed_poster
- **Core idea:** Generate and compress captions inside an LMM-assisted image-text coding pipeline and fine-tune an LIC network with semantic-perceptual objectives.
- **Datasets:** Kodak and paper-specific low-rate evaluation sets; verify exact split
- **Metrics:** LPIPS; LPIPS BD-rate; PSNR; MS-SSIM; bitrate
- **Reported result:** Author reports 41.58% LPIPS BD-rate improvement over compared methods.
- **Difference / caveat:** Depends on LMM, caption coder and restoration model; pin every weight and count caption bytes, prompts and container overhead.
- **Implementation:** official; https://github.com/tokkiwa/ImageTextCoding
- **Primary source:** https://arxiv.org/abs/2411.13033
- **Research priority:** P1

### Neural Image Compression with Text-guided Encoding for both Pixel-level and Perceptual Fidelity (2024, arXiv)

- **Evidence status:** preprint
- **Core idea:** Use text-adaptive image encoding and joint image-text training while avoiding a text-conditioned generative decoder.
- **Datasets:** MS-COCO 30k and additional paper datasets; verify exact splits
- **Metrics:** PSNR; MS-SSIM; LPIPS; bitrate
- **Reported result:** Author reports the best LPIPS among compared baselines while retaining competitive pixel fidelity.
- **Difference / caveat:** Human/machine captions may be side information; reproduce with full caption-rate accounting and exact caption-generation policy.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2403.02944
- **Research priority:** P1

### SELIC: Semantic-Enhanced Learned Image Compression via High-Level Textual Guidance (2025, IEEE ICME)

- **Evidence status:** peer_reviewed
- **Core idea:** Use frozen BLIP and BERT branches to derive semantic features and fuse them with image latents before quantization and entropy coding.
- **Datasets:** CLIC; LIU4K; Kodak; Tecnick
- **Metrics:** PSNR; MS-SSIM; BD-rate; encode/decode time; parameters
- **Reported result:** Author reports about 0.1–0.15 dB gain over its non-semantic baseline and -4.9% BD-rate versus VVC on Kodak.
- **Difference / caveat:** Encoder invokes image-to-text/text encoders; verify whether all semantic influence is represented in the serialized latent and reproduce timing/rate under a matched codec protocol.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2504.01279
- **Research priority:** P1

## Image Coding for Machines

### Recognition-Aware Learned Image Compression (2022, arXiv)

- **Evidence status:** preprint
- **Core idea:** Joint rate-distortion and recognition objective.
- **Datasets:** classification datasets
- **Metrics:** classification accuracy; bpp; distortion
- **Reported result:** Reports large low-rate classification gains over traditional codecs in its setup.
- **Difference / caveat:** Task-specific and classification-centric.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2202.00198
- **Research priority:** P2

### TransTIC: Transferring Transformer-based Image Compression from Human Perception to Machine Perception (2023, ICCV)

- **Evidence status:** peer_reviewed
- **Core idea:** Instance-specific encoder prompts and task-specific decoder prompts adapt a frozen Transformer codec.
- **Datasets:** multiple machine vision tasks
- **Metrics:** task accuracy; bpp; reconstruction
- **Reported result:** Strong transfer without full codec fine-tuning.
- **Difference / caveat:** Prompt/side information rate and task-model dependence require care.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/ICCV2023/html/Chen_TransTIC_Transferring_Transformer-based_Image_Compression_from_Human_Perception_to_Machine_ICCV_2023_paper.html
- **Research priority:** P1

### Rate-Distortion-Cognition Controllable Versatile Neural Image Compression (2024, ECCV)

- **Evidence status:** peer_reviewed
- **Core idea:** Single model controls rate, reconstruction distortion and cognition with scalable residual branch.
- **Datasets:** classification; detection; segmentation benchmarks
- **Metrics:** rate; distortion; task accuracy
- **Reported result:** Flexible Pareto control across machine tasks.
- **Difference / caveat:** OCR sequence integrity is not explicitly modeled.
- **Implementation:** paper_reports_public_verify; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2407.11700
- **Research priority:** P1

### Bridging Compressed Image Latents and Multimodal Large Language Models (2025, ICLR)

- **Evidence status:** peer_reviewed
- **Core idea:** Lightweight transform neck and surrogate visual-encoder loss connect codec latents to MLLMs.
- **Datasets:** MLLM understanding benchmarks
- **Metrics:** MLLM task scores; bpp; reconstruction
- **Reported result:** Avoids full MLLM in codec training loop.
- **Difference / caveat:** Paper URL and exact model versions should be recorded from repo before citation.
- **Implementation:** official; https://github.com/NYCU-MAPL/BridgingCompressionMLLM
- **Primary source:** https://github.com/NYCU-MAPL/BridgingCompressionMLLM
- **Research priority:** P1

### Deep Image Coding for Machines with Semantic-Guided Self-Supervision (2025, verify)

- **Evidence status:** publication_status_verify
- **Core idea:** Compressed-domain multi-task representation with semantic-guided self-supervision.
- **Datasets:** classification; detection and related tasks
- **Metrics:** task performance; bpp
- **Reported result:** Designed for machine consumption without full RGB reconstruction.
- **Difference / caveat:** Verify definitive title/venue from the pinned repository.
- **Implementation:** official_or_author; https://github.com/Tongji-MIC-Lab/DICM
- **Primary source:** https://github.com/Tongji-MIC-Lab/DICM
- **Research priority:** P2

### Test-Time Fine-Tuning of Image Compression Models for Multi-Task Adaptability (2025, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Instance-specific SVD-LoRA adaptation with singular-value side information.
- **Datasets:** closed/open-set machine tasks
- **Metrics:** task performance; bpp; side overhead
- **Reported result:** Improves adaptability to unseen machine tasks.
- **Difference / caveat:** Test-time compute and adapter bitrate must be counted.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2025/html/Park_Test-Time_Fine-Tuning_of_Image_Compression_Models_for_Multi-Task_Adaptability_CVPR_2025_paper.html
- **Research priority:** P2

### Towards Unified Human Perception and Machine Understanding: Token Flow Guided Compression Framework (2026, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Variable-length 1D visual tokens, token-flow propagation and direct LVLM consumption.
- **Datasets:** LVLM understanding and reconstruction benchmarks
- **Metrics:** LVLM task scores; perceptual quality; bpp
- **Reported result:** Reports strong LVLM understanding at comparable rates.
- **Difference / caveat:** OCR exact-sequence preservation remains a distinct requirement.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Towards_Unified_Human_Perception_and_Machine_Understanding_Token_Flow_Guided_CVPR_2026_paper.html
- **Research priority:** P1

## Standards for Human/Machine Coding

### JPEG AI — ISO/IEC 6048-1:2025 / ITU-T T.840.1 Core Coding System (2025, International Standard)

- **Evidence status:** standard
- **Core idea:** First end-to-end learning-based international image coding standard; compact single stream, progressive, ROI and machine-consumption scope.
- **Datasets:** JPEG AI evaluation datasets
- **Metrics:** subjective quality; objective metrics; machine tasks; conformance
- **Reported result:** Official overview reports about 30% rate reduction at equal subjective quality relative to conventional standards.
- **Difference / caveat:** Reference software, profile and conformance version must be pinned; standards/patent terms differ from code license.
- **Implementation:** reference_software; https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software
- **Primary source:** https://jpeg.org/jpegai/
- **Research priority:** P0

## OCR-Aware and Text-Preserving Compression

### Deep Image Compression Using Scene Text Quality Assessment (2023, Pattern Recognition)

- **Evidence status:** peer_reviewed
- **Core idea:** Scene-text image quality assessor guides iterative variable-rate compression.
- **Datasets:** scene text quality and compression datasets
- **Metrics:** quality prediction MAE/correlation; subjective/objective text quality
- **Reported result:** Reports better text quality and quality-assessment accuracy than compared regressors.
- **Difference / caveat:** Iterative, assessor and codec are not a unified end-to-end utility allocation system.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2305.11373
- **Research priority:** P1

### Learned Image Compression with Text Quality Enhancement (2024, ICIP)

- **Evidence status:** peer_reviewed
- **Core idea:** Text logit loss aligns OCR outputs between original and reconstruction.
- **Datasets:** two screenshot datasets
- **Metrics:** CER; WER; BD-rate; image quality
- **Reported result:** Reports average BD-rate -32.64% at fixed CER and -28.03% at fixed WER.
- **Difference / caveat:** Single OCR dependence, screenshot focus and cross-OCR transfer need evaluation.
- **Implementation:** official_code_not_found; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2402.08643
- **Research priority:** P0

### TFIC: End-to-End Text-Focused Image Compression for Coding for Machines (2025, arXiv)

- **Evidence status:** preprint
- **Core idea:** Frozen OCR supplies sequence supervision to a Transformer codec.
- **Datasets:** approximately 20k synthetic train and small synthetic test reported
- **Metrics:** OCR accuracy; bpp; encoder time
- **Reported result:** Reports very large OCR gains in selected very-low-rate settings and encoder faster than OCR.
- **Difference / caveat:** Small synthetic evaluation, single OCR and no strong cross-OCR evidence.
- **Implementation:** official_code_not_found; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2503.19495
- **Research priority:** P0

## Robustness and Security

### MALICE: Manipulation Attacks on Learned Image Compression (2022, arXiv)

- **Evidence status:** preprint_or_venue_verify
- **Core idea:** Adversarial inputs induce bitrate or reconstruction failures.
- **Datasets:** multiple LIC models and image sets
- **Metrics:** bitrate inflation; distortion; attack success
- **Reported result:** Reports severe white-box and measurable black-box bitrate amplification.
- **Difference / caveat:** Threat model and codec versions must be matched.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2205.13253
- **Research priority:** P1

### Backdoor Attacks Against Deep Image Compression via Adaptive Frequency Trigger (2023, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Frequency-adaptive trigger preserves clean performance while activating compression failure.
- **Datasets:** LIC benchmark datasets
- **Metrics:** clean RD; triggered failure; stealth
- **Reported result:** Demonstrates supply-chain/backdoor risk in learned codecs.
- **Difference / caveat:** Locate exact CVF paper URL before formal citation.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2023/
- **Research priority:** P2

### Robust Neural Image Compression (2023, verify)

- **Evidence status:** publication_status_verify
- **Core idea:** Adversarial training/robustification for neural image compression.
- **Datasets:** LIC adversarial benchmarks
- **Metrics:** clean RD; adversarial RD
- **Reported result:** Public implementation useful as defense baseline.
- **Difference / caveat:** Verify exact paper metadata from repository.
- **Implementation:** author_code; https://github.com/tongxyh/ImageCompression_Adversarial
- **Primary source:** https://github.com/tongxyh/ImageCompression_Adversarial
- **Research priority:** P1

### Adversarial Attacks on Learned Image Compression: Rate-Distortion Vulnerability Analysis (2024, arXiv)

- **Evidence status:** preprint
- **Core idea:** Separates rate and distortion attacks and diagnoses hyperprior/IGDN vulnerabilities.
- **Datasets:** multiple LIC models
- **Metrics:** rate attack; distortion attack; transferability
- **Reported result:** Shows broad vulnerability across architectures.
- **Difference / caveat:** Use actual bitstream rate, not likelihood only, when reproducing.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2405.07717
- **Research priority:** P1

### Towards Robust Learned Image Compression: Benchmarking Common Corruptions (2024, OpenReview)

- **Evidence status:** publication_status_verify
- **Core idea:** Corruption benchmark for LIC robustness.
- **Datasets:** CLIC-C; Kodak-C
- **Metrics:** RD under 15 corruptions; frequency diagnostics
- **Reported result:** Provides standardized corruption evaluation.
- **Difference / caveat:** Confirm final publication status/version.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openreview.net/forum?id=FxRfAIj4s2
- **Research priority:** P1

### Bitstream Collision Attacks on Learned Image Compression (2025, arXiv)

- **Evidence status:** preprint
- **Core idea:** Construct semantically distinct inputs that map to the same or colliding bitstream representation.
- **Datasets:** LIC models
- **Metrics:** collision success; perceptual/semantic divergence
- **Reported result:** Raises authenticity and semantic-integrity concerns.
- **Difference / caveat:** Preprint; inspect exact threat constraints.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2503.19817
- **Research priority:** P2

### Control Your View: High-Resolution Global Semantic Manipulation in Learned Image Compression (2026, arXiv)

- **Evidence status:** preprint
- **Core idea:** Studies high-resolution global semantic manipulation attacks against learned image compression systems.
- **Datasets:** LIC benchmark images and manipulated semantics
- **Metrics:** semantic manipulation success; RD impact
- **Reported result:** Shows LIC can be vulnerable not only to RD degradation but also high-level semantic manipulation after compression.
- **Difference / caveat:** Threat constraints and exact codec targets need verification; directly motivates semantic-integrity tests for text-bearing images.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2605.08727
- **Research priority:** P1

### On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors (2026, CVPR Workshop)

- **Evidence status:** peer_reviewed_workshop
- **Core idea:** Evaluate bit-flip robustness of RCC-style diffusion compressors and propose a more robust Turbo-DDCM variant.
- **Datasets:** Kodak; DIV2K
- **Metrics:** PSNR; LPIPS; FID; corrupted output rate
- **Reported result:** Authors report RCC-based diffusion compressors are substantially more robust to bit flips than classical and learned codecs.
- **Difference / caveat:** Focuses on diffusion/RCC representations; OSCAR `.oscr` text enhancement still needs direct bit-flip tests.
- **Implementation:** not_confirmed; https://openaccess.thecvf.com/content/CVPR2026W/AIGENS/html/Vaisman_On_the_Robustness_of_Diffusion-Based_Image_Compression_to_Bit-Flip_Errors_CVPRW_2026_paper.html
- **Primary source:** https://arxiv.org/abs/2604.05743
- **Research priority:** P1

## RAW / ISP-Aware Compression

### Invertible Image Signal Processing (2021, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Invertible RAW↔sRGB ISP with differentiable JPEG simulation.
- **Datasets:** paired RAW-RGB camera datasets
- **Metrics:** RAW reconstruction; RGB quality; JPEG robustness
- **Reported result:** Supports RAW recovery from rendered/compressed RGB.
- **Difference / caveat:** Not a pure learned bitstream codec; camera-specific data matter.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2021/html/Xing_Invertible_Image_Signal_Processing_CVPR_2021_paper.html
- **Research priority:** P2

### RAWtoBit: A Fully End-to-End Camera ISP and Image Compression Framework (2022, ECCV)

- **Evidence status:** peer_reviewed
- **Core idea:** Jointly optimize RAW-to-RGB ISP and compression with dual-teacher distillation.
- **Datasets:** RAW-RGB datasets
- **Metrics:** RGB quality; bpp; ISP quality
- **Reported result:** Improves over separately optimized ISP and codec pipelines.
- **Difference / caveat:** Sensor generalization and data pairing are major limitations.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2208.07639
- **Research priority:** P1

### Raw Image Reconstruction with Learned Compact Metadata (2023, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Transmit compact learned metadata alongside sRGB for RAW recovery.
- **Datasets:** RAW-RGB datasets
- **Metrics:** RAW reconstruction; metadata rate; RGB quality
- **Reported result:** Scalable representation for display RGB and recoverable RAW.
- **Difference / caveat:** All metadata rate must be included.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Raw_Image_Reconstruction_With_Learned_Compact_Metadata_CVPR_2023_paper.html
- **Research priority:** P1

### CoDISP: Exploring Compressed-Domain Camera ISP with RGB-Guided Encoder (2024, CVPR Workshop)

- **Evidence status:** peer_reviewed_workshop
- **Core idea:** Perform ISP operations in compressed domain with shared latent.
- **Datasets:** RAW/RGB ISP tasks
- **Metrics:** ISP task quality; memory; compression
- **Reported result:** Targets mobile memory and integrated ISP processing.
- **Difference / caveat:** Workshop result; evaluate against current full pipelines.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2024W/MAI/html/Zhang_CoDISP_Exploring_Compressed_Domain_Camera_ISP_with_RGB-guided_Encoder_CVPRW_2024_paper.html
- **Research priority:** P2

### ParamISP: Learned Forward and Inverse ISPs using Camera Parameters (2024, CVPR)

- **Evidence status:** peer_reviewed
- **Core idea:** Condition ISP on ISO, exposure, aperture and focal parameters.
- **Datasets:** multi-camera RAW-RGB
- **Metrics:** forward/inverse ISP quality
- **Reported result:** Improves sensor/capture adaptation.
- **Difference / caveat:** Metadata availability and normalization vary by camera.
- **Implementation:** verify; no verified public implementation
- **Primary source:** https://openaccess.thecvf.com/content/CVPR2024/html/Kim_ParamISP_Learned_Forward_and_Inverse_ISPs_using_Camera_Parameters_CVPR_2024_paper.html
- **Research priority:** P2

## RAW Coding for Machines

### ρ-Vision: Direct RAW Image Compression and Computer Vision (2024, IEEE TPAMI)

- **Evidence status:** peer_reviewed
- **Core idea:** Direct RAW compression and machine vision with unpaired RAW simulation.
- **Datasets:** MultiRAW and cross-camera sets
- **Metrics:** compression; detection; cross-sensor generalization
- **Reported result:** Includes MultiRAW dataset and code.
- **Difference / caveat:** License and camera-data terms must be checked.
- **Implementation:** official; https://github.com/NJUVISION/rho-vision
- **Primary source:** https://github.com/NJUVISION/rho-vision
- **Research priority:** P1

## Ocr Aware Semantic Side Channel

### TextBoost: Boosting Scene Text Fidelity in Ultra-low Bitrate Image Compression (2026, arXiv)

- **Evidence status:** preprint
- **Core idea:** Transmit filtered OCR strings/coordinates as auxiliary guidance, render a guidance map and fuse it in the decoder for small-text fidelity.
- **Datasets:** TextOCR; ICDAR 2015; Kodak
- **Metrics:** DET F1; E2E text spotting F1; PSNR; LPIPS; MS-SSIM; bpp
- **Reported result:** Author reports up to 60.6% higher text-recognition F1 at comparable PSNR/bpp.
- **Difference / caveat:** Track B comparison only unless every OCR string/geometry/filter/header byte plus encoder OCR cost and privacy risk is counted; do not compare directly to pure-image OSCAR Track A.
- **Implementation:** not_confirmed; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2603.04115
- **Research priority:** P0

## Survey

### Advances in Diffusion-Based Generative Compression (2026, arXiv)

- **Evidence status:** preprint
- **Core idea:** Unifying review of diffusion-based generative lossy compression and open challenges.
- **Datasets:** review
- **Metrics:** rate-distortion-perception; common randomness; diffusion compression taxonomy
- **Reported result:** No new codec result; useful as a conceptual map for diffusion compression baselines.
- **Difference / caveat:** Survey preprint; cite for taxonomy rather than performance claims.
- **Implementation:** not_applicable; no verified public implementation
- **Primary source:** https://arxiv.org/abs/2601.18932
- **Research priority:** P2

# Live Literature Check

**Date:** 2026-06-25  
**Purpose:** record a live web check for recent LIC, generative compression, VQ compression, and robustness/security leads beyond the bootstrap registry. These notes are not a final related-work section; use `paper_registry.csv` for structured status labels.

## Added To Registry

The following primary-source entries were added to `literature/paper_registry.csv`:

- `chen2025_cmic`: content-adaptive Mamba/SSM LIC, arXiv preprint, code not confirmed.
- `li2025_scaling1b`: LIC scaling-law study up to 1B parameters, arXiv preprint.
- `niu2025_hvqcgic`: VQ hyperprior entropy modeling for controllable generative compression, arXiv preprint.
- `yang2026_diffusion_review`: diffusion-based generative compression survey, arXiv preprint.
- `relic2026_regionadaptive`: region-adaptive spatially varying diffusion codec, arXiv preprint.
- `jia2026_codlite`: real-time one-step diffusion codec, arXiv preprint with announced official code.
- `vaisman2026_bitflip_diffusion`: CVPRW 2026 bit-flip robustness study for diffusion-based compression.

## Technical Takeaways For OSCAR-LIC

1. **Scaling is not the immediate route.** The 1B-parameter LIC scaling work is useful as a frontier signal, but OSCAR-LIC's current bottleneck is sparse text-sensitive assignment precision, not global RD capacity.
2. **Content-adaptive sequence/graph priors are converging.** CMIC and GLIC both argue that static Euclidean neighborhoods are suboptimal. This supports future candidate-conditioned local neighborhoods, but not a large backbone change before the OSCAR text stream is stable.
3. **VQ entropy modeling is becoming more mature.** HVQ-CGIC/HyperVQ-style VQ hyperpriors are directly relevant to OSCAR's compact code-index payloads. The immediate lesson is to make code assignment entropy/utility aware, not only residual-MSE nearest-center assignment.
4. **Generative compression is moving toward practical one-step decoders.** CoD-Lite and related one-step diffusion codecs weaken the old latency objection, but do not solve exact text fidelity or hallucination. For OSCAR-LIC, these remain optional local refiner ideas after deterministic OCR preservation works.
5. **Region-adaptive generative codecs overlap with rate allocation.** Region-adaptive diffusion formalizes spatially varying importance maps. OSCAR should keep its distinct claim: marginal OCR utility per actual bit, not saliency/ROI importance alone.
6. **Bitstream robustness is becoming a first-class topic.** The diffusion bit-flip study and bitstream collision line both reinforce that OSCAR's optional `TEXT_GATE` / `TEXT_MAIN` sections need direct corruption, collision, and graceful-fallback tests.

## Immediate Research Implication

The literature still supports a conservative first-paper path:

- keep corrected MLIC++ as the frozen base;
- keep actual `.oscr` byte accounting as mandatory for promotion;
- improve candidate-conditioned code assignment and no-op precision before adding a heavy generative decoder;
- treat generative/VQ/diffusion work as comparison baselines and future optional extensions;
- add robustness tests once clean OCR gains exceed the promoted pooled image-diff gate.

## Primary URLs Checked

- CMIC: https://arxiv.org/abs/2508.02192
- LIC scaling to 1B: https://arxiv.org/abs/2508.09075
- HVQ-CGIC / HyperVQ: https://arxiv.org/abs/2512.07192
- Diffusion generative compression review: https://arxiv.org/abs/2601.18932
- Region-adaptive generative compression: https://arxiv.org/abs/2604.01122
- CoD-Lite: https://arxiv.org/abs/2604.12525 and https://github.com/microsoft/GenCodec/CoD_Lite
- Diffusion bit-flip robustness: https://arxiv.org/abs/2604.05743 and https://openaccess.thecvf.com/content/CVPR2026W/AIGENS/html/Vaisman_On_the_Robustness_of_Diffusion-Based_Image_Compression_to_Bit-Flip_Errors_CVPRW_2026_paper.html

## Additional Live Checks On 2026-06-26

These are fresh positioning notes from primary paper/project pages. They should be moved into `paper_registry.csv` only after status/license/code availability is checked carefully.

- ARCHE: autoregressive residual compression with hyperprior/excitation. Source: https://arxiv.org/html/2603.10188
- HiDE: hierarchical dictionary entropy modeling. Source: https://arxiv.org/html/2603.06766v1
- EF-LIC: entropy-coding-free LIC. Source: https://arxiv.org/html/2605.23323v1
- FlowCodec: one-step flow prior for ultra-low-bitrate LIC. Source: https://arxiv.org/html/2606.21030v1
- DiffO: one-step diffusion-oriented generative compression with VQ residual ideas and rate-adaptive noise. Source: https://arxiv.org/html/2506.16572v1
- DLF: dual-generative latent fusion for extreme image compression. Source: https://openaccess.thecvf.com/content/ICCV2025/papers/Xue_DLF_Extreme_Image_Compression_with_Dual-generative_Latent_Fusion_ICCV_2025_paper.pdf
- CADC: content-adaptive diffusion-based generative image compression. Source: https://openaccess.thecvf.com/content/CVPR2026/papers/Sheng_CADC_Content_Adaptive_Diffusion-Based_Generative_Image_Compression_CVPR_2026_paper.pdf
- VLIC: vision-language models as perceptual judges for human-aligned image compression. Source: https://openaccess.thecvf.com/content/CVPR2026/papers/Sargent_VLIC_Vision-Language_Models_As_Perceptual_Judges_for_Human-Aligned_Image_Compression_CVPR_2026_paper.pdf
- Text-aware learned compression / TFIC-style direction: frozen OCR objectives for low-rate text fidelity. Source: https://arxiv.org/html/2503.19495v2
- PICD: perceptual image compression with diffusion rendering; important Track B comparison because text is encoded separately/losslessly. Source: https://arxiv.org/html/2505.05853v1
- Bitstream collision attacks on neural image compression. Source: https://arxiv.org/html/2503.19817v1

### Additional Technical Implications

1. Recent entropy-prior work reinforces that a strong base codec should remain competitive, but OSCAR-LIC's differentiator is not another global prior. It is actual-byte, plaintext-free text enhancement allocation.
2. Diffusion/flow/VQ generative codecs are rapidly improving low-bitrate perceptual quality, yet exact text preservation remains risky unless text is transmitted as side information. OSCAR-LIC should keep Track A separate from PICD-style semantic side-channel methods.
3. VLM-judged perceptual metrics may be valuable for qualitative figures, but they cannot replace CER/WER/exact text metrics for OCR preservation.
4. Security work around bitstream collisions and bit flips strengthens the need for `.oscr` CRC, optional-section recovery profiles, and adversarial/robustness tests before any security-aware claim.
5. For the current Eval300/Eval600 assignment bottleneck, the literature suggests keeping the model simple and local: improve code assignment and no-op calibration before importing diffusion or large transformer modules.

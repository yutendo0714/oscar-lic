# Live Literature Check

**Date:** 2026-06-26  
**Purpose:** record the fresh web check requested for current LIC, OCR-aware compression, generative low-bitrate compression, and security/robustness leads. This note is a planning artifact; structured status labels live in `literature/paper_registry.csv`.

## Registry Updates

The following entries were added or refined in `literature/paper_registry.csv`:

- `wang2026_textboost`: OCR-guided ultra-low-bitrate scene-text compression. Track B only unless every text/geometry/metadata byte and encoder OCR cost is counted.
- `cao2026_eflic`: entropy-coding-free LIC. Official inference repository found at `https://github.com/SevenCTHU/EF-LIC`; checkpoint/hash and actual index-file accounting still need a baseline smoke before use.
- `huang2026_flowcodec`: one-step flow prior for generative ultra-low-bitrate compression using large pretrained text-to-image priors and lightweight LoRA adaptation.
- `liang2026_controlview`: high-resolution global semantic manipulation attack against LIC systems.

## Technical Takeaways For OSCAR-LIC

1. **TextBoost is the closest new OCR-preservation comparator, but it is not a pure-image method.** It transmits OCR-derived text and geometry as guidance, so it belongs in Track B. OSCAR-LIC should cite it as strong evidence that text fidelity matters at ultra-low bitrate, while keeping the primary Track A claim distinct: no plaintext string, token, box, or prompt side-channel.
2. **EF-LIC reinforces that actual representation bytes, not entropy estimates, matter.** Removing entropy coding shifts the rate-accounting question to serialized vector/index syntax. Before using EF-LIC as a baseline, we need checkpoint SHA256, exact output file format, and complete-file bpp on our frozen smoke subset.
3. **FlowCodec moves generative compression toward simpler one-step adaptation.** Its latent-compression plus latent-transport split is attractive for optional perceptual refinement, but exact text preservation remains risky. For OSCAR, it is a low-bitrate generative baseline/extension candidate, not a replacement for the deterministic utility stream.
4. **Semantic manipulation attacks widen the robustness scope.** `Control Your View` suggests learned codecs can be attacked at the semantic level, not only through PSNR/rate degradation. OSCAR's robustness plan should include text-specific semantic integrity tests: names, amounts, IDs, and OCR strings under input perturbation and bitstream corruption.
5. **Current assignment work should not pivot to a larger global backbone.** The newest papers support richer priors in general, but N123-N125 show the local bottleneck is no-headroom verification and clean hard-positive supervision. The immediate OSCAR path remains simple, policy-level, byte-counted candidate selection.

## Primary Sources Checked

- TextBoost arXiv: https://arxiv.org/abs/2603.04115
- EF-LIC arXiv: https://arxiv.org/abs/2605.23323
- EF-LIC official inference repository: https://github.com/SevenCTHU/EF-LIC
- FlowCodec arXiv: https://arxiv.org/abs/2606.21030
- Control Your View arXiv: https://arxiv.org/abs/2605.08727
- Lai et al. text-logit loss baseline: https://arxiv.org/abs/2402.08643

## OSCAR-LIC Action Items

- Do not treat TextBoost side-channel results as Track A evidence; add a future Track B baseline row only with byte-counted OCR text/geometry overhead.
- Keep EF-LIC as a P2 practical baseline candidate; first action is checkpoint retrieval, SHA256 recording, license review, and complete-file bpp smoke if we decide to use it.
- Add semantic-manipulation and text-integrity attack cases to the robustness backlog after clean Eval300 assignment gains beat the current actual `.oscr` floor.
- Continue next research work on a materially different verifier objective or additional clean supervision, not another score/risk threshold sweep over N117/N120 features.

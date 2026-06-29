# Paper Outline

## Working title

**OSCAR-LIC: Actual-Byte Text Utility Enhancement for Frozen-Base Learned Image Compression**

## Abstract claim template

Existing text-aware codecs either optimize a fixed recognizer or uniformly prioritize detected text. OSCAR-LIC keeps a strong learned image codec frozen and adds a plaintext-free optional enhancement stream whose transmitted symbols are selected and assigned by counterfactual OCR utility under complete byte accounting. The current evidence supports a compact frozen-base path with exact base-only decode and actual-bitstream OCR gains, while the open technical problem is a candidate-conditioned group/no-op assignment head that captures more oracle code-effect headroom without false changes.

Do not insert performance numbers until the result registry marks them valid.

## Sections

1. Introduction: exact text semantics as sparse high-value information
2. Related work: LIC, ICM, OCR-aware, generative screen coding
3. Problem definition: pure-image versus semantic side-channel tracks
4. Method:
   - frozen MLIC++ base adapter and `.oscr` optional sections
   - counterfactual utility labels for candidate selection and code assignment
   - compact codebook/index payloads with complete-byte accounting
   - group/no-op calibrated assignment head
5. Experimental protocol:
   - cross-OCR, cross-domain, actual bitstreams
   - promotion requires compress/decompress, not likelihood bpp
6. Results:
   - corrected MLIC++/CompressAI actual-bitstream baselines
   - compact codebook selected-candidate results
   - learned assignment at unchanged payload size
   - utility/latent/window negative controls
   - generalization, complexity, robustness
7. Limitations and privacy
8. Conclusion

## Main figures

- `.oscr` stream layout and base-only decode
- selected latent candidates and assignment choices
- actual bpp/OCR trade-off curves
- false-change/missed-oracle policy-error chart
- small-text success/failure gallery
- rate breakdown and latency

## Main novelty sentence

The novelty is not “compression plus OCR loss.” It is a frozen-base, fully counted optional visual stream that learns **state-dependent text utility and code assignment under actual transmitted bytes**, while preserving base decoder compatibility and guarding against recognizer-specific false changes.

## Current Claim Boundary

The strongest current result is not yet a full cross-domain codec superiority claim. It is:

- a corrected MLIC++ actual-byte baseline and `.oscr` adapter;
- compact text-sensitive enhancement streams that can preserve PARSeq and improve Tesseract;
- a learned code-index assignment policy that improves Tesseract at unchanged actual payload size;
- a clear assignment oracle gap and negative evidence showing that flat latent vectors, standalone window CNNs and scalar no-op penalties are insufficient.

The next paper-critical result should show that the group/no-op assignment model beats the current pooled image-diff actual-bitstream baseline, not merely that it improves AUC/AP.

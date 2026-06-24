# Paper Outline

## Working title

**OSCAR-LIC: OCR-Invariant Scalable Learned Image Compression via Counterfactual Utility-Guided Bit Allocation**

## Abstract claim template

Existing text-aware codecs optimize a fixed OCR or uniformly prioritize detected text. OSCAR-LIC estimates the marginal recognition utility of transmitted latent units and builds a plaintext-free scalable enhancement stream. It is trained with multiple OCR teachers but evaluated on held-out recognizers and domains under complete bit accounting.

Do not insert performance numbers until the result registry marks them valid.

## Sections

1. Introduction: exact text semantics as sparse high-value information
2. Related work: LIC, ICM, OCR-aware, generative screen coding
3. Problem definition: pure-image versus semantic side-channel tracks
4. Method:
   - multi-teacher invariant representation
   - counterfactual utility labels/predictor
   - budget allocation and entropy-coded enhancement stream
5. Experimental protocol:
   - cross-OCR, cross-domain, actual bitstreams
6. Results:
   - main RD-OCR curves
   - utility ablations
   - generalization, complexity, robustness
7. Limitations and privacy
8. Conclusion

## Main figures

- architecture and stream layout
- utility map versus text mask and selected units
- BD-Rate@CER curves for held-out OCR
- small-text failure/success gallery
- rate breakdown and latency

## Main novelty sentence

The novelty is not “compression plus OCR loss.” It is the explicit learning and coding of **state-dependent marginal OCR value per actual transmitted bit**, combined with recognizer-invariant supervision and a fully counted scalable visual stream.

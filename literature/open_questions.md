# Open Research Questions

## Highest-value questions for the first paper

1. Does an oracle based on `Δ OCR loss / Δ actual bits` beat an equally budgeted text mask once gate and index bytes are counted?
2. How state-dependent is candidate utility? Does the ranking remain stable as packets accumulate, or do strong interactions invalidate a static predictor?
3. Which teacher representation transfers best: token logits, visual encoder features, CTC alignments, or stroke topology?
4. Can the deployment predictor avoid a full OCR call while retaining most oracle regret advantage?
5. Is the gain concentrated in tiny text and numeric fields, or broadly distributed?
6. Does a learned enhancement stream preserve evidence for unseen OCR systems, or merely create ensemble-specific preprocessing?
7. At what side-rate fraction does text benefit saturate, and what happens to non-text quality?

## Methodological questions

- Should utility optimize mean teacher loss, worst-teacher loss, or a risk-sensitive mixture?
- Is utility better predicted per spatial tile, channel group, bit-plane, VQ token, or packet?
- Can gate syntax be inferred from the base latent sufficiently to reduce explicit side bits?
- How should marginal benefit be estimated when arithmetic-coder byte lengths are discontinuous?
- Does a Shapley approximation produce meaningfully better labels than greedy trajectories for its cost?
- Can topology targets reduce the need for generative refinement?

## Safety and deployment questions

- Which packet errors produce silent numeric substitutions rather than visible corruption?
- Can text-focused latents leak private text more readily than ordinary image latents?
- How robust is a compressed-domain OCR adapter to downstream model upgrades?
- What decoder behavior is safest when enhancement CRC fails: base fallback, abstention, or uncertainty flag?

## Scope-control rule

An interesting question is not automatically part of the first paper. The first paper keeps only experiments necessary to establish cross-OCR utility-guided scalable coding. RAW, full privacy training, MLLM integration and generative refinement remain extensions unless the core method is already complete.

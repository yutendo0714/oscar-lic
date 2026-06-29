# Claim–Evidence Matrix

The machine-readable source is `experiments/claim_evidence.yaml`.

| Claim | Required comparison | Primary evidence | Claim must be weakened/rejected when |
|---|---|---|---|
| Utility beats masks | U010–U013 | held-out OCR at matched complete bytes | oracle itself fails strong heuristic controls |
| Predictor learns marginal value | U100–U101 | regret plus downstream OCR | rank correlation does not translate to OCR |
| Multi-teacher is model-invariant | O010–O021 | unseen/worst OCR | gains are teacher-only |
| Stream is scalable | S001–S020 | independent base, progressive rate/OCR | side rate or harmful packets dominate |
| Cross-domain superiority | X010–X020 | BD-Rate@CER/WER, visual and compute controls | fewer than two domains meet threshold |
| Robust text layer | R100–R110 | ΔCER at matched FEC bytes | uniform FEC is not worse |

A module diagram, attractive reconstruction or lower training loss is not evidence for these claims by itself.

## Current Evidence Status

As of the Eval300 assignment experiments on 2026-06-25, the strongest defensible claim is narrower than the full paper target:

| Claim area | Current status | Evidence | Do not claim yet |
|---|---|---|---|
| Actual-byte optional stream | Supported for the current prototype | MLIC++ base-only `.oscr` decode works; counted `TEXT_GATE`/`TEXT_MAIN` streams are evaluated with complete-file bpp | broad scalability across budgets/domains |
| Compact text-sensitive enhancement | Supported as a conditional-go direction | learned selection plus K16/K64 codebook streams improve Tesseract while preserving PARSeq on Eval300 held-out val splits | final codec superiority or full RD curve |
| Learned assignment at fixed payload size | Supported but small | pooled image-diff group gate keeps actual bpp unchanged and improves Tesseract by `-8` unicode / `-8` latin edits with PARSeq unchanged | closing most top-8 assignment oracle headroom |
| Candidate-conditioned utility learning | Partially supported | code-only tensor reranker is a weak positive substrate, but all flat/branch/window latent variants remain below the promoted actual-bitstream gate | that high AUC/AP implies useful OCR policy |
| Multi-teacher invariance | Partially supported | PARSeq-safe targets and held-out Tesseract comparisons prevent the worst teacher-only failure modes seen earlier | robust invariance across OCR families/languages |
| Robustness/security | Not yet supported beyond design | bitstream format and base-only decode make tests feasible | robustness to bit flips, adversarial inputs or privacy attacks |

Promotion evidence must remain actual-bitstream first. Any new assignment or selector model should be considered exploratory until it is exported to explicit `.oscr`, decoded, and compared with PARSeq/Tesseract row-wise against nearest and the current pooled image-diff baseline.

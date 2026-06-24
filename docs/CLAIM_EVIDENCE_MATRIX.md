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

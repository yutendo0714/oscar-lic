# Architecture Decision Records

## ADR-001: Primary comparison track is pure visual coding

**Status:** accepted  
**Decision:** Main results prohibit lossless text strings and OCR tokens in the bitstream.  
**Rationale:** Semantic side-channels solve a different problem and create rate/privacy confounds.  
**Consequence:** PICD is compared in a separate semantic-assisted track.

## ADR-002: MLIC++ is the first high-performance base codec

**Status:** accepted  
**Decision:** Start with MLIC++ official repository at the pinned snapshot in `baselines/registry.yaml`.  
**Rationale:** Public implementation, checkpoints, actual compression path and strong entropy model.  
**Alternative:** DCAE is retained as a second base due to dictionary prior compatibility.

## ADR-003: Separate environments for codec and OCR teachers

**Status:** accepted  
**Decision:** Do not force all dependencies into one environment. Cache OCR labels/features in versioned manifests where gradient flow is not needed; use a dedicated joint environment only for differentiable teacher experiments.  
**Rationale:** Legacy CompressAI/PyTorch and modern OCR stacks can conflict.

## ADR-004: Oracle feasibility precedes predictor development

**Status:** accepted  
**Decision:** No large utility predictor training before oracle allocation beats mask baselines under matched total rate.  
**Rationale:** A predictor cannot exceed an uninformative target.

## ADR-005: Actual bitstream is primary rate

**Status:** accepted  
**Decision:** Main tables use serialized total bytes. Estimated entropy is diagnostic only.  
**Rationale:** Gate, header, padding and coder overhead matter strongly at low bitrate.

## ADR-006: Held-out OCR is a primary metric, not appendix

**Status:** accepted  
**Decision:** At least one recognizer family and one OCR system not used for training must appear in the main table.  
**Rationale:** Single-model gains can be adversarial preprocessing rather than information preservation.

## ADR template

```text
## ADR-###: Title
Status: proposed | accepted | superseded | rejected
Context:
Decision:
Alternatives:
Consequences:
Evidence:
Date:
```

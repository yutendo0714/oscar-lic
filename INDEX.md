# Repository Index

This file is a compact map. The authoritative agent contract is `AGENTS.md`.

## Start here

1. `AGENTS.md` — mandatory operating rules, tracks and gates
2. `PROJECT_SPEC.md` — falsifiable hypotheses, contribution boundary, success/No-Go criteria
3. `STATUS.md` / `STATE.yaml` — current truth and next gate
4. `TASKS.md`, `BLOCKERS.md`, `DECISIONS.md`, `NEGATIVE_RESULTS.md` — research operations
5. `PROMPT_TO_AGENT.md` — minimal instruction to give an AI research agent
6. `docs/HANDOFF.md` — current 2026-06-25 OSCAR-LIC result, fixed assets and next move
7. `experiments/manifests/oscar_lic_12h_repro_manifest_2026_06_25.yaml` — 12h artifact/checkpoint/W&B/hash manifest

## Source-of-truth map

| Subject | Authoritative file(s) |
|---|---|
| research hypothesis and scope | `PROJECT_SPEC.md`, `experiments/hypotheses.yaml` |
| agent behavior and permissions | `AGENTS.md` |
| current state | `STATUS.md`, `STATE.yaml`, `docs/HANDOFF.md` |
| 12h research summary | `experiments/reports/oscar_lic_12h_research_summary_2026_06_25.md`, `experiments/manifests/oscar_lic_12h_repro_manifest_2026_06_25.yaml` |
| current promoted result | `docs/HANDOFF.md`, `experiments/reports/eval300_assignment_tensor_reranker_diagnostics_2026_06_25.md` |
| next model config | `configs/training/assignment_group_noop_window_v0.yaml` |
| experiment schedule/status | `experiments/EXPERIMENT_MATRIX.csv` |
| run provenance | `experiments/run_manifest.schema.json`, `scripts/make_experiment_manifest.py` |
| literature evidence | `literature/paper_registry.csv`, `literature/claims_to_sources.yaml`, `docs/LIC_RESEARCH_SURVEY_2026_06_25.md` |
| public-code pins | `literature/repositories.yaml`, `baselines/locks/` |
| baseline definitions | `baselines/registry.yaml`, `baselines/expected_results.yaml` |
| checkpoints | `baselines/checkpoint_manifest.yaml` |
| OCR train/evaluation separation | `baselines/ocr_models.yaml`, `configs/ocr/` |
| datasets, permissions and paths | `data/registry.yaml`, `data/manifests/`, `data/licenses/` |
| OCR normalization | `evaluation/OCR_NORMALIZATION.md` |
| rate, metric and statistical protocol | `evaluation/PROTOCOL.md`, `evaluation/METRICS.md`, `evaluation/statistical_analysis.md` |
| proposed architecture and losses | `docs/architecture.md`, `docs/losses.md`, `docs/utility_definition.md` |
| bitstream syntax | `docs/BITSTREAM_SPEC.md`, `src/oscarlic/bitstream.py` |
| threat/privacy model | `docs/THREAT_MODEL.md`, `docs/privacy_ethics.md` |
| paper claims and required evidence | `docs/CLAIM_EVIDENCE_MATRIX.md`, `experiments/claim_evidence.yaml` |
| packaging/integrity | `MANIFEST.sha256`, `RELEASE_INFO.json`, `scripts/build_release.py` |

## Executable bootstrap surface

The bootstrap package contains deterministic utilities for:

- repository/registry/schema validation;
- deterministic project-owned toy data and immutable JSONL manifests;
- Unicode-aware CER/WER/NED/exact-match;
- actual file-size bpp and section accounting;
- transparent no-extrapolation BD-rate;
- budgeted utility-allocation controls and a small exact/greedy oracle;
- experimental `.oscr` container packing, parsing and CRC validation;
- experiment provenance manifests;
- pinned external-repository checkout without automatically executing third-party code.

Neural codec and OCR training code is not fabricated before the selected upstream implementations pass reproduction gates. The expected adapters, tensors, losses and stages are fully specified under `docs/` and `configs/`.

## Current boundary

This workspace is now operational for continuing the corrected-MLIC++ frozen-base OSCAR-LIC assignment research. It still does not contain publication-approved dataset licenses, proprietary credentials or a full paper RD reproduction. Their unresolved hashes, licenses and hardware-dependent facts are explicit blockers rather than guessed values.

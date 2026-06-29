# Literature and Repository Search Log

## Snapshot: 2026-06-24

Primary sources checked include arXiv/OpenReview/CVF pages, the official JPEG AI site and official/author Git repositories. Current CVPR 2026 entries were verified against CVF Open Access where listed.

### Verified current items of special relevance

- JPEG AI Part 1 published as ISO/IEC 6048-1:2025 / ITU-T T.840.1.
- CVPR 2026: GLIC, block-based artifact-free LIC, practical learned codec study, RDVQ, CoD, CADC, AEIC, VLIC, TFGC.
- Priority branch pins recorded for CompressAI, MLIC, FTIC, DCAE, LALIC, MambaIC, PARSeq and RDVQ.

### Freshness rule

When this log is older than 30 days, search the following before claiming state of the art:

- `learned image compression 2026 2027 arxiv`
- `OCR-aware image compression`
- `image coding for machines OCR text`
- `CVPR ICCV ECCV ICLR NeurIPS image compression`
- repository releases/issues for all P0/P1 assets
- JPEG AI reference software tags and standard parts

Add new entries to `paper_registry.csv`; do not overwrite historical status without a dated note.

### Semantic text-guided compression update — 2026-06-24

Added verified primary-source entries for TGIC (AAAI 2023), TACO/text-guided encoding (arXiv 2024), LMM-driven image-text coding (VCIP 2024 poster; official repository pinned), and SELIC (ICME 2025). These are catalogued separately from OCR-preserving LIC because caption-level semantics do not guarantee exact text-string fidelity. A mask-guided text-aware LIC lead remains in `unverified_leads.md` pending an authoritative source.

## Live check: 2026-06-25

The user explicitly requested latest LIC and security/robustness coverage, so a live web check was performed despite the bootstrap snapshot being fresh. Primary arXiv/CVF/GitHub pages were checked for:

- content-adaptive SSM/Mamba LIC: CMIC (`2508.02192`);
- large-scale LIC: scaling HPCM-style models to 1B parameters (`2508.09075`);
- VQ/generative entropy modeling: HVQ-CGIC / HyperVQ (`2512.07192`);
- diffusion compression survey and practical one-step codecs: diffusion generative compression review (`2601.18932`), CoD-Lite (`2604.12525`);
- region-adaptive generative compression (`2604.01122`);
- bitstream/bit-flip robustness for diffusion-based compression (`2604.05743`, CVPRW 2026 AIGENS).

Structured entries were appended to `paper_registry.csv`, and the concise research implication note is `literature/live_literature_check_2026_06_25.md`.

## Live check: 2026-06-26

The user requested continued latest LIC/security coverage, so another live check was performed against primary arXiv/GitHub pages. The following new or newly verified items were recorded:

- TextBoost (`2603.04115`): OCR text/geometry guided ultra-low-bitrate scene-text compression. Important Track B comparator, not a pure-image Track A baseline.
- EF-LIC (`2605.23323`): entropy-coding-free learned image compression; official inference repository found at `https://github.com/SevenCTHU/EF-LIC`.
- FlowCodec (`2606.21030`): one-step flow prior for ultra-low-bitrate generative compression using large pretrained priors.
- Control Your View (`2605.08727`): high-resolution semantic manipulation attack against LIC systems.
- Lai et al. text-logit loss (`2402.08643`) was rechecked as the closest older OCR-loss baseline for context.

Structured entries were appended/refined in `paper_registry.csv`, and the concise research implication note is `literature/live_literature_check_2026_06_26.md`.

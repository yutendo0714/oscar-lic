# Current Status

**Last updated:** 2026-06-25  
**Stage:** Workspace foundation validated / external reproduction pending  
**Current gate:** G1 — Base codec reproduction

## Completed in this artifact

- Research question, hypotheses, contribution boundary and No-Go conditions defined.
- Literature snapshot assembled through 2026-06-24, with 58 structured records and explicit evidence status.
- Priority repositories and verified branch-head pins recorded.
- Dataset, license-risk, evaluation, OCR normalization, bitstream and experiment protocols created.
- Full experiment matrix and machine-readable run/report schemas created.
- Executable CER/WER/NED, actual-bpp, BD-rate, manifest, repository-bootstrap and `.oscr` CRC utilities included.
- Structural validation passed in the artifact environment.
- Unit test result: **26 passed**.
- Environment snapshot written to `environment/resolved_environment.json`.

## Completed in this workspace on 2026-06-25

- Copied the external `oscar_lic_package` skeleton into `/workspace/oscar-lic` as the active project tree.
- Preserved `/workspace/oscar-lic/oscar_lic_package` as an ignored external snapshot.
- Built `.venv` for bootstrap/analysis and `.venv-mlic` for MLIC++ reproduction.
- Verified CUDA visibility in `.venv-mlic` on NVIDIA GeForce RTX 4070 Ti SUPER.
- Installed PyTorch 2.0.1+cu118 and CompressAI 1.2.0b3 in the MLIC++ environment.
- Cloned pinned `compressai`, `mlic`, `parseq`, and `dcae` repositories under `external/`.
- Linked available `/dpl` datasets into `data/raw/` and generated local manifests for MLIC-Train-100K and CLIC.
- Created an offline wandb setup run: `wandb/offline-run-20260625_042120-n0ryq4yw`.
- Added foundation notes in `docs/FOUNDATION_SETUP_2026_06_25.md`, `docs/OSCAR_LIC_UNDERSTANDING.md`, `docs/NEXT_EXPERIMENT_PLAN.md`, and `literature/state_of_the_art_2026_06_25.md`.
- Downloaded corrected MLIC++ lambda 0.0250 checkpoint and recorded SHA256/size in `baselines/checkpoint_manifest.yaml`.
- Ran actual compress/decompress smoke on two CLIC validation images with complete stream-byte bpp accounting; see `experiments/reports/mlicpp_smoke_2026_06_25.md`.

## Truthfully unresolved

- External repositories are cloned at pinned commits but their training/evaluation code has not been executed.
- Dataset licenses/terms have not been approved for the eventual institution/user.
- Scientific benchmark datasets have not been downloaded and local approved split/checksum manifests do not yet exist; a five-image project-owned toy set is available for smoke tests only.
- Checkpoints have not been downloaded; therefore their local SHA256 and sizes remain unresolved.
- No MLIC++ or other codec result has been reproduced on target hardware yet.
- No OCR model bundle has been frozen locally.
- No OSCAR-LIC neural training has been run.
- PARSeq and DCAE checkpoint weights are not downloaded or hashed yet.

## Immediate objective

Resolve B01–B03 sufficiently to run `B001` and `B010`:

1. use the existing project-owned toy set for `B001` and separately approve scientific benchmark data;
2. run MLIC++ one operating point on a frozen validation subset;
3. reproduce CompressAI actual-bitstream smoke;
4. freeze the first internal regression point.

In parallel, lock one train OCR teacher and one held-out evaluator, then run `O002` only after checkpoint/data provenance is complete.

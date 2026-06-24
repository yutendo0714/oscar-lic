# Foundation Setup Log

**Date:** 2026-06-25 JST  
**wandb run:** `wandb/offline-run-20260625_042120-n0ryq4yw`  
**Primary goal:** make `/workspace/oscar-lic` usable as the active OSCAR-LIC research workspace while preserving `oscar_lic_package` as an external snapshot.

## Project Layout

The active project tree now lives directly under `/workspace/oscar-lic`:

- `src/oscarlic/`: lightweight research utilities for rate, bitstream containers, metrics, allocation, and registries.
- `configs/`: model, data, training, evaluation, robustness, and ablation configs.
- `scripts/`: validation, manifest, literature, environment, and metric utilities.
- `docs/`, `evaluation/`, `literature/`, `baselines/`, `experiments/`: research protocols and records.
- `external/`: pinned third-party repositories.
- `oscar_lic_package/`: preserved external snapshot; ignored by the active validator and `.gitignore`.

## Environments

### Bootstrap / analysis environment

Path: `.venv`

Purpose:

- repository validation;
- metrics;
- manifests;
- literature rendering;
- wandb setup logging.

Key pins:

- Python 3.10.12
- NumPy 2.2.6
- Pillow 12.2.0
- pytest 9.0.2
- wandb 0.28.0
- gdown 6.1.0

Lock:

- `environment/locks.bootstrap-py310.txt`
- SHA256: `f9fbb67fef3cbec08c6006e9f6cd11d907af59b089359c565802c9da4ff37ec8`

Note: the original package pinned NumPy 2.3.5, which requires Python >=3.11. For MLIC++ compatibility, the active bootstrap env uses Python 3.10 and NumPy 2.2.6.

### MLIC++ reproduction environment

Path: `.venv-mlic`

Purpose:

- MLIC++ import and checkpoint reproduction;
- CompressAI 1.2.0b3 actual bitstream experiments;
- later OSCAR-LIC base adapter work.

Key pins:

- Python 3.10.12
- PyTorch 2.0.1+cu118
- torchvision 0.15.2+cu118
- CompressAI 1.2.0b3
- NumPy 1.26.4
- Pillow 10.4.0
- wandb 0.28.0

Lock:

- `environment/locks.mlic-py310-cu118.txt`
- SHA256: `bef8407e437e4254aa1b2aed59834f274763bb2b223801556994c905af7c9e4f`

GPU check:

- `torch.cuda.is_available() == True`
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- Driver: 570.148.08
- CUDA runtime: 11.8 through PyTorch wheel

## External Repositories

Resolved lock: `baselines/locks/resolved_repositories.json`

- `compressai`: `81e018a2c68109d579020ba4bfa80378a366de28`
- `mlic`: `4aa9d2a512eb9db382172058e6c278fb0d376e17`
- `parseq`: `1902db043c029a7e03a3818c616c06600af574be`
- `dcae`: `e2525a00467cbc326045674c7e5e0f1d9964604b`

These repositories were cloned with blob filtering and checked out at fixed commits. No third-party training/evaluation code was executed during clone.

## Data

Local dataset links are recorded in `data/local_paths.yaml` and linked into `data/raw/`.

Available local sources include:

- `/dpl/mlic_train_100k`
- `/dpl/clic`
- `/dpl/openimages`
- `/dpl/veritext_raw/textocr`
- `/dpl/veritext_raw/icdar2013`
- `/dpl/veritext_raw/icdar2015`
- `/dpl/total_text`

Generated manifests:

- `data/manifests/mlic_train_100k_train512x512.local.jsonl`: 102274 images, existence-validated.
- `data/manifests/clic.local.jsonl`: 352 images, existence-validated.

These are local availability manifests, not final publication manifests. License status and split rules in `data/registry.yaml` remain authoritative.

## Validation Results

Commands passed:

- `.venv/bin/python scripts/validate_repo.py`
- `.venv/bin/python scripts/render_literature.py --check`
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python scripts/validate_dataset_manifest.py data/manifests/mlic_train_100k_train512x512.local.jsonl --root data/raw/mlic_train_100k/train512x512`
- `.venv/bin/python scripts/validate_dataset_manifest.py data/manifests/clic.local.jsonl --root data/raw/clic`

Unit tests:

- 26 passed.

Warnings:

- checkpoint hashes are intentionally unresolved for MLIC++, PARSeq, and DCAE until weights are downloaded and hashed.

## Next Required Foundation Work

1. Download one corrected MLIC++ checkpoint, preferably lambda 0.0250 first, and record URL, file size, SHA256, model class, and architecture variant.
2. Run a real MLIC++ compress/decompress smoke on toy or CLIC validation images and report actual complete-file bpp.
3. Freeze a PARSeq teacher checkpoint and at least one held-out OCR evaluator.
4. Implement a base codec adapter boundary rather than modifying MLIC++ internals broadly.
5. Add wandb logging hooks to actual training/evaluation scripts before the first learned run.

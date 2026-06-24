# Bootstrap Audit

**Date:** 2026-06-24  
**Scope:** repository structure and lightweight utilities only; no external model/data execution

## Results

- `python scripts/validate_repo.py`: passed
- structured YAML/JSON files parsed: 43 at validation time
- warnings: unresolved checkpoint hashes for MLIC++, PARSeq and DCAE, intentionally retained as blockers
- `python -m pytest -q`: 26 passed
- literature renderer check: 58 records in sync
- toy dataset manifest: 5 hashed records validated
- environment capture: written to `environment/resolved_environment.json`
- CUDA in validation container: unavailable

## What this audit proves

- required research documents and machine-readable registries exist;
- structured files parse;
- experiment IDs and registry IDs are unique under current validators;
- Unicode normalization and sequence metric regression cases pass;
- complete-file bpp arithmetic passes;
- the experimental `.oscr` header/table/CRC implementation round-trips and detects corruption;
- the linear no-extrapolation BD-rate helper passes a known 50% rate case.

## What this audit does not prove

- external code builds or matches its paper;
- any dataset may legally be used at the user's institution;
- checkpoints are authentic or safe;
- MLIC++ RD performance is reproduced;
- OCR results or proposed-method gains exist;
- the proposed utility hypothesis is correct.

Those items are deliberately guarded by G1–G8.

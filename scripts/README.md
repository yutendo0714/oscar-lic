# Scripts

| Script | Purpose |
|---|---|
| `validate_repo.py` | Required-file, YAML/JSON, registry and config-reference validation |
| `validate_registry.py` | Focused data/baseline/experiment registry checks |
| `verify_environment.py` | Captures platform, packages, CUDA and command availability |
| `bootstrap_external_repos.py` | Clones selected repositories at explicit pins; never runs them |
| `build_dataset_manifest.py` | Creates immutable path/size/hash JSONL manifests |
| `compute_text_metrics.py` | Computes versioned CER/WER/NED/exact metrics |
| `compute_bpp.py` | Computes actual bpp from complete file bytes |
| `compute_bd_rate.py` | Transparent no-extrapolation linear BD-rate utility |
| `inspect_oscr.py` | CRC-validates and inspects `.oscr` research containers |
| `make_experiment_manifest.py` | Captures config/environment/data provenance before a run |
| `render_literature.py` | Renders `catalog.md` and safe BibTeX stubs from the CSV registry |
| `next_task.py` | Selects the next unblocked experiment/task from project state |
| `update_experiment_status.py` | Updates experiment state without silently rewriting result evidence |
| `build_release.py` | Generates release metadata, SHA-256 manifest and deterministic ZIP |

The neural training/codec scripts are intentionally not faked in this bootstrap snapshot. They are implemented only after the selected external baseline passes G1, so interfaces can be adapted to the verified code rather than an assumed API.

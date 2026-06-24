# CompressAI Smoke Reproduction

## Frozen source

- Repository: `InterDigitalInc/CompressAI`
- Commit: `81e018a2c68109d579020ba4bfa80378a366de28`
- Snapshot package version in `pyproject.toml`: `1.2.9.dev0`
- License: BSD 3-Clause Clear style; repository license explicitly does not grant patent rights

This modern snapshot must not be installed into the legacy MLIC++ environment.

## Build

```bash
python scripts/bootstrap_external_repos.py --selected compressai
# create/activate environment/compressai_snapshot.yml after local CUDA choice
cd external/compressai
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest -q -m "not slow and not pretrained" tests
```

Record compiler and build logs because entropy-coder extensions are part of the result.

## Official interfaces

The frozen README provides:

```bash
python examples/codec.py --help
python -m compressai.utils.eval_model pretrained /path/to/images -a ARCH -q QUALITY...
python -m compressai.utils.eval_model checkpoint /path/to/images -a ARCH -p CHECKPOINT...
```

For `B001`, use the codec interface that physically writes a bitstream and retain the file. Verify whether `eval_model` reports entropy estimates or actual coding for the selected arguments; do not assume.

## Initial model

Use `bmshj2018-hyperprior`, MSE, quality 3 for the five-image smoke test. This is an operational smoke point, not a claim of SOTA or exact reproduction of the original TensorFlow paper.

## Required checks

- model zoo weight URL and local SHA256 recorded
- encode/decode round trip succeeds
- decoded shape equals original
- complete-file bytes counted
- model revision and quality index recorded
- CPU/GPU numerical differences measured if both are used
- result labeled framework reference, not original-author byte reproduction

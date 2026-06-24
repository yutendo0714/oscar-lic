# Reproducibility Requirements

## Artifact identity

Each result bundle contains:

```text
manifest.yaml
config_resolved.yaml
environment.lock
external_repositories.lock.json
dataset_manifest.sha256
checkpoint_manifest.yaml
metrics/*.json
logs/stdout.txt
logs/stderr.txt
bitstream_manifest.jsonl
```

Large bitstreams/checkpoints may be stored outside the repository, but their URI and SHA256 remain in the manifest.

## Randomness

Record Python, NumPy, PyTorch CPU/CUDA and data-loader seeds. Deterministic mode is used for regression tests; performance runs may use nondeterministic kernels only if documented and repeated.

## Floating point

Record precision (FP32/TF32/FP16/BF16), autocast, cudnn flags and compiler settings. Entropy coding probabilities must be decoder-compatible across encoder/decoder; avoid platform-dependent unquantized probability tables.

## Patches

External code changes live in `baselines/patches` or a fork commit. Never edit `external/` without recording diff and commit. A workaround that changes model output invalidates direct comparison unless applied consistently.

## Result invalidation

Invalidate and rerun when:

- a primary metric bug is found
- rate omitted a stream
- test data informed training/tuning
- wrong model/checkpoint was used
- decoder output does not correspond to stored bitstream

Keep invalid results with reason; do not silently replace.

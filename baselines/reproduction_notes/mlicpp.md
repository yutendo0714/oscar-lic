# MLIC++ Reproduction Note

## Frozen source

- Repository: `JiangWeibeta/MLIC`
- Commit: `4aa9d2a512eb9db382172058e6c278fb0d376e17`
- Environment stated by repository: CompressAI `1.2.0b3`, PyTorch `2.0.1`
- License recorded: Apache-2.0

## Critical architecture/checkpoint distinction

The official README states that the 2024 corrected checkpoint uses `LatentResidualPrediction` and `SynthesisTransform`, not the `Old` classes. Never infer compatibility from the filename. Record the chosen checkpoint SHA256 and a successful strict `state_dict` load.

## Official test path at the frozen commit

- launcher: `MLIC++/playground/test.sh`
- Python entry: `MLIC++/playground/test.py`
- coding implementation: `MLIC++/utils/testing.py`

The provided shell file contains machine-specific absolute paths. Recreate the command with local paths rather than executing it unchanged:

```bash
cd external/mlic/MLIC++/playground
export PYTHONPATH=..:${PYTHONPATH}
CUDA_VISIBLE_DEVICES=0 python test.py   -exp B010_mlicpp_smoke   --gpu_id 0   -c /approved/path/checkpoint.pth.tar   -d /approved/path/dataset_root
```

Before running, inspect `python test.py --help` and archive the output because argument semantics can be surprising.

## Verified behavior of the frozen test code

- Instantiates `MLICPlusPlus`.
- Loads `checkpoint['state_dict']` with strict default behavior.
- Uses `ImageFolder(args.dataset, split="tecnick")`; the dataset root must match that expected directory layout. This hard-coded split makes the script unsuitable for arbitrary folders without a transparent wrapper or patch.
- Pads right/bottom with zeros to a multiple of 64.
- Serializes original `H,W`, latent shape and entropy-coded strings into a file.
- Computes bpp from complete produced file size divided by original `H*W`.
- Crops decoded output back to original dimensions.
- Calls `update_resolutions(16,16)` before coding to avoid resolution leakage.
- Compresses the first image once as warm-up and again for the measured path.

## Reproduction hazards

1. **Dataset adapter:** do not relabel Kodak as `tecnick` without recording the wrapper. Prefer a project-side evaluation adapter that accepts an immutable image manifest while preserving the codec calls exactly.
2. **Checkpoint safety:** `torch.load` uses Python pickle. Download only from the registered author source, hash it before loading, and inspect in an isolated environment.
3. **Timing:** repository-reported `cost_time` may exclude Python I/O/container writing. Project timing must separately measure model, entropy coding and complete wall time.
4. **Rounded logs:** recompute per-image bpp from retained files, not rounded log strings.
5. **Padding:** denominator remains original dimensions; verify byte-identical decode and exact output shape.
6. **Resolution state:** omitting `update_resolutions` can leak resolution statistics or change results.
7. **Old/new code:** never mix old checkpoint classes with corrected architecture.

## B010 acceptance artifact

Store:

```text
experiments/B010/
  manifest.json
  command.txt
  environment.json
  checkpoint.sha256
  source_manifest.jsonl
  bitstreams/
  reconstructions/
  per_image.jsonl
  summary.json
  stdout.log
  reproduction_report.md
```

G1 is passed only after actual produced files, decoded outputs and immutable provenance exist. The published BD-rate table is not itself a regression target until the complete evaluation protocol matches.

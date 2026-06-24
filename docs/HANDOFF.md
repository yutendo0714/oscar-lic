# Research Handoff

## What an agent may do immediately

1. Read `AGENTS.md` in full.
2. Re-run structural tests.
3. Inspect registries and external licenses.
4. Execute dry-run repository bootstrap commands.
5. Create deterministic toy images and manifests.
6. Prepare, but not silently approve, dataset/checkpoint acquisition.

## What requires local facts

The directory cannot truthfully pre-fill:

- the institution's acceptance of dataset terms;
- credentials or private storage paths;
- target GPU memory and compute budget;
- bytes and hashes of checkpoints not yet downloaded;
- reproduced numerical baselines;
- author responses or unreleased code.

The agent must record these facts rather than infer them.

## First useful deliverable

The first scientific deliverable is not an OSCAR-LIC model. It is an immutable `B010` package containing:

- exact MLIC repository commit;
- environment lock and hardware record;
- corrected checkpoint SHA256;
- five source-image hashes;
- encoder/decoder command;
- actual compressed files;
- decoded images;
- per-image total bytes, PSNR and timing;
- a reproduction report with deviations.

Only after this package exists should differentiable OCR integration begin.

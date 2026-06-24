# OSCAR-LIC Understanding Note

**Date:** 2026-06-25  
**Scope:** interpretation of the copied `oscar_lic_package` research skeleton and the implementation strategy for this workspace.

## Core Problem

OSCAR-LIC targets pure visual learned image compression for images containing text. The key question is:

> Can we preserve OCR-relevant visual evidence under a strict total bit budget, without transmitting plaintext text, OCR tokens, or lossless box coordinates?

This matters because low-bitrate codecs often spend too few bits on tiny strokes, holes, diacritics, punctuation, and digit-like structures. These regions can occupy few pixels but dominate semantic correctness.

## Central Novelty To Protect

The proposal has three primary contribution axes:

1. **OCR-model-invariant objective:** train with multiple OCR teachers and teacher dropout, then evaluate on held-out OCR families.
2. **Counterfactual OCR utility per actual bit:** rank transmission candidates by measured or distilled `delta OCR loss / delta serialized bits`, not by text mask, saliency, confidence, or estimated entropy alone.
3. **Plaintext-free scalable text enhancement stream:** keep base image decoding independent, and add an optional sparse enhancement stream whose gate/index/payload/header/CRC bytes are all counted.

The paper should not become "a diffusion text renderer" or "OCR side-channel compression." Those are useful baselines/extensions, but Track A forbids sending recognized strings.

## Architecture Reading

The intended stream is:

- `B0`: independently decodable base image stream, initially MLIC++.
- `B1`: optional sparse enhancement stream for text-preserving residual evidence.

The reference path is:

1. Base codec encodes image into base latent/hyperlatent.
2. Utility features are derived from base/hyper features, cheap textness, uncertainty, and candidate-rate estimates.
3. Candidate units are initially `8x8` latent spatial tiles times channel groups of 8.
4. A budget allocator selects units under a target enhancement budget.
5. Gate/index syntax is serialized and counted.
6. Selected enhancement payload is entropy-coded conditionally on decoded base features.
7. Decoder fuses a bounded residual into the base reconstruction.

The cleanest first implementation is:

- frozen MLIC++ adapter;
- dense or oracle-masked continuous feature residual;
- exact gate/index serialization prototype;
- greedy counterfactual oracle on small crops;
- learned utility predictor only after oracle beats controls.

## Evaluation Invariants

The following are non-negotiable:

- Primary rate is `8 * complete_file_bytes / (original_H * original_W)`.
- Estimated likelihood bpp is diagnostic only.
- `B0` must decode when `B1` is absent.
- No Track A bitstream metadata may contain plaintext transcription, OCR tokens, or lossless boxes.
- Gate/index/header/padding/CRC/FEC/model-profile bytes are included.
- Training teachers and held-out OCR evaluators must be separated.
- Final claims require cross-domain text evaluation and non-text visual sanity checks.

## Why MLIC++ Is The Right First Base

MLIC++ is a strong base because it has:

- official code and public checkpoints;
- actual compress/decompress path;
- multi-reference entropy modeling with channel, local, and global references;
- a natural place to attach a conditional enhancement entropy model.

The risk is implementation fragility: MLIC++ has old/new architecture variants and corrected checkpoints. The workspace must pin commit, checkpoint URL, SHA256, model class, and exact actual-bitstream command before using results.

## Simplicity Bias

The first publishable path should be simple:

- continuous learned residual first, not VQ tokens;
- deterministic fusion first, not diffusion refiner;
- greedy oracle first, not a large predictor;
- one base codec first, then DCAE/JPEG AI as supporting baselines.

Complexity should be introduced only when it answers a measured bottleneck:

- if oracle beats masks but predictor is weak, improve utility features;
- if residual capacity is insufficient under oracle masks, improve enhancement representation;
- if actual index overhead dominates, redesign packet syntax;
- if held-out OCR transfer fails, revisit teacher invariance and topology losses.

## Early Go/No-Go Gates

The most important early gate is G4:

1. Freeze base codec and a small candidate layout.
2. Generate exact greedy oracle trajectories on representative text crops.
3. Compare oracle allocation against uniform text mask, detector confidence, uncertainty, residual energy, and random controls at matched total bytes.
4. Evaluate with held-out OCR.
5. Proceed to learned utility predictor only if oracle gains survive side-bit accounting.

If this gate fails, a larger model is unlikely to save the core hypothesis.

## Current Workspace Status

- `/workspace/oscar-lic/oscar_lic_package` remains an external snapshot.
- The project has been copied into `/workspace/oscar-lic` as the active research tree.
- Pinned external repositories are cloned under `external/`.
- Bootstrap and MLIC++ reproduction venvs are available.
- `/dpl` datasets are linked into `data/raw/` and lightweight local manifests exist.
- An offline wandb foundation run records the initial setup.


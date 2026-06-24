# Task Queue

## P0 — Bootstrap state

- [x] Run `python scripts/validate_repo.py`.
- [x] Run unit tests; 26 tests passed in the artifact environment.
- [x] Record bootstrap hardware/software with `scripts/verify_environment.py`.
- [x] Validate CER/WER normalization on hand-calculated examples.
- [x] Validate complete-file bpp and OSCAR container accounting with unit tests.
- [x] Repeat G0 validation in the target compute environment.
- [ ] Review `LICENSES.md` and approve only the datasets required for the first smoke run.
- [x] Clone P0 external repositories at registered commits.
- [ ] Review cloned external repository code/licenses before executing third-party experiments.
- [x] Download one corrected MLIC++ checkpoint; record source, size and SHA256.
- [x] Create and hash a deterministic project-owned five-image smoke set (`data/toy`).
- [x] Build bootstrap and MLIC++ reproduction venvs.
- [x] Verify CUDA visibility in MLIC++ reproduction venv.
- [x] Link available `/dpl` datasets into `data/raw` and generate local availability manifests.
- [x] Create an offline wandb foundation run.
- [x] Run MLIC++ corrected-checkpoint actual compress/decompress smoke on two CLIC images.

## P1 — Baseline foundation

- [ ] Reproduce CompressAI hyperprior actual-bitstream smoke (`B001`).
- [ ] Reproduce MLIC++ one operating point (`B010`) on a frozen validation subset.
- [ ] Freeze internal regression values and tolerances in `baselines/expected_results.yaml`.
- [ ] Lock PARSeq teacher checkpoint and preprocessing.
- [ ] Lock at least one held-out OCR evaluator and detector mode.
- [ ] Measure original-image OCR upper bound and detector failure rate (`O002`).
- [ ] Implement single-teacher OCR-loss baseline (`O010`).
- [ ] Implement multi-teacher no-utility baseline (`O020`).

## P2 — Center hypothesis

- [ ] Integrate exact candidate transmission unit with verified MLIC++ feature shapes.
- [ ] Implement greedy counterfactual oracle on small crops (`U001`).
- [ ] Benchmark oracle label cost on 100 crops and update B05.
- [ ] Compare oracle against uniform-text, detector-confidence, uncertainty and random controls.
- [ ] Estimate interaction error through sampled subsets / approximate Shapley analysis.
- [ ] Make and record the G4 Go/No-Go decision.

## P3 — Proposed model, only after G4

- [ ] Train lightweight utility predictor.
- [ ] Implement entropy-coded gate/index stream.
- [ ] Implement conditional text enhancement latent.
- [ ] Verify base-only independent decoding and packet truncation.
- [ ] Train at two budgets and test held-out OCR.

## P4 — Full evaluation

- [ ] Scene, screen and document domains with approved datasets.
- [ ] 4–6 overlapping operating points and 3 training seeds.
- [ ] Actual total-rate and per-section breakdown.
- [ ] text-size, script, numeric and domain stratification.
- [ ] complexity, memory and optional energy measurements.
- [ ] corruption, bit error and model-drift tests.
- [ ] image-level paired bootstrap confidence intervals.

## P5 — Optional extensions after core result

- [ ] One-step local glyph refiner.
- [ ] Unequal error protection.
- [ ] Compressed-domain OCR adapter.
- [ ] Privacy attacker.
- [ ] RAW/low-light text extension.

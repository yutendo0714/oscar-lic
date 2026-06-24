# International-Conference Experiment Plan

## Phase 0: Infrastructure

Deliverables:

- valid repository and environment manifests
- actual bitstream bpp tool
- OCR normalization unit tests
- 5-image codec/OCR smoke report

Gate: G0–G2.

## Phase 1: Reproduce and standardize baselines

Experiments:

- CompressAI hyperprior and joint AR
- MLIC++ public checkpoint
- JPEG AI VM if buildable
- original image OCR upper bounds
- classical codec operating points: JPEG, WebP/AVIF/JPEG XL/BPG/VTM/SCC as licensed/available

Output: rate curves with all commands, versions and actual bytes.

## Phase 2: OCR loss baselines

Same base, data and bpp grid:

- RD only
- text-weighted distortion
- single PARSeq loss
- single CRNN/CTC loss
- multi-teacher loss
- teacher dropout

Main question: does held-out OCR improve or only the optimized teacher?

## Phase 3: Oracle utility feasibility

Candidate set and budget held constant:

- random ranking
- detector confidence
- base uncertainty
- edge energy
- uniform text allocation
- exact greedy oracle
- approximate Shapley diagnostic

Run on 100–1000 validation images/crops before predictor training. If oracle does not win, stop and redefine.

## Phase 4: Learned utility and stream

Ablations:

- utility feature inputs
- ranking vs regression loss
- mean vs uncertainty-aware selection
- candidate tile/group size
- gate coding method
- text latent channel count
- base frozen vs adapters vs tail fine-tuning

## Phase 5: Generalization

Axes:

- held-out OCR architecture
- scene/screen/document
- font family
- text height bins: ≤8, 9–16, 17–32, >32 px
- Latin versus held-out scripts
- curved/vertical/perspective text
- unseen application/site templates

## Phase 6: Optional generative refiner

Compare no refiner, deterministic CNN refiner, one-step flow/diffusion. Record exact-match changes and seed/platform variance. Do not hide cases where perceptual quality rises but text changes.

## Phase 7: Robustness and privacy

- common corruptions
- input rate/distortion attacks
- random bit error rates 1e-6 to 1e-3
- bursts 8–256 bits
- packet loss 0.1–5%
- decoder model version mismatch
- bitstream-to-text privacy attacker

## Phase 8: Submission package

- main tables from immutable result JSON
- 3 seeds and 95% image-bootstrap CI
- per-image scatter and Pareto plots
- failure gallery selected by predefined criteria
- environment/container locks
- code/data instructions and license review

## Primary table design

Rows: codecs.  
Columns per domain and rate region:

- total bpp
- CER/WER/exact word accuracy for train and held-out OCR
- PSNR/MS-SSIM/LPIPS/DISTS
- encoder/decoder latency including entropy coder and utility predictor
- gate/text side bpp

Semantic-assisted methods appear in a separate table.

## Statistical plan

The image/document is the resampling unit. Use paired bootstrap for method differences. For multiple primary comparisons, control familywise error or declare a small preregistered set of primary hypotheses. Do not treat word crops from the same image as independent samples.

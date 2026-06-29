# Eval600 -> Eval300 Top-8 Assignment External Transfer

**Date:** 2026-06-26  
**Stage:** G5 diagnostic / compact-symbol assignment transfer  
**Track:** A, pure visual bitstream diagnostic  
**Conclusion:** valid negative result. Eval600-disjoint top-8 assignment labels improve row metrics on Eval300 seed1, but the learned policies do not transfer as safe OCR-preserving assignment policies.

## Purpose

The promoted compact OSCAR-LIC path still leaves large code-index assignment headroom at unchanged payload size. This run tests whether external Eval600 assignment labels can train the existing group/no-op latent-window assignment head and transfer to clean Eval300 seed1 validation groups.

This is not a promoted compression result. It is a pre-`.oscr` policy diagnostic over a fixed top-8 assignment artifact. Promotion still requires counted `.oscr` encode/decode and row-wise OCR.

## Inputs

- Merged feature artifact: `experiments/features/eval600_oraclesafe123_plus_eval300_seed1val_top8_windows_2026_06_25.npz`
- Feature SHA256: `6478fd87249e30afb5b6659b49a8341254bb0c34d5f7297ff4fa2849c5c2b9fa`
- Train groups: 123 Eval600 oracle-safe selected groups
- Validation groups: 75 Eval300 seed1 clean-val groups
- Validation oracle headroom: 17 changed groups, Tesseract unicode edit delta `-24`, PARSeq delta `0`, same code-index payload size

## Code

- `scripts/train_assignment_group_noop_window.py` SHA256: `2dbe5399fb6937ab53f0400a73df2a1867881e15816c41f79789b2cbf2b091b5`
- `scripts/merge_assignment_feature_npz.py` SHA256: `eb8a4148364386e5dc5b78515061e182a6b06d49815e8b64c63b6d629636465a`

## Runs

### Oracle-choice target

Output: `experiments/results/eval600_to_eval300_assignment_group_noop_window_oraclechoice_2026_06_25.json`  
SHA256: `93c5ebfe2a255f2899855b02eb71348e2d394e5a1700fb1b7f98c82455a7c4ec`  
Scores: `experiments/results/eval600_to_eval300_assignment_group_noop_window_oraclechoice_scores_2026_06_25.jsonl`  
W&B offline run: `3vj4lavk`

Key aggregate metrics over model seeds 0/1/2:

- Val row AUC/AP: `0.8408 / 0.5416`
- Val group AUC: `0.5737`
- Threshold `0.5`: mean changed groups `24.33`, Tesseract delta `-2.33`, false changes `16.67`, wrong changes `5.00`, Tesseract-worsen groups `2.67`, exact oracle groups `2.67`
- Train-tuned threshold with strict train constraints: mean changed groups `31.33`, Tesseract delta `-1.67`, false changes `22.67`, wrong changes `5.67`, Tesseract-worsen groups `3.67`, exact oracle groups `3.00`

### Safe soft-utility target

Output: `experiments/results/eval600_to_eval300_assignment_group_noop_window_softutility_2026_06_25.json`  
SHA256: `0b041697d10d1a3aa476d313827d26196ebd011a7a2082e4d1066fb5d6b00072`  
Scores: `experiments/results/eval600_to_eval300_assignment_group_noop_window_softutility_scores_2026_06_25.jsonl`  
W&B offline run: `sulkrjq6`

Key aggregate metrics over model seeds 0/1/2:

- Val row AUC/AP: `0.8149 / 0.3867`
- Val group AUC: `0.6041`
- Threshold `0.5`: mean changed groups `23.33`, Tesseract delta `-1.00`, false changes `16.33`, wrong changes `5.33`, Tesseract-worsen groups `2.67`, exact oracle groups `1.67`
- Train-tuned threshold with strict train constraints: mean changed groups `0.67`, Tesseract delta `0.00`, false changes `0.00`, wrong changes `0.67`, Tesseract-worsen groups `0.33`, exact oracle groups `0.00`

### Consensus selector diagnostics

Oracle-choice consensus output: `experiments/results/eval600_to_eval300_assignment_score_consensus_oraclechoice_2026_06_25.json`  
SHA256: `c5e886c3edf244bb13516da846f7d30bbea1aa0b842ef270e0dd63db4440a226`  
W&B offline run: `1m472jil`

- Consensus train-tuned val policy: changed groups `37`, Tesseract delta `-5`, false changes `28`, wrong changes `6`, Tesseract-worsen groups `2`, exact oracle groups `3`

Soft-utility consensus output: `experiments/results/eval600_to_eval300_assignment_score_consensus_softutility_2026_06_25.json`  
SHA256: `0d81b42523093d07fc8cb9c3894f5a230996f0580e86b03968da0e23018854b0`  
W&B offline run: `ndreov1l`

- Consensus train-tuned val policy: changed groups `35`, Tesseract delta `+1`, false changes `25`, wrong changes `9`, Tesseract-worsen groups `5`, exact oracle groups `1`

## Interpretation

The external top-8 assignment labels contain useful row-level signal, but the current group/no-op window head remains poorly calibrated for safe deployment. The main failure is not lack of top-8 recall: Eval300 seed1 has `-24` edit oracle headroom with 17 exact changed groups. The failure is no-headroom / false-change suppression and hard-positive transfer.

The training substrate is also biased toward oracle-safe selected Eval600 groups. It contains rank-0 groups, but it underrepresents hard negative images where no candidate should be touched. This likely explains why train-tuned consensus can be perfect on Eval600 train groups yet emits 25-28 false changes on Eval300.

## Next Action

Do not promote these policies to actual `.oscr`. The next assignment substrate should add Eval600 hard no-headroom negatives before another architecture sweep:

1. select Eval600 source images with no PARSeq-safe Tesseract-improving candidate;
2. generate top-8 assignment variants for representative high-risk candidates;
3. run PARSeq/Tesseract OCR and merge them as hard negative groups;
4. retrain the same simple head first, then only increase model capacity if false-change suppression improves.

This keeps the OSCAR-LIC novelty axis intact: compact plaintext-free enhancement, actual-byte accounting, and marginal OCR utility per bit, while attacking the current calibration bottleneck directly.

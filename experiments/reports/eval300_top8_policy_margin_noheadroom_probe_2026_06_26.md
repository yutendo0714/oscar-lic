# Eval300 Top8 Policy-Margin No-Headroom Probe

Date: 2026-06-26

This is a diagnostic-only probe. It uses held-out Tesseract-derived assignment labels inside Eval300, so it must not be promoted as deployable training or selection.

## Purpose

N064 showed that neighborhood support exists but top1/policy selection is unsafe. This probe asks whether a stricter policy-level objective can suppress no-headroom false changes while preserving hard-positive assignment changes.

The run uses the nearest-plus-top8 latent-window artifact and changes the objective away from plain row cross-entropy:

- row target: `safe_tesseract_soft_bce`
- group target: `strong_tesseract` with threshold `-2`
- explicit false-change, PARSeq-harm and Tesseract-nonimprove penalties
- hard-positive row/group margins
- matched positive-vs-no-headroom pairwise margins
- train threshold constrained to zero false changes, zero wrong changes and zero Tesseract-worsened groups

Only `val_seed=1` was run as a probe before deciding whether this family deserves a full cross-seed sweep.

## Result

- Row validation AUC/AP: `0.88695 / 0.58937`
- Group validation AUC: `0.46991`
- Oracle on this split: `17` changes, `-24` Tesseract edits, PARSeq unchanged.
- Default threshold `0.5`: `3` changed groups, `0` exact oracle changes, `2` false changes, `1` wrong change, mean Tesseract delta `+1.67`.
- Train-tuned threshold: mean `2.33` changed groups, `0.33` exact oracle changes, `1.33` false changes, `0.67` wrong changes, mean Tesseract delta `+0.33`.
- Nearest/no-op remains better than the learned policies on this probe.

Train-side thresholding can look clean: model seeds `0` and `2` select `18/225` useful train changes with `0` false/wrong changes and `-42` Tesseract edits. That calibration does not transfer to the held-out val shard.

## Interpretation

This is a valid negative probe for this policy-margin/no-headroom configuration. It reduces aggressiveness relative to earlier row-argmax failures, but mostly by collapsing toward no-op; it does not recover the top8 oracle headroom and still emits false/wrong changes on val.

Do not run counted `.oscr` promotion from this checkpoint family. A full cross-seed sweep of this exact objective is not justified unless the next experiment adds genuinely new calibration evidence or more training domains rather than just retuning these weights.

# Eval300 Assignment Tensor Reranker Diagnostics

Date: 2026-06-25

Scope: tensor-backed nearest-plus-top4 code-index assignment rerankers for the frozen MLIC++/OSCAR-LIC K64 utility-weighted codebook path.

## Promotion Baseline

The current learned assignment baseline remains the pooled image-diff group gate over the codegeom oracle-choice scorer:

- actual bpp unchanged versus nearest assignment: mean `3.849649`, enhancement mean `0.028989`;
- PARSeq unchanged exactly across seed0/1/2 val75;
- Tesseract improves by `-8` unicode and `-8` latin-alnum edit characters versus nearest;
- policy shape: `5/75` changed groups, `4` exact oracle hits, `1` false change, `0` wrong changes.

This is the precision floor. A new assignment scorer must beat the `-8/-8` actual-bitstream OCR result without reintroducing broad false or wrong changes.

## Tensor Artifacts

- Base tensor artifact: `experiments/features/eval300_assignment_top4_codeeffect_tensors_seed0_1_2_2026_06_25.npz`, SHA256 `95a895e7771bbc24324d593fcf730b4bbfcb7394f0e8f18eff32fa717ee68ee2`.
- Base metadata: SHA256 `f6be9db954eb9a2654b4ff8b5a1db1807795296915051fa31abc80506b000b6c`.
- Latent-context tensor artifact: `experiments/features/eval300_assignment_top4_codeeffect_latentctx_tensors_seed0_1_2_2026_06_25.npz`, SHA256 `7cddeff996536b65b2492b1c93bb31e20b5ad17be7c7df6bddebd3e752e13094`.
- Latent metadata: SHA256 `bf5c5edb4b6927e30542a63b2747019594ab0254639018a30f418950f12f8b7e`.

Both artifacts contain `300` groups, `1213` valid candidate rows and at most `5` candidates per group. The latent-context artifact adds `latent_yhat`, `latent_target` and `latent_residual` tensors with shape `[300, 5, 512]`.

## Result Summary

All numbers below are table-level Eval300 val75 policy deltas versus nearest assignment unless explicitly marked as actual-bitstream. PARSeq delta is `0` for these learned tensor runs.

| Variant | Tesseract delta | Changed groups | False changes | Worsen groups | SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Code-only listwise h96, 3 seeds | `-5.67` mean, best `-7` | `16.67` | `13.33` | `2.33` | `797d20405a492330ee5238fa19b52e5f8f6eda4d2f0680315b63758616dbc3d3` |
| Code-only h96, 10 seeds | `-4.6` mean, best `-7` | `16.5` | `13.2` | `2.7` | `de6e11a66ae0d9c72319d21dc228eeaf5544b62ab2f639fe9be01eaba73c3a2d` |
| Val-tuned margin gate, best seed1 | `-8` | `11` | `7` | `1` | `260d1e533ee8c9e3da0afd897a4643de85a01a547836df774be6c603e38e5bc7` |
| Train-tuned margin gate, best seed1 | `-7` | `17` | `13` | `2` | `57a494873dac11901bfff40ddc9f96157a7a573e404914cae54701330011620a` |
| Non-improve/harm logit penalties | `-5.33` to `-2.33` | high | high | high | see `STATUS.md` |
| Group-change margin loss | `-5.0`, `-3.67`, `-4.67` | ~`16` | ~`13` | high | `5ff7ff6c...`, `272a73b6...`, `d9890e47...` |
| Latent concat h96 | `-2.0` | `12.33` | `10.67` | `2.0` | `04e97f4568be60a4d99b85e62f7509cc2af54782020205bd0e0b0972614c629f` |
| Latent concat h64 regularized | `-2.0` | `13.33` | `10.67` | `2.67` | `c0ba9d9421adbd0ed6e32ac00d26d7643af0eb60f2afbf6b139b24f1ecccb9ad` |
| Latent branch h96 | `-2.0` | `20.0` | `17.0` | `3.0` | `669d242e1cd535f71a2375416082638eeb0ad39ff97732d988256f8895cfc455` |
| Latent branch h64 regularized | `-5.33` | `21.67` | `17.33` | `3.0` | `3f5cce26920c9a1faf94e2dd09686f0f02c952a85dd5e23e1d5650591ce74ef9` |
| Latent CNN h96 | `-5.0` | `20.33` | `16.33` | `3.0` | `63071ca328e96c25ae471c271d3b47e47c9b50e10111657803960b5452d5cffd` |
| Latent CNN h64 regularized | `-4.0` | `19.0` | `15.33` | `3.33` | `163ef08bf78200cda56b052c709035d90a7f009b26ed742a81b38895aeb8bf22` |
| Structured latent-window CNN h96 | `-4.33` | `18.0` | `14.67` | `3.0` | `20640cdd1f7a7e2f46567973a7454fb7b0f9de60a5e37954c15c8f4c8b1089ad` |
| Structured latent-window CNN h64 regularized | `-4.0` | `17.33` | `14.33` | `3.0` | `d7d24224aadbca4c6a5d01a632e23f51819fb61bcf2a7a0411c3f4b38a0651c7` |
| Structured latent-window CNN h96, no-change weight 2.0 | `-3.0` | `16.67` | `14.0` | `3.0` | `d1f0dadd73a2bf06bc45e10f29431f0faecd0d8fc6d4f0119f2924c1182de9f1` |
| Structured latent-window CNN h96 margin gate, best seed0 | `-6` | `20` | `15` | `3` | `12c2a1bec95d3c9a9043d67fa9c0d26f0a85db8c5483d7751de20b949c33c726` |

## Interpretation

The tensor substrate is useful, but the current MLP/branch/CNN models are not promotion candidates.

Main reasons:

- The best post-hoc val-tuned margin gate can tie aggregate `-8` Tesseract edits, but it does so with `11` changes, `7` false changes and one Tesseract-worsen group. The promoted actual-bitstream image-diff gate reaches the same aggregate gain with only `5` changes and one false change.
- Train-tuned calibration falls back to best `-7`, so the `-8` margin-gated result is not robust enough for a paper claim.
- Direct scalar losses for non-improve, harm and group-change calibration do not clean up false changes. They mostly reshuffle high-risk non-nearest choices under the same representation.
- Raw flattened latent vectors are actively misleading: concat, branch and small CNN variants can raise ranking diagnostics while degrading policy-level OCR relative to the simpler code-only listwise smoke.
- Structured local latent-window export works and the trainer can consume it, but a plain pooled window-CNN branch is still below the code-only tensor smoke. Increasing the scalar no-change penalty raises validation AUC/AP but further weakens policy-level OCR, and post-hoc margin gating reaches only best `-6` with many false changes. The missing piece is likely the objective/calibration and group interaction, not only spatial access to neighboring latent cells.
- The promoted image-diff policy misses oracle cases because the scorer ranks the wrong code first, not because the gate rejects oracle-best choices. Relaxing gates will expose bad scorer choices unless candidate-conditioned ranking improves.

## Decision

Do not promote any tensor/latent reranker from this batch.

Keep the actual-bitstream pooled image-diff group gate as the current learned baseline. Treat the tensor artifacts and training code as the next substrate, not as a deployable result.

## Next Move

The next model should be architectural, not another scalar loss or seed sweep:

1. Add a true group/no-op head sharing the code-effect representation instead of using post-hoc margin gates.
2. Replace flat `[512]` latent vectors with structured local latent-token windows around the selected tile/channel group, but train them through an explicit group/no-op head rather than a standalone pooled CNN branch.
3. Use listwise top4 reranking with nearest as an explicit no-op anchor and report false/wrong/missed oracle groups for every run.
4. Promote only after explicit policy export, counted `.oscr` reconstruction, PARSeq/Tesseract OCR and pairwise comparison against nearest and the current image-diff baseline.

## Verification

Final workspace checks after this diagnostic report:

- GPU visible: NVIDIA GeForce RTX 4070 Ti SUPER, driver `570.148.08`.
- Python compile passed for `scripts/export_assignment_reranker_tensors.py`, `scripts/train_assignment_codeeffect_reranker.py`, `scripts/analyze_assignment_tensor_margin_gate.py` and `scripts/validate_repo.py`.
- Unit tests passed: `34 passed`.
- Repository structural validation passed with the expected unresolved-DCAE warning only.

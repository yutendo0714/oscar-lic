# Architecture Decision Records

## ADR-001: Primary comparison track is pure visual coding

**Status:** accepted  
**Decision:** Main results prohibit lossless text strings and OCR tokens in the bitstream.  
**Rationale:** Semantic side-channels solve a different problem and create rate/privacy confounds.  
**Consequence:** PICD is compared in a separate semantic-assisted track.

## ADR-002: MLIC++ is the first high-performance base codec

**Status:** accepted  
**Decision:** Start with MLIC++ official repository at the pinned snapshot in `baselines/registry.yaml`.  
**Rationale:** Public implementation, checkpoints, actual compression path and strong entropy model.  
**Alternative:** DCAE is retained as a second base due to dictionary prior compatibility.

## ADR-003: Separate environments for codec and OCR teachers

**Status:** accepted  
**Decision:** Do not force all dependencies into one environment. Cache OCR labels/features in versioned manifests where gradient flow is not needed; use a dedicated joint environment only for differentiable teacher experiments.  
**Rationale:** Legacy CompressAI/PyTorch and modern OCR stacks can conflict.

## ADR-004: Oracle feasibility precedes predictor development

**Status:** accepted  
**Decision:** No large utility predictor training before oracle allocation beats mask baselines under matched total rate.  
**Rationale:** A predictor cannot exceed an uninformative target.

## ADR-005: Actual bitstream is primary rate

**Status:** accepted  
**Decision:** Main tables use serialized total bytes. Estimated entropy is diagnostic only.  
**Rationale:** Gate, header, padding and coder overhead matter strongly at low bitrate.

## ADR-006: Held-out OCR is a primary metric, not appendix

**Status:** accepted  
**Decision:** At least one recognizer family and one OCR system not used for training must appear in the main table.  
**Rationale:** Single-model gains can be adversarial preprocessing rather than information preservation.

## ADR-007: Promotion requires actual `.oscr` byte accounting

**Status:** accepted  
**Decision:** Assignment, codebook and enhancement variants are promoted only after real compress/decompress through the counted `.oscr` path, using complete transmitted bytes for bpp.  
**Rationale:** Estimated entropy and row metrics repeatedly failed to predict actual policy behavior at low bitrate and small optional payloads.  
**Evidence:** Current promoted pooled image-diff group gate preserves PARSeq and improves Tesseract at measured actual bpp; several high-AUC alternatives were negative controls.

## ADR-008: Next assignment model must learn no-op jointly

**Status:** accepted  
**Decision:** The next assignment experiment should use a small group-aware model with a shared code-effect/window encoder and explicit group/no-op head.  
**Rationale:** Score coverage shows the remaining headroom is code assignment ranking, while naive scalar penalties, flat latent features and post-hoc margin gates do not recover the top-8 oracle safely.  
**Evidence:** `docs/HANDOFF.md`, `configs/training/assignment_group_noop_window_v0.yaml`.

## ADR-009: Do not promote group/no-op window v0

**Status:** accepted  
**Decision:** Keep the pooled image-diff actual-bitstream gate as the learned baseline; do not promote the current group/no-op window v0, strong-target variant, reward-softmax row-target variant, candidate-only BCE variant, tree-ranker diagnostic, or 10-seed v0 sweep.  
**Rationale:** These variants improve some row metrics but remain policy-imprecise, with many false changes and no Tesseract gain beyond the promoted `-8/-8` actual-bitstream baseline. Candidate-only row reweighting and non-neural tabular rankers do not close the gap.  
**Evidence:** `experiments/reports/eval300_group_noop_assignment_followup_2026_06_25.md`, W&B `9z4trvnq`, `qwbu6asw`, `nay55x83`, `o882h3gb`.

## ADR-010: Cross-fit group-gate diagnostics are not promotion evidence

**Status:** accepted  
**Decision:** Treat real-seed cross-fit group-gate results as calibration diagnostics until the selected policies are exported through counted `.oscr` streams and evaluated with row-wise OCR.  
**Rationale:** The fixed threshold `0.5` cross-fit is positive at table level, and train-tuned thresholds are stronger, but table-level assignment metrics do not include all actual stream/reconstruction effects. Prior row/table metrics have repeatedly overestimated policy quality.  
**Evidence:** `experiments/reports/eval300_group_gate_realseed_crossfit_2026_06_25.md`, W&B `z2cse6yf`.

## ADR-011: OSCR decode is fail-closed unless a recovery profile is explicit

**Status:** accepted  
**Decision:** Keep default `.oscr` parsing fail-closed on any CRC mismatch. Base-layer recovery after optional enhancement damage must be implemented as an explicit per-section recovery/UEP profile, not by globally disabling CRC verification.  
**Rationale:** A one-byte `TEXT_MAIN` corruption leaves the `BASE_MAIN` payload independently decodable in no-CRC salvage mode, but a one-byte `BASE_MAIN` corruption with CRC bypass triggered an MLIC++ CUDA out-of-memory path.  
**Evidence:** `experiments/reports/oscr_bitstream_robustness_smoke_2026_06_25.md`, W&B `4369gwqs`.

## ADR-012: Cross-fit exports need clean-val precheck before actual streams

**Status:** accepted  
**Decision:** Before running counted `.oscr` promotion for any mixed cross-fit export, first filter the policy to the exact clean split-compatible rows and require the table delta to beat the current promoted actual-bitstream baseline.  
**Rationale:** The real-seed cross-fit train-tuned group gate had strong mixed 100-group table gains, but after filtering to Eval300 val75-compatible rows the best Tesseract delta was only `-3`, below the promoted `-8` actual-bitstream baseline.  
**Evidence:** `experiments/reports/eval300_group_gate_clean_val_filter_2026_06_25.md`, W&B `36wwlrx8`.

## ADR-013: Multi-candidate allocation must model diminishing returns

**Status:** accepted  
**Decision:** Do not train or promote a multi-candidate OSCAR allocator that assumes per-candidate OCR utilities add independently. Future multi-candidate policies need subset-aware labels, diminishing-return features, or an explicit interaction penalty/control.  
**Rationale:** A 5-image raw-residual actual `.oscr` smoke showed strong non-additivity: two individually useful candidates predicted `-8` Tesseract unicode edit errors additively, but the pair achieved only `-1`. PARSeq also saturated after either single candidate fixed the same error.  
**Evidence:** `experiments/reports/eval300_interaction_smoke5_2026_06_25.md`, W&B `fhk5uy8l`.

## ADR-014: Top-8 assignment training must include hard no-headroom negatives and policy-budget objectives

**Status:** accepted  
**Decision:** Do not promote or seed-sweep the current top-8 group/no-op latent-window head on oracle-safe-only labels. Future top-8 assignment work must train on a hard-negative-balanced substrate and optimize for policy-level false-change budgets or matched positive-vs-hard-negative contrast before counted `.oscr` promotion.  
**Rationale:** Eval600 oracle-safe-only transfer produced useful row AUC/AP but too many Eval300 false changes. Adding 120 hard no-headroom groups reduced false changes and improved AUC/AP, showing the data direction is right, but the current head still misses hard positives and remains below the promoted actual-bitstream baseline.  
**Evidence:** `experiments/reports/eval600_to_eval300_top8_assignment_external_transfer_2026_06_26.md`, `experiments/reports/eval600_hard_noheadroom_top8_assignment_transfer_2026_06_26.md`, W&B `3vj4lavk`, `x94exiuz`, `5mtvr2r1`.

## ADR-015: Matched pairwise assignment losses are shortlist diagnostics, not promotion objectives

**Status:** accepted  
**Decision:** Do not continue seed-sweeping or scalar-weight sweeping matched pairwise losses on the current single-score top-8 group/no-op head. Treat the pairwise head as a shortlist diagnostic only. Future work must add an explicit no-op/top-k policy stage, direct false/wrong-change budget optimization, or richer local code-effect evidence before any counted `.oscr` promotion attempt.  
**Rationale:** Matched pairwise losses improve Eval300 seed1 row AP (`0.6962`) and show that 10-13/17 useful changes appear in the scorer top4, but deployed threshold/train-tuned/consensus policies recover only `-1` to `-2.33` Tesseract edits and remain far below the promoted actual-bitstream baseline.  
**Evidence:** `experiments/reports/eval600_hardneg_pairwise_assignment_objective_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_pairwise_assignment_objective_2026_06_26.yaml`, W&B `nfy2esqe`, `4yx9140t`, `ebb7tvge`, `semgweeh`.

## ADR-016: Post-hoc policy budgets do not replace deployable code-effect evidence

**Status:** accepted  
**Decision:** Do not attempt counted `.oscr` promotion by applying only post-hoc thresholds, score consensus, or budget-fraction selection to the current Eval600-trained score files. Future assignment promotion attempts must improve the deployable evidence/model or calibration data, not just the policy knob on the same scores.  
**Rationale:** Budgeted selection can perfectly select Eval600 train positives under zero false/wrong/worsen constraints, but the same score boundary does not transfer to Eval300. Best Eval300 table deltas remain around `-2` Tesseract edits and require many false changes, below the promoted actual-bitstream gate.  
**Evidence:** `experiments/reports/eval600_hardneg_policy_budget_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_policy_budget_selector_2026_06_26.yaml`, W&B `guj1r9l7`.

## ADR-017: Tiny source supplements are insufficient calibration data

**Status:** accepted  
**Decision:** Do not continue by appending only tiny source-specific supplements to the same single-score top-8 group/no-op latent-window head. Future calibration-data work must be larger, source/domain-balanced and validated externally, or must pair the data with richer deployable candidate-local code-effect evidence.  
**Rationale:** Adding 20 IAM groups to the hard-negative-balanced Eval600 transfer artifact provides relevant data, including 5 useful assignment-change groups, but Eval300 policies still recover only `-1` Tesseract edit on average under train-tuned thresholds with `11.67` false changes. Row AUC/AP remains high (`0.8966/0.6642`), so the failure is not solved by small data append alone.  
**Evidence:** `experiments/reports/eval600_hardneg_iam_supplement20_assignment_transfer_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_iam_supplement20_assignment_transfer_2026_06_26.yaml`, W&B `iikeyxg2`.

## ADR-018: More same-kind calibration labels do not rescue the current single-score assignment head

**Status:** accepted  
**Decision:** Do not keep appending no-safe Eval600 groups to the same single-score group/no-op latent-window head as the primary route to promotion. The next assignment attempt must change the decision structure or deployable evidence: for example a separate high-precision no-op rejector, top-k shortlist verifier, policy-level verifier, or richer candidate-local code-effect predictor.  
**Rationale:** A larger 210-group source-balanced no-safe supplement adds 19 top-8 assignment oracle-change groups and raises the merged artifact to 548 groups, but Eval300 transfer remains weak. Train-tuned thresholds average only `-1.33` Tesseract edits with `10.0` false changes; threshold `0.5` is slightly harmful on average despite high val AUC/AP (`0.8969/0.6807`).  
**Evidence:** `experiments/reports/eval600_hardneg_balanced210_assignment_transfer_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_assignment_transfer_2026_06_26.yaml`, W&B `5q05gb1z`.

## ADR-019: Score-only top-k verifiers are not enough

**Status:** accepted  
**Decision:** Do not continue assignment promotion attempts that use only the current first-stage score/rank/vote/group-probability JSONL features in a second-stage verifier, budget selector, threshold, consensus rule, linear head or shallow MLP head.  
**Rationale:** A separate top-k verifier can perfectly fit Eval600 train groups under strict false/wrong/worsen constraints, but Eval300 candidate and group AUCs fall far below `0.5` and transferred policies miss all useful oracle changes. The top-M upper remains strong, so recall exists; the missing ingredient is externally transferable candidate-local code-effect evidence.  
**Evidence:** `experiments/reports/eval600_hardneg_balanced210_topk_score_verifier_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_topk_score_verifier_2026_06_26.yaml`, W&B `blrnzrcc`.

## ADR-020: Image-diff evidence must improve the scorer, not only the gate

**Status:** accepted  
**Decision:** Do not rely on an external Eval600-trained image-diff group gate as the next assignment promotion route. If reconstructed-image/code-effect evidence is reused, it should enter the candidate-level scorer/reranker or be paired with source/domain-disjoint calibration, not only a group-level accept/reject gate over the current scorer.  
**Rationale:** The external image-diff group gate has weak Eval300 AUC/AP (`0.4980/0.2267`) and either collapses to no useful changes under fixed thresholds or inherits bad scorer choices under train-tuned thresholds. The group gate cannot recover oracle codes that the scorer fails to rank first.  
**Evidence:** `experiments/reports/eval600_hardneg_balanced210_imgdiff_group_gate_external_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_imgdiff_group_gate_external_2026_06_26.yaml`, W&B `f2omvl4p`.

## ADR-021: Candidate row AUC is not enough for assignment promotion

**Status:** accepted  
**Decision:** Do not promote or seed-sweep the external diff-crop candidate scorer, and do not treat high candidate-row AUC/AP as sufficient evidence for OSCAR-LIC assignment progress. Future assignment models must be selected by policy-level OCR safety and external calibration: Tesseract/PARSeq deltas, false/wrong changes, missed oracle groups, and eventually counted `.oscr` bpp.  
**Rationale:** The diff-crop candidate scorer reaches Eval300 row AUC/AP `0.8791/0.6051`, but all learned deployment policies worsen Tesseract (`+0.67` to `+1.67`) and select zero improving groups while the oracle remains `-24`. This closes the immediate "put image-diff into the candidate scorer" loophole and points to policy-level objectives or verifiers over richer non-score evidence.  
**Evidence:** `experiments/reports/eval600_hardneg_balanced210_external_diffcrop_candidate_scorer_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_external_diffcrop_candidate_scorer_2026_06_26.yaml`, W&B `xem245nk`.

## ADR-022: Safe-improve label swapping is not a standalone fix

**Status:** accepted  
**Decision:** Do not continue by swapping rare OCR-safe-improvement labels into the same diff-crop candidate scorer. Safe-improve supervision should be used only with explicit abstention, policy-level false/wrong-change constraints, uncertainty-aware selection or source/domain-disjoint calibration.  
**Rationale:** Changing the diff-crop scorer target from `oracle_choice` to `safe_improve` lowers Eval300 AUC/AP to `0.5186/0.0676` and makes policies more aggressive and harmful (`+3.33` Tesseract edits under always-best). The label is semantically closer to the desired policy but not externally learnable with this scorer alone.  
**Evidence:** `experiments/reports/eval600_hardneg_balanced210_external_diffcrop_safeimprove_candidate_scorer_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_external_diffcrop_safeimprove_candidate_scorer_2026_06_26.yaml`, W&B `k5xbgn7o`.

## ADR-023: Next assignment selector should operate on a top-4 shortlist with abstention

**Status:** accepted  
**Decision:** The next assignment experiment should not trust any single top-1 scorer. It should keep a top-4 candidate shortlist, include an explicit no-op/abstention option, and penalize harmful non-nearest choices directly under external validation.  
**Rationale:** The N036 score-rank audit shows that diff-crop evidence still carries shortlist information: oracle-choice scores keep mean `11.67/17` oracle and `14.33/17` safe-improve groups in top-4, while safe-improve scores keep `14/17` safe groups in top-4. The failure is top-1 calibration: oracle-choice has zero improving top-1 groups, and safe-improve produces mean `4.0` harmful top-1 groups.  
**Evidence:** `experiments/reports/eval600_hardneg_balanced210_diffcrop_score_rank_audit_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_diffcrop_score_rank_audit_2026_06_26.yaml`, W&B `ccm9qx83`.

## ADR-024: The current top-4 selector is an upper-bound substrate, not a learned policy

**Status:** accepted  
**Decision:** Keep the union top-4 shortlist construction as a diagnostic substrate, but do not promote or seed-sweep the first learned top-4 MLP selector. Future selector work must improve calibration rather than capacity: out-of-fold score features, source/domain-disjoint calibration, uncertainty-aware abstention or simpler high-precision rules.  
**Rationale:** The shortlist oracle reaches `-19.33` Tesseract edits with PARSeq unchanged, close to the full top-8 oracle `-24`. The learned selector, however, recovers `0` exact oracle changes and worsens Tesseract by `+1.11` with `9.56` false/wrong changes.  
**Evidence:** `experiments/reports/eval600_hardneg_balanced210_top4_policy_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_top4_policy_selector_2026_06_26.yaml`, W&B `33xqtxjk`.

## ADR-025: In-sample score calibration is invalid for selector transfer

**Status:** accepted  
**Decision:** Do not tune score-based top-k thresholds, budgets, MLP selectors or simple rules on score outputs from models trained on the same Eval600 groups. Future score-based selector diagnostics must use out-of-fold/cross-fit score generation or a separate source/domain-disjoint calibration set.  
**Rationale:** The diff-crop budget selector selects 76 Eval600 train changes with zero false/wrong/worsen events and `-100` Tesseract edits, but the transferred Eval300 threshold selects no oracle changes and only false/neutral rows. This explains the N037 MLP failure and makes in-sample Eval600 score calibration unsuitable for promotion-oriented selector design.  
**Evidence:** `experiments/reports/eval600_hardneg_balanced210_diffcrop_budget_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_diffcrop_budget_selector_2026_06_26.yaml`, W&B `qay9jfvo`.

## ADR-026: Top-4 shortlist is a validated actual-bitstream upper-bound substrate

**Status:** accepted  
**Decision:** Keep the score-seed-2 union top-4 shortlist as the main assignment-selector substrate, but treat oracle selection within that shortlist as a non-deployable upper bound only. The next promotable path must learn or calibrate a no-op-aware selector for this shortlist without using held-out OCR oracle labels.  
**Rationale:** The shortlist oracle survives real counted `.oscr` compress/decompress at unchanged bpp (`3.849649` mean, `0.028989` enhancement bpp), preserves PARSeq exactly, and improves Tesseract by `-20` unicode / `-17` latin-alnum edits versus nearest and `-12` / `-9` versus the current best learned actual stream. This confirms that the remaining headroom is in selector calibration, not payload accounting.  
**Evidence:** `experiments/reports/eval300_shortlist_oracle_top4_scoreseed2_actual_oscr_2026_06_26.md`, manifest `experiments/manifests/eval300_shortlist_oracle_top4_scoreseed2_actual_oscr_2026_06_26.yaml`, W&B `ftpgkenc`.

## ADR-027: Multi-candidate allocation needs stop/continue utility

**Status:** accepted  
**Decision:** Refine OSCAR-LIC's "marginal OCR utility per bit" into conditional marginal utility given already-selected enhancement units. Any future multi-candidate stream allocator must include an explicit stop/continue decision, subset-aware labels, learned interaction terms, or diminishing-return regularization before adding second/third candidates.  
**Rationale:** N069, N070, N071 and N072 show that positive individual utility and monotone PSNR are not enough. In the 20-image pair audit, Tesseract pair utility retained only `16.95%` of additive single-candidate gain and was worse than the best single on `11/20` samples. In the 10-image 3-candidate all-subset smoke, the best one-candidate policy beat every pair/triple policy for Tesseract, while the subset oracle chose one candidate on most samples. Scaling to 40 images strengthens the conclusion: Tesseract unicode best single gives `-63` edits, best pair only `-36`, triple `-32`, and the subset oracle chooses a one-candidate policy for `36/40` unicode samples; PARSeq chooses one candidate on `40/40`. Conditional edge analysis shows the structural reason: Tesseract unicode base-to-single edges improve by `-179` edits, while single-to-pair edges worsen by `+178` edits.  
**Consequence:** Do not implement independent top-k summation for OSCAR multi-candidate enhancement, even as a "simple first version". A simple model is still preferred, but it must be a simple conditional/stopping model rather than a simple additive model. Any learned multi-candidate allocator should beat the N074 `first_only` stop baseline before scaling conditional labels or promotion attempts.  
**Evidence:** `experiments/reports/eval300_interaction_smoke20_detail_audit_2026_06_26.md`, `experiments/reports/eval300_interaction_subset3_smoke10_tesseract_subset_oracle_2026_06_26.md`, `experiments/reports/eval300_interaction_subset3_smoke40_tesseract_subset_oracle_2026_06_26.md`, `experiments/reports/eval300_interaction_subset3_smoke40_tesseract_conditionals_2026_06_26.md`, `experiments/reports/eval300_interaction_subset3_smoke40_stop_baselines_2026_06_26.md`, manifests `experiments/manifests/eval300_interaction_smoke20_detail_audit_2026_06_26.yaml`, `experiments/manifests/eval300_interaction_subset3_smoke10_2026_06_26.yaml`, `experiments/manifests/eval300_interaction_subset3_smoke40_2026_06_26.yaml`, `experiments/manifests/eval300_interaction_subset3_smoke40_conditionals_2026_06_26.yaml` and `experiments/manifests/eval300_interaction_subset3_smoke40_stop_baselines_2026_06_26.yaml`, W&B `828bl6tt`, `sem19sup`, `90cff84m`, `bzrbijit`, `iidwqg03`.

## ADR-028: Do not scale shallow conditional-edge allocators

**Status:** accepted  
**Decision:** Do not generate larger multi-candidate conditional-edge label tables for transition/candidate/rate/PSNR/source-feature allocators, and do not promote shallow feature allocators that only match N074 `first_only`.  
**Rationale:** N075 shows that transition-only exactly reproduces the `first_only` floor, while adding candidate identity, selected-set bits, enhancement bpp, PSNR gain and source/domain lowers edge F1 and does not improve deployable policy behavior. A non-deployable OCR-state upper-control improves edge delta but still fails the coherent greedy subset policy versus the fixed one-candidate `single_a` policy.  
**Consequence:** Future multi-candidate work must first introduce materially richer selected-set/candidate-local code-effect evidence, a compact symbol representation whose second/third units have real conditional OCR gain, or an explicit verifier that beats the fixed one-candidate policy. Until then, the higher-leverage path is one-candidate assignment/code-effect improvement and compact symbol training.  
**Evidence:** `experiments/reports/eval300_interaction_subset3_smoke40_feature_allocator_2026_06_26.md`, manifest `experiments/manifests/eval300_interaction_subset3_smoke40_feature_allocator_2026_06_26.yaml`, W&B `s4jy1ggg`.

## ADR-029: Existing assignment score families are not a safe threshold substrate

**Status:** accepted  
**Decision:** Do not continue by threshold-sweeping, budget-sweeping or reweighting the current code-only, diff-crop, latent-window, latent-context or tabular image-diff score families on the actual Eval300 assignment failure bank. The next assignment verifier must introduce materially different candidate-local code-effect evidence, improve first-stage candidate scoring, or use an OCR/text-aware verifier whose deployable features are not just these failed score confidences.  
**Rationale:** N077 joins the N076 actual `.oscr` failure bank with top8 deployable candidate features and five score families. The 11 recoverable shortlist-oracle changes are never ranked first by code-only/diff-crop/latent-window/latent-context scores, while the one current bad change is ranked first by every score family with high confidence. This is not a calibration-threshold problem; the score ordering itself is wrong for the high-value cases.  
**Consequence:** Future work should not spend cycles on more threshold knobs over these scores. Use N077's contact sheets and feature audit to design new evidence: glyph-shape repair descriptors, candidate-local code-effect comparison beyond scalar residual geometry, a better first-stage top-k generator, or a fixed OCR/text verifier with documented provenance and held-out-evaluator discipline.  
**Evidence:** `experiments/reports/eval300_actual_assignment_failure_feature_audit_2026_06_26.md`, manifest `experiments/manifests/eval300_actual_assignment_failure_feature_audit_2026_06_26.yaml`, W&B `wqnd7ezy`.

## ADR-030: Top-8 shortlist is the current actual-bitstream upper-bound target

**Status:** accepted  
**Decision:** Treat the score-seed-2 top-8 shortlist oracle as the strongest current non-deployable assignment upper bound, and make future assignment work target top-8 recall plus safe no-op selection rather than only the previously validated top-4 substrate.  
**Rationale:** N079 verifies the top-8 oracle through real counted `.oscr` compress/decompress at the same mean actual bpp as nearest/current/top-4 streams (`3.849649` total, `0.028989` enhancement). It recovers all `17/75` oracle changes, keeps PARSeq unchanged, improves Tesseract by `-24` unicode / `-19` latin edits versus nearest and adds `-4` / `-2` over the top-4 actual oracle. N081 adds paired-bootstrap support: nearest->top8 and current->top8 Tesseract unicode/latin CIs exclude zero at unchanged actual byte-counted bpp, while top4->top8 remains observed-positive but CI-touching-zero. This closes the two top-4 shortlist recall misses without changing payload size, but avoids overclaiming statistical separation from top4 at the current sample size.  
**Consequence:** The next deployable selector/generator should improve first-stage top-k evidence and no-op safety for top-8 candidates. Top-4 remains useful for high-precision substrate analysis and statistically cleaner top4-vs-current evidence, but it is no longer the upper bound for assignment headroom. Existing-score threshold sweeps remain rejected by ADR-029.  
**Evidence:** `experiments/reports/eval300_shortlist_oracle_top8_scoreseed2_actual_oscr_2026_06_26.md`, manifest `experiments/manifests/eval300_shortlist_oracle_top8_scoreseed2_actual_oscr_2026_06_26.yaml`, W&B `lky54qz5`; bootstrap report `experiments/reports/eval300_top8_actual_oscr_ocr_bootstrap_ci_2026_06_26.md`, manifest `experiments/manifests/eval300_top8_actual_oscr_ocr_bootstrap_ci_2026_06_26.yaml`, W&B `7ubvieta`.

## ADR-031: Source-modulo OOF scores may define a candidate shortlist, not a selector

**Status:** accepted  
**Decision:** Use source-modulo out-of-fold top-8 scores only as a leakage-controlled first-stage candidate-generator diagnostic/substrate. Do not use source-modulo in-train clean-val scores as validation evidence, and do not promote source-modulo top-score policies without a separate no-op-aware selector and counted `.oscr` validation.  
**Rationale:** N117 filters each source-mod score file to the source-index slice that was held out during that model's training. This OOF shortlist improves clean-val first-stage coverage beyond deterministic `topk_rank<=4`: `oof_score_top4` reaches `14/17` exact oracle changes and Tesseract `-21`, while `topk4_union_oof_top4` reaches `15/17` and `-22` versus deterministic rankcap4 at `12/17` and `-17`. However, these numbers use an oracle chooser inside the shortlist and do not solve no-op/change acceptance.  
**Consequence:** The next selector should operate on the N117 OOF shortlist, include nearest/no-op explicitly, and be judged by policy-level false/wrong/missed groups before any actual `.oscr` export. Existing ADR-029/ADR-030 still block threshold sweeps over non-OOF or in-train score families.  
**Evidence:** `experiments/reports/eval300_top8_source_mod_oof_shortlist_2026_06_26.md`, manifest `experiments/manifests/eval300_top8_source_mod_oof_shortlist_2026_06_26.yaml`, W&B `lz93tl1o`.

## ADR-032: Separate change proposal from no-op risk on the OOF shortlist

**Status:** accepted  
**Decision:** Do not keep treating nearest/no-op as an ordinary candidate in the same shallow multiclass or candidate-probability selector over the N117 OOF shortlist. Future N117-based policies should separate non-nearest change proposal/ranking from a calibrated risk-abstention decision, or add materially richer candidate-local evidence before rejoining no-op and candidate choice.  
**Rationale:** N118 showed that logistic, HistGradient and random-forest nearest/no-op-inclusive selectors cannot safely consume the N117 shortlist headroom. N119 audits the failure mode: the topk4-union-OOF-top4 shortlist covers `15/17` validation oracle changes, but for all three rebuilt N118 models the covered oracle candidate scores are below nearest in `15/15` cases. Random forest can rank the oracle above other wrong non-nearest candidates in `8/15` cases, so the useful signal is not entirely absent; it is being suppressed at the nearest/no-op boundary.  
**Consequence:** The next experiment should first design a policy-level risk or abstention objective around the N117 shortlist, preferably with hard-positive mining and explicit false/wrong costs, or add candidate-local evidence that estimates OCR-useful non-nearest changes before comparing to nearest. Do not seed/capacity sweep the N118 shallow selector family.  
**Evidence:** `experiments/reports/eval300_top8_oof_shortlist_policy_selector_2026_06_26.md`, manifest `experiments/manifests/eval300_top8_oof_shortlist_policy_selector_2026_06_26.yaml`, W&B `ehwksv4h`; follow-up audit `experiments/reports/eval300_top8_oof_shortlist_policy_failure_audit_2026_06_26.md`, manifest `experiments/manifests/eval300_top8_oof_shortlist_policy_failure_audit_2026_06_26.yaml`, W&B `p5okugfr`.

## ADR-033: Weighted scalar proposal tuning is exhausted for N117/N120

**Status:** accepted  
**Decision:** Stop class-weight and hard-positive/no-op-negative weight sweeps over the current N117 OOF shortlist scalar proposal features. Keep the N120 two-stage structure only as a scaffold for richer proposal evidence, not as a knob-tuning target.  
**Rationale:** N120 validates that separating non-nearest proposal from no-op risk can recover a safe partial policy (`2/17`, Tesseract `-5`, PARSeq `0`, no false/wrong), but N121 shows risk scores are capped at two exact accepts under zero-bad validation. N122 then sweeps 36 proposal weighting settings and still cannot exceed that safe `-5` policy; higher proposal rank1 settings create risk-gated false/wrong pressure or do not improve accepted exact changes.  
**Consequence:** The next assignment-selector work should introduce materially richer candidate-local code-effect evidence, listwise/pairwise proposal objectives with new representations, or a new first-stage shortlist source. Do not continue by changing scalar sample weights on the same OOF/rate/codebook/image-diff feature set.  
**Evidence:** N120 report `experiments/reports/eval300_top8_oof_shortlist_two_stage_risk_2026_06_26.md`, W&B `kf6ekni7`; N121 audit `experiments/reports/eval300_top8_oof_two_stage_risk_miss_audit_2026_06_26.md`, W&B `ll9iwhpy`; N122 report `experiments/reports/eval300_top8_oof_weighted_proposal_risk_2026_06_26.md`, manifest `experiments/manifests/eval300_top8_oof_weighted_proposal_risk_2026_06_26.yaml`, W&B `87vmfylo`.

## ADR-034: Do not widen the N117 latent-window proposal path

**Status:** accepted  
**Decision:** Do not continue by seed/capacity/epoch/dropout sweeping the high-capacity latent/code/window proposal model over the N117 OOF shortlist, and do not treat the N123 latent-window proposal score as a promotable selector substrate. Keep N120's two-stage separation as a useful scaffold, but pivot to a simpler source-robust anchor, a different verification objective, or substantially more clean supervision before adding more local latent capacity.  
**Rationale:** N123 adds candidate-local latent windows, raw code/latent vectors and deployable table features to the non-nearest proposal stage while keeping teacher/OCR outputs excluded from inference features. Despite materially richer evidence, the model overfits train to near-zero listwise loss and weakens validation proposal rank1 to `4/17` compared with N120's `6/17`; the ensemble all-proposal policy has `58` false changes and `13` wrong changes, and train-tuned risk gates collapse to no-op across all budgets. This is the same structural failure as earlier high-capacity verifier/ranker diagnostics: more local latent evidence without a stronger source-robust objective increases memorization, not safe utility.  
**Consequence:** Future assignment-selector work should favor low-variance priors, cross-fit/source-robust calibration, or an explicitly policy-level verifier that can use candidate-local evidence without turning no-headroom groups into false changes. N123 should be cited as evidence against "just make the latent-window proposal larger."  
**Evidence:** N123 report `experiments/reports/eval300_top8_oof_latent_proposal_risk_2026_06_26.md`, manifest `experiments/manifests/eval300_top8_oof_latent_proposal_risk_2026_06_26.yaml`, W&B `rlldzgi3`.

## ADR-035: N117 proposal anchors are no longer the main bottleneck to tune

**Status:** accepted  
**Decision:** Stop tuning deterministic rank/score-fusion proposal anchors and shallow N120-style risk gates over the N117 OOF shortlist. Keep N117 as a candidate-generation upper-control, but the next promotable path must change the verification/calibration objective or add substantially more clean no-headroom supervision.  
**Rationale:** N124 tests the low-variance alternative to N123: deterministic topk/rate, OOF-rank, reciprocal-rank and fused anchors. The topk/rate anchors restore proposal rank1 to `6/17`, tying N120's best rank1, but all-proposal use is false-heavy (`58` false groups), and train-tuned risk transfer either leaks false/wrong changes or collapses to no-op. The best Tesseract validation policy is only `-2` with `2/17` exact useful changes plus `3` false and `1` wrong changes, still below the current actual-bitstream `-8` floor and less safe than the current learned stream.  
**Consequence:** The project should now treat proposal-rank improvement alone as insufficient. Future work should directly model no-headroom/false-change risk, use cross-fit calibration with a different target, add larger clean supervision for no-op boundaries, or redesign the OSCAR utility target around policy-level false/wrong costs. More anchors or risk thresholds over the same N117 features are not a productive next step.  
**Evidence:** N124 report `experiments/reports/eval300_top8_oof_deterministic_proposal_risk_2026_06_26.md`, manifest `experiments/manifests/eval300_top8_oof_deterministic_proposal_risk_2026_06_26.yaml`, W&B `lfs66wt4`.

## ADR-036: Source-OOF calibration does not rescue the shallow N117 risk gate

**Status:** accepted  
**Decision:** Stop source-OOF thresholding experiments over the current N120-style shallow risk features for deterministic N117 proposal anchors. Any next calibration work must change the risk representation/target or add substantially more clean no-headroom supervision.  
**Rationale:** N125 calibrates risk thresholds on source-held-out train scores, then applies those thresholds to full-train risk models on validation. This reduces the false/wrong leakage seen in N124 mostly by abstaining: topk and z-fused anchors collapse to no-op, while the best oof-rank/random-forest case reaches only `1/17` exact useful changes with `1` false and `1` wrong change, Tesseract `-1`, PARSeq `0`. It is worse than N120's safe partial policy and far below the actual-bitstream floor.  
**Consequence:** Treat shallow proposal-score/margin risk features as insufficient under source shift. The next work should either design a policy-level verifier with richer no-headroom evidence, generate more clean no-headroom/hard-positive labels, or move back to compact symbol learning where the proposal distribution itself has lower false-change pressure.  
**Evidence:** N125 report `experiments/reports/eval300_top8_oof_source_oof_risk_calibration_2026_06_26.md`, manifest `experiments/manifests/eval300_top8_oof_source_oof_risk_calibration_2026_06_26.yaml`, W&B `whucl72w`.

## ADR-037: Treat N117 verifier work as supervision-limited

**Status:** accepted  
**Decision:** Do not implement another N117/N120 selector unless it explicitly changes the supervision source, the verifier objective, or the candidate representation for no-headroom suppression. The next promotable verifier must be judged against the N126 target: at least `3` exact useful validation changes with near-zero bad moves, because fewer exact accepts cannot beat the current actual-bitstream floor.  
**Rationale:** N126 shows the N117 topk4-union-OOF-top4 shortlist still covers `15/17` validation oracle-change groups and has a covered-oracle upper control of Tesseract `-22`, but the train/validation label geometry is highly imbalanced. Validation has `58` no-headroom groups, only `15` positive shortlist rows, and `24.8` candidate negatives per positive. Source/modulo breakdowns show the positive signal is concentrated in a few sources/mod slices while no-headroom rows exist everywhere.  
**Consequence:** Future verifier work should be framed as clean supervision acquisition, no-headroom-aware objective design, or compact utility-symbol learning. More score/risk calibration over the same N117 features is not a research step; it is a repeat of a measured imbalance.  
**Evidence:** N126 report `experiments/reports/eval300_top8_verifier_supervision_gap_2026_06_26.md`, manifest `experiments/manifests/eval300_top8_verifier_supervision_gap_2026_06_26.yaml`, W&B `gma1wetq`.

## ADR-038: Use N127 hard cohorts as supervision, not as policy evidence

**Status:** accepted  
**Decision:** The N127 hard-cohort export may be used to train the next N117 verifier, but only as train-only supervision. It must not be reported as validation performance, used as a selector policy, or used to tune thresholds on held-out rows. Any verifier trained from it must include an explicit feature-scrubbing check that excludes OCR strings, references, edit deltas, evaluator outputs and validation rows from inference features.  
**Rationale:** N126 shows the raw N117 shortlist is dominated by no-headroom groups and candidate negatives. N127 reduces that to a usable train-only contrastive set: `60` covered train oracle positives and `162` matched no-headroom negatives (`2.7` negatives per positive), with validation excluded. This supplies cleaner no-headroom supervision without weakening the OCR-protocol separation, provided label/audit fields are not promoted into deployable inputs.  
**Consequence:** The next N117 experiment should train a verifier/objective against N127's hard positives and matched no-headroom negatives, then evaluate on the frozen validation shortlist. Promotion remains blocked unless the table-level validation policy reaches at least `3` exact useful changes with near-zero false/wrong moves and then survives counted `.oscr` compress/decompress.  
**Evidence:** N127 report `experiments/reports/eval300_top8_verifier_hard_cohorts_2026_06_26.md`, rows `experiments/results/eval300_top8_verifier_hard_cohorts_train_2026_06_26.jsonl`, manifest `experiments/manifests/eval300_top8_verifier_hard_cohorts_2026_06_26.yaml`, W&B `zmzewu8d`.

## ADR-039: N127 alone is insufficient for shallow deployable verifiers

**Status:** accepted  
**Decision:** Do not continue by seed/model/threshold sweeping the N128 scrubbed scalar/tree hard-cohort verifier family or the N129 linear matched-pairwise hard-cohort scorer. N127 may remain an auxiliary supervision source, but the next verifier must add a materially different policy-level objective, richer local visual/code-effect evidence, direct no-headroom suppression, or more clean supervision.  
**Rationale:** N128 satisfies the leakage constraints: `227` selected features, no leak-like OCR/reference/oracle features, train-only hard cohorts, and train source-mod OOF thresholding. Despite that, the best validation policy is no-op (`0/17` exact useful changes, false/wrong `0`, Tesseract `0`, PARSeq `0`). Forcing all groups to change shows the model has not learned a safe useful-change score: it creates `58` false changes and `14-15` wrong changes, with only `2-3/17` exact useful changes. N129 changes the objective to matched pairwise ranking and reaches `1.0` train pairwise accuracy, but the transferred policies still collapse to no-op or a false validation change.  
**Consequence:** The remaining problem is not just label imbalance, feature scrubbing, or row-vs-pairwise loss; it is a policy/objective and evidence problem. Future work should explicitly optimize group-level false/wrong/missed tradeoffs or return to compact utility-symbol learning where the proposal distribution itself carries lower false-change pressure. No counted `.oscr` export should be attempted from N128/N129.  
**Evidence:** N128 report `experiments/reports/eval300_top8_hard_cohort_verifier_2026_06_26.md`, result `experiments/results/eval300_top8_hard_cohort_verifier_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_hard_cohort_verifier_2026_06_26.yaml`, W&B `98v72lq6`; N129 report `experiments/reports/eval300_top8_hard_cohort_pairwise_verifier_2026_06_26.md`, result `experiments/results/eval300_top8_hard_cohort_pairwise_verifier_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_hard_cohort_pairwise_verifier_2026_06_26.yaml`, W&B `y61bs9uu`.

## ADR-040: Compact optional symbol streams are required but not a selector fix

**Status:** accepted  
**Decision:** The next OSCAR utility-symbol implementation should combine gate/index/code information into a compact optional section or otherwise amortize optional-section headers before any low-bitrate claim. However, do not treat this byte-packing work as a substitute for solving candidate utility selection and no-headroom suppression.  
**Rationale:** N130 audits existing counted `.oscr` streams for current, top-4 oracle and top-8 oracle policies across seed0/1/2. All three have identical actual bpp (`3.849649`) and enhancement payload bpp (`0.028989`), confirming OCR differences are selector choices rather than extra payload. For selected word-crop streams, the logical enhancement is only about `3` bytes on average, while optional section table and fixed gate/code headers lift the selected optional total to `103` bytes. A compact combined optional section would save about `0.064242` bpp over the 225-stream dataset for this pattern. N131 then implements a `TEXT_SYMBOLS` compact codebook section and confirms on 5 actual MLIC++ latent-codebook streams that split `[2,10,12]` and compact `[2,14]` streams preserve base payloads and reconstructions exactly while saving `60` bytes per selected stream. N132 scales this to the current learned Eval300 seed0/1/2 streams, N133 regenerates top-4/top-8 oracle streams, and N134 adds the nearest-assignment baseline. At identical selected-stream budgets, compact packing lowers mean actual total bpp from `3.849649` to `3.774026`, enhancement payload bpp from `0.028989` to `0.003781`, and selected optional bytes from `103` to `43` for nearest/current/top4/top8 alike. N135 then joins these matched compact rates with existing OCR bootstrap evidence.  
**Consequence:** Future actual-rate comparisons should report both logical utility-symbol bytes and container overhead. Compact packing now has Eval300-scale evidence for the nearest baseline, current learned policy and top4/top8 oracle upper bounds and should be used for future reported OSCAR-LIC codebook streams. The research bottleneck after N126-N129 remains verifier/objective quality; selector claims still require held-out OCR and counted compress/decompress evidence. Any new selected policy still needs its own compact actual-byte accounting before rate claims.  
**Evidence:** N130 report `experiments/reports/eval300_oscr_section_accounting_current_top4_top8_2026_06_26.md`, result `experiments/results/eval300_oscr_section_accounting_current_top4_top8_2026_06_26.json`, manifest `experiments/manifests/eval300_oscr_section_accounting_current_top4_top8_2026_06_26.yaml`, W&B `po2m40fu`; N131 report `experiments/reports/mlicpp_latent_codebook_compact_section_smoke5_2026_06_26.md`, result `experiments/results/mlicpp_latent_codebook_compact_section_smoke5_compare_2026_06_26.json`, manifest `experiments/manifests/mlicpp_latent_codebook_compact_section_smoke5_2026_06_26.yaml`, W&B `8wefk30c`; N132 report `experiments/reports/eval300_current_compact_section_accounting_2026_06_26.md`, result `experiments/results/eval300_current_compact_section_accounting_2026_06_26.json`, manifest `experiments/manifests/eval300_current_compact_section_accounting_2026_06_26.yaml`, W&B `ll3kk3zf`; N133 report `experiments/reports/eval300_compact_section_accounting_current_top4_top8_2026_06_26.md`, result `experiments/results/eval300_compact_section_accounting_current_top4_top8_2026_06_26.json`, manifest `experiments/manifests/eval300_compact_section_accounting_current_top4_top8_2026_06_26.yaml`, W&B `cz3ts55g`; N134 report `experiments/reports/eval300_compact_section_accounting_nearest_current_top4_top8_2026_06_26.md`, result `experiments/results/eval300_compact_section_accounting_nearest_current_top4_top8_2026_06_26.json`, manifest `experiments/manifests/eval300_compact_section_accounting_nearest_current_top4_top8_2026_06_26.yaml`, W&B `fjr97j6p`; N135 report `experiments/reports/eval300_compact_rate_ocr_summary_nearest_current_top4_top8_2026_06_26.md`, result `experiments/results/eval300_compact_rate_ocr_summary_nearest_current_top4_top8_2026_06_26.json`, manifest `experiments/manifests/eval300_compact_rate_ocr_summary_nearest_current_top4_top8_2026_06_26.yaml`, W&B `ruuwcyk6`.

## ADR-041: Center-level utility priors are diagnostic, not a deployable assignment policy

**Status:** accepted  
**Decision:** Do not continue by sweeping scalar center-utility-biased code assignment thresholds or treating codebook-center utility as the main selector. Future code-index assignment must be candidate-conditioned or policy-level, with explicit no-headroom/false-change suppression and actual compact `.oscr` byte accounting.  
**Rationale:** N136 regenerated the utility-weighted K64 train codebooks with valid nonzero `center_utility` metadata and tested utility-biased assignment on the held-out Eval300 val75 splits with compact `TEXT_SYMBOLS` streams. The codebook centers are unchanged from the earlier K64 utility-weighted runs, isolating the diagnostic to assignment choice rather than a new symbol set. Relative-error thresholds `1.01` and `1.05` changed `7/75` and `23/75` codes respectively, but both produced the same held-out OCR outcome: PARSeq unchanged and Tesseract only `-1` unicode char versus nearest, while being `+7` chars worse than the current compact policy at the same mean actual bpp `3.774026` and enhancement payload bpp `0.003781`.  
**Consequence:** Center utility can remain in codebook artifacts for analysis, but it should not drive the next promotion attempt. The next compact-symbol experiment should learn candidate-local utility, no-op risk, or policy-level assignment decisions; a global center prior is too coarse for the OSCAR-LIC marginal-utility claim.  
**Evidence:** N136 report `experiments/reports/eval300_centerutilv2_assignment_compact_2026_06_26.md`, result `experiments/results/eval300_centerutilv2_assignment_compact_2026_06_26.json`, manifest `experiments/manifests/eval300_centerutilv2_assignment_compact_2026_06_26.yaml`, W&B `i0aqw89l`; N137 rank/eligibility audit `experiments/reports/eval300_center_utility_assignment_rank_audit_2026_06_26.md`, result `experiments/results/eval300_center_utility_assignment_rank_audit_2026_06_26.json`, manifest `experiments/manifests/eval300_center_utility_assignment_rank_audit_2026_06_26.yaml`, W&B `4wdrz9ws`.

## ADR-042: Exception centers need no-headroom gating before OCR claims

**Status:** accepted  
**Decision:** Utility-positive residual exception centers are a promising compact-symbol proposal change, but appending them to the codebook is not enough. Future exception-center experiments must include an explicit exception selector or no-headroom verifier that can choose those centers before any OCR evaluation or promotion claim.  
**Rationale:** N138 shows that adding train oracle-change residual exception centers can solve part of the relative-error feasibility bottleneck from N137, but it also creates false-change pressure. K4/K8 exception centers make `16/17` validation oracle-change residuals select an exception center, yet also make `31-36` no-headroom validation groups select a close exception center. K16 is a better feasibility tradeoff, with `9/17` useful oracle residuals and `13` no-headroom close opportunities. N139 then generated actual compact `.oscr` streams for K80=K64+16 under the existing nearest assignment path; all three validation seeds transmitted selected candidates (`75` total) but `assignment_changed_total=0`, so the exception centers were never used.  
**Consequence:** Do not run more nearest-assignment K80/K-plus-center OCR sweeps. The next actual-stream smoke should first prove that a policy selects exception centers on table-level diagnostics without opening no-headroom false changes. Only then should decoded reconstructions, PARSeq/Tesseract row-wise OCR, and compact actual-byte accounting be run.  
**Evidence:** N138 report `experiments/reports/eval300_utility_exception_codebook_feasibility_2026_06_26.md`, result `experiments/results/eval300_utility_exception_codebook_feasibility_2026_06_26.json`, manifest `experiments/manifests/eval300_utility_exception_codebook_feasibility_2026_06_26.yaml`, W&B `ukcxz6kl`; N139 report `experiments/reports/eval300_k80_exception16_compact_stream_audit_2026_06_26.md`, result `experiments/results/eval300_k80_exception16_compact_stream_audit_2026_06_26.json`, manifest `experiments/manifests/eval300_k80_exception16_compact_stream_audit_2026_06_26.yaml`, W&B `bzg5slyt`.

## ADR-043: Distance-only exception-center selection is not source-robust

**Status:** accepted  
**Decision:** Do not promote or OCR-evaluate exception-center policies whose only evidence is residual distance, distance margin or codebook center utility. Future exception-center selection must add candidate-local visual/code-effect evidence or a policy-level verifier that is validated on held-out groups before actual `.oscr` generation.  
**Rationale:** N140 evaluates K80 exception-center selector proxies on the Eval300 train/validation top8 residual artifact. The distance threshold `extra_relative_error <= 1.0` looks excellent on train (`44/63` oracle-change groups covered, `1` no-headroom false, precision `0.9778`) but is false-heavy on validation (`9/17` covered, `13` no-headroom false, precision `0.4091`). Train-tuned thresholds either no-op at false budget 0 or leak `19-45` validation false changes. Logistic, HistGradient and random-forest scores over distance/margin/center-utility features also no-op on validation under train-tuned false budgets. N141 then runs the conservative `rel<=1.0` policy through actual compact `.oscr` and OCR on seed1/2 val150: rate drops because only `22` candidates are selected instead of current's `50`, but PARSeq is unchanged and Tesseract worsens by `+3` unicode / `+6` latin-alnum aggregate chars versus current.  
**Consequence:** The exception-center branch should not proceed by threshold polishing, source-agnostic score calibration, shallow learned distance models, or sparse exception-only selection. The next valid step is a different verifier representation, for example source/nearest/candidate local glyph patches plus latent/code-effect tensors and explicit false-change/replace/drop penalties, and it must first show nonzero useful validation changes with near-zero no-headroom false changes before OCR streams.  
**Evidence:** N140 report `experiments/reports/eval300_exception_center_selector_audit_2026_06_26.md`, result `experiments/results/eval300_exception_center_selector_audit_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_center_selector_audit_2026_06_26.yaml`, W&B `nfck52vf`; N141 report `experiments/reports/eval300_exception_center_rel100_actual_ocr_2026_06_26.md`, result `experiments/results/eval300_exception_center_rel100_actual_ocr_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_center_rel100_actual_ocr_2026_06_26.yaml`, W&B `x7imee7s`.

## ADR-044: Exception centers must be current-preserving replace/drop policies

**Status:** accepted  
**Decision:** Do not treat sparse exception-center selection as the next OSCAR-LIC policy. Future exception-center work must use the current compact stream as the default action and explicitly choose among keep-current, replace-with-exception, drop-current and no-op actions. Lower rate from dropping current selections is a secondary rate-saving result unless OCR is preserved or improved relative to current under actual serialized bytes.  
**Rationale:** N141 shows the conservative exception policy selects only `22` candidates versus current compact's `50`, reducing total bpp from `3.823677` to `3.793834` and enhancement bpp from `0.003645` to `0.001563`, but Tesseract worsens by `+3` unicode and `+6` latin-alnum chars versus current while PARSeq is unchanged. On the selected rows, exception replacements are not net-positive for Tesseract unicode and worsen latin by `+2`. N142's contact sheet confirms most current/exception pairs are visually and OCR-wise neutral, with one improvement and one regression. N145 decomposes the executed policy into current-relative actions: `22` replacements, `28` drops and `100` keep-base rows. Only `5/150` rows are strictly beneficial across Tesseract/PARSeq profiles, while `7` are harmful and `1` is mixed; most aggregate Tesseract harm comes from dropped current selections.  
**Consequence:** The next exception-center experiment should first build a table-level current-preserving action policy with explicit penalties for false replacement and harmful current drops. It must beat or tie current OCR before actual `.oscr` export, then use compact `TEXT_SYMBOLS` streams and complete-file bpp. Do not rerun sparse exception-only OCR policies, distance thresholds or shallow distance/margin/center-utility selectors as promotion attempts.  
**Evidence:** Design plan `docs/EXCEPTION_CENTER_REPLACEMENT_PLAN_2026_06_26.md`; N141 report `experiments/reports/eval300_exception_center_rel100_actual_ocr_2026_06_26.md`, result `experiments/results/eval300_exception_center_rel100_actual_ocr_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_center_rel100_actual_ocr_2026_06_26.yaml`, W&B `x7imee7s`; N142 contact sheet `experiments/figures/eval300_exception_center_rel100_contact_sheet_2026_06_26/selected_tesseract_unicode.png`, manifest `experiments/manifests/eval300_exception_center_rel100_contact_sheet_2026_06_26.yaml`; N145 report `experiments/reports/eval300_exception_center_current_relative_labels_2026_06_26.md`, result `experiments/results/eval300_exception_center_current_relative_labels_2026_06_26.json`, table `experiments/results/eval300_exception_center_current_relative_labels_2026_06_26.jsonl`, manifest `experiments/manifests/eval300_exception_center_current_relative_labels_2026_06_26.yaml`, W&B `tvpq70xz`; N146 report `experiments/reports/eval300_exception_current_relative_feature_gap_2026_06_26.md`, result `experiments/results/eval300_exception_current_relative_feature_gap_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_current_relative_feature_gap_2026_06_26.yaml`, W&B `qhqzghzd`.

## ADR-045: Mine close current-only hard negatives before exception replace/drop training

**Status:** accepted  
**Decision:** Do not train an exception-center replace/drop verifier from the current N143 train split alone. Before verifier training, create or mine train hard negatives that resemble validation current-only drop-risk rows, especially close-to-exception current-only examples and IAM handwriting examples.  
**Rationale:** N148 shows a strong train/validation mismatch among current-only rows. Validation current-only rows are all close to exception centers (`53/53` have `extra_relative_error <= 1.5`, median `1.0733`), while train current-only rows are mostly far away (`14/180` have `extra_relative_error <= 1.5`, median `3445.97`). N145/N146 then show that dropping close current-only validation rows is exactly where aggregate Tesseract harm appears (`+3` unicode, `+4` latin on seed1/2), with IAM carrying most harmful/mixed cases. N151 confirms that mined train-use probes can produce useful supervision: aggregate Tesseract improves at lower bpp with PARSeq unchanged, but `4/32` probe rows are harmful, so a verifier needs those hard negatives rather than only positive/neutral controls. N152 expands the same actual-label path to `53` train-use probes and strengthens the positive signal (`3.332659 -> 3.248499` weighted bpp, Tesseract `-15/-11`, PARSeq unchanged), but still leaves `4/53` harmful rows across SynthText, ICDAR and IAM-like cases. N153 names those four harmful rows and packages image/rate/OCR evidence for them, making them a concrete pre-training acceptance test rather than an aggregate warning. N154 then shows why a single safety threshold is the wrong objective: a zero-harm deployable-feature veto keeps `-13/-12` Tesseract but loses most byte saving, while one/two-harm frontiers reach `-17/-14` with materially better rate. N155 shows that a tiny learned veto over the same 53 rows cannot reproduce that frontier; it collapses to all-veto or no-op under leave-one-out scoring.  
**Consequence:** The next data step is not a larger verifier over the same N143 rows, direct thresholding on exception distance, or a seed/model sweep over the 53-row learned veto. It is a hard-negative/counterfactual labeling step plus a stronger objective/representation: generate or select close current-only train probes, include IAM-heavy and short/dark text cases, and label current-vs-drop/replacement behavior under actual compact streams before fitting a deployable replace/drop policy. Any diagnostic verifier should report the N154 safety/utility frontier, not only a single aggregate OCR delta.  
**Evidence:** N148 report `experiments/reports/eval300_exception_current_only_distribution_gap_2026_06_26.md`, result `experiments/results/eval300_exception_current_only_distribution_gap_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_current_only_distribution_gap_2026_06_26.yaml`, W&B `rsngelv2`; N151 report `experiments/reports/eval300_exception_counterfactual_probe_actual_ocr_2026_06_26.md`, result `experiments/results/eval300_exception_counterfactual_probe_actual_ocr_2026_06_26.json`, label table `experiments/results/eval300_exception_counterfactual_probe_actual_ocr_2026_06_26.jsonl`, manifest `experiments/manifests/eval300_exception_counterfactual_probe_actual_ocr_2026_06_26.yaml`, W&B `jvzqm6hf`; N152 report `experiments/reports/eval300_exception_counterfactual_probe_actual_ocr_v2_rel2_2026_06_26.md`, result `experiments/results/eval300_exception_counterfactual_probe_actual_ocr_v2_rel2_2026_06_26.json`, label table `experiments/results/eval300_exception_counterfactual_probe_actual_ocr_v2_rel2_2026_06_26.jsonl`, manifest `experiments/manifests/eval300_exception_counterfactual_probe_actual_ocr_v2_rel2_2026_06_26.yaml`, W&B `y6x0wjkp`; N153 report `experiments/reports/eval300_exception_probe_harmful_evidence_2026_06_26.md`, result `experiments/results/eval300_exception_probe_harmful_evidence_2026_06_26.json`, table `experiments/results/eval300_exception_probe_harmful_evidence_2026_06_26.jsonl`, manifest `experiments/manifests/eval300_exception_probe_harmful_evidence_2026_06_26.yaml`, W&B `whghgit7`; N154 report `experiments/reports/eval300_exception_probe_veto_rule_search_2026_06_26.md`, result `experiments/results/eval300_exception_probe_veto_rule_search_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_probe_veto_rule_search_2026_06_26.yaml`, W&B `8skettkp`; N155 report `experiments/reports/eval300_exception_probe_veto_diagnostic_2026_06_26.md`, result `experiments/results/eval300_exception_probe_veto_diagnostic_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_probe_veto_diagnostic_2026_06_26.yaml`, W&B `vwffygur`; N145/N146 reports as above.

## ADR-046: Use N152 rel<=2 as the active exception probe frontier

**Status:** accepted  
**Decision:** Treat N152 rel<=2 as the active exception replace/drop probe frontier for the next verifier design. Keep N156 rel<=4 as auxiliary hard-negative/neutral evidence and contact-sheet material, but do not promote it to the main training/evaluation frontier or keep widening distance thresholds for more rows.  
**Rationale:** N156 expands the actual-labeled probe set from N152's `53` training-use rows to `63`, but the additional rows are mostly neutral. Aggregate Tesseract remains positive (`-14` unicode / `-11` latin) and PARSeq unchanged, yet labels shift only from N152's `16` beneficial / `33` neutral / `4` harmful to N156's `17` beneficial / `41` neutral / `5` harmful. The zero-harm deployable-feature veto weakens slightly (`-12/-12`, seven beneficial rows vetoed) compared with N152/N154 (`-13/-12`, six beneficial rows vetoed). N157 then confirms the mechanism directly: all 53 N152 rows are preserved, and the 10 added rel<=4 rows are only `1` beneficial, `8` neutral and `1` harmful; added rows alone save `430` bytes but worsen Tesseract unicode by `+1`. The one/two-harm frontiers are informative, but they do not justify replacing N152 as the sharper substrate.  
**Consequence:** The next useful step is richer current-preserving verifier design or deliberately targeted new hard-negative labels, not wider rel threshold sampling. Any learned verifier should still be judged against the N154/N156 harm-budget frontiers and actual compact OCR against current before promotion.  
**Evidence:** N156 actual OCR report `experiments/reports/eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26.md`, result `experiments/results/eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26.yaml`, W&B `8pfcdtsv`; N156 harmful evidence `experiments/reports/eval300_exception_probe_harmful_evidence_v3_rel4_2026_06_26.md`, W&B `pxz0ztxx`; N156 veto frontier `experiments/reports/eval300_exception_probe_veto_rule_search_v3_rel4_2026_06_26.md`, W&B `oui8k7ce`; N157 expansion comparison `experiments/reports/eval300_exception_probe_expansion_v2_to_v3_rel4_comparison_2026_06_26.md`, result `experiments/results/eval300_exception_probe_expansion_v2_to_v3_rel4_comparison_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_probe_expansion_v2_to_v3_rel4_comparison_2026_06_26.yaml`, W&B `uprhtlh3`; N152/N154 evidence as in ADR-045.

## ADR template

```text
## ADR-###: Title
Status: proposed | accepted | superseded | rejected
Context:
Decision:
Alternatives:
Consequences:
Evidence:
Date:
```

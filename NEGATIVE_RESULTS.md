# Negative and Invalid Results

The 2026-06-25 research pass produced several valid negative controls. Keep them visible so future work does not chase attractive row metrics that fail actual policy evaluation.

## N001 — Row AUC/AP did not predict OCR policy gain
- Date: 2026-06-25
- Result: multiple assignment rankers reached high validation AUC/AP but worsened policy-level Tesseract or introduced many false changes.
- Evidence: `experiments/reports/eval300_assignment_tensor_reranker_diagnostics_2026_06_25.md`
- Lesson: promote only through actual `.oscr` reconstruction, complete-byte bpp and row-wise OCR comparison.

## N002 — Flat latent/context widening was misleading
- Date: 2026-06-25
- Result: flat 512-D latent concatenation, simple latent side branch and wide residual-coordinate tables did not beat the promoted actual-bitstream gate; some variants were harmful despite strong AP.
- Evidence: `docs/HANDOFF.md`, `experiments/results/eval300_assignment_tensor_reranker_diagnostics_summary_2026_06_25.json`
- Lesson: use structured local latent/token context with explicit no-op calibration, not scalar/coordinate feature widening.

## N003 — Standalone structured-window CNN was not enough
- Date: 2026-06-25
- Result: structured local latent-window artifacts were finite and useful as a substrate, but h96/h64 window-CNN policies reached only about `-4.33` / `-4.0` mean Tesseract edits with many false changes.
- Evidence: W&B `s325nhay`, `experiments/manifests/oscar_lic_12h_repro_manifest_2026_06_25.yaml`
- Lesson: next model needs a group-aware no-op head and policy calibration, not only a pooled local window branch.

## N004 — Scalar no-change penalties and margin gates were insufficient
- Date: 2026-06-25
- Result: no-change weight `2.0` and post-hoc margin gates over window scores failed to beat the promoted `-8/-8` Tesseract baseline at comparable precision.
- Evidence: W&B `gcyb3dth`, `l1z3244s`
- Lesson: no-op should be learned as an explicit group decision tied to the row scorer.

## N005 — Group/no-op window v0 did not hide a strong seed
- Date: 2026-06-25
- Result: expanding the structured-window group/no-op v0 from 3 to 10 model seeds did not reveal a promotion candidate. The best fixed-threshold seed reaches only `-5` Tesseract edits versus the promoted actual-bitstream `-8`, and mean false changes remain high (`12.5` at threshold `0.5`).
- Evidence: `experiments/reports/eval300_group_noop_assignment_followup_2026_06_25.md`, W&B `9z4trvnq`
- Lesson: do not spend more seed sweeps on this architecture without a stronger representation or calibration split.

## N006 — Strong-improvement and reward-softmax targets were harmful
- Date: 2026-06-25
- Result: supervising the group head only on strong Tesseract improvements (`<= -2`) missed nearly all validation oracle changes, while Tesseract reward-softmax row supervision increased false/wrong changes and worsened Tesseract.
- Evidence: `experiments/reports/eval300_group_noop_assignment_followup_2026_06_25.md`, W&B `9z4trvnq`
- Lesson: scalar target reshaping is not enough; row assignment needs better local glyph/code-effect evidence and no-harm calibration.

## N007 — Leave-real-seed-out calibration did not rescue the code-only group/no-op head
- Date: 2026-06-25
- Result: using `group_keys[:,0]` as a leave-real-seed-out diagnostic kept large oracle headroom (`-29` to `-40` Tesseract edits), but learned policies recovered only a few edits and retained false/wrong changes.
- Evidence: `experiments/reports/eval300_group_noop_assignment_followup_2026_06_25.md`, W&B `qwbu6asw`
- Lesson: cross-split calibration is necessary for future claims, but the current row scorer still lacks enough candidate-conditioned evidence.

## N008 — Tree rankers increased row AUC/AP without policy gain
- Date: 2026-06-25
- Result: ExtraTrees and HistGradientBoosting over summarized code/latent/window features reached validation AUC/AP around `0.94/0.88`, but still failed policy-level OCR gains. ExtraTrees mostly selected no-op; budgeted policies produced many false changes and no useful Tesseract improvement.
- Evidence: `experiments/reports/eval300_group_noop_assignment_followup_2026_06_25.md`, W&B `nay55x83`
- Lesson: non-neural tabular nonlinearity does not solve the assignment problem; promotion must remain tied to policy-level OCR and actual `.oscr` tests.

## N009 — Candidate-only BCE row supervision did not improve assignment precision
- Date: 2026-06-25
- Result: restricting row BCE to non-nearest candidates with positive weighting improved the window variant's validation AUC/AP to `0.767/0.431`, but the best train-tuned policy recovered only `-4.0` Tesseract edit characters while sending `10.33` false changes on average.
- Evidence: `experiments/reports/eval300_group_noop_assignment_followup_2026_06_25.md`, W&B `o882h3gb`
- Lesson: the no-op row is not the sole cause of failure; assignment needs candidate-conditioned evidence for safe OCR effect, not only candidate-positive reweighting.

## N010 — Mixed real-seed cross-fit policy gains vanished on clean val rows
- Date: 2026-06-25
- Result: real-seed cross-fit train-tuned group-gate policies looked strong on mixed 100-group held-out tables, but filtering them to the clean Eval300 val75-compatible rows reduced the best three-shard Tesseract delta to only `-3` edits with PARSeq unchanged. Fixed threshold `0.5` filtered policies reached only `-2` at best.
- Evidence: `experiments/reports/eval300_group_gate_clean_val_filter_2026_06_25.md`, W&B `36wwlrx8`, `experiments/results/eval300_group_gate_clean_val_policy_filter_summary_2026_06_25.json`
- Lesson: do not launch counted `.oscr` promotion for cross-fit exports before a clean split-compatible table precheck. Mixed train+val group diagnostics can overstate deployable gain.

## N011 — Fixed promoted gate plus alternate reranker did not improve the policy
- Date: 2026-06-25
- Result: keeping the promoted pooled image-diff gate's 5 changed groups fixed and swapping only the code choice to h96/h128 code-effect or group/no-op reranker scores tied at best (`-8` Tesseract edits, PARSeq unchanged) and did not beat the promoted actual-bitstream gate. The gate-conditioned top4 oracle is only `-9`.
- Evidence: `experiments/reports/eval300_hybrid_imgdiff_gate_codeeffect_2026_06_25.md`, W&B `qd8ol5zr`, `experiments/results/eval300_assignment_hybrid_imgdiff_gate_codeeffect_h96_seed0_9_2026_06_25.json`
- Lesson: do not spend more effort swapping rankers only inside the old high-precision gate's change set. The next gain must safely expand the change set while preserving low false/wrong-change rates.

## N012 — Score-consensus thresholds did not safely expand the change set
- Date: 2026-06-25
- Result: tuning score, margin and vote thresholds over h96/h128/group-noop reranker seeds on train groups failed on clean Eval300 val groups. The best variants reached only `-5` or `-4` Tesseract edits and produced 12-22 false changes, even when train false-change allowance was set to zero.
- Evidence: `experiments/reports/eval300_score_consensus_selector_2026_06_25.md`, W&B `rm9fssf0`, `experiments/results/eval300_assignment_score_consensus_h96_seed0_9_false0_2026_06_25.json`
- Lesson: agreement among current reranker scores is not enough for a deployable selector. The next selector needs stronger image/latent/text-aware evidence, not only score post-processing.

## N013 — Scalar stroke/projection image features weakened the group gate
- Date: 2026-06-25
- Result: adding dark/ink fractions, row/column projections and dark/edge-region difference features raised some classifier-looking signals but reduced policy-level gain to `-5` Tesseract edits, with more false/wrong changes than the promoted `-8` image-diff gate.
- Evidence: `experiments/reports/eval300_imgstroke_group_gate_2026_06_25.md`, W&B `t4m1mhub`, `experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgstroke_seed0_1_2_2026_06_25.json`
- Lesson: adding more scalar image statistics to the current group gate is not enough. Future improvements need candidate-local evidence or supervision that directly targets safe extra oracle-change groups.

## N014 — Direct PARSeq teacher features tied but did not improve the promoted gate
- Date: 2026-06-25
- Result: PARSeq teacher confidence/edit-distance features produced two model seeds that tie `-8` Tesseract edits, but they require 7 changed groups and 3 false changes, worse than the promoted actual-bitstream gate's 5 changed groups and 1 false change.
- Evidence: `experiments/reports/eval300_parseq_teacher_feature_gate_2026_06_25.md`, W&B `csp3watg` and `fcd0lfon`, `experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_parseqfeat_seed0_1_2_2026_06_25.json`
- Lesson: OCR-teacher evidence is relevant, but direct inference-time PARSeq features are not a promotion path without distillation and encoder-cost accounting.

## N015 — PARSeq teacher target distillation did not calibrate a safe group gate
- Date: 2026-06-25
- Result: Distilling PARSeq teacher edit/norm-edit/confidence-gain targets into the deployable pooled image-diff group gate failed policy evaluation. Fixed thresholds `0.5/0.8` collapsed to no-op, while train-tuned thresholds recovered only `-3` Tesseract edits and introduced about `15` false changes, `4` wrong changes and `4` Tesseract-worsen groups.
- Evidence: `experiments/reports/eval300_parseq_teacher_target_distill_2026_06_25.md`, W&B `ibzb6tu4`, `experiments/results/eval300_assignment_group_gate_parseq_teacher_confgain_target_e100_seed0_1_2_2026_06_25.json`
- Lesson: Teacher OCR evidence must be distilled at candidate-local/code-effect level with explicit no-op precision. Group-level scalar target swapping is not enough, even when validation AUC is moderate.

## N016 — Candidate-local PARSeq teacher row scoring did not transfer to safe OCR policy
- Date: 2026-06-25
- Result: A small row MLP trained on PARSeq teacher-derived targets with deployable scalar codebook/image-diff features reached about `0.58` val AUC and `0.46` AP for confidence-style teacher labels, but policy-level behavior failed. Unconstrained policies changed nearly all groups and produced 54-58 false changes; train-tuned precision constraints collapsed to no-op or near no-op.
- Evidence: `experiments/reports/eval300_teacher_row_scorer_2026_06_25.md`, W&B `ifaqyaeh`, `experiments/results/eval300_assignment_teacher_row_scorer_confgain_seed0_1_2_2026_06_25.json`
- Lesson: Candidate-local teacher distillation needs richer deployable local evidence such as image/code-effect patches or latent-window tensors plus explicit group no-op precision. Scalar row scoring is not enough.

## N017 — Latent-window PARSeq teacher targets worsened assignment policy
- Date: 2026-06-25
- Result: Adding PARSeq teacher target arrays to the structured latent-window code-effect artifact did not improve policy selection. The top4 artifact still has oracle headroom (`-15` Tesseract edits), but models trained on `teacher_parseq_combo_gain` or `teacher_parseq_conf_gain` worsen Tesseract by `+7.67` to `+9.33` edits under argmax and remain worse after margin gating.
- Evidence: `experiments/reports/eval300_window_teacher_target_2026_06_25.md`, W&B `c4mctgjh`, `experiments/results/eval300_assignment_codeeffect_window_parseqteacher_confgain_seed0_1_2_2026_06_25.json`
- Lesson: Richer latent-window evidence does not make PARSeq confidence gain a good primary assignment target. Use actual safe assignment labels as the main target and treat OCR-teacher signals, if any, as auxiliary regularization.

## N018 — Top8 latent-window actual-label reranker still failed policy precision
- Date: 2026-06-25
- Result: Exporting a nearest-plus-top8 latent-window artifact fixed shortlist recall (`0` oracle-missing groups), but the actual-label latent-window reranker still failed promotion. Row AUC/AP reached `0.8918/0.6124`, yet argmax recovered only `-4` Tesseract edits with `11.67` false changes, and the best margin precheck reached only `-6` edits with `6` false changes.
- Evidence: `experiments/reports/eval300_top8_window_oraclelabel_precheck_2026_06_25.md`, W&B `vyfi6rhr`, `experiments/results/eval300_assignment_codeeffect_window_top8_oraclelabels_margin_2026_06_25.json`
- Lesson: Candidate recall was a real bottleneck, but fixing it is not enough. The next model needs explicit calibrated change/no-change or policy-level false-change control, not row argmax plus post-hoc margin.

## N019 — Top8 group/no-op window head overfit train and failed clean val
- Date: 2026-06-25
- Result: The explicit group/no-op window head on the top8 artifact reached high row metrics and perfectly fit train oracle-change groups under the selected threshold, but failed clean val. Default threshold `0.5` gives only `-4` Tesseract edits with `11.33` false changes; train-tuned threshold falls to `-2.67` edits with `13.67` false changes.
- Evidence: `experiments/reports/eval300_top8_group_noop_window_2026_06_25.md`, W&B `vb3tnf4r`, `experiments/results/eval300_group_noop_window_top8_default_seed0_1_2_2026_06_25.json`
- Lesson: Simply adding a group/no-op head to the top8 latent-window model is not enough. Retrying needs cross-fit calibration, a direct policy-level false-change loss, or more data.

## N020 — Top8 policy errors show the current group head is not the bottleneck
- Date: 2026-06-25
- Result: Clean-val policy error analysis shows the row-only top8 latent-window model and explicit group/no-op head fail almost identically. Group/no-op has `12` missed oracle, `12` false change, `2` wrong change and `3` exact oracle groups; row-only has `12` missed oracle, `11` false change, `2` wrong change and `3` exact oracle groups.
- Evidence: `experiments/reports/eval300_top8_policy_error_analysis_2026_06_25.md`, W&B `klj2113f`, `experiments/results/eval300_group_noop_window_top8_default_seed0_policy_errors_2026_06_25.json`, `experiments/results/eval300_codeeffect_window_top8_oraclelabels_seed0_policy_errors_2026_06_25.json`
- Lesson: Do not rerun the same row-argmax/margin or group/no-op architecture. The next top8 model must explicitly train no-headroom/rank-0 suppression and hard-positive recall, preferably with cross-fit calibration or more data.

## N021 — Tail-margin losses did not solve top8 policy precision
- Date: 2026-06-25
- Result: Direct rank0/no-headroom and hard-positive margin losses preserved high row AUC/AP but did not improve clean-val assignment policy enough. The best threshold-0.5 variant reaches only mean `-4.33` Tesseract edits with `12.0` false changes and `3.33` wrong changes; rank0-heavy and code-only variants are weaker.
- Evidence: `experiments/reports/eval300_top8_tail_margin_diagnostics_2026_06_25.md`, W&B `d7y7j8dn`, `experiments/results/eval300_group_noop_window_top8_tailmargin_r1_hp1_seed0_1_2_2026_06_25.json`
- Lesson: Do not simply increase tail-margin weights. The failure likely needs more labeled groups, a separate high-precision no-headroom rejector, source/difficulty-aware calibration, or richer candidate-local evidence judged by policy errors.

## N022 — Source-disjoint top8 split exposed poor assignment generalization
- Date: 2026-06-25
- Result: The default top8 group/no-op model becomes harmful when validation is source-index-disjoint (`source_index % 5 == 4`). Threshold `0.5` worsens Tesseract by `+2.33` edits and PARSeq by `+1.0`, with `6.67` false changes and `5.67` wrong changes, despite row AP `0.6233`.
- Evidence: `experiments/reports/eval300_top8_source_mod_split_diagnostic_2026_06_25.md`, W&B `x1to33x8`, `experiments/results/eval300_group_noop_window_top8_source_mod5_r4_seed0_1_2_2026_06_25.json`
- Lesson: Current Eval300 labels are too small/correlated for robust assignment generalization. Treat source-disjoint splits as diagnostics and prioritize more data or stronger calibration before increasing model capacity.

## N023 — Eval600 patch-listwise gate recovers little oracle headroom
- Date: 2026-06-25
- Result: Eval600 adds 9600 no-overlap single-candidate rows and a strong Tesseract oracle (`0.4181 -> 0.3591` full-split CER), but the existing `tesseract_parseq_safe` patch-listwise gate recovers only a tiny validation gain. Across three internal group splits, train-tuned selection preserves PARSeq but improves Tesseract CER only `0.4162 -> 0.4139` at higher bpp, far below the oracle (`0.3605` on the corresponding validation splits).
- Evidence: `experiments/reports/eval600_utility_table_patch_gate_2026_06_25.md`, W&B `x4brl6zw`, `experiments/results/candidate_patch_listwise_gate_tess_parseq_safe_eval600_noeval300_h32_do20_e400_seed0_1_2_2026_06_25.json`
- Lesson: More one-candidate rows help avoid residual-RMS harm but do not solve high-precision selection. Do not spend the next cycle on seed sweeps of this same gate; use fixed external-split evaluation, high-precision no-headroom rejection, source/difficulty calibration, or compact-symbol/top8 assignment training.

## N024 — Eval600-to-Eval300 external patch gate still captures little oracle headroom
- Date: 2026-06-25
- Result: Training the same `tesseract_parseq_safe` patch-listwise gate on the Eval300-disjoint Eval600 table and evaluating on fixed Eval300 preserves PARSeq but remains too weak for promotion. Train-tuned/constrained selection improves Eval300 Tesseract CER only `0.4326 -> 0.4257` at `3.9339` actual single-candidate bpp, while the raw single-candidate oracle reaches `0.3672` at `4.3483` bpp. Fixed threshold `0.5` is slightly harmful (`0.4328` Tesseract CER).
- Evidence: `experiments/reports/eval600_to_eval300_external_patch_gate_2026_06_25.md`, W&B `lykbfyhg`, `experiments/results/candidate_patch_listwise_gate_external_eval600_to_eval300_tess_parseq_safe_h32_do20_e400_seed0_1_2_2026_06_25.json`
- Lesson: Fixed external-split support is now available, and it confirms that this one-candidate patch-gate family should not be promoted or seed-swept. Future work needs stronger no-headroom/source calibration, more data, or compact-symbol/top8 assignment objectives with explicit false-change control.

## N025 — Aggregate no-headroom proxies are too coarse for safe selection
- Date: 2026-06-25
- Result: Eval600-trained group-level no-headroom/headroom proxies transfer weakly to Eval300. Image/base-reconstruction aggregate features have nontrivial signal (`image_only` AUC about `0.65`, AP about `0.35-0.36` versus `0.233` prevalence), but top-ranked groups still contain many harmful candidates. Adding PARSeq base features or source one-hot improves AP only modestly and remains diagnostic, not deployable.
- Evidence: `experiments/reports/eval600_to_eval300_no_headroom_proxy_2026_06_25.md`, W&B `22stj9mv`, `experiments/results/candidate_no_headroom_proxy_eval600_to_eval300_2026_06_25.json`
- Lesson: The no-headroom hypothesis is real, but aggregate features alone are too coarse. Do not bolt this proxy onto the current patch gate; use it to design explicit false-change/no-headroom supervision for richer top8/compact-symbol candidate evidence.

## N026 — Eval600 top-8 assignment labels did not transfer without hard no-headroom negatives
- Date: 2026-06-26
- Experiment ID: exploratory G5 assignment transfer diagnostic
- Hypothesis: External Eval600 top-8 assignment labels can train the current group/no-op latent-window assignment head and recover Eval300 seed1 assignment headroom.
- Configuration: Train on `experiments/features/eval600_oraclesafe123_plus_eval300_seed1val_top8_windows_2026_06_25.npz` with Eval600 groups; validate on Eval300 seed1 clean-val groups. Tested `oracle_choice`, `safe_tesseract_soft_bce`, and score-consensus selectors.
- Result: Eval300 seed1 oracle headroom is large (`-24` Tesseract edit characters, PARSeq unchanged, 17 changed groups), but learned policies remain unsafe. `oracle_choice` threshold 0.5 reaches only mean `-2.33` Tesseract edits with `16.67` false changes and `5.00` wrong changes. Train-tuned `oracle_choice` is worse (`-1.67` edits, `22.67` false changes). `safe_tesseract_soft_bce` either produces many false changes at threshold 0.5 or collapses to near no-op when train-tuned. Consensus reaches at best `-5` edits but emits `28/75` false changes.
- Why it failed or was negative: The Eval600 training substrate was selected from oracle-safe candidate groups and underrepresents hard images where no assignment should be changed. Row AUC/AP remains insufficient for policy promotion under the no-headroom/false-change bottleneck.
- Valid negative result or invalid run: valid negative result.
- Evidence paths: `experiments/reports/eval600_to_eval300_top8_assignment_external_transfer_2026_06_26.md`, W&B offline runs `3vj4lavk`, `sulkrjq6`, `1m472jil`, `ndreov1l`.
- Should it be retried? Only after adding Eval600 hard no-headroom negative groups to the top-8 assignment substrate or changing the objective to train directly on a source/difficulty-disjoint high-precision no-change criterion.

## N027 — Hard no-headroom negatives alone did not make the current top-8 assignment head promotable
- Date: 2026-06-26
- Experiment ID: `E_eval600_hard_noheadroom_top8_assignment_transfer_2026_06_26`
- Hypothesis: Adding Eval600 hard no-headroom negatives to oracle-safe top-8 assignment labels will make the current group/no-op latent-window head safe enough for Eval300 promotion.
- Configuration: Selected 120 Eval600 no-safe-source, PARSeq-exact, Tesseract-distance<=2 hard no-headroom groups; only 4/120 have an assignment oracle code change. Merged them with 123 Eval600 oracle-safe groups and Eval300 seed1 clean-val groups.
- Result: This improves row metrics and reduces false changes, but still fails promotion. `oracle_choice` val AUC/AP improves to `0.8932/0.6410` and false changes at threshold 0.5 drop to `9.33`, but Tesseract delta remains only `-2.0` versus the `-24` oracle and promoted `-8` baseline. Soft utility collapses to no-op when train-tuned. Hard-positive margins recover at most a seed-level `-6` but with `11+` false changes and unstable seeds.
- Why it failed or was negative: The data fix addresses no-headroom rejection but the current head cannot simultaneously recover hard positives and keep the false-change floor low. Loss reweighting reopens too many no-change groups.
- Valid negative result or invalid run: valid negative result for this architecture/objective; partial positive data result for hard-negative substrate construction.
- Evidence paths: `experiments/reports/eval600_hard_noheadroom_top8_assignment_transfer_2026_06_26.md`, manifest `experiments/manifests/eval600_hard_noheadroom_top8_assignment_transfer_2026_06_26.yaml`, W&B `x94exiuz`, `3vzcvcb3`, `5mtvr2r1`, `ljgxm9kc`, `u5mrf5q3`.
- Should it be retried? Not as another seed sweep or scalar margin tweak. Retry only with source/difficulty-matched pairwise contrast, policy-level false-change budget training, or richer candidate-local code-effect evidence.

## N028 — Matched pairwise losses did not make the hard-negative-balanced top-8 assignment head promotable
- Date: 2026-06-26
- Experiment ID: `E_eval600_hardneg_pairwise_assignment_objective_2026_06_26`
- Hypothesis: Source/difficulty-matched positive-vs-hard-negative pairwise losses will improve the current group/no-op head enough to recover Eval300 seed1 assignment oracle headroom while keeping false changes low.
- Configuration: Train on `experiments/features/eval600_oraclesafe123_hardnoheadroom120_plus_eval300_seed1val_top8_windows_2026_06_26.npz` with 123 oracle-safe Eval600 groups, 120 hard no-headroom Eval600 groups and 75 Eval300 seed1 validation groups. Tested group+row matched pairwise weights (`0.5/0.5`) and row-heavy matched pairwise weight (`2.0`) using the updated group/no-op trainer.
- Result: Pairwise group+row improves validation row AUC/AP to `0.8961/0.6962`, but threshold `0.5` reaches only `-1.0` Tesseract edit characters with `8.0` false changes; train-tuned selection reaches only `-1.33` edits with `11.33` false changes. Row-heavy pairwise reaches at best `-2.33` with `12.67` false changes, and score-consensus is either too conservative (`-1`) or harmful (`+4`). The non-deployable top-M upper control is useful diagnostically: top-4 contains 10-13/17 oracle changes (`-15` to `-18` Tesseract edits), but top-1/gated policies remain weak.
- Why it failed or was negative: Matched contrast improves shortlist quality but does not solve final selection/no-op calibration. The current single-score head still cannot convert top-4 candidate recall into a high-precision deployed policy.
- Valid negative result or invalid run: valid negative result for matched pairwise losses on the current single-score group/no-op architecture; partial positive evidence for a top-4 shortlist plus explicit no-op/policy-budget selector.
- Evidence paths: `experiments/reports/eval600_hardneg_pairwise_assignment_objective_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_pairwise_assignment_objective_2026_06_26.yaml`, W&B `nfy2esqe`, `4yx9140t`, `ebb7tvge`, `semgweeh`, `lpfpu1kd`, `79v3dcbn`.
- Should it be retried? Do not rerun the same pairwise weights or seed-sweep this single-score head. Retry only with an explicit top-k shortlist/no-op policy model, policy-level false/wrong-change constraints, or richer local code-effect evidence, and validate externally before counted `.oscr` promotion.

## N029 — Post-hoc policy-budget selection did not transfer safely from Eval600 to Eval300
- Date: 2026-06-26
- Experiment ID: `E_eval600_hardneg_policy_budget_selector_2026_06_26`
- Hypothesis: A top-k/no-op policy-budget selector tuned on hard-negative-balanced Eval600 score files can convert useful shortlist coverage into a safer Eval300 policy without retraining the scorer.
- Configuration: Added `scripts/analyze_assignment_policy_budget_selector.py`. Tuned shortlist size, ranking metric and change budget on Eval600 train groups under PARSeq-safe selection and train-side false/wrong/Tesseract-worsen constraints, then evaluated by score-threshold transfer and budget-fraction transfer on Eval300 seed1.
- Result: Train scores are perfectly separable under the budget criterion (`70/243` train changes, `-98` Tesseract edits, zero false/wrong/worsen), but Eval300 transfer fails. Oracle-choice strict threshold gives 12 changes with `0` Tesseract net gain and 6 false / 3 wrong / 3 worsened groups; fraction transfer gives `-2` edits but 12 false / 5 wrong / 5 worsened groups. Pairwise strict threshold collapses to no-op, while fraction transfer gives only `-1` with 14 false / 4 wrong. Capped and top4-only budgets remain weak; best table delta is `-2`, still with 7+ false changes.
- Why it failed or was negative: The current score distributions are externally miscalibrated. Train-side budget tuning can select true positives, but the learned score/no-op boundary does not transfer to Eval300 precision requirements.
- Valid negative result or invalid run: valid negative result for post-hoc budgeted selection over these score files.
- Evidence paths: `experiments/reports/eval600_hardneg_policy_budget_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_policy_budget_selector_2026_06_26.yaml`, W&B `l52j5hun`, `guj1r9l7`.
- Should it be retried? Not with the same score files and threshold/budget post-processing. Retry only after adding richer deployable candidate-local code-effect evidence, source/domain-disjoint calibration data, or a genuinely different policy architecture.

## N030 — IAM source-balanced supplement20 did not fix Eval300 assignment calibration
- Date: 2026-06-26
- Experiment ID: `E_eval600_hardneg_iam_supplement20_assignment_transfer_2026_06_26`
- Hypothesis: Adding a small IAM-only source-balanced calibration supplement to the Eval600 hard-negative top-8 assignment substrate will improve the current group/no-op latent-window head's Eval300 no-op boundary.
- Configuration: Added `--include-source-name` to `scripts/select_assignment_hard_noheadroom_candidates.py`; selected 20 additional IAM groups after excluding the prior oracle-safe123 and hard-noheadroom120 selections. Generated 160 top-8 K64 utility-weighted reconstructions, ran PARSeq and Tesseract PSM8 OCR, built an assignment table, exported compatible radius-1 latent windows and merged the supplement into the 318-group hard-negative-balanced transfer artifact for 338 groups total.
- Result: The supplement is relevant but too small. It contains 5/20 PARSeq-safe Tesseract-improving assignment-change groups, with oracle Tesseract delta `-6` and PARSeq delta `0`. Training still fails Eval300 policy use: validation AUC/AP is `0.8966/0.6642`, but threshold `0.5` has mean Tesseract delta `0.0` with `9.67` false changes, and train-tuned thresholds reach only `-1.0` Tesseract edit with `11.67` false changes.
- Why it failed or was negative: Small source balancing does not repair the external calibration boundary for the current single-score head. The model still overfits train separation and cannot convert row metrics into a high-precision no-op/change decision on Eval300.
- Valid negative result or invalid run: valid negative result for tiny source-specific supplementation of the current architecture/objective.
- Evidence paths: `experiments/reports/eval600_hardneg_iam_supplement20_assignment_transfer_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_iam_supplement20_assignment_transfer_2026_06_26.yaml`, W&B `m9hf0uo6`, `8eej9xt7`, `iikeyxg2`.
- Should it be retried? Not as another tiny per-source append or seed sweep. Retry only with larger source/domain-balanced calibration data, source-disjoint validation, richer candidate-local code-effect evidence, or a different no-op architecture.

## N031 — Balanced210 no-safe calibration data still did not rescue the current assignment head
- Date: 2026-06-26
- Experiment ID: `E_eval600_hardneg_balanced210_assignment_transfer_2026_06_26`
- Hypothesis: A substantially larger source-balanced no-safe Eval600 supplement will give the current group/no-op latent-window assignment head enough calibration data to transfer safely to Eval300.
- Configuration: Selected 210 remaining Eval600 no-safe groups after excluding oracle-safe123, hard-noheadroom120 and IAM supplement20; required PARSeq base exactness, Tesseract base distance <= 6 and max 40 groups per source. Generated 1680 top-8 K64 assignment reconstructions, ran PARSeq/Tesseract OCR, built the assignment table, exported radius-1 latent windows and merged them into the previous 338-group artifact for 548 groups total.
- Result: The supplement is useful data but not a promotion path. It contains 19/210 assignment oracle-change groups and 76 safe improving rows. PARSeq remains exact over all 1680 candidates and Tesseract PSM8 has CER `0.2672`. The trained head still fails Eval300 policy use: threshold `0.5` is slightly harmful on average (`+0.33` Tesseract edits with `7.67` false changes), and train-tuned thresholds average only `-1.33` edits with `10.0` false changes. Val AUC/AP remains high (`0.8969/0.6807`), again showing row metrics do not imply safe policy.
- Why it failed or was negative: More same-kind no-safe calibration labels do not fix the current single-score decision boundary. The model still cannot separate hard positives from false changes at the precision needed to beat the promoted actual-bitstream baseline.
- Valid negative result or invalid run: valid negative result for larger source-balanced no-safe supplementation of the current architecture/objective.
- Evidence paths: `experiments/reports/eval600_hardneg_balanced210_assignment_transfer_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_assignment_transfer_2026_06_26.yaml`, W&B `cucaosnh`, `lkvvfxzo`, `5q05gb1z`.
- Should it be retried? Not by adding more similar no-safe groups to this head. Retry with a different decision structure, such as a separate no-op rejector, top-k verifier, policy-level verifier, or richer candidate-local code-effect evidence.

## N032 — Score-only top-k verifier does not transfer from Eval600 to Eval300
- Date: 2026-06-26
- Experiment ID: `E_eval600_hardneg_balanced210_topk_score_verifier_2026_06_26`
- Hypothesis: A second-stage verifier over top-k assignment shortlist score features can turn the useful Eval300 top-M upper recall into a safer deployed policy.
- Configuration: Added `scripts/train_assignment_topk_score_verifier.py`. The verifier uses only deployable first-stage score aggregates, nearest margins, model-seed vote fractions, rank/top-k positions and group-probability aggregates. OCR deltas are used only for labels, train-threshold constraints and evaluation. Tested top-2 MLP, top-4 MLP, top-4 linear and top-8 MLP-fast strict policies over the N031 score JSONL.
- Result: The top-M upper is still strong (`top2=-15`, `top4=-17`, `top8=-24` Tesseract edits on Eval300 seed1), but the learned verifier is externally anti-calibrated. Eval300 candidate AUC/AP is only `0.2858/0.0517` for top-2, `0.3212/0.0328` for top-4 MLP, `0.2560/0.0270` for top-4 linear and `0.3464/0.0259` for top-8. Learned policy transfer never improves Tesseract: top-4 MLP averages `+0.33` edits and top-8 averages `+2.33` edits with false changes.
- Why it failed or was negative: A separate verifier changes the decision structure, but because it only sees the current score/rank/vote features it inherits the same Eval600-to-Eval300 calibration failure. Shortlist recall is not the bottleneck; deployable code-effect evidence is.
- Valid negative result or invalid run: valid negative result for score-only top-k verifier variants. One full-grid top-8 run was interrupted and excluded; the reported top-8 evidence is the completed fast strict run.
- Evidence paths: `experiments/reports/eval600_hardneg_balanced210_topk_score_verifier_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_topk_score_verifier_2026_06_26.yaml`, W&B `blrnzrcc`.
- Should it be retried? Not with the same score JSONL features, thresholds, budgets, consensus, linear heads or shallow MLP heads. Retry only with richer candidate-local code-effect evidence or source/domain-disjoint calibration labels.

## N033 — External image-diff group gate does not rescue Eval600-to-Eval300 assignment transfer
- Date: 2026-06-26
- Experiment ID: `E_eval600_hardneg_balanced210_imgdiff_group_gate_external_2026_06_26`
- Hypothesis: The lightweight reconstructed-image/code-effect features that helped the Eval300 pooled image-diff group gate can act as a stronger no-op rejector when trained on Eval600 hard-negative-balanced top-8 assignment labels and transferred to Eval300 seed1.
- Configuration: Added `scripts/prepare_external_assignment_table.py` to merge the 4 Eval600 assignment tables with Eval300 seed1 rows and repair `real_seed` keys. Augmented the merged 548-group, 4384-row table with `img_*` features, then trained `scripts/train_assignment_group_gate.py` with image features for assignment scorer seeds 0/1/2 and gate seeds 0/1/2.
- Result: The image-diff gate itself does not transfer: Eval300 gate AUC/AP is only `0.4980/0.2267`. Fixed thresholds `0.5/0.8` recover no oracle changes and give Tesseract deltas of `0`, `+1`, `+1` for assignment scorer seeds 0/1/2. Train-tuned gates are no better than accepting the bad scorer top rows: seed0 reaches only `-2` edits with 5 false and 1 wrong change, while seeds1/2 worsen by `+2` edits.
- Why it failed or was negative: The group gate can reject bad changes, but it cannot promote oracle codes that the current scorer does not rank first. External image-diff group features are not enough unless the candidate-level code-effect scorer/reranker itself improves.
- Valid negative result or invalid run: valid negative table-level result; no actual `.oscr` stream was produced because the policy is below no-op/promoted baselines.
- Evidence paths: `experiments/reports/eval600_hardneg_balanced210_imgdiff_group_gate_external_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_imgdiff_group_gate_external_2026_06_26.yaml`, W&B `f2omvl4p`.
- Should it be retried? Not as another group-gate threshold or seed sweep over the same scorer. Retry image-diff evidence only as part of a candidate-level scorer/reranker or with source/domain-disjoint calibration.

## N034 — External diff-crop candidate scorer raises row AUC but harms policy
- Date: 2026-06-26
- Experiment ID: `E_eval600_hardneg_balanced210_external_diffcrop_candidate_scorer_2026_06_26`
- Hypothesis: Moving deployable reconstructed-image/code-effect evidence into the candidate-level scorer, via local diff-crops and `img_*` features, can recover Eval300 seed1 top-8 assignment oracle choices after Eval600 hard-negative-balanced training.
- Configuration: Trained `scripts/train_assignment_patch_ranker.py` on `experiments/results/eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl` with `--image-mode diff_crop --include-image-features`, target `oracle_choice`, `bce_listwise`, model seeds 0/1/2, and Eval300 seed1 as fixed external validation.
- Result: Row metrics look strong (`val_auc=0.8791`, `val_ap=0.6051`), but deployed policies are harmful. Always-best changes `7.33` groups on average and worsens Tesseract by `+1.67` edit characters with zero improving groups; top-changed-3 still worsens by `+1.0`; train-tuned conservative threshold worsens by `+0.67`. PARSeq is unchanged, but the Eval300 oracle remains `-24` Tesseract edits across 17 groups.
- Why it failed or was negative: Candidate-local diff-crop evidence can classify rows, but the resulting scores are not calibrated for safe external policy selection. The model selects only neutral or harmful changes on Eval300 while missing all useful oracle-change groups.
- Valid negative result or invalid run: valid negative result for this candidate-level diff-crop scorer; no counted `.oscr` promotion stream was produced because table-level policy is worse than no-op.
- Evidence paths: `experiments/reports/eval600_hardneg_balanced210_external_diffcrop_candidate_scorer_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_external_diffcrop_candidate_scorer_2026_06_26.yaml`, W&B `xem245nk`.
- Should it be retried? Not as another seed, threshold or small-capacity sweep. Retry only with a policy-level safety objective, an explicit no-op/verifier stage over richer non-score candidate evidence, and source/domain-disjoint external calibration.

## N035 — Safe-improve target does not rescue the external diff-crop scorer
- Date: 2026-06-26
- Experiment ID: `E_eval600_hardneg_balanced210_external_diffcrop_safeimprove_candidate_scorer_2026_06_26`
- Hypothesis: Replacing `oracle_choice` supervision with the stricter `label_tesseract_parseq_safe_improves` target can align the diff-crop candidate scorer with safe OCR policy behavior.
- Configuration: Reused the N034 diff-crop candidate scorer setup and 548-group image-diff table, but trained with `--target safe_improve`, `bce_listwise`, model seeds 0/1/2 and fixed Eval300 seed1 external validation.
- Result: Train AUC/AP remains high (`0.9977/0.9699`), but Eval300 candidate AUC/AP collapses to `0.5186/0.0676`. Policy behavior is worse than no-op: always-best changes 23 groups and worsens Tesseract by `+3.33`; train-tuned conservative threshold changes 21 groups and worsens by `+3.0`; top-changed-3 still worsens by `+1.0`. PARSeq remains unchanged, while the oracle remains `-24` Tesseract edits.
- Why it failed or was negative: Rare safe-improvement labels fit Eval600 but do not transfer to Eval300 with this representation/objective. The scorer becomes aggressive enough to select a few improving rows, but harmful rows dominate the net policy effect.
- Valid negative result or invalid run: valid negative result for safe-improve target swapping on the external diff-crop candidate scorer.
- Evidence paths: `experiments/reports/eval600_hardneg_balanced210_external_diffcrop_safeimprove_candidate_scorer_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_external_diffcrop_safeimprove_candidate_scorer_2026_06_26.yaml`, W&B `k5xbgn7o`.
- Should it be retried? Not as another label, seed, threshold or capacity sweep. Retry safe-improve supervision only inside an explicit policy-level selector or uncertainty-aware verifier with source/domain-disjoint calibration.

## N037 — First learned top-4 policy selector fails despite strong shortlist oracle
- Date: 2026-06-26
- Experiment ID: `E_eval600_hardneg_balanced210_top4_policy_selector_2026_06_26`
- Hypothesis: A selector over nearest plus top-4 candidates from the oracle-choice and safe-improve diff-crop scorers, with explicit no-op and harmful-candidate margin penalty, can convert the strong top-4 shortlist into safe Eval300 policy gain.
- Configuration: Added `scripts/train_assignment_topk_policy_selector.py`. The candidate set is nearest/no-op plus the union of top-4 rows from N034 and N035 score files. The selector uses deployable code/image-diff/scorer features, listwise oracle-choice supervision and a train-side harmful-candidate margin. Ran 3 score-model seeds x 3 selector seeds on the fixed Eval600 -> Eval300 seed1 split.
- Result: The shortlist oracle is strong (`-19.33` Tesseract edits, PARSeq `0`, `14.33/17` exact oracle changes, `2.67` missed full-oracle changes), but the learned selector fails. Argmax and train-tuned margin both worsen Tesseract by `+1.11` edits on average, with `9.56` false changes, `9.56` wrong changes and `0` exact oracle changes.
- Why it failed or was negative: The candidate set has most of the needed information, but the learned selection boundary over in-sample Eval600 score features does not transfer. The selector learns to choose non-oracle non-nearest candidates rather than abstaining or selecting useful shortlist rows.
- Valid negative result or invalid run: valid negative result for this first top-4 MLP selector; positive diagnostic for the union shortlist upper bound.
- Evidence paths: `experiments/reports/eval600_hardneg_balanced210_top4_policy_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_top4_policy_selector_2026_06_26.yaml`, W&B `33xqtxjk`.
- Should it be retried? Do not seed-sweep this MLP selector. Retry only with better calibrated high-precision selection, such as source/domain-disjoint calibration, uncertainty-aware abstention, or a simpler rule learned from out-of-fold score features.

## N038 — Diff-crop budget rules do not transfer without out-of-fold score calibration
- Date: 2026-06-26
- Experiment ID: `E_eval600_hardneg_balanced210_diffcrop_budget_selector_2026_06_26`
- Hypothesis: A strict high-precision budget/threshold rule over the N034/N035 diff-crop score files can avoid the MLP selector's overfitting and transfer useful top-4 shortlist choices to Eval300.
- Configuration: Reused `scripts/analyze_assignment_policy_budget_selector.py` with N034 and N035 score JSONLs as voters. Tuned top-m, rank metric, budget and threshold on Eval600 train groups under zero false-change, wrong-change, Tesseract-worsen and PARSeq-worsen constraints; evaluated threshold and budget-fraction transfer on Eval300 seed1.
- Result: Train tuning is perfect-looking: top-4 `score_mean` selects 76/473 changes, Tesseract delta `-100`, and zero false/wrong/worsen events. Eval300 transfer selects no exact oracle changes. Threshold transfer changes 2 groups with Tesseract delta `0` and 2 false changes; fraction transfer changes 12 groups with Tesseract delta `+1`, 12 false changes and 1 Tesseract-worsen group.
- Why it failed or was negative: The score boundary is calibrated to in-sample Eval600 score outputs. It does not identify Eval300 oracle changes even with strict train-side safety constraints.
- Valid negative result or invalid run: valid negative result for budget/threshold rules over these in-sample diff-crop score files.
- Evidence paths: `experiments/reports/eval600_hardneg_balanced210_diffcrop_budget_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_hardneg_balanced210_diffcrop_budget_selector_2026_06_26.yaml`, W&B `qay9jfvo`.
- Should it be retried? Not over the same in-sample score files. Retry score-based selection only after generating out-of-fold/cross-fit scores or using a source/domain-disjoint calibration set.

## N039 — Same-Eval300 leave-real-seed-out calibration still does not rescue the top-4 selector
- Date: 2026-06-26
- Experiment ID: `E_eval300_realseed_lro_top4_policy_selector_2026_06_26`
- Hypothesis: If calibration labels come from the same Eval300 distribution but different `real_seed` shards, the score-seed-2 top-4 selector can recover the shortlist oracle without Eval600-to-Eval300 calibration shift.
- Configuration: Added `scripts/train_assignment_topk_policy_selector_split_diagnostic.py` as a non-mutating companion to the historical top-4 selector script. It trains on two Eval300 `real_seed` shards and validates on the held-out shard, using nearest plus the union of top-4 rows from N034/N035 scores. Corrected run uses score model seed `2` and selector seeds `0,1,2`.
- Result: The shortlist oracle remains useful (`-6.67` Tesseract edits per 25-group shard on average, PARSeq `0`, `5.0` exact oracle changes, `0.67` missed changes), but the learned selector is negative. Argmax and train-tuned margin both average `+0.33` Tesseract edits, only `0.11` exact oracle changes, `1.67` false changes and `1.67` wrong changes. Per split, the useful `real_seed=1/2` oracle headroom is `-7/-13`, while learned argmax gives `+0.67/+0.33`.
- Why it failed or was negative: The failure is not only Eval600-to-Eval300 distribution shift. Even near-distribution calibration labels do not make the current shallow score-feature MLP identify hard-positive code changes or abstain reliably.
- Valid negative result or invalid run: valid negative result for the corrected score-seed-2 run. The initial JSON `eval300_realseed_lro_top4_policy_selector_scoreseed2_2026_06_26.json` is retained as an invalid diagnostic artifact because argparse append defaults unintentionally mixed score seeds `0/1/2/2`.
- Evidence paths: `experiments/reports/eval300_realseed_lro_top4_policy_selector_2026_06_26.md`, manifest `experiments/manifests/eval300_realseed_lro_top4_policy_selector_2026_06_26.yaml`, W&B `vtyajbnv`.
- Should it be retried? Do not retry shallow selectors over the same score/rank features, even with same-domain real-seed calibration. Retry only with out-of-fold first-stage scores, richer candidate-local deployable code-effect evidence, explicit uncertainty/no-op policy objectives, or source/domain-disjoint high-precision calibration.

## N040 — Small diff-crop patch selector is still unsafe under Eval300 leave-real-seed-out calibration
- Date: 2026-06-26
- Experiment ID: `E_eval300_realseed_lro_top4_patch_selector_2026_06_26`
- Hypothesis: Adding deployable candidate-local visual evidence, via nearest/candidate/difference diff-crop patches, can rescue the score-seed-2 top-4 selector under same-Eval300 leave-real-seed-out calibration.
- Configuration: Added `scripts/train_assignment_topk_patch_selector_split_diagnostic.py`. It keeps the N039 top-4 shortlist, adds a small CNN over `96x32` grayscale diff-crops, concatenates existing scalar score/code/image features, and trains the same listwise oracle-choice target with harmful-candidate margin. Ran score model seed `2`, selector seeds `0,1,2`, validation real-seeds `0,1,2`.
- Result: The patch selector is slightly less bad than N039 but remains a valid negative result. Shortlist oracle stays at `-6.67` Tesseract edits per shard, PARSeq `0`, `5.0` exact changes. Patch argmax/train-tuned margin average `+0.11` Tesseract edits, `0.33` exact oracle changes, `1.89` false changes and `1.89` wrong changes. `real_seed=1` has a weak positive `-0.33` average, but `real_seed=2` worsens to `+0.67`.
- Why it failed or was negative: Local visual code-effect patches add a small amount of useful signal, but not enough to control false/wrong non-nearest changes or recover hard positives. The margin remains unhelpful, matching argmax.
- Valid negative result or invalid run: valid negative result for this small patch-CNN selector; no actual `.oscr` promotion stream was produced.
- Evidence paths: `experiments/reports/eval300_realseed_lro_top4_patch_selector_2026_06_26.md`, manifest `experiments/manifests/eval300_realseed_lro_top4_patch_selector_2026_06_26.yaml`, W&B `ly4wl7xd`.
- Should it be retried? Do not capacity-sweep this small patch-CNN selector as the next primary path. Retry candidate-local visual evidence only with a stronger policy-level false-change objective, out-of-fold first-stage scores, richer code-effect representation, or source/domain-disjoint calibration.

## N041 — Source-OOF first-stage scores do not rescue the top-4 selector
- Date: 2026-06-26
- Experiment ID: `E_eval600_source_oof_to_eval300_top4_selector_2026_06_26`
- Hypothesis: Replacing in-sample Eval600 first-stage score features with source-modulo out-of-fold scores will remove the N037/N038 calibration artifact and let the top-4 selector transfer safely to Eval300 seed1.
- Configuration: Added `scripts/train_assignment_patch_ranker_source_oof_scores.py` and `scripts/merge_assignment_oof_scores.py`. Trained OOF diff-crop rankers on Eval600 `source_index % 5` folds for `oracle_choice` and `safe_improve`, merged OOF Eval600 scores with external Eval300 score rows, then trained the same score-seed-2 top-4 selector on Eval600 and evaluated Eval300 seed1.
- Result: OOF score calibration is more honest but still not promotable. The top-4 shortlist oracle remains strong on Eval300 (`-20` Tesseract edits, 15 exact oracle changes, PARSeq unchanged), but the OOF selector argmax averages `+0.67` Tesseract edits with `4.33` false/wrong changes and only `0.33` exact oracle changes. Train-tuned margin is conservative but misses all useful changes: `+0.33` Tesseract edits, `1.0` false/wrong changes, `0` exact oracle changes.
- Why it failed or was negative: In-sample score calibration was a real flaw, but not the full bottleneck. With OOF first-stage scores, the current second-stage objective still cannot learn a high-precision hard-positive detector or a safe no-op boundary from the available score/patch scalar evidence.
- Valid negative result or invalid run: valid negative result for OOF score calibration as the only change; no counted `.oscr` promotion stream was produced.
- Evidence paths: `experiments/reports/eval600_source_oof_to_eval300_top4_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_source_oof_to_eval300_top4_selector_2026_06_26.yaml`, W&B `mew0ag4h`, `14y9ta7l`, `w74f4anb`.
- Should it be retried? Do not rerun this OOF selector as a seed, threshold or small-capacity sweep. Reuse the OOF score infrastructure only inside a genuinely different policy-level objective, uncertainty-aware abstention model, source/domain-disjoint calibration setup, or richer candidate-local code-effect representation.

## N042 — OCR reward-policy loss does not rescue the source-OOF top-4 selector
- Date: 2026-06-26
- Experiment ID: `E_eval600_source_oof_to_eval300_top4_reward_policy_selector_2026_06_26`
- Hypothesis: Keeping the N041 source-OOF top-4 shortlist but replacing oracle-choice classification with a train-side OCR edit reward/cost objective will directly suppress false changes and recover hard positives.
- Configuration: Added `scripts/train_assignment_topk_reward_policy_selector.py`. The reward gives nearest/no-op zero value, starts non-nearest value from negative Tesseract delta, penalizes non-improving/false changes, strongly penalizes PARSeq worsening, and adds a Tesseract-worsen penalty. Trained selector seeds `0,1,2` on Eval600 with score model seed `2` OOF inputs and evaluated Eval300 seed1.
- Result: The reward oracle remains strong and is slightly better than the label oracle (`-21` vs `-20` Tesseract edits, PARSeq unchanged), but the learned policy is harmful. Argmax averages `+4.0` Tesseract edits with `14.33` false/wrong changes and only `0.33` exact oracle changes. The original safety margin collapses to no-op, while reward-tuned margin still worsens by `+3.33` edits with `10.33` false/wrong changes.
- Why it failed or was negative: The reward objective preserves headroom but does not make the current scalar score/rank/image-diff evidence separable enough. The selector either over-selects harmful non-nearest rows or abstains entirely.
- Valid negative result or invalid run: valid negative result for changing only the loss/objective over the same source-OOF top-4 feature set; no counted `.oscr` promotion stream was produced.
- Evidence paths: `experiments/reports/eval600_source_oof_to_eval300_top4_reward_policy_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_source_oof_to_eval300_top4_reward_policy_selector_2026_06_26.yaml`, W&B `s7wqdw5n`.
- Should it be retried? Do not sweep small reward weights over this same feature set. Retry only after changing the deployable evidence materially, such as richer candidate-local code-effect representation, calibrated uncertainty-aware abstention, or a verifier that sees more than scalar score/rank features.

## N043 — Source-OOF score budget rules collapse to no-op
- Date: 2026-06-26
- Experiment ID: `E_eval600_source_oof_to_eval300_top4_budget_selector_2026_06_26`
- Hypothesis: The strict train-zero-false budget rule rejected in N038 may work if the score boundary is tuned on source-modulo OOF first-stage scores instead of in-sample Eval600 scores.
- Configuration: Reused `scripts/analyze_assignment_policy_budget_selector.py` with N041 mixed OOF oracle-choice and safe-improve scores. Tuned top-m `2/3/4` and score/margin/vote rank metrics under zero false-change, wrong-change, Tesseract-worsen and PARSeq-worsen train constraints, then transferred by threshold and by budget fraction to Eval300 seed1.
- Result: OOF calibration removes the fake train confidence but also removes useful recall. The tuned rule selects only `1/473` train groups for `-2` Tesseract edits and zero false/wrong/worsen events. On Eval300, both threshold and fraction transfer select zero changes, giving no improvement and missing all 17 oracle-change groups.
- Why it failed or was negative: Score-only OOF abstention is safe but too conservative. It cannot identify Eval300 hard positives at the precision/recall tradeoff needed for promotion.
- Valid negative result or invalid run: valid negative result for source-OOF score budget/threshold rules; no counted `.oscr` promotion stream was produced.
- Evidence paths: `experiments/reports/eval600_source_oof_to_eval300_top4_budget_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_source_oof_to_eval300_top4_budget_selector_2026_06_26.yaml`, W&B `p4nohfm9`.
- Should it be retried? Do not rerun score-only budget/threshold selectors over the same OOF score files. Retry only with richer candidate-local evidence or a verifier that materially changes the input signal.

## N044 — Source-OOF external small patch selector remains unsafe
- Date: 2026-06-26
- Experiment ID: `E_eval600_source_oof_to_eval300_top4_patch_selector_2026_06_26`
- Hypothesis: Combining source-OOF first-stage scores with candidate-local nearest/candidate/difference image patches may provide richer deployable evidence than scalar score/rank features and recover Eval300 hard positives.
- Configuration: Added `scripts/train_assignment_topk_patch_selector_external_diagnostic.py` as a non-mutating companion to the N040 patch diagnostic. It trains a small 3-layer CNN over `96x32` diff-crops plus scalar selector features on all non-Eval300-seed1 groups and validates on Eval300 seed1, using score model seed `2` OOF inputs and selector seeds `0,1,2`.
- Result: The shortlist oracle remains strong (`-20` Tesseract edits, 15 exact oracle changes, PARSeq unchanged), but the patch selector is still unsafe. Argmax averages `+3.0` Tesseract edits with `5.33` false/wrong changes and `0` exact oracle changes. Train-tuned margin is more conservative but still worsens by `+1.0` with `1.33` false/wrong changes and `0` exact oracle changes.
- Why it failed or was negative: The small scratch diff-crop CNN makes the policy less aggressive than the reward/scalar selectors, but it still cannot distinguish hard positives from false non-nearest changes under external calibration.
- Valid negative result or invalid run: valid negative result for source-OOF plus small local patch evidence; no counted `.oscr` promotion stream was produced.
- Evidence paths: `experiments/reports/eval600_source_oof_to_eval300_top4_patch_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_source_oof_to_eval300_top4_patch_selector_2026_06_26.yaml`, W&B `hdxd4tkj`.
- Should it be retried? Do not capacity-sweep this same scratch patch-CNN family. Retry only with materially stronger evidence, such as structured MLIC++ latent/codebook features under a different calibration scheme, or a fixed pretrained visual encoder with recorded weight provenance and license risk.

## N045 — Fixed AlexNet patch embeddings reduce aggressiveness but recover no oracle changes
- Date: 2026-06-26
- Experiment ID: `E_eval600_source_oof_to_eval300_top4_alexnet_patch_selector_2026_06_26`
- Hypothesis: Replacing the scratch patch-CNN in N044 with a fixed pretrained visual encoder over nearest/candidate/difference diff-crops may provide stronger candidate-local evidence for the source-OOF top-4 selector.
- Configuration: Added `scripts/train_assignment_topk_alexnet_patch_selector.py`. It freezes torchvision AlexNet ImageNet1K V1 features from the preexisting local checkpoint `/root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`, concatenates those patch embeddings with scalar selector features, trains selector seeds `0,1,2` on all non-Eval300-seed1 groups, and validates on Eval300 seed1. The AlexNet checkpoint SHA256 is `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` and is recorded in `baselines/checkpoint_manifest.yaml`; license terms still require manual publication review.
- Result: The shortlist oracle remains strong (`-20` Tesseract edits, 15 exact oracle changes, PARSeq unchanged), but the learned policy is still negative. Argmax and train-tuned margin both average `+0.67` Tesseract edits, `3.0` false/wrong changes and `0` exact oracle changes. PARSeq remains unchanged.
- Why it failed or was negative: Generic ImageNet visual features make the policy less aggressive than the scratch patch selector, but they still do not identify the OCR-useful code changes. This suggests the missing evidence is OCR/text/code-effect specific rather than generic natural-image recognition.
- Valid negative result or invalid run: valid negative result for fixed AlexNet patch embeddings as the only representation change; no counted `.oscr` promotion stream was produced.
- Evidence paths: `experiments/reports/eval600_source_oof_to_eval300_top4_alexnet_patch_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_source_oof_to_eval300_top4_alexnet_patch_selector_2026_06_26.yaml`, W&B `idwilglc`.
- Should it be retried? Do not capacity-sweep this AlexNet selector. Retry fixed pretrained evidence only if switching to an OCR/text-specific verifier with reviewed provenance, or combine it with OSCAR-LIC-native local latent/codebook code-effect evidence under a different calibration scheme.

## N046 — Multi-evidence summary selector invalid due to NaN feature column
- Date: 2026-06-26
- Experiment ID: `E_eval600_source_oof_to_eval300_top4_multievidence_selector_2026_06_26`
- Hypothesis: Fusing source-OOF score features, scalar image/code features, fixed AlexNet patch embeddings and OSCAR-LIC latent/codebook summary features may produce a safer high-precision top-4 selector than any single weak evidence family.
- Configuration: Added `scripts/train_assignment_topk_multievidence_selector.py`. It joins all 4384 rows to the merged top8 latent-window artifact with zero missing rows, extracts 70 latent/code summary features per row, concatenates them with the N041 selector features and the N045 cached AlexNet features, and trains the same explicit-nearest top-k MLP on Eval600/non-Eval300-seed1 groups. Two variants were run: `safe` with harmful-margin weight `1.5`, and `recall` with harmful-margin weight `0.0`.
- Result: Post-run audit found one all-NaN feature column (`200`) with `4384` NaN values. The source was the upstream latent-window artifact's all-NaN `baseline_score` array. All runs had `train_margin=nan`, so the apparent no-op collapse is not reliable model evidence.
- Why it failed or was negative: This is a feature-cache bug, not a valid model result.
- Valid negative result or invalid run: invalid run. It is retained for audit and superseded by N048.
- Evidence paths: `experiments/reports/eval600_source_oof_to_eval300_top4_multievidence_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_source_oof_to_eval300_top4_multievidence_selector_2026_06_26.yaml`, W&B `chn5v4td`, `mkjhaz2x`.
- Should it be retried? Use the sanitized N048 rerun for model conclusions.

## N047 — Hard-positive weighted summary selector invalid due to NaN feature column
- Date: 2026-06-26
- Experiment ID: `E_eval600_source_oof_to_eval300_top4_balanced_multievidence_selector_2026_06_26`
- Hypothesis: The N046 no-op collapse is caused mainly by nearest-dominant label imbalance, so hard-positive group weighting and oracle-vs-nearest margin losses should move the multi-evidence selector toward useful top-4 oracle changes.
- Configuration: Added `scripts/train_assignment_topk_balanced_selector.py`, reusing the N046 461-D feature cache. Train split has 70 oracle-change groups and 403 no-headroom groups. Ran a `balanced` objective with oracle-change weight `8`, positive margin weight `1`, and weak noheadroom/harmful margins `0.2`; also ran a `positive-only` objective with oracle-change weight `32`, positive margin weight `2`, and no suppression margins.
- Result: This run consumed the N046 feature cache before sanitization. It inherited the all-NaN feature column (`200`) and all runs had `train_margin=nan`, so the apparent no-op collapse is not reliable model evidence.
- Why it failed or was negative: This is a feature-cache bug, not a valid model result.
- Valid negative result or invalid run: invalid run. It is retained for audit and superseded by N048.
- Evidence paths: `experiments/reports/eval600_source_oof_to_eval300_top4_balanced_multievidence_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_source_oof_to_eval300_top4_balanced_multievidence_selector_2026_06_26.yaml`, W&B `sybz8479`, `n8b6dxc9`.
- Should it be retried? Use the sanitized N048 rerun for model conclusions.

## N048 — Clean multi-evidence summary selector is harmful after NaN sanitization
- Date: 2026-06-26
- Experiment ID: `E_eval600_source_oof_to_eval300_top4_multievidence_clean_rerun_2026_06_26`
- Hypothesis: After sanitizing the NaN feature column from N046/N047, summary-level multi-evidence features and hard-positive weighting may produce a useful top-4 selector.
- Configuration: Added `scripts/sanitize_feature_cache.py` and generated `eval600_source_oof_to_eval300_top4_multievidence_features_clean_2026_06_26.npz` by replacing NaN/inf with `0.0`. The sanitizer recorded `4384` NaNs in feature column `200`. Reused `scripts/train_assignment_topk_balanced_selector.py` with a standard CE variant and a balanced variant (`oracle-change weight 8`, positive margin 1, weak noheadroom/harmful margins).
- Result: The clean selector moves away from no-op but is harmful. Standard CE changes `3.67` groups on average, all false/wrong, with Tesseract `+1.67`. Balanced weighting changes `6.0` groups, recovers only `0.67` exact oracle changes, creates `5.33` false/wrong changes, and still worsens Tesseract by `+1.33`. PARSeq remains unchanged. The shortlist oracle remains `-20` Tesseract edits.
- Why it failed or was negative: Summary-level evidence contains some signal but does not separate safe hard positives from harmful non-nearest changes under Eval600-to-Eval300 transfer.
- Valid negative result or invalid run: valid negative result for the corrected clean-feature summary-MLP selector; no counted `.oscr` promotion stream was produced.
- Evidence paths: `experiments/reports/eval600_source_oof_to_eval300_top4_multievidence_clean_rerun_2026_06_26.md`, manifest `experiments/manifests/eval600_source_oof_to_eval300_top4_multievidence_clean_rerun_2026_06_26.yaml`, W&B `x0dbl88o`, `pr68eztu`.
- Should it be retried? Do not continue summary-feature MLP selectors or simple reweighting. Retry only with structured latent/code-token models, OCR/text-specific verification, or better first-stage candidate scoring.

## N049/N050 — Candidate scoring, not top-4 coverage alone, blocks Eval600-to-Eval300 selection
- Date: 2026-06-26
- Experiment ID: `E_eval600_to_eval300_candidate_scoring_nonnearest_ranker_2026_06_26`
- Hypothesis: The source-OOF top-4 selectors may fail mainly because nearest/no-op is over-scored; a non-nearest-only candidate scorer should recover changed oracle candidates while a thresholded abstention rule falls back to nearest.
- Configuration: Added `scripts/audit_topk_candidate_scoring.py` and `scripts/train_assignment_nonnearest_patch_ranker.py`. The audit uses the N041 mixed source-OOF oracle-choice and safe-improve scores on Eval300 seed1. The new ranker trains only on non-nearest rows with a small diff-crop CNN, model seeds `0,1,2`, and thresholded nearest fallback for both `oracle_choice` and `safe_improve` targets.
- Result: The top-4 shortlist still has strong headroom: full oracle changes 17 groups for `-24` Tesseract edits, and the top-4 shortlist oracle changes 15 groups for `-20`. However, oracle-choice score argmax recovers `0` exact changes and worsens Tesseract by `+1`; safe-improve score argmax recovers only `2` exact changes but makes `21` false/wrong changes and worsens by `+3`. The non-nearest-only ranker overfits train (`train_auc` about `0.997-1.000`) but transfers weakly: oracle-choice target reaches only `0.590` val AUC / `0.041` AP and safe-improve target `0.572` / `0.093`. Its best mean Tesseract direction is only `-0.33` for the safe-improve utility threshold, with about `17.33` false/wrong changes.
- Why it failed or was negative: Removing nearest from row-level scoring does not make rare safe text-improving edits separable from false changes under Eval600-to-Eval300 transfer. Candidate coverage is often adequate, but candidate scoring/calibration remains the bottleneck.
- Valid negative result or invalid run: valid negative result for candidate-scoring audit plus non-nearest-only diff-crop scorer; no counted `.oscr` promotion stream was produced.
- Evidence paths: `experiments/reports/eval600_to_eval300_candidate_scoring_nonnearest_ranker_2026_06_26.md`, manifest `experiments/manifests/eval600_to_eval300_candidate_scoring_nonnearest_ranker_2026_06_26.yaml`, W&B `rym86loh`.
- Should it be retried? Do not retry nearest-vs-nonnearest score separation with the same small diff-crop CNN, score-threshold, safe/utility target swap, or scalar top-4 selector stack. Retry candidate scoring only with materially different evidence or calibration: structured latent/code-token modeling, OCR/text-specific fixed verifier with reviewed provenance, or broader source/domain-balanced labels.

## N051 — Source/length/difficulty strata do not yield a simple top-4 rescue rule
- Date: 2026-06-26
- Experiment ID: `E_eval600_source_oof_to_eval300_top4_failure_strata_2026_06_26`
- Hypothesis: The N049 top-4 scoring failure may be concentrated enough by source, reference length, or base OCR difficulty to motivate a simple high-precision filter before building a heavier selector.
- Configuration: Added `scripts/analyze_top4_failure_strata.py`, reading the source-OOF top-4 score files and Eval300 seed1 imgdiff assignment table. It analyzes the same 75 groups as N049 by source, source split, reference-length bin, nearest Tesseract/PARSeq distance bin, image area, and oracle rank bins. OCR deltas are analysis labels only.
- Result: The shortlist oracle remains strong (`15/75` changed groups, `-20` Tesseract edits, PARSeq unchanged), but simple strata are not selective enough. IAM contains `10/17` oracle-change groups and `-11` shortlist-oracle edits, yet safe-score argmax creates `13` false/wrong IAM changes and worsens IAM Tesseract by `+2`. Nearest Tesseract distance `3-5` contains `9` oracle-change groups and the only `2` safe-score exact hits, but also `9` safe-score false changes. Short words (`1-4`) create `11` safe-score false changes despite only `5` oracle-change groups.
- Why it failed or was negative: Source, length and base-OCR difficulty reveal where the pressure is, but they cannot separate useful code-effect changes from false/wrong non-nearest assignments. The tempting hand rule "change IAM/mid-difficulty groups" would also trigger many false changes.
- Valid negative result or invalid run: valid diagnostic negative for simple strata-based filtering; no counted `.oscr` promotion stream was produced.
- Evidence paths: `experiments/reports/eval600_source_oof_to_eval300_top4_failure_strata_2026_06_26.md`, manifest `experiments/manifests/eval600_source_oof_to_eval300_top4_failure_strata_2026_06_26.yaml`, W&B `ox7xrbee`.
- Should it be retried? Do not replace the selector with source/length/base-OCR-distance filtering. Use these strata only for calibration/evaluation splits and for designing richer OCR/text-aware or structured latent/code-effect evidence.

## N053 — Direct PARSeq encoder features do not rescue the source-OOF top-4 selector
- Date: 2026-06-26
- Experiment ID: `E_eval600_source_oof_to_eval300_top4_parseq_encoder_diagnostic_2026_06_26`
- Hypothesis: Direct PARSeq training-teacher encoder-memory features may provide the OCR/text-aware evidence missing from scalar score/rank, image-diff, generic patch, AlexNet and summary-feature top-4 selectors.
- Configuration: Added `scripts/extract_parseq_topk_feature_cache.py`, extracting PARSeq encoder memory mean/std/max/min embeddings plus candidate-minus-nearest relative features for the source-OOF top-4 assignment substrate. The final cache has `4384` rows and `4755` features, using checkpoint `/root/.cache/torch/hub/checkpoints/parseq-bb5792a6.pt` with SHA256 `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`. Reused `scripts/train_assignment_topk_balanced_selector.py` with a standard CE variant and a balanced hard-positive variant. PARSeq is a train teacher here, not held-out evidence or a deployable free feature.
- Result: The shortlist oracle remains strong (`15/75` changed groups, `-20` Tesseract edits, PARSeq unchanged), but the learned selectors fail. CE argmax/train-tuned margin change only `0.67` groups on average, recover `0` exact oracle changes, and produce no Tesseract gain. Balanced weighting changes `4.33` groups, still recovers `0` exact oracle changes, creates `4.33` false/wrong changes, worsens Tesseract by `+1.67`, and has `1.33` Tesseract-worsen groups on average.
- Why it failed or was negative: Flattened OCR-teacher encoder features do not make safe hard-positive code assignment separable under Eval600-to-Eval300 transfer. Forcing recall via hard-positive weighting mainly selects false/wrong non-nearest changes.
- Valid negative result or invalid run: valid negative diagnostic for direct PARSeq encoder feature concatenation into the current top-4 MLP selector; no counted `.oscr` promotion stream was produced.
- Evidence paths: `experiments/reports/eval600_source_oof_to_eval300_top4_parseq_encoder_diagnostic_2026_06_26.md`, manifest `experiments/manifests/eval600_source_oof_to_eval300_top4_parseq_encoder_diagnostic_2026_06_26.yaml`, W&B `mw4lraiv`.
- Should it be retried? Do not retry direct PARSeq encoder feature appends, shallow MLPs, or hard-positive reweighting over this feature cache. Reuse teacher evidence only under a materially different design, such as auxiliary distillation into deployable features, a structured OCR/text-specific verifier, or explicit source/domain-disjoint policy calibration with OCR encoder cost accounted where applicable.

## N054 — Hand-crafted topology features give weak signal but not safe top-4 selection
- Date: 2026-06-26
- Experiment ID: `E_eval600_source_oof_to_eval300_top4_topology_selector_2026_06_26`
- Hypothesis: Deployable text-structure features computed from source, nearest reconstruction and candidate reconstruction may provide safer code-effect evidence than scalar score/rank, generic patch, AlexNet, summary, non-nearest diff-crop or direct PARSeq encoder features.
- Configuration: Added `scripts/extract_topology_topk_feature_cache.py`, extracting connected-component counts, hole/euler proxies, dark row/column projections, distance-transform stroke-width approximations, boundary/Sobel statistics, candidate-vs-nearest dark-pixel additions/removals and local diff-bbox topology. The cache has `4384` rows, `143` existing selector/scalar dims, `299` topology dims and `442` total dims. Trained CE and balanced top-4 selectors with `scripts/train_assignment_topk_balanced_selector.py`.
- Result: The topology features carry weak useful signal but remain far below promotion quality. CE changes `6.0` groups, recovers `0.33` exact oracle changes, worsens `0.67` Tesseract groups and gives only `-0.33` Tesseract edits with `5.67` false/wrong changes. Balanced changes `10.67` groups, recovers `0.67` exact oracle changes and gives `-1.33` Tesseract edits, but emits `10.0` false/wrong changes. PARSeq remains unchanged. The shortlist oracle is still `-20` Tesseract edits with `15` exact oracle changes.
- Why it failed or was negative: Hand-crafted topology evidence has a weak directionally useful signal, but the shallow selector cannot reach the precision needed for assignment promotion. The result is dominated by false/wrong non-nearest changes and remains far below the current actual-bitstream learned baseline (`-8`) and the shortlist oracle (`-20`).
- Valid negative result or invalid run: valid negative result for topology-only shallow top-4 selector promotion; no counted `.oscr` promotion stream was produced.
- Evidence paths: `experiments/reports/eval600_source_oof_to_eval300_top4_topology_selector_2026_06_26.md`, manifest `experiments/manifests/eval600_source_oof_to_eval300_top4_topology_selector_2026_06_26.yaml`, W&B `zfmoo46f`, `ozo0gg97`.
- Should it be retried? Do not seed/capacity sweep this topology-only shallow selector. Reuse topology features only as auxiliary evidence inside a materially different high-precision abstention/verifier objective, source/domain-disjoint calibration setup, or structured latent/code-token model.

## N055 — CRNN/ABINet train-teacher confidence does not transfer to safe assignment selection
- Date: 2026-06-26
- Experiment ID: `E_eval600_eval300_strhub_trainteacher_transfer_2026_06_26`
- Hypothesis: Additional STRHub train teachers, CRNN and ABINet, may provide OCR/text-aware confidence or edit-distance signals that identify Tesseract-useful top-8 assignment changes without using held-out OCR at inference or for training labels.
- Configuration: Locked CRNN checkpoint SHA256 `679d0e315af22d813c6d5a86287ee5705a9a8dfe37378349359442b8b8953264` and ABINet checkpoint SHA256 `1d1e373e6d08bde09500a015436b87ae0f4c9bbd844c8c4fc7a8d9e3628325a7` from the STRHub/PARSeq model hub. Ran recognizer-only OCR on Eval300 originals and on the hard-negative-balanced Eval600 plus Eval300 seed1 top8 candidate table, then analyzed confidence and distance-confidence policies.
- Result: CRNN/ABINet are useful locked train-teacher assets, but direct teacher improvements are too sparse and confidence policies are unsafe. On Eval300 seed1, only `1/75` groups improve either train teacher, while Tesseract has `17/75` oracle-change groups. CRNN confidence changes `67/75` groups, recovers `0` exact oracle changes, creates `54` false changes and worsens Tesseract by `+2` edits. ABINet confidence changes `68/75`, recovers `3` exact oracle changes, creates `52` false changes and worsens Tesseract by `+3`. Mean confidence is also harmful (`+1` with `52` false changes).
- Why it failed or was negative: Train-teacher confidence is dense but poorly calibrated for candidate assignment utility. It mostly selects non-nearest changes in no-headroom groups. The non-deployable `trainteacher_no_worse_heldout_oracle` upper bound shows CRNN/ABINet no-worse constraints do not exclude Tesseract-useful choices, but the actual choice still uses held-out Tesseract oracle labels and cannot be used for training or selection.
- Valid negative result or invalid run: valid negative result for direct CRNN/ABINet confidence, distance-confidence and sparse edit-improvement labels as primary selector signals; positive asset lock for train-teacher diversity.
- Evidence paths: `experiments/reports/eval600_eval300_strhub_trainteacher_transfer_2026_06_26.md`, manifest `experiments/manifests/eval600_eval300_strhub_trainteacher_transfer_2026_06_26.yaml`, W&B `ccwg6z4j`.
- Should it be retried? Do not retry train-teacher confidence argmax, mean-confidence ensembling, or sparse CRNN/ABINet edit-gain labels as the main selector target. Reuse CRNN/ABINet only as auxiliary train-teacher constraints, dense teacher-loss/NLL signals against ground truth, synthetic perturbation labelers, or an explicitly costed OCR/text verifier.

## N056 — CRNN/ABINet teacher loss is dense but unsafe as an assignment selector
- Date: 2026-06-26
- Experiment ID: `E_eval600_eval300_strhub_trainteacher_loss_transfer_2026_06_26`
- Hypothesis: Per-candidate CRNN/ABINet teacher loss against ground-truth text may be a better calibrated dense train-teacher proxy than confidence for selecting OCR-useful assignment code changes.
- Configuration: Added `scripts/run_strhub_teacher_loss.py` to compute per-sample CRNN CTC loss and ABINet cross-entropy loss against STRHub charset-adapted ground truth, without changing OCR metric references. Added `scripts/analyze_strhub_teacher_loss_transfer.py` to evaluate loss-decrease policies on the hard-negative-balanced Eval600 plus Eval300 seed1 top8 assignment table.
- Result: Teacher loss is dense but over-permissive. On Eval300 seed1, `75/75` groups have some CRNN or ABINet loss-improving candidate, while only `17/75` have held-out Tesseract oracle changes. CRNN loss argmax changes `67/75` groups, creates `52` false changes and worsens Tesseract by `+3`. ABINet loss argmax changes `66/75`, creates `50` false changes and gives no net Tesseract improvement. Mean loss argmax reaches only `-2` edits with `51` false changes. The non-deployable loss-no-worse held-out oracle changes only `5` groups and reaches `-6` edits, below the current actual-bitstream learned baseline.
- Why it failed or was negative: Lower teacher loss mostly detects small recognizer-preference changes, not safe held-out OCR utility. Hard filtering by both-teacher loss no-worse removes much of the Tesseract headroom. This rules out both direct loss argmax and simple no-worse loss filtering as primary policy rules.
- Valid negative result or invalid run: valid negative result for teacher-loss argmax and hard teacher-loss no-worse filtering; positive infrastructure for future auxiliary-loss diagnostics.
- Evidence paths: `experiments/reports/eval600_eval300_strhub_trainteacher_loss_transfer_2026_06_26.md`, manifest `experiments/manifests/eval600_eval300_strhub_trainteacher_loss_transfer_2026_06_26.yaml`, W&B `e5bqv22c`, `9pk49iuy`, `s4xivdqz`.
- Should it be retried? Do not use teacher-loss argmax, mean-loss ensembling or hard CRNN+ABINet loss-no-worse filters as the selector. Reuse teacher losses only as auxiliary regularization or diagnostics inside a materially different verifier/abstention objective with external validation.

## N057 — Source OCR pseudo-label loss is deployable-style but still too permissive
- Date: 2026-06-26
- Experiment ID: `E_eval600_eval300_strhub_source_pseudolabel_loss_transfer_2026_06_26`
- Hypothesis: Using CRNN/ABINet predictions on the source image as pseudo-labels may provide an encoder-deployable OCR self-consistency verifier for candidate assignment choices.
- Configuration: Added `scripts/build_strhub_pseudolabel_splits.py` to create a 548-group source-image OCR split and teacher-specific pseudo-label candidate tables. Ran CRNN/ABINet source OCR, then per-candidate CRNN/ABINet pseudo-label losses with empty source predictions allowed as valid pseudo-labels.
- Result: Pseudo-label loss is still far denser than held-out utility. On Eval300 seed1, all `75/75` groups have at least one CRNN or ABINet pseudo-loss-improving candidate, while only `17/75` have Tesseract oracle changes. CRNN pseudo-loss argmax changes `66` groups, creates `51` false changes and gives only `-1` Tesseract edit. ABINet pseudo-loss argmax changes `66`, creates `50` false changes and gives `-1`. Mean pseudo-loss argmax reaches `-3` with `50` false changes. The non-deployable pseudo-loss-no-worse held-out oracle reaches `-7`, still below the current learned actual-bitstream baseline.
- Why it failed or was negative: Source OCR self-consistency mostly preserves the train teacher's own reading and does not separate OCR-useful code-effect changes from harmless/harmful visual perturbations. Hard self-consistency filtering discards too much of the held-out Tesseract headroom.
- Valid negative result or invalid run: valid negative result for source-pseudo-label loss argmax and hard pseudo-loss no-worse filtering; positive infrastructure for future encoder-side verifier diagnostics.
- Evidence paths: `experiments/reports/eval600_eval300_strhub_source_pseudolabel_loss_transfer_2026_06_26.md`, manifest `experiments/manifests/eval600_eval300_strhub_source_pseudolabel_loss_transfer_2026_06_26.yaml`, W&B `4cx9p4rf`, `l24e394t`, `vcrn793q`, `bu151m61`, `smxrubsn`.
- Should it be retried? Do not use source-pseudo-label loss argmax, mean pseudo-loss ensembling, or hard pseudo-loss no-worse filtering as the selector. Reuse source pseudo-labels only as auxiliary diagnostics inside a materially different abstention/verifier model with external validation.

## N059 — Fixed train-teacher vetoes do not stabilize the current best changes
- Date: 2026-06-26
- Experiment ID: `E_eval300_current_best_changed_pair_teacher_veto_2026_06_26`
- Hypothesis: The current best's single Tesseract-worsened changed sample may be removable by a fixed CRNN/ABINet/PARSeq teacher-loss or confidence veto without sacrificing the useful changed samples.
- Configuration: Added `scripts/build_changed_ocr_pair_split.py` to extract the five current-best changed Tesseract pairs from the nearest-vs-current comparison, then ran `scripts/run_strhub_teacher_loss.py` with propagated pair metadata for CRNN, ABINet and PARSeq on the 10 nearest/current images. Added `scripts/analyze_changed_pair_teacher_veto.py` to evaluate fixed veto rules over teacher loss deltas, confidence deltas and teacher prediction changes.
- Result: The current changed set has 4 Tesseract-improving changes and 1 Tesseract-worsening change for net `-8`. Fixed teacher vetoes do not isolate the bad sample. `veto_if_crnn_abinet_loss_both_increase` removes the worsened sample but also removes one useful improvement, reducing the net to `-6`. `veto_if_mean_loss_increases` removes two useful improvements and the worsened sample, reducing the net to `-3`. Confidence-decrease, teacher-prediction-change and no-teacher-loss-improvement vetoes remove nothing and keep `-8`.
- Why it failed or was negative: Train-teacher losses are nearly flat on these tiny code-effect changes, and their signs do not align with held-out Tesseract utility. The only rules that catch the bad change also throw away useful changes.
- Valid negative result or invalid run: valid negative diagnostic; not a promotion experiment because rules are interpreted against held-out Tesseract deltas and no new counted `.oscr` stream was produced.
- Evidence paths: `experiments/reports/eval300_current_best_changed_pair_teacher_veto_2026_06_26.md`, manifest `experiments/manifests/eval300_current_best_changed_pair_teacher_veto_2026_06_26.yaml`, W&B `d2hzt8qt`, `ui1wrmt9`, `vxspk149`, aggregate W&B `0jzdckyo`.
- Should it be retried? Do not add fixed CRNN/ABINet/PARSeq teacher-loss or confidence veto overlays to the current best policy. Reuse teacher loss only as auxiliary calibration evidence inside a materially different verifier or abstention objective.

## N064 — Top8 feature-neighborhood support is not enough for safe assignment selection
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_feature_neighbor_audit_2026_06_26`
- Hypothesis: If the top8 structured latent/code-effect features already contain enough transferable evidence, a simple group-disjoint KNN neighborhood policy over code geometry, image-diff, latent summaries, or PARSeq-teacher-augmented summaries should recover useful assignment changes without another learned selector.
- Configuration: Added `scripts/analyze_top8_feature_neighbor_audit.py` and evaluated 7 feature families over the 300-group / 2400-row Eval300 top8 table and latent-window artifact with k=`5,15,31`. For train rows, KNN scores exclude rows from the same group. Policies include oracle-label argmax, train-tuned oracle-margin, and train-tuned safe-improve thresholding. Tesseract-derived labels are diagnostic only and are not a deployable selector signal.
- Result: Neighborhood support is visible but not policy-grade. Many feature spaces keep oracle candidates in top4 (`~0.61-0.76` top4 recall), but top1/policy selection collapses to no-op or false/wrong changes. The best safe-threshold policy is `imgdiff_scalar`, k=`15`, with mean Tesseract delta `-1.0`, PARSeq delta `0.0`, `1.0` changed group, `0.0` exact oracle groups, `0.33` false changes and `0.67` wrong changes over the three seed-held-out top8 diagnostic splits. The most aggressive useful argmax, `codegeom_onehot` k=`5`, reaches `-3.33` mean Tesseract edits but emits `18.67` false changes and `7.0` wrong changes.
- Why it failed or was negative: The existing scalar/summary feature neighborhoods can place useful candidates in a shortlist but cannot make a high-precision abstention decision. This reinforces the earlier AUC/policy gap: the representation has partial recall signal but insufficient no-headroom suppression and hard-positive top1 calibration.
- Valid negative result or invalid run: valid negative diagnostic, not a counted `.oscr` promotion run. The population is the top8 seed-held-out diagnostic table, not the exact 225-row actual-bitstream current-best/shortlist evaluation.
- Evidence paths: `experiments/reports/eval300_top8_feature_neighbor_audit_2026_06_26.md`, manifest `experiments/manifests/eval300_top8_feature_neighbor_audit_2026_06_26.yaml`, W&B `lapwwjuu`.
- Should it be retried? Do not promote or repeat KNN/feature-neighborhood selection over the same scalar/summary feature families. A retry needs materially different evidence, such as a structured OCR/text-aware verifier, conditional code-effect modeling, or additional source/domain-disjoint calibration data.

## N067 — Policy-margin no-headroom suppression still fails to recover top8 assignment headroom
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_policy_margin_noheadroom_probe_2026_06_26`
- Hypothesis: A stricter policy-level objective with strong no-headroom suppression, hard-positive margins and matched positive-vs-no-headroom pairwise margins may convert the top8 latent-window row signal into a high-precision assignment policy.
- Configuration: Ran `scripts/train_assignment_group_noop_window.py` on `experiments/features/eval300_assignment_top8_codeeffect_latentctx_windows_parseqfeat_seed0_1_2_2026_06_25.npz` for `val_seed=1`, model seeds `0,1,2`, 500 epochs. Used `safe_tesseract_soft_bce` row target, `strong_tesseract` group target with delta `-2`, group positive weighting, `false_change_margin_weight=2.0`, PARSeq harm and Tesseract non-improve penalties, hard-positive row/group margins, matched-pairwise group/row margins, `parseq_safe`, and train threshold constraints requiring zero false changes, zero wrong changes and zero Tesseract-worsened groups.
- Result: Row validation AUC/AP remains high (`0.88695 / 0.58937`), but group validation AUC is poor (`0.46991`). The oracle on this split is `17` changed groups and `-24` Tesseract edits. Default threshold `0.5` changes only `3` groups on average, recovers `0` exact oracle changes, emits `2` false changes and `1` wrong change, and worsens Tesseract by `+1.67`. The train-tuned threshold averages `0.33` exact oracle changes but still worsens by `+0.33` with false/wrong changes.
- Why it failed or was negative: The objective can look clean on train, selecting useful train changes with zero false/wrong changes for some seeds, but that calibration does not transfer to the held-out val shard. It suppresses aggressiveness mostly by collapsing toward no-op while still failing to isolate useful val changes.
- Valid negative result or invalid run: valid negative diagnostic. It is non-deployable because the training target uses held-out Tesseract-derived assignment labels, and no counted `.oscr` stream was produced.
- Evidence paths: `experiments/reports/eval300_top8_policy_margin_noheadroom_probe_2026_06_26.md`, manifest `experiments/manifests/eval300_top8_policy_margin_noheadroom_probe_2026_06_26.yaml`, W&B `vg421itb`.
- Should it be retried? Do not run counted promotion or a simple cross-seed/weight sweep of this exact policy-margin/no-headroom objective. A retry needs genuinely new calibration evidence, more source/domain-disjoint training data, or a different verifier rather than just retuning the same margin weights.

## N068 — Actual-policy consensus over the pooled image-diff gates does not reveal hidden headroom
- Date: 2026-06-26
- Experiment ID: `E_eval300_actual_policy_consensus_audit_2026_06_26`
- Hypothesis: The two already-exported pooled image-diff gate policies may contain complementary actual-bitstream changes; AND/OR/consensus could improve precision or recover more top-4 shortlist headroom without retraining.
- Configuration: Added `scripts/analyze_actual_policy_consensus_audit.py` and compared `pooled_imgdiff_model1_threshold05`, `pooled_imgdiff_model2_threshold08`, deterministic AND/OR combinations, and the non-deployable score-seed-2 top-4 shortlist oracle over the same 75 Eval300 val groups. Held-out Tesseract labels are used only to categorize already-evaluated rows.
- Result: Same-change consensus is exactly identical to the promoted current-best policy: 5 changed groups, 4 exact oracle changes, 1 false change, Tesseract delta `-8`, PARSeq delta `0`. Policy B adds two changes over policy A, but they are one false change and one wrong/neutral Tesseract case; union-style policies still have Tesseract delta `-8` but increase policy error count to 2 false + 1 wrong. The shortlist oracle remains far stronger with 15 exact changes and Tesseract delta `-20`.
- Why it failed or was negative: This gate family is already saturated on the Eval300 actual-bitstream split. The extra policy-B changes do not recover useful shortlist-oracle headroom, and consensus does not remove the current false change.
- Valid negative result or invalid run: valid diagnostic negative; no new counted `.oscr` stream was produced because all combinations are table-level combinations of already-exported actual policies, and the result does not beat the current best.
- Evidence paths: `experiments/reports/eval300_actual_policy_consensus_audit_2026_06_26.md`, manifest `experiments/manifests/eval300_actual_policy_consensus_audit_2026_06_26.yaml`, W&B `nd3gr4iq`.
- Should it be retried? Do not promote or retune AND/OR/consensus/threshold rules over this same pooled image-diff gate family. Retry only if the underlying candidate-local evidence or verifier objective changes materially.

## N069 — Independent additive multi-candidate utility is invalid on interaction smoke20
- Date: 2026-06-26
- Experiment ID: `E300-INTERACTION-SMOKE20-DETAIL-AUDIT`
- Hypothesis: If two individually useful enhancement candidates have approximately additive OCR utility, a multi-candidate allocator can rank candidates independently and sum marginal gains.
- Configuration: Added `scripts/analyze_interaction_smoke_details.py` and re-analyzed the existing counted Eval300 raw-residual interaction smoke20 outputs per sample for Tesseract PSM8 and PARSeq. The analysis compares base, single A, single B and pair AB OCR distances, joins actual stream bpp/PSNR rows, and computes additive expected delta, observed pair delta, interaction residual, pair-vs-best-single delta and conditional second-candidate deltas.
- Result: Tesseract unicode is strongly subadditive. Single A and B sum to `-59` edit errors, but pair AB gives only `-10`; the pair retains only `16.95%` of additive gain, is worse than the best single candidate on `11/20` samples, and adding B after A has net `+21` edit errors rather than additional improvement. Latin-normalized Tesseract is similar (`15.00%` additive-gain retention; pair worse than best single on `7/20`). PARSeq saturates: pair AB ties the best single candidate on all `20/20` samples under both profiles.
- Why it failed or was negative: OCR utility is conditional on already-selected latent units. The second candidate often duplicates, cancels or perturbs the first candidate's OCR effect rather than adding independent utility.
- Valid negative result or invalid run: valid negative result for independent additive top-k utility as a multi-candidate allocation assumption. It is a high-rate raw-residual diagnostic, not a promoted compression method.
- Evidence paths: `experiments/reports/eval300_interaction_smoke20_detail_audit_2026_06_26.md`, `experiments/results/eval300_interaction_smoke20_detail_audit_2026_06_26.json`, manifest `experiments/manifests/eval300_interaction_smoke20_detail_audit_2026_06_26.yaml`, W&B `828bl6tt`.
- Should it be retried? Do not design future multi-candidate OSCAR allocation by summing independent marginal utilities. A retry should use conditional/diminishing-return utility, explicit subset-aware scoring, or learned interaction terms, and should still report actual transmitted bytes.

## N070 — Adding more individually useful candidates can reduce OCR gain despite higher PSNR
- Date: 2026-06-26
- Experiment ID: `E300-INTERACTION-SUBSET3-SMOKE10`
- Hypothesis: If each candidate is individually safe-improving, transmitting larger subsets should be at least competitive with the best single candidate, especially as PSNR rises monotonically.
- Configuration: Added `scripts/build_interaction_subset_policies.py` to select 10 Eval300 images with three distinct safe-improving raw-residual candidates and export all nonempty subsets. Generated counted `.oscr` streams for single A/B/C, pair AB/AC/BC and triple ABC using `scripts/run_mlicpp_latent_residual_stub.py`, then ran PARSeq and Tesseract PSM8. Added `scripts/analyze_interaction_subset_oracle.py` to compute fixed-policy metrics and a non-deployable all-subset OCR oracle.
- Result: Tesseract unicode rejects the monotonic-subset assumption. The best fixed one-candidate policy (`single_a`) gives `-20` edit errors at `4.8440` actual bpp. The best pair (`pair_ac`) gives only `-14` at `6.3172` bpp, and the triple gives `-9` at `7.7885` bpp, even though average PSNR delta rises from `0.0574` to `0.1145` to `0.1660` dB. The subset oracle chooses one candidate on `8/10` unicode samples and `9/10` latin samples. PARSeq also saturates: under the tie-break, the oracle chooses one candidate on `10/10` samples.
- Why it failed or was negative: Candidate residuals interact destructively or redundantly for OCR. Visual distortion metrics can improve monotonically while OCR utility is non-monotonic. Positive individual marginal utility is therefore insufficient evidence to transmit another candidate.
- Valid negative result or invalid run: valid diagnostic negative for monotonic positive-marginal top-k allocation. It is a high-rate raw-residual smoke, not a promoted compression method.
- Evidence paths: `experiments/reports/eval300_interaction_subset3_smoke10_tesseract_subset_oracle_2026_06_26.md`, `experiments/reports/eval300_interaction_subset3_smoke10_parseq_subset_oracle_2026_06_26.md`, manifest `experiments/manifests/eval300_interaction_subset3_smoke10_2026_06_26.yaml`, W&B `sem19sup`.
- Should it be retried? Do not add second/third enhancement candidates solely because individual utility is positive or PSNR improves. Future retries need explicit stop/continue supervision, conditional utility labels, learned interaction terms, or subset-aware oracle distillation.

## N071 — Smoke40 confirms one-candidate stop dominates multi-candidate raw residual subsets
- Date: 2026-06-26
- Experiment ID: `E300-INTERACTION-SUBSET3-SMOKE40`
- Hypothesis: The N070 smoke10 result may be a tiny-sample artifact; with 40 images, larger subsets of three individually safe-improving candidates may become competitive because PSNR rises monotonically.
- Configuration: Reused `scripts/build_interaction_subset_policies.py` and `scripts/analyze_interaction_subset_oracle.py` to select 40 Eval300 images with three distinct safe-improving raw-residual candidates. Generated counted `.oscr` streams for single A/B/C, pair AB/AC/BC and triple ABC using corrected MLIC++ lambda `0.0250`, tile `4`, quant step `0.05`, residual scale `0.25`, then ran PARSeq and Tesseract PSM8 OCR.
- Result: The larger diagnostic strengthens N070. For Tesseract unicode, the best fixed single policy is `single_a` with `-63` edit errors at `6.0395` actual bpp. The best pair is weaker (`pair_ac`, `-36` at `8.0560` bpp), and the triple is weaker still (`-32` at `10.0711` bpp), even though PSNR delta rises from `0.0515` to `0.1322` to `0.1903` dB. The subset oracle chooses one candidate on `36/40` unicode samples and `37/40` latin samples. PARSeq's subset oracle chooses `single_a` on `40/40` samples under both profiles.
- Why it failed or was negative: More enhancement bytes and better PSNR do not imply more OCR utility. For these raw-residual candidates, second/third units are usually redundant or harmful after the first useful unit, so an additive positive-marginal allocator would over-send.
- Valid negative result or invalid run: valid diagnostic negative for monotonic positive-marginal multi-candidate allocation. It is high-rate raw-residual evidence, not a promoted compression method.
- Evidence paths: `experiments/reports/eval300_interaction_subset3_smoke40_tesseract_subset_oracle_2026_06_26.md`, `experiments/reports/eval300_interaction_subset3_smoke40_parseq_subset_oracle_2026_06_26.md`, manifest `experiments/manifests/eval300_interaction_subset3_smoke40_2026_06_26.yaml`, W&B `90cff84m`.
- Should it be retried? Not by scaling the same independent top-k raw-residual subsets again. Retry only with explicit conditional stop/continue labels, subset-aware utility prediction, learned interaction terms, or a compact symbol representation whose second/third units have demonstrable conditional OCR gain under actual transmitted bytes.

## N072 — First-candidate and second-candidate utilities have opposite signs in smoke40
- Date: 2026-06-26
- Experiment ID: `E300-INTERACTION-SUBSET3-SMOKE40-CONDITIONALS`
- Hypothesis: A single candidate utility score may still be usable if it is interpreted as an add-one-candidate marginal score at any stage of allocation.
- Configuration: Added `scripts/analyze_interaction_subset_conditionals.py` and applied it to the N071 all-subset smoke40 outputs. The script computes conditional edges from base to singles, singles to pairs and pairs to triple for Tesseract PSM8 and PARSeq, preserving the counted `.oscr` stream bpp/PSNR rows.
- Result: Tesseract unicode has a clear sign flip. Base-to-single edges improve by `-179` edit errors over 120 edges, while single-to-pair edges worsen by `+178` edit errors over 240 edges at comparable added enhancement bpp and PSNR gain. Only `4/40` samples have a best pair that beats the best single, and stopping after the best single costs only `9` edit errors versus the non-deployable best-any subset oracle. PARSeq is fully saturated under best-subset selection: stopping after the best single has `0` regret under both normalization profiles.
- Why it failed or was negative: The conditional context changes the sign and magnitude of utility. A shared additive score cannot distinguish "send the first useful evidence unit" from "send another unit that perturbs or duplicates already restored glyph evidence."
- Valid negative result or invalid run: valid diagnostic negative for stage-agnostic add-one scoring. It uses high-rate raw residuals and OCR-derived subset labels, so it is design evidence rather than a deployable method.
- Evidence paths: `experiments/reports/eval300_interaction_subset3_smoke40_tesseract_conditionals_2026_06_26.md`, `experiments/reports/eval300_interaction_subset3_smoke40_parseq_conditionals_2026_06_26.md`, manifest `experiments/manifests/eval300_interaction_subset3_smoke40_conditionals_2026_06_26.yaml`, W&B `bzrbijit`.
- Should it be retried? Do not train a stage-agnostic additive multi-candidate utility head. Retry by modeling allocation stage and selected-set context explicitly, for example separate first-candidate and continue heads, subset-aware distillation, or a stop-token policy.

## N075 — Shallow feature conditional allocators do not beat first-only
- Date: 2026-06-26
- Experiment ID: `E300-INTERACTION-SUBSET3-SMOKE40-FEATURE-ALLOCATOR`
- Hypothesis: Cross-fit feature models over transition, candidate identity, selected-set context, actual enhancement bpp, PSNR gain and source/domain can identify when to continue beyond the first candidate and beat the N074 `first_only` floor.
- Configuration: Added `scripts/analyze_conditional_edge_feature_allocator.py`. It trains pure-NumPy logistic models with image-index-modulo cross-fitting over the N073 480-row edge table. Feature modes were `transition`, `transition_candidate_rate_psnr`, `transition_candidate_rate_psnr_source`, and a non-deployable `transition_candidate_rate_psnr_ocr_state_upper` upper-control using OCR source distances. Evaluation reports both N074-style edge selection metrics and greedy image-level stop/continue subset policies.
- Result: Deployable-lite features do not beat the floor. Transition-only exactly reproduces N074 `first_only` (F1 `0.8498`, Tesseract delta `-179`, PARSeq delta `-13`). Adding candidate/rate/PSNR/source features lowers F1 to `0.8169-0.8244` and gives weaker Tesseract deltas (`-176` to `-177`). The OCR-state upper-control improves edge Tesseract delta to `-190`, but its greedy subset policy chooses two triples and drops to `-54` Tesseract edits, below the fixed one-candidate `single_a` policy at `-63`.
- Why it failed or was negative: The edge table's shallow transition/rate/PSNR/source features cannot identify the rare useful continue edges without admitting many harmful or redundant continue decisions. Even non-deployable OCR-state distances do not translate into a better coherent subset policy, so the bottleneck is conditional candidate-local code-effect evidence rather than another small classifier over the current edge features.
- Valid negative result or invalid run: valid diagnostic negative. It uses OCR-derived labels and a high-rate raw-residual smoke table, so it is design evidence rather than a deployable method.
- Evidence paths: `experiments/reports/eval300_interaction_subset3_smoke40_feature_allocator_2026_06_26.md`, `experiments/results/eval300_interaction_subset3_smoke40_feature_allocator_2026_06_26.json`, manifest `experiments/manifests/eval300_interaction_subset3_smoke40_feature_allocator_2026_06_26.yaml`, W&B `s4jy1ggg`.
- Should it be retried? Do not scale the same shallow feature allocator or generate larger multi-candidate labels to feed it. Retry multi-candidate allocation only with materially richer selected-set/candidate-local evidence, a compact symbol representation whose second/third units have conditional OCR gain, or a verifier that can beat the fixed one-candidate policy before counted promotion.

## N077 — Existing score families cannot safely recover the actual shortlist headroom
- Date: 2026-06-26
- Experiment ID: `E300-ACTUAL-ASSIGNMENT-FAILURE-FEATURE-AUDIT`
- Hypothesis: The N076 failure-bank targets may already be separable by existing deployable candidate scores or simple score ranks, so a threshold/budget rule over current code-only, diff-crop, latent-window, latent-context or tabular image-diff score files might recover shortlist-oracle changes while rejecting the one bad current change.
- Configuration: Added `scripts/analyze_actual_assignment_failure_features.py`. The script joined the 75-case N076 actual `.oscr` failure bank with the seed0/1/2 top8 codegeom+image-diff feature table, five existing score JSONL families, and contact sheets for recover/reject/first-stage-miss/good-change strata. OCR outcomes were used only to define held-out diagnostic strata.
- Result: The audit has zero missing candidate-feature rows. Across the 11 recoverable shortlist-oracle targets, none is ranked first by code-only, diff-crop, latent-window or latent-context scores (`rank<=1` count `0`), and only `6/11` are within rank `<=4`. The single current bad change to reject is rank `1` under all five score families with very high scores (`0.8876-0.9999`). First-stage misses sit at top8 rank `6`, outside the current top4 score families.
- Why it failed or was negative: The current score families are confidently wrong on the abstention floor while under-ranking the recoverable OCR-improving shortlist candidates. This is the worst shape for threshold tuning: increasing recall admits the known bad change, while high precision misses most recoverable headroom.
- Valid negative result or invalid run: valid diagnostic negative. It does not define a deployable selector and does not use held-out OCR strings as features.
- Evidence paths: `experiments/reports/eval300_actual_assignment_failure_feature_audit_2026_06_26.md`, `experiments/results/eval300_actual_assignment_failure_feature_audit_2026_06_26.json`, contact sheets under `experiments/figures/eval300_actual_assignment_failure_feature_audit_2026_06_26/`, manifest `experiments/manifests/eval300_actual_assignment_failure_feature_audit_2026_06_26.yaml`, W&B `wqnd7ezy`.
- Should it be retried? Do not threshold-sweep or budget-sweep these same score families. Retry only after adding materially different candidate-local code-effect evidence, a better first-stage candidate generator/scorer, or an OCR/text-aware verifier whose features are independent of the failed score confidence.

## N083 — Existing score families remain wrong on the top8 target bank
- Date: 2026-06-26
- Experiment ID: `E_eval300_actual_assignment_failure_feature_audit_top8_2026_06_26`
- Hypothesis: After updating the target inventory from top4 to top8, the same score families might become sufficient because top8 closes the old first-stage miss bucket.
- Configuration: Reran `scripts/analyze_actual_assignment_failure_features.py` on the N082 top8 actual failure bank, joining the same seed0/1/2 top8 codegeom+image-diff table and the same five existing score files: code-only, diff-crop, latent-window, latent-context and tabular image-diff.
- Result: The negative survives. Existing score families rank `0/13` top8 recover targets first, and only `5-6/13` are in rank `<=4`. The single current bad change is rank `1` under every score family. Several recover targets are top8 ranks `6/7` and are absent from top4 score sources.
- Why it failed or was negative: Updating the upper-bound target does not fix the score-ordering problem. The failed scores remain confidently attracted to the bad current change and insufficiently rank the deep OCR-improving candidates.
- Valid negative result or invalid run: valid diagnostic negative. OCR outcomes define held-out strata only; no held-out OCR strings are used as deployable features.
- Evidence paths: `experiments/reports/eval300_actual_assignment_failure_feature_audit_top8_2026_06_26.md`, `experiments/results/eval300_actual_assignment_failure_feature_audit_top8_2026_06_26.json`, contact sheets under `experiments/figures/eval300_actual_assignment_failure_feature_audit_top8_2026_06_26/`, manifest `experiments/manifests/eval300_actual_assignment_failure_feature_audit_top8_2026_06_26.yaml`, W&B `vtufh0fw`.
- Should it be retried? Do not retry threshold, budget, consensus, rank calibration, shallow MLP calibration or score-only verifier variants over these score files. Retry only with a materially new candidate-local code-effect representation or a properly costed OCR/text-aware verifier.

## N085 — Glyph-code-only logistic verifier collapses toward no-op
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_glyphcode_verifier_crossfit_2026_06_26`
- Hypothesis: The N084 glyph-code differential features can form a compact verifier with explicit no-op suppression that recovers top8 oracle changes while avoiding the known bad current change.
- Configuration: Added `scripts/train_top8_glyphcode_verifier.py`. The verifier trains balanced logistic regression with real-seed cross-fitting over N084 glyph-code features plus rank/no-op indicators, then tunes a train-fold margin threshold against false/wrong/missed policy costs.
- Result: The verifier is safe-ish only because it barely moves. It changes `1.0` group per fold on average, keeps `19.33` groups exactly, but misses `5.0` oracle-change groups, creates `0.33` false changes and `0.33` wrong changes, and reaches only mean Tesseract delta `-0.67` with PARSeq delta `0`. Seed0 worsens Tesseract by `+1`, seed1 improves `-3`, and seed2 is neutral.
- Why it failed or was negative: The features contain rank-support signal, but the simple linear verifier cannot separate no-op groups from deep-rank useful changes at policy precision. Train-fold margin tuning collapses toward abstention instead of recovering the top8 oracle headroom.
- Valid negative result or invalid run: valid negative result. The initial `.venv` attempt failed before producing outputs because scikit-learn was unavailable; the recorded run uses `.venv-mlic` with `sklearn==1.4.2` and complete manifest hashes.
- Evidence paths: `experiments/reports/eval300_top8_glyphcode_verifier_crossfit_2026_06_26.md`, `experiments/results/eval300_top8_glyphcode_verifier_crossfit_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_glyphcode_verifier_crossfit_2026_06_26.yaml`, W&B `2hkuk3tl`.
- Should it be retried? Do not retry glyph-code-only logistic/margin calibration, thresholds, or small linear variants. Retry only by combining glyph-code evidence with materially richer candidate-local code-effect/OCR-text evidence and a different policy-level decision structure.

## N086 — Nearest-biased glyph-code scalar score suppresses all recover targets
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_glyphcode_verifier_score_audit_2026_06_26`
- Hypothesis: The N085 failure can be explained by verifier score ordering and margins, revealing whether glyph-code features should be retried as a scalar score or kept only as auxiliary evidence.
- Configuration: Added `scripts/analyze_top8_glyphcode_verifier_score_audit.py`, reproduced the N085 real-seed cross-fit logistic verifier, and stored per-group selected status, oracle score rank, oracle-vs-nearest score margin and standardized coefficient summaries.
- Result: The scalar score is structurally nearest-biased. `is_nearest` has the largest positive mean standardized coefficient (`1.4542`). For the 13 recover-top8 oracle-change groups, the oracle candidate is never score-rank 1, only `3/13` are score-rank <=2 and `7/13` are score-rank <=4; the mean oracle-minus-nearest score margin is `-0.8163`. The original top8 shortlist contains all recover candidates, so this is a verifier scoring failure, not first-stage recall.
- Why it failed or was negative: A single binary oracle-code score must reward nearest/no-op rows for 57 no-op groups while also finding rare non-nearest hard positives. The logistic verifier resolves that conflict by learning a strong nearest prior, which suppresses exactly the recover candidates OSCAR needs.
- Valid negative result or invalid run: valid diagnostic negative. It explains N085 and does not define a promoted selector.
- Evidence paths: `experiments/reports/eval300_top8_glyphcode_verifier_score_audit_2026_06_26.md`, `experiments/results/eval300_top8_glyphcode_verifier_score_audit_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_glyphcode_verifier_score_audit_2026_06_26.yaml`, W&B `c1hjhcsa`.
- Should it be retried? Do not retry a single glyphcode-only nearest-vs-candidate scalar score. Retry with a two-stage or multi-head decision where no-op acceptance and hard-positive candidate scoring are separated, and where glyph-code features are auxiliary evidence rather than the only score source.

## N087 — Non-nearest-only glyph-code ranking is still too weak
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_glyphcode_nonnearest_ranker_2026_06_26`
- Hypothesis: If N085/N086 failed mainly because nearest/no-op positives dominate the scalar score, removing nearest rows from candidate scoring should recover hard-positive top8 oracle candidates.
- Configuration: Added `scripts/train_top8_glyphcode_nonnearest_ranker.py`. The ranker trains only on non-nearest rows using glyph-code features plus top8 rank, then evaluates candidate ranking separately from no-op acceptance. The oracle-change-only policy changes groups only when the held-out oracle code differs from nearest, so it isolates candidate ranking and is not deployable.
- Result: Removing nearest rows does not rescue glyph-code scalar ranking. Among 13 recover-top8 groups, only `1/13` oracle codes are non-nearest score-rank 1, `3/13` are rank <=2 and `8/13` are rank <=4. The oracle-change-only policy makes 12 wrong recover changes; recover best-nonnearest Tesseract delta sums to only `-1` edit character. Seed1 reaches `-3` with 5 wrong changes, while seed2 changes 11 groups for zero net Tesseract gain.
- Why it failed or was negative: The nearest prior is not the only bottleneck. Glyph-code scalar features have some top4 support but do not identify the exact OCR-useful code among neighboring non-nearest candidates with enough precision.
- Valid negative result or invalid run: valid diagnostic negative. It is not a promoted selector and does not use actual `.oscr` streams beyond the existing verified candidate table.
- Evidence paths: `experiments/reports/eval300_top8_glyphcode_nonnearest_ranker_2026_06_26.md`, `experiments/results/eval300_top8_glyphcode_nonnearest_ranker_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_glyphcode_nonnearest_ranker_2026_06_26.yaml`, W&B `ra6wajli`.
- Should it be retried? Do not retry glyph-code-only non-nearest logistic ranking, thresholds, or shallow scalar variants. Future two-stage assignment needs richer structured candidate-local code-effect or OCR/text-aware evidence, with glyph-code features at most auxiliary.

## N089 — Tiny evidence-cache patch ranker does not beat the glyphcode floor
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_evidence_cache_nonnearest_ranker_2026_06_26`
- Hypothesis: A small Conv2d patch/code-effect ranker over source/nearest/candidate/diff channels plus codebook/glyph scalars can improve non-nearest candidate ranking over N087's glyphcode-only floor.
- Configuration: Added `scripts/train_top8_evidence_cache_nonnearest_ranker.py`. It trains a tiny Conv2d branch plus tabular MLP on the N088 evidence cache, using BCE over non-nearest rows only and real-seed cross-fit evaluation. The oracle-change-only policy isolates candidate ranking and is not deployable.
- Result: The cache/model does not improve the candidate-ranking floor. Recover-top8 oracle rank1 remains `1/13`, while rank<=4 drops from N087's `8/13` to `5/13`. The oracle-change-only policy still produces many wrong changes: seed1 has 6 changed groups, 6 wrong changes and Tesseract `-1`; seed2 has 11 changed groups, 9 wrong changes and Tesseract `-1`.
- Why it failed or was negative: The first patch/code-effect cache is useful infrastructure, but the tiny BCE ranker is too weak and likely under-supervised for sparse hard positives. Attaching a no-op acceptor to this ranker would inherit a poor candidate scorer.
- Valid negative result or invalid run: valid diagnostic negative. The run used CUDA successfully and produced a complete manifest; no actual `.oscr` promotion streams were generated because the candidate-ranking precheck failed.
- Evidence paths: `experiments/reports/eval300_top8_evidence_cache_nonnearest_ranker_2026_06_26.md`, `experiments/results/eval300_top8_evidence_cache_nonnearest_ranker_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_evidence_cache_nonnearest_ranker_2026_06_26.yaml`, W&B `dn4pvr3o`.
- Should it be retried? Do not retry the same tiny BCE cache ranker or attach a no-op gate to it. Retry with pair/listwise ranking, stronger hard-positive supervision, broader training data or richer latent/code-token context.

## N090 — Listwise evidence-cache ranking still cannot identify recover codes
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_evidence_cache_listwise_ranker_2026_06_26`
- Hypothesis: N089 may have failed because row-wise BCE is misaligned with the group ranking objective; a listwise CE plus pairwise-margin loss over oracle-change non-nearest groups should rank the true top8 oracle code higher.
- Configuration: Added `scripts/train_top8_evidence_cache_listwise_ranker.py`. It trains the same tiny Conv2d patch branch plus tabular MLP over the N088 evidence cache, but uses held-out-real-seed listwise CE on oracle-change groups only, with five model seeds per fold and pairwise margin auxiliary. It evaluates only an oracle-change-only isolation policy, not a deployable no-op selector.
- Result: The listwise objective does not rescue candidate ranking. Recover-top8 oracle rank1 falls to `0/13`, rank<=2 is `2/13`, and rank<=4 is `7/13`, still below N087's glyphcode-only rank<=4 floor of `8/13`. The oracle-change-only policy makes all 13 recover groups wrong changes; seed1 has 6 wrong changes and seed2 has 10 wrong changes, with only `-1` Tesseract edit on each.
- Why it failed or was negative: The bottleneck is not just BCE-vs-listwise objective mismatch. The tiny N088 cache has only 17 oracle-change training groups per full pool and does not expose enough OCR/text-conditioned evidence to distinguish hard-positive code effects from neighboring non-nearest alternatives.
- Valid negative result or invalid run: valid diagnostic negative. The run used CUDA successfully, logged W&B offline, hash-verified the manifest, and did not generate actual `.oscr` promotion streams because the rank precheck failed.
- Evidence paths: `experiments/reports/eval300_top8_evidence_cache_listwise_ranker_2026_06_26.md`, `experiments/results/eval300_top8_evidence_cache_listwise_ranker_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_evidence_cache_listwise_ranker_2026_06_26.yaml`, W&B `l35bw7qy`.
- Should it be retried? Do not retry tiny N088-cache BCE/listwise/pairwise variants or attach a no-op acceptor to them. Retry only with broader candidate supervision, richer latent/code-token context, or OCR/text-aware train-teacher features that are not tuned on held-out OCR labels.

## N092 — Train-safe tabular score thresholds do not transfer to validation
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_trainval_tabular_threshold_policy_2026_06_26`
- Hypothesis: N091's broader train-partition tabular ranker has enough candidate-order signal that a conservative train-tuned no-op/abstention threshold can recover useful validation changes while suppressing false/wrong changes.
- Configuration: Added `scripts/evaluate_top8_trainval_tabular_threshold_policy.py`. It retrains the N091 ranker, sweeps best-nonnearest score thresholds on train-partition groups, and evaluates selected policies on the val partition without using validation labels for tuning.
- Result: The threshold score is not calibrated across train/val. The train zero-error-safe threshold changes 20 useful train groups for Tesseract `-27`, but on val it changes one false group, recovers no oracle changes and gets Tesseract `0`. The one-error train policy changes 36 train groups for `-46`, but val again recovers no useful changes and makes 3 false changes. The aggressive lowest-threshold control gets val Tesseract `-7`, but only by changing 72 groups with 65 false/wrong selections, still below the counted current `-8` floor and far below publication precision.
- Why it failed or was negative: Candidate ranking signal exists, but a single score threshold is a poor no-op acceptor. Score scale and hard-positive/no-headroom separation shift between train and val, so train-safe thresholds abstain from the useful validation changes while aggressive thresholds mostly fire on false no-headroom groups.
- Valid negative result or invalid run: valid policy-calibration negative. It uses train-only threshold tuning, logs W&B offline, hash-verifies the manifest, and intentionally does not generate actual `.oscr` streams because the pre-promotion policy criterion failed.
- Evidence paths: `experiments/reports/eval300_top8_trainval_tabular_threshold_policy_2026_06_26.md`, `experiments/results/eval300_top8_trainval_tabular_threshold_policy_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_trainval_tabular_threshold_policy_2026_06_26.yaml`, W&B `nme6oouw`.
- Should it be retried? Do not retry single best-score threshold sweeps over the same N091 scalar score. Retry only with a richer no-op acceptor using score margins, group uncertainty, source/domain calibration, OCR/text-aware train-teacher verification, or explicit hard no-headroom negative modeling.

## N093 — Small group acceptors over N091 score summaries overfit train and miss val positives
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_trainval_group_acceptor_2026_06_26`
- Hypothesis: N092 may have failed because a single best-candidate score threshold is too weak; a group-level acceptor using score margins, score distribution statistics, source/domain one-hot and best-candidate deployable tabular features should learn a better no-op decision.
- Configuration: Added `scripts/train_top8_trainval_group_acceptor.py`. It retrains the N091 ranker, builds one group feature vector per train/val group from N091 score statistics plus deployable scalar/tabular candidate features, trains logistic, HistGradientBoosting and random forest acceptors on train groups, and threshold-tunes only on train.
- Result: All acceptors overfit train and fail validation. Each model's train-selected policy accepts all 63 useful train changes with no false/wrong changes and Tesseract `-87`, but validation accepts only 2-3 false no-headroom groups, recovers zero exact changed groups and gets Tesseract `0`. This is worse than N091's oracle-change-only `-12` and does not approach the counted current `-8` floor.
- Why it failed or was negative: The hard-positive/no-headroom calibration shift is not solved by small post-hoc models over the current score/tabular evidence. The train split contains separable artifacts that do not align with validation positives.
- Valid negative result or invalid run: valid calibration negative. Inputs exclude OCR deltas, oracle labels and OCR prediction text; labels are used only for train supervision and evaluation. The run logged W&B offline, hash-verified the manifest and intentionally did not generate actual `.oscr` streams.
- Evidence paths: `experiments/reports/eval300_top8_trainval_group_acceptor_2026_06_26.md`, `experiments/results/eval300_top8_trainval_group_acceptor_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_trainval_group_acceptor_2026_06_26.yaml`, W&B `qjbxkbp7`.
- Should it be retried? Do not retry small logistic/tree acceptors over the same N091 score summaries and scalar/tabular evidence. Retry only with materially new evidence, such as OCR/text-aware train-teacher verification, richer latent/code-token context, or a source/domain split diagnostic that explains the train-to-val mismatch.

## N095 — Fixed train-teacher string consensus collapses to no-op
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_trainval_teacher_consensus_2026_06_26`
- Hypothesis: A stricter OCR/text-aware verifier using PARSeq, CRNN and ABINet agreement may provide safe no-op calibration for Eval300 top8 assignment without using held-out Tesseract labels.
- Configuration: Extracted CRNN and ABINet OCR/loss for all `2400` Eval300 top8 trainval candidate rows with `scripts/run_strhub_teacher_loss.py`, then evaluated fixed teacher-consensus rules in `scripts/analyze_top8_trainval_teacher_consensus.py`. Rules used PARSeq/CRNN/ABINet string edit deltas and optional CRNN/ABINet teacher-loss no-worse guards. Held-out Tesseract deltas were used only for evaluation.
- Result: The strict teacher-consensus rules are safe only because they almost never move. On the validation partition, the best fixed policy changes `0/75` groups, recovers `0/17` oracle changes and gets Tesseract delta `0`. The looser any-string-improve rules change only one no-headroom group and still recover no useful changes. No val oracle-change row passes the strict consensus filters (`0/17`).
- Why it failed or was negative: CRNN/ABINet/PARSeq string-level agreement is too insensitive to the code-index changes that help held-out Tesseract on this top8 substrate. It is the opposite failure mode of teacher-loss argmax: loss is too permissive, while strict string consensus is too conservative.
- Valid negative result or invalid run: valid diagnostic negative. It is not a counted `.oscr` promotion and is explicitly non-deployable unless teacher inference cost/distillation is accounted.
- Evidence paths: `experiments/reports/eval300_top8_trainval_teacher_consensus_2026_06_26.md`, `experiments/results/eval300_top8_trainval_teacher_consensus_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_trainval_teacher_consensus_2026_06_26.yaml`, W&B `3ufv2ktz`/`36z5gtmk`/`cw71blzg`.
- Should it be retried? Do not retry fixed PARSeq/CRNN/ABINet string-consensus rules, teacher string-delta argmax, or teacher string no-worse filters as the selector. Reuse the full Eval300 teacher-output cache only for auxiliary regularization, diagnostic stratification or a verifier with materially richer deployable candidate-local code-effect evidence.

## N096 — N091 tabular candidate ranking is not source-robust enough
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_trainval_leave_source_out_ranker_2026_06_26`
- Hypothesis: The partial-positive N091 candidate ranker may generalize across source domains, leaving no-op acceptance as the only bottleneck.
- Configuration: Added `scripts/analyze_top8_trainval_leave_source_out_ranker.py`. For each source, trained the same deployable-feature tabular listwise ranker on all other sources and evaluated candidate ranking on the held-out source. This is a diagnostic split, not a deployment selector; no counted `.oscr` streams were generated.
- Result: Source-disjoint candidate ranking is much weaker than the ordinary N091 split. Across `80` held-out-source oracle-change groups, only `20` are score-rank 1 and `54` are rank<=4; the first-stage top8 contains all `80`. The oracle-change-only isolation policy has Tesseract delta `-33` and PARSeq `0`, but makes `60` wrong changed selections. IAM and ICDAR2015 retain some signal, while TextOCR/ICDAR2013 are especially weak.
- Why it failed or was negative: The N091 score learned useful in-split artifacts but does not reliably identify the exact OCR-useful code under source holdout. Therefore a no-op acceptor on top of this scorer would inherit a candidate-ranking transfer problem, not just an abstention problem.
- Valid negative result or invalid run: valid diagnostic negative. The first `.venv` attempt failed before outputs because torch was unavailable; the recorded run uses `.venv-mlic` and has a verified manifest.
- Evidence paths: `experiments/reports/eval300_top8_trainval_leave_source_out_ranker_2026_06_26.md`, `experiments/results/eval300_top8_trainval_leave_source_out_ranker_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_trainval_leave_source_out_ranker_2026_06_26.yaml`, W&B `po46bnrf`.
- Should it be retried? Do not use N091's ordinary split result as source-robust evidence. Retry with source/domain-balanced calibration labels, richer latent/code-token candidate context, or a source-aware objective; do not merely tune thresholds or small acceptors on the same scalar score.

## N097 — Feature widening and source-balanced loss do not fix source-robust top8 ranking
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_trainval_source_robust_feature_ablation_2026_06_26`
- Hypothesis: N096's source-transfer failure may come from code one-hot/source shortcuts or unbalanced source loss; removing shortcut-prone features or using source-balanced loss might recover robust candidate ranking.
- Configuration: Added `scripts/analyze_top8_trainval_source_robust_feature_ablation.py`. For each held-out source, compared five deployable feature families (`all_tabular`, `continuous_no_onehot`, `codegeom_no_onehot`, `imgdiff_only`, `topk_error_only`) under standard and source-balanced listwise losses. The completed diagnostic used `80` CPU epochs and one model seed; a first `250`-epoch GPU attempt was interrupted before outputs because Python group-loss loops were too slow, and no metrics from that attempt were used.
- Result: The simplest first-stage prior wins: `topk_rank + assignment_relative_error` reaches rank1 `36/80` and rank<=4 `67/80`, exactly matching the first-stage top8 rank counts, with oracle-change-only Tesseract `-44` and PARSeq `-1`. Broader features are worse: `all_tabular` standard is rank1 `23/80`, rank<=4 `57/80`; source-balanced all-tabular improves rank1 only to `27/80` and rank<=4 drops to `53/80`. Image-diff-only is intermediate (`31/80` rank1 under source-balanced) but still has `49` wrong changed selections.
- Why it failed or was negative: Shallow models over wider scalar/code/image features appear to learn source/code-specific shortcuts and underperform the simple top-k/rate prior under source holdout. Source-balanced loss changes the trade-off but does not solve wrong-candidate selection or no-op safety.
- Valid negative result or invalid run: completed run is a valid diagnostic negative for feature widening/source-balanced loss alone. The interrupted first attempt is explicitly recorded as interrupted and excluded from metrics. No counted `.oscr` streams were generated because this is still an oracle-change-only candidate-ranking isolation study.
- Evidence paths: `experiments/reports/eval300_top8_trainval_source_robust_feature_ablation_2026_06_26.md`, `experiments/results/eval300_top8_trainval_source_robust_feature_ablation_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_trainval_source_robust_feature_ablation_2026_06_26.yaml`, W&B `ny4rt902`.
- Should it be retried? Do not retry wider all-tabular feature families or source-balanced loss alone as a promotion path. Keep the simple top-k/rate prior as a source-robust anchor, and retry only with materially richer candidate-local OCR/latent evidence plus a separate no-op/verifier.

## N098 — Top-k prior plus shallow train-tuned no-op acceptor is not enough
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_topk_prior_noop_policy_2026_06_26`
- Hypothesis: N097 showed that `topk_rank + assignment_relative_error` is the most source-robust candidate-order prior; fixing the candidate to this top prior and training/tuning a shallow no-op acceptor on train may yield a deployable table-level policy.
- Configuration: Added `scripts/evaluate_top8_topk_prior_noop_policy.py`. For each group, selected the top non-nearest first-stage prior candidate, then evaluated train-tuned one-dimensional feature thresholds and shallow exact-change classifiers. Inputs exclude OCR/oracle/label/source/reference text features; labels are used only for train tuning and evaluation.
- Result: The raw top-prior candidate is not a selector: changing all validation groups gives Tesseract `+3`, exact changed `6/17`, and `69` false/wrong changes. The best train-tuned policy (`feature_le_img_source_nearest_edge_density_delta`, train best-Tesseract/PARSeq-safe) reaches only validation Tesseract `-2`, exact changed `1/17`, false `6`, wrong `0`, PARSeq `0`. Zero/one-error train-safe policies mostly collapse to no-op on validation.
- Why it failed or was negative: The top-k/rate prior is useful as an ordering anchor, but shallow acceptors still cannot separate high-yield hard positives from no-headroom false changes. The same train-to-val calibration problem persists even with a simpler candidate prior.
- Valid negative result or invalid run: valid table-level policy negative. It is explicitly below the current actual-bitstream `-8` Tesseract floor and therefore no `.oscr` export was generated.
- Evidence paths: `experiments/reports/eval300_top8_topk_prior_noop_policy_2026_06_26.md`, `experiments/results/eval300_top8_topk_prior_noop_policy_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_topk_prior_noop_policy_2026_06_26.yaml`, W&B `cs1us9r1`.
- Should it be retried? Do not retry shallow top-k-prior one-dimensional/logistic no-op acceptors. Keep the top-k/rate prior as an anchor only, and retry with richer candidate-local verifier evidence or new source/domain-balanced calibration labels.

## N099 — Top-k prior plus latent/code-token verifier still overfits
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_topk_prior_latent_verifier_2026_06_26`
- Hypothesis: The N097 top-k/rate prior may become deployable if the acceptor sees materially richer local code-effect evidence: code-center geometry, latent target/residual vectors and local latent-window tensors.
- Configuration: Added `scripts/train_top8_topk_prior_latent_verifier.py`. For each group, fixed the candidate to the top non-nearest top-k/rate prior and trained a small verifier with core scalar features, code-center MLPs, latent-vector MLPs and a local latent-window CNN. Targets were `exact_changed` and `safe_improve`; thresholds were tuned only on the train partition. OCR/oracle labels are used only for train supervision and evaluation, not as input features.
- Result: The richer verifier fits train but fails validation. The best `exact_changed` seed changes 30 train groups with 30 exact changes and no false/wrong changes (`-41` Tesseract), but validation changes only 3 groups, recovers 1 exact useful change, makes 2 false/wrong changes and reaches Tesseract `-3`, PARSeq `0`. Other exact seeds are similar, and all `safe_improve` target runs collapse to no-op on validation.
- Why it failed or was negative: The candidate-local latent/code evidence is not enough under the current train/val split and objective. It improves train memorization but does not separate hard positives from no-headroom groups well enough to beat the promoted actual-bitstream baseline (`-8` Tesseract, PARSeq unchanged).
- Valid negative result or invalid run: valid diagnostic negative for the completed 225/75 split. A first local attempt is invalid and excluded because the partition join omitted `seed` from the feature artifact key, producing a wrong `221/79` split; no metrics from that invalid attempt were logged or used.
- Evidence paths: `experiments/reports/eval300_top8_topk_prior_latent_verifier_2026_06_26.md`, `experiments/results/eval300_top8_topk_prior_latent_verifier_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_topk_prior_latent_verifier_2026_06_26.yaml`, W&B `zmk5957j`.
- Should it be retried? Do not retry this latent/code-token verifier as a seed/capacity sweep or shallow objective tweak. Future attempts need new calibration evidence, explicit policy-level false/wrong-change budget optimization, harder no-headroom negatives, or teacher-output auxiliary regularization distilled into deployable code-effect evidence.

## N100 — Source-OOF calibration makes the top-k verifier safe only by no-op
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_topk_prior_oof_calibrated_verifier_2026_06_26`
- Hypothesis: N099's train overfit may be caused by in-sample threshold tuning; tuning thresholds on train-source out-of-fold scores, with hard-negative weights and pairwise margins, may produce a safer top-k-prior verifier.
- Configuration: Added `scripts/train_top8_topk_prior_oof_calibrated_verifier.py`. Candidate choice is still fixed to the N097 top-k/rate prior. For each target (`exact_changed`, `safe_improve`), loss mode (`standard`, `hardneg4`, `hardneg8`) and pairwise weight (`0`, `0.25`), trained source-domain OOF models on the six train sources and selected a threshold on OOF train scores under false/wrong/PARSeq/Tesseract budget constraints. Also trained a full-train model and applied the same OOF threshold.
- Result: Source-domain OOF calibration is too conservative. The best OOF-ensemble validation policy changes `0/75` groups and gets Tesseract delta `0`, PARSeq `0`. Applying the same OOF threshold to a full-train model changes `5` validation groups, recovers `2` exact useful changes, has `3` false/wrong changes and reaches only Tesseract `-2`, PARSeq `0`. This is again below the promoted actual-bitstream `-8` floor.
- Why it failed or was negative: OOF calibration prevents in-sample train memorization, but the score boundary no longer admits useful validation changes. Full-train scoring with the OOF threshold recovers a couple of positives but still admits too many false/wrong groups and does not improve over N098/N099.
- Valid negative result or invalid run: valid diagnostic negative for the `--group-key source` run. A first local run with default `source_index` is superseded and excluded because that key is image/group index rather than source-domain label; it was overwritten before W&B logging or manifest creation.
- Evidence paths: `experiments/reports/eval300_top8_topk_prior_oof_calibrated_verifier_2026_06_26.md`, `experiments/results/eval300_top8_topk_prior_oof_calibrated_verifier_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_topk_prior_oof_calibrated_verifier_2026_06_26.yaml`, W&B `he3yxldr`.
- Should it be retried? Do not retry top-k-prior latent/code verification by adding only source-OOF thresholds, hard-negative scalar weights, pairwise margins, seeds or capacity. Retry only with a different policy-level objective, larger/source-balanced calibration labels, or train-teacher auxiliary supervision that improves deployable candidate evidence rather than only threshold calibration.

## N101 — Larger same-kind source-balanced calibration still collapses the top-k verifier to no-op
- Date: 2026-06-26
- Experiment ID: `E_eval600_to_eval300_top8_topk_prior_oof_calibrated_verifier_2026_06_26`
- Hypothesis: N100 may have been underpowered by the small Eval300 train partition; using the larger merged Eval600 oracle-safe, hard-noheadroom, IAM-supplement and balanced-no-safe artifact with source-domain OOF calibration may provide enough source-balanced calibration for the top-k-prior latent/code verifier.
- Configuration: Reused `scripts/train_top8_topk_prior_oof_calibrated_verifier.py` on `experiments/features/eval600_oraclesafe123_hardnoheadroom120_iamsupp20_balanced210_plus_eval300_seed1val_top8_windows_2026_06_26.npz` and `experiments/results/eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_2026_06_26.jsonl`. The split contains `393` train, `109` val and `46` unused test groups. Candidate choice is still fixed to the N097 top-k/rate prior; source-domain OOF thresholds are tuned on train only for `exact_changed` and `safe_improve` targets, three hard-negative loss modes and two pairwise weights.
- Result: Every tested condition collapses to no-op on validation. Both the OOF ensemble and the full-train model with the OOF threshold change `0/109` validation groups, recover `0/21` exact useful changes and produce Tesseract `0`, PARSeq `0`. This is safer than harmful selection but does not approach the promoted actual-bitstream baseline (`-8` Tesseract, PARSeq unchanged).
- Why it failed or was negative: Adding more same-kind source-balanced calibration labels to this verifier family raises the threshold rather than improving recoverability. The model still lacks deployable evidence or objective structure that separates Eval300 hard positives from no-headroom groups under a policy-level false/wrong budget.
- Valid negative result or invalid run: valid diagnostic negative. The run uses train-only OOF calibration, logs W&B offline, hash-verifies the manifest and intentionally does not generate actual `.oscr` streams because the pre-promotion table policy fails.
- Evidence paths: `experiments/reports/eval600_to_eval300_top8_topk_prior_oof_calibrated_verifier_2026_06_26.md`, `experiments/results/eval600_to_eval300_top8_topk_prior_oof_calibrated_verifier_2026_06_26.json`, manifest `experiments/manifests/eval600_to_eval300_top8_topk_prior_oof_calibrated_verifier_2026_06_26.yaml`, W&B `43an1dgl`.
- Should it be retried? Do not retry this top-k-prior latent/code verifier by adding only larger same-kind calibration labels, OOF thresholds, hard-negative scalar weights, pairwise weights, seeds or capacity. Future data scaling must pair new labels with materially new candidate-local code-effect evidence, a genuine policy-level false/wrong objective, or teacher-output regularization distilled into deployable features.

## N102 — Teacher-loss auxiliary ranker is only a small unstable candidate-ranking gain
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_latent_teacher_aux_ranker_2026_06_26`
- Hypothesis: N095's fixed teacher consensus was too conservative, but CRNN/ABINet teacher loss may still help if distilled as auxiliary supervision into a deployable latent/code candidate ranker rather than used directly as a selector.
- Configuration: Added `scripts/train_top8_latent_teacher_aux_ranker.py`. It trains a top-8 listwise latent/code ranker over the Eval300 train partition, using CRNN/ABINet teacher-loss utility only through an auxiliary prediction head. Teacher predictions/loss/confidence, reference length, PARSeq nearest distance and Tesseract nearest distance are excluded from selector inputs. Tested teacher weights `0`, `0.05`, `0.20` with three model seeds.
- Result: The best teacher-auxiliary run (`teacher_weight=0.20`, seed `0`) improves validation exact changed/rank1 from the weight-0 baseline `7/17` to `8/17`, with Tesseract `-12` and PARSeq `0` under oracle-change-only evaluation. The improvement is unstable: the other `0.20` seeds drop to `4/17`; all runs still need oracle knowledge to decide which groups change, and the best run makes `9` wrong candidate selections among the `17` oracle-change groups.
- Why it failed or was negative: Teacher auxiliary loss carries some representation signal, but the current objective does not produce a stable deployable candidate scorer or a no-op/change acceptor. The result does not beat the current actual-bitstream policy under a deployable policy check, because no no-op acceptor is evaluated and wrong-candidate pressure remains high.
- Valid negative result or invalid run: valid diagnostic partial-positive/non-promotion result. A first local run was superseded before W&B/manifest because it reported `exact` without separating correct nearest/no-op from exact changed groups; the recorded run fixes the metric and hash-verifies the manifest.
- Evidence paths: `experiments/reports/eval300_top8_latent_teacher_aux_ranker_2026_06_26.md`, `experiments/results/eval300_top8_latent_teacher_aux_ranker_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_latent_teacher_aux_ranker_2026_06_26.yaml`, W&B `bbfhzt6p`.
- Should it be retried? Do not promote this ranker, attach a shallow threshold directly, or treat the seed-0 teacher-weight gain as robust. Retry only with source-disjoint/cross-fit stability checks, an explicit policy-level no-op/false-wrong objective, or a stronger candidate-local representation while retaining the rule that teacher signals are auxiliary and not inference inputs.

## N103 — Teacher auxiliary helps aggregate OCR utility more than exact oracle-code rank
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_latent_teacher_aux_source_loo_2026_06_26`
- Hypothesis: The small N102 teacher-auxiliary gain may be source-robust, or may reveal that teacher loss optimizes OCR utility rather than exact oracle-code identity.
- Configuration: Added `scripts/analyze_top8_latent_teacher_aux_source_loo.py`. For each source domain, held that source out as validation and trained the N102 latent/code ranker on the remaining sources. Compared teacher weights `0` and `0.20` with one model seed. This is a diagnostic split over existing Eval300 groups, not a deployable selector.
- Result: Teacher weight `0.20` improves aggregate held-out-source Tesseract utility from `-41` to `-46`, but exact changed groups drop from `24/80` to `23/80`, rank<=4 drops from `59/80` to `57/80`, and wrong choices increase from `56` to `57`. PARSeq changes from `-1` to `0`.
- Why it failed or was negative: Exact oracle-code rank is not the right sole objective if teacher auxiliary loss steers the model toward OCR-useful alternative codes. However, wrong-candidate pressure remains far too high for deployment, and no no-op/false-change acceptor is evaluated.
- Valid negative result or invalid run: valid diagnostic partial-positive/non-promotion result. It is source-held-out and hash-verified, but uses oracle-change-only evaluation and does not export counted `.oscr` streams.
- Evidence paths: `experiments/reports/eval300_top8_latent_teacher_aux_source_loo_2026_06_26.md`, `experiments/results/eval300_top8_latent_teacher_aux_source_loo_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_latent_teacher_aux_source_loo_2026_06_26.yaml`, W&B `byfrp6jn`.
- Should it be retried? Do not optimize only exact oracle-code rank or attach a shallow threshold. Retry with a policy-level OCR-utility objective that explicitly penalizes false/wrong/no-op errors and then validate with a deployable no-op acceptor before any actual `.oscr` export.

## N104 — No-op-inclusive train OCR utility hard targets do not transfer
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_latent_policy_utility_selector_2026_06_26`
- Hypothesis: A no-op-inclusive latent/code selector trained directly on train-partition OCR utility labels may solve the false/no-op boundary that exact-code rankers and post-hoc acceptors miss.
- Configuration: Added `scripts/train_top8_latent_policy_utility_selector.py`. It uses train Tesseract/PARSeq deltas to define a group target among nearest/no-op and top-8 candidates, with penalties for PARSeq and Tesseract worsening. This is non-promotable because the same OCR family used for validation utility defines the train target.
- Result: The best validation seed changes `7/75` groups, but only `2` are exact useful changes; `4` are false changes, `1` is a wrong change, `14` oracle-change groups are missed, Tesseract delta is `0`, PARSeq delta is `0`, and `2` samples worsen under Tesseract. Other seeds are worse (`+1` and `+2` Tesseract).
- Why it failed or was negative: Direct train OCR utility targets change the objective shape but do not solve train-to-validation no-headroom calibration. The model learns to move on several false groups while missing most useful validation changes.
- Valid negative result or invalid run: valid non-promotable diagnostic. It logs W&B and hash-verifies the manifest, but it intentionally does not export actual `.oscr` streams because the table policy fails and uses train Tesseract/PARSeq utility labels.
- Evidence paths: `experiments/reports/eval300_top8_latent_policy_utility_selector_2026_06_26.md`, `experiments/results/eval300_top8_latent_policy_utility_selector_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_latent_policy_utility_selector_2026_06_26.yaml`, W&B `nabzwagk`.
- Should it be retried? Do not retry hard train-Tesseract utility targets as the main selector or claim. Retry only with teacher-separated utility, cross-fit pseudo-utility, explicit no-headroom modeling, and held-out OCR validation before any counted `.oscr` export.

## N105 — Hard teacher-loss utility targets do not solve the no-op boundary
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_latent_teacher_utility_selector_2026_06_26`
- Hypothesis: N104 may fail because train Tesseract/PARSeq labels are too evaluator-specific; CRNN/ABINet teacher-loss utility could provide a teacher-separated no-op-inclusive target that transfers to held-out Tesseract/PARSeq validation while keeping teacher outputs out of inference inputs.
- Configuration: Added `scripts/train_top8_latent_teacher_utility_selector.py`. The model uses the same deployable latent/code evidence as the N102/N104 family, defines group targets from CRNN/ABINet loss utility among nearest/no-op and top8 candidates, and sweeps `min_gain` values `0.0`, `0.05`, `0.10` over three model seeds. Teacher losses are train targets only, not selector inputs.
- Result: The policy remains non-promotable. Aggressive `min_gain=0.0` settings change `69-75/75` validation groups but recover only `2-3` exact useful changes and create `54-58` false changes plus `12-15` wrong changes. The best conservative setting changes `4/75` groups, recovers `0` exact useful changes, creates `3` false changes and `1` wrong change, misses `16/17` oracle-change groups, leaves Tesseract and PARSeq aggregate deltas at `0`, and worsens one Tesseract sample.
- Why it failed or was negative: Teacher-loss utility is dense and does not define a sharp deployable no-headroom boundary when used as a hard policy target. Tightening `min_gain` suppresses false changes by collapsing recall, while loosening it changes many no-op groups.
- Valid negative result or invalid run: valid diagnostic negative. It is teacher-separated and logs W&B/manifest hashes, but intentionally exports no counted `.oscr` streams because the pre-promotion table policy fails.
- Evidence paths: `experiments/reports/eval300_top8_latent_teacher_utility_selector_2026_06_26.md`, `experiments/results/eval300_top8_latent_teacher_utility_selector_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_latent_teacher_utility_selector_2026_06_26.yaml`, W&B `l6uvc0zp`.
- Should it be retried? Do not retry direct hard CRNN/ABINet teacher-loss utility targets, min-gain sweeps, or seed/capacity sweeps as the selector. Use teacher loss only as auxiliary representation shaping or cross-fit calibration evidence inside an explicit false/no-headroom detector with held-out OCR validation and actual-bitstream promotion gates.

## N106 — Source-OOF guards over teacher-aux ranker scores are too conservative
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_teacher_aux_crossfit_guard_2026_06_26`
- Hypothesis: N102/N103's teacher-auxiliary ranker may become deployable if no-op/change acceptance is trained separately on source-held-out ranker predictions rather than in-sample scores.
- Configuration: Added `scripts/train_top8_teacher_aux_crossfit_guard.py`. For teacher weights `0.0` and `0.20`, trained the latent/code teacher-auxiliary ranker in source-OOF fashion over train groups, then trained group acceptors (`logistic`, `hist_gradient`, `random_forest`, plus direct top-score thresholds) on deployable ranker-derived group features. Thresholds were tuned on OOF train predictions with false/wrong budgets `0`, `1`, and `2`; validation used a full-train ranker and no teacher/OCR outputs as inference inputs.
- Result: The guard suppresses almost all changes. Most policies collapse to no-op. The best validation policy is logistic with budget `1`: `2/75` changed groups, `1` exact useful change, `1` false change, `0` wrong changes, `16/17` missed oracle-change groups, Tesseract `-1`, PARSeq `0`, and no Tesseract-worse groups. This is far below the current counted actual-bitstream baseline (`-8` Tesseract, PARSeq unchanged).
- Why it failed or was negative: Source-OOF calibration removes the unsafe in-sample confidence but leaves too little separable high-precision signal in the ranker-derived group features. It improves safety by rejecting nearly everything rather than learning the hard-positive/no-headroom boundary.
- Valid negative result or invalid run: valid diagnostic negative. Manifest hashes check, W&B is logged, and counted `.oscr` export is intentionally blocked because the table policy fails.
- Evidence paths: `experiments/reports/eval300_top8_teacher_aux_crossfit_guard_2026_06_26.md`, `experiments/results/eval300_top8_teacher_aux_crossfit_guard_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_teacher_aux_crossfit_guard_2026_06_26.yaml`, W&B `e5hx93e1`.
- Should it be retried? Do not seed/capacity/threshold-sweep shallow acceptors over the same OOF teacher-auxiliary ranker scores. Retry only with materially new false/no-headroom evidence, richer deployable candidate representations, or a different calibration target that does not reduce to ranker score summaries.

## N107 — CRNN/ABINet logit summaries are not enough for a deployable OCR-aware selector
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_strhub_logit_feature_selector_2026_06_26`
- Hypothesis: Direct CRNN/ABINet logit-distribution summaries may expose richer OCR/text-aware evidence than strings, confidence, or scalar loss, enabling a selector to distinguish recoverable top8 assignment changes from no-headroom groups.
- Configuration: Added `scripts/extract_strhub_logit_features.py` and `scripts/train_top8_strhub_logit_feature_selector.py`. Extracted 62-dimensional logit/probability summaries for all `2400` Eval300 top8 trainval candidate reconstructions from CRNN and ABINet, then trained logistic, HistGradientBoosting and random forest selectors on raw, candidate-minus-nearest and absolute-delta logit features. Evaluated exact-oracle and safe-improve targets with train-tuned false/wrong budgets `0`, `1`, `2`.
- Result: Train-tuned deployable policies are either no-op or harmful. The best validation policy is no-op (`0` changed, Tesseract `0`). The moving exact-oracle logistic policy changes `7` groups but recovers `0` exact useful changes and worsens Tesseract by `+6`. Oracle-change-only candidate ranking is also insufficient: exact-oracle logistic gives rank1 `5/17`, rank<=4 `11/17`, Tesseract `-1` with `12` wrong changes; safe-improve random forest gives Tesseract `-3` but only `2/17` rank1 and `15` wrong changes.
- Why it failed or was negative: Fixed logit summary statistics carry some candidate-ranking information but not a calibrated no-headroom boundary. They also lose sequence alignment detail, so the selector cannot reliably tell useful glyph-level changes from OCR-neutral or harmful text perturbations.
- Valid negative result or invalid run: valid diagnostic negative. It records feature-cache hashes, model checkpoint hashes and W&B, and intentionally exports no counted `.oscr` streams because the table policy fails.
- Evidence paths: `experiments/reports/eval300_top8_strhub_logit_feature_selector_2026_06_26.md`, `experiments/results/eval300_top8_strhub_logit_feature_selector_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_strhub_logit_feature_selector_2026_06_26.yaml`, W&B `lm0rfc66`.
- Should it be retried? Do not retry shallow classifiers or thresholds over the same fixed logit-summary features. Retry only with sequence-aligned OCR evidence, attention/character-position features, a materially different false/no-headroom label, or a learned verifier whose encoder OCR cost is explicitly reported.

## N112 — Binned CRNN/ABINet sequence profiles still do not give a deployable selector
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_strhub_sequence_feature_selector_2026_06_26`
- Hypothesis: N107 may have failed because global logit summaries erase character-position information; binned CRNN/ABINet sequence profiles might preserve enough local OCR evidence to rank useful top8 code changes and calibrate no-op decisions.
- Configuration: Added `scripts/extract_strhub_sequence_profile_features.py` to extract 8-bin sequence profiles over max probability, margin, entropy, normalized predicted ID, transition and blank/pad probability. Each CRNN/ABINet cache has `2400` rows and `194` features; the selector combines raw, candidate-minus-nearest and absolute-delta features for a total dimension of `1169`. Added `scripts/train_top8_strhub_sequence_feature_selector.py`, reusing N107's train-tuned false/wrong budgets over exact-oracle and safe-improve targets.
- Result: Sequence profiles improve some oracle-change-only utility but do not transfer to a deployable policy. The best oracle-change-only run reaches Tesseract `-11` and PARSeq `0`, but it makes `13` wrong candidate selections among the `17` oracle-change validation groups and worsens one Tesseract sample. Train-tuned validation policies collapse to no-op or select false-heavy changes; the best selected policy changes `0/75` groups and gets Tesseract `0`, PARSeq `0`.
- Why it failed or was negative: Position-binned teacher evidence is still not a calibrated no-headroom detector. It can rank some OCR-useful alternatives when forced to change oracle-change groups, but the wrong-candidate rate and train-to-val threshold collapse remain too high for actual `.oscr` promotion.
- Valid negative result or invalid run: valid diagnostic negative. Teacher OCR outputs are inference-time features in this diagnostic, so any future use would need encoder OCR cost accounting or distillation; no counted `.oscr` streams are exported because the table policy fails the current `-8` Tesseract floor.
- Evidence paths: `experiments/reports/eval300_top8_strhub_sequence_feature_selector_2026_06_26.md`, `experiments/results/eval300_top8_strhub_sequence_feature_selector_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_strhub_sequence_feature_selector_2026_06_26.yaml`, W&B `7yx6ljan`.
- Should it be retried? Do not seed/capacity/threshold-sweep shallow selectors over these binned sequence-profile features. If OCR-aware evidence is revisited, use it as auxiliary representation shaping, a costed OCR verifier with a different no-op/headroom target, or sequence-level alignment inside a model that explicitly optimizes policy-level false/wrong/missed tradeoffs.

## N115 — Clean rankcap4 tabular candidate chooser improves utility but remains wrong-heavy
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_rankcap4_candidate_chooser_2026_06_26`
- Hypothesis: N114 shows a clean topk-rankcap<=4 candidate set has a strong upper control (`12/17` exact changes, Tesseract `-17`). A second-stage chooser trained only on deployable tabular codebook/image-diff features may recover enough of this headroom without using provenance-sensitive source_mod5 scores.
- Configuration: Added `scripts/train_top8_rankcap_candidate_chooser.py`. The candidate set is non-nearest rows with `topk_rank <= 4` from the Eval300 top8 trainval table. Features are `topk_rank`, `assignment_relative_error`, `candidate_slot`, `codebook_*` and `img_*` numeric columns, with OCR/label/oracle/reference/source IDs excluded. Trained logistic, HistGradientBoosting and random-forest row classifiers on train groups whose oracle code is inside the rankcap set; evaluated validation under oracle-change-only covered and all-change isolation policies.
- Result: The best model is `random_forest_d4`, with validation rank1 `7/17`, rank<=4 `12/17`, covered-only Tesseract `-13`, PARSeq `0`, but still `5` wrong changes and `5` missed oracle groups. Logistic and HistGradient reach only `6/17` rank1 and Tesseract `-8`/`-9` with `6` wrong and `5` missed.
- Why it failed or was negative: It confirms that the clean rankcap4 shortlist contains useful headroom and that a shallow chooser can improve over raw topk order, but the wrong-candidate rate remains far above the current actual-bitstream precision floor. Thresholding or exporting this table policy would not be scientifically defensible.
- Valid negative result or invalid run: valid diagnostic non-promotion result. It is leakage-controlled relative to source_mod5, hash-verified, and W&B-logged, but it uses oracle-change-only evaluation and exports no counted `.oscr` streams.
- Evidence paths: `experiments/reports/eval300_top8_rankcap4_candidate_chooser_2026_06_26.md`, `experiments/results/eval300_top8_rankcap4_candidate_chooser_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_rankcap4_candidate_chooser_2026_06_26.yaml`, W&B `euh2kyhd`.
- Should it be retried? Do not capacity/seed sweep shallow tabular rankcap choosers. Retry only with materially richer candidate-local representation, teacher-auxiliary representation learning, or a clean first-stage generator that increases held-out rankcap coverage before candidate choice.

## N116 — Local latent-window rankcap chooser does not beat the shallow tabular chooser
- Date: 2026-06-26
- Experiment ID: `E_eval300_top8_rankcap4_window_chooser_2026_06_26`
- Hypothesis: N115 may be limited because tabular codebook/image-diff summaries miss local code-effect structure; a small CNN over MLIC++ latent windows (`y_hat`, target, residual and target-minus-y_hat) might pick the correct candidate inside the clean `topk_rank <= 4` set.
- Configuration: Added `scripts/train_top8_rankcap_window_chooser.py`. Used `experiments/features/eval300_assignment_top8_codeeffect_latentctx_windows_parseqfeat_seed0_1_2_2026_06_25.npz`, concatenating local latent-window tensors into 128 channels plus core/code/latent summary features. Trained listwise CE plus pairwise margin over train groups whose oracle code is inside rankcap4, with 3 model seeds on CUDA. Teacher/OCR outputs were not inference inputs.
- Result: Best seed reaches validation rank1 `7/17`, rank<=4 `12/17`, covered-only Tesseract `-7`, PARSeq `0`, with `5` wrong and `5` missed groups. Other seeds are weaker (`5/17` or `6/17`, Tesseract `-6/-7`). This is worse than N115's shallow random forest Tesseract `-13` at the same wrong/missed counts.
- Why it failed or was negative: Local latent windows in a small CNN overfit the tiny 55 train-positive rankcap groups and do not solve the candidate tie-breaker among visually similar code effects. The issue is not just absence of local latent context; the model needs a different representation/objective or more clean candidate-supervision data.
- Valid negative result or invalid run: valid diagnostic negative. Manifest hashes check, W&B is logged, GPU was available, and no counted `.oscr` export is run because the table policy is below the current actual-bitstream floor.
- Evidence paths: `experiments/reports/eval300_top8_rankcap4_window_chooser_2026_06_26.md`, `experiments/results/eval300_top8_rankcap4_window_chooser_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_rankcap4_window_chooser_2026_06_26.yaml`, W&B `uunsr40q`.
- Should it be retried? Do not seed/capacity sweep this small latent-window CNN. Retry only with a materially different objective such as teacher-auxiliary representation shaping, sequence/text-aware differentiable supervision, or more clean first-stage/rankcap labels.

## N118 — No-op-inclusive selector cannot use the N117 OOF shortlist safely
- Date: 2026-06-26
- Experiment ID: `eval300_top8_oof_shortlist_policy_selector_2026_06_26`
- Hypothesis: N117 improves clean held-out first-stage candidate coverage, so a nearest/no-op-inclusive selector trained on OOF score features plus deployable tabular candidate evidence may convert that larger shortlist into validation policy gain.
- Configuration: Added `scripts/train_top8_oof_shortlist_policy_selector.py`. Candidate set is nearest/no-op plus non-nearest `topk_rank <= 4` union source-modulo OOF-score top4. Features include OOF score/rank/margins, topk/rate priors and deployable codebook/image-diff scalars, excluding OCR outputs, references and held-out deltas. Trained logistic, HistGradient and random forest selectors on train groups; evaluated argmax and train-tuned non-nearest-vs-nearest margin policies on clean validation.
- Result: Validation policies do not recover the N117 candidate-set headroom. Logistic argmax changes one false group and recovers `0/17` useful changes; budget-tuned policies collapse to no-op. HistGradient argmax is no-op, and budget policies select one false plus one wrong change with Tesseract `+2`. Random forest argmax is no-op; its budget1 policy recovers only `1/17` exact useful change but adds `2` false and `1` wrong changes, for Tesseract `+2`.
- Why it failed or was negative: The first-stage OOF shortlist contains useful candidates, but ordinary one-positive-per-group classification still learns a no-op boundary that either collapses or opens false/wrong groups. Candidate-set coverage and no-op safety remain separate bottlenecks.
- Valid negative result or invalid run: valid diagnostic negative. Manifest hashes check and W&B is logged; no counted `.oscr` export is attempted because table policy is below the current actual-bitstream floor.
- Evidence paths: `experiments/reports/eval300_top8_oof_shortlist_policy_selector_2026_06_26.md`, `experiments/results/eval300_top8_oof_shortlist_policy_selector_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_oof_shortlist_policy_selector_2026_06_26.yaml`, W&B `ehwksv4h`.
- Should it be retried? Do not seed/capacity sweep shallow classifiers over the same N117 shortlist features. Retry only with a materially different no-op objective, policy-level false/wrong risk optimization, richer candidate-local evidence, or a held-out calibration mechanism that does not reduce to candidate probability margins.
- Follow-up audit: N119 (`p5okugfr`) confirms the failure mode. The N117 union shortlist covers `15/17` validation oracle changes, but covered oracle candidates score below nearest in `15/15` cases for logistic, HistGradient and random forest. Random forest can rank oracle above wrong non-nearest in `8/15`, so future work should separate non-nearest proposal from nearest/no-op risk abstention.

## N122 — Hard-positive/no-op-negative weighting does not improve the N120 proposal path
- Date: 2026-06-26
- Experiment ID: `eval300_top8_oof_weighted_proposal_risk_2026_06_26`
- Hypothesis: N121 showed that N120 is bottlenecked by proposal misses, so upweighting positive and hard-positive non-nearest rows while downweighting no-oracle negatives may improve proposal rank1 and let the existing two-stage risk gate exceed the `-5` safe partial policy.
- Configuration: Added `scripts/train_top8_oof_shortlist_weighted_proposal_risk.py`. Reused the N117 OOF shortlist and N120 two-stage structure. Swept `36` hist-gradient proposal weighting settings over positive weight, hard-positive bonus and no-oracle negative weight, then evaluated hist-gradient and random-forest risk gates with train-tuned false/wrong budgets.
- Result: Weighting moves proposal rank1 somewhat (`3/17` to `8/17`) but does not improve the safe risk policy. The best zero-false/wrong policies remain `2/17` exact useful changes, Tesseract `-5`, PARSeq `0`, tying N120 and staying below the current actual-bitstream learned floor (`-8`). Settings with higher proposal rank1 often introduce wrong or false changes after risk gating.
- Why it failed or was negative: The current scalar/OFF-score proposal features can be reweighted into different wrong candidates, but the risk stage cannot safely convert the extra proposal movement into additional exact changes. The next bottleneck is representation/candidate-local evidence, not class-weight tuning.
- Valid negative result or invalid run: valid diagnostic negative. Manifest hashes check; no counted `.oscr` export was attempted because no table policy beat the current actual-bitstream floor.
- Evidence paths: `experiments/reports/eval300_top8_oof_weighted_proposal_risk_2026_06_26.md`, `experiments/results/eval300_top8_oof_weighted_proposal_risk_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_oof_weighted_proposal_risk_2026_06_26.yaml`, W&B `87vmfylo`.
- Should it be retried? Do not run more scalar class-weight, hard-positive-weight or no-op-negative-weight sweeps over the same N117/N120 feature family. Retry only with materially richer candidate-local evidence, a different proposal objective such as listwise/pairwise ranking with new features, or a new shortlist source.

## N123 — High-capacity latent/code/window proposal over the N117 shortlist overfits and worsens held-out proposal rank
- Date: 2026-06-26
- Experiment ID: `eval300_top8_oof_latent_proposal_risk_2026_06_26`
- Hypothesis: N122 may have failed because the scalar OOF/rate/codebook/image-diff proposal features lack local candidate evidence. Adding deployable candidate-local latent/code/window tensors to the non-nearest proposal stage, while keeping the N120 separate risk-abstention stage, may improve proposal rank1 and unlock more safe accepted changes.
- Configuration: Added `scripts/train_top8_oof_shortlist_latent_proposal_risk.py`. The candidate set is nearest plus non-nearest `topk_rank <= 4` union source-modulo OOF-score top4. The proposal model uses local MLIC++ latent windows (`y_hat`, target, residual and target-minus-y_hat), raw code/latent vectors, deployable table features and code/latent summary statistics; `teacher_parseq_*`, OCR deltas, OCR strings, references, predictions and oracle labels are excluded from inference features. Three CUDA seeds are ensembled, then the existing N120 logistic/HistGradient/random-forest risk gates are train-thresholded.
- Result: Validation proposal rank1 is only `4/17` and rank<=4 is `13/17`, worse than N120's hist-gradient rank1 `6/17` and rank<=4 `14/17`. All-proposal use recovers `4/17` exact changes but makes `58` false changes and `13` wrong changes. Every train-tuned risk model and budget collapses to no-op on validation (`0/17` exact useful changes, Tesseract `0`, PARSeq `0`).
- Why it failed or was negative: The latent/window model reaches near-zero train listwise loss but does not generalize; adding high-capacity local latent evidence on the same tiny train-positive set increases overfit rather than source-robust candidate ranking. Risk abstention cannot rescue a proposal distribution that either moves too many no-headroom groups or fails the hard positives.
- Valid negative result or invalid run: valid diagnostic negative. GPU was visible, W&B is logged, manifest hashes the feature NPZ/table/score files/scripts, and no counted `.oscr` export was attempted because the table policy does not beat the current actual-bitstream floor.
- Evidence paths: `experiments/reports/eval300_top8_oof_latent_proposal_risk_2026_06_26.md`, `experiments/results/eval300_top8_oof_latent_proposal_risk_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_oof_latent_proposal_risk_2026_06_26.yaml`, W&B `rlldzgi3`.
- Should it be retried? Do not seed/capacity/epoch/dropout sweep this high-capacity latent/code/window proposal scaffold, and do not only resweep the same N120 risk gates. Retry only with a materially different objective/representation, substantially more clean supervision, or a simpler source-robust anchor that reduces overfit before risk calibration.

## N124 — Deterministic N117 proposal anchors still fail no-op/risk transfer
- Date: 2026-06-26
- Experiment ID: `eval300_top8_oof_deterministic_proposal_risk_2026_06_26`
- Hypothesis: N123 may have failed primarily because a high-capacity proposal model overfit the small train-positive set. A low-variance deterministic proposal anchor over the same N117 OOF shortlist, especially `topk_rank + assignment_relative_error` or OOF-rank anchors, may preserve source-robust candidate ranking and let the separate N120 risk stage safely accept useful changes.
- Configuration: Added `scripts/analyze_top8_oof_deterministic_proposal_risk.py`. It evaluates seven deterministic non-nearest proposal orders over the N117 union shortlist: top-k/rate, relative-error, OOF-score, OOF-rank, reciprocal-rank fusion, union-membership and z-score fusion. Each anchor is paired with the same N120 logistic/HistGradient/random-forest train-tuned risk gates. No OCR strings, predictions, references, held-out OCR deltas or oracle labels are inference features.
- Result: The best deterministic topk/rate anchors recover validation proposal rank1 `6/17` and rank<=4 `12/17`, matching N120's rank1 but still making `58` false changes and `11` wrong changes if applied to all groups. Risk transfer is not safe enough: the best observed Tesseract policy is the OOF-rank anchor with logistic risk, `2/17` exact useful changes, `3` false changes, `1` wrong change, Tesseract `-2`, PARSeq `0`. Conservative variants collapse to no-op or select false/wrong neutral changes.
- Why it failed or was negative: Low-variance anchoring avoids N123's proposal overfit but does not solve no-headroom calibration. The risk stage is trained to be safe on train groups, yet validation either leaks false/wrong changes or abstains from nearly all useful changes. This localizes the remaining bottleneck to verification/calibration objective or supervision coverage, not just proposal ranker capacity.
- Valid negative result or invalid run: valid diagnostic negative. Manifest hashes check, W&B is logged, and no counted `.oscr` export was attempted because no table policy beats the current actual-bitstream floor.
- Evidence paths: `experiments/reports/eval300_top8_oof_deterministic_proposal_risk_2026_06_26.md`, `experiments/results/eval300_top8_oof_deterministic_proposal_risk_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_oof_deterministic_proposal_risk_2026_06_26.yaml`, W&B `lfs66wt4`.
- Should it be retried? Do not run more deterministic rank/score fusion anchors or N120 risk threshold sweeps over the same N117 shortlist features. Retry only with a materially different risk objective/calibration target, substantially more clean no-headroom supervision, or a selector that directly models policy-level false/wrong costs without reusing the same shallow risk transfer.

## N125 — Source-OOF risk calibration over N117 anchors collapses recall
- Date: 2026-06-26
- Experiment ID: `eval300_top8_oof_source_oof_risk_calibration_2026_06_26`
- Hypothesis: N124 may have leaked false/wrong validation changes because risk thresholds were tuned on in-sample train risk scores. Source-held-out risk calibration over train sources may set more transferable thresholds while retaining useful deterministic proposals.
- Configuration: Added `scripts/analyze_top8_oof_source_oof_risk_calibration.py`. It reuses deterministic N117 proposal anchors (`topk_rank_anchor`, `oof_rank_anchor`, `z_fused_anchor`), trains risk models in source-held-out folds on train risk rows, tunes thresholds on source-OOF train scores, then fits full-train risk models and applies the OOF-calibrated threshold to validation. No OCR strings, predictions, references, held-out OCR deltas or oracle labels are inference features.
- Result: Source-OOF calibration is more conservative but still not useful. Topk and z-fused anchors collapse to no-op under the best budget summaries (`0/17` exact, Tesseract `0`). The best moving case is oof-rank with random-forest risk: `1/17` exact useful change, `1` false, `1` wrong, Tesseract `-1`, PARSeq `0`. This is worse than N120's safe partial `-5` and far below the current actual-bitstream `-8` floor.
- Why it failed or was negative: In-sample thresholding contributed to N124's false/wrong leakage, but source-OOF thresholding removes recall rather than recovering safe hard positives. Shallow risk features over proposal scores/margins remain unable to distinguish useful hard positives from no-headroom groups across source shifts.
- Valid negative result or invalid run: valid diagnostic negative. Manifest hashes check, W&B is logged, and no counted `.oscr` export was attempted because no table policy beats the current actual-bitstream floor.
- Evidence paths: `experiments/reports/eval300_top8_oof_source_oof_risk_calibration_2026_06_26.md`, `experiments/results/eval300_top8_oof_source_oof_risk_calibration_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_oof_source_oof_risk_calibration_2026_06_26.yaml`, W&B `whucl72w`.
- Should it be retried? Do not repeat source-OOF thresholding over the same N120 risk features/proposal anchors. Retry only with a materially different verification representation, an objective directly optimized for false/wrong/missed policy tradeoffs, or more clean no-headroom supervision.

## N128 — N127 hard-cohort verifier collapses to no-op under source-OOF thresholding
- Date: 2026-06-26
- Experiment ID: `eval300_top8_hard_cohort_verifier_2026_06_26`
- Hypothesis: N127's train-only hard cohorts (`60` covered oracle positives, `162` matched no-headroom negatives) may provide cleaner no-headroom supervision than the raw N117 shortlist. A scrubbed verifier trained on these cohorts, with thresholds tuned only on train source-modulo OOF predictions, might accept at least `3` exact validation changes with near-zero false/wrong moves.
- Configuration: Added `scripts/train_top8_hard_cohort_verifier.py`. Candidate set is non-nearest `topk_rank <= 4` union OOF-score top4 from the N117 shortlist. The feature scrub selected `227` numeric deployable features (`topk/rate`, OOF score/rank/margins, codebook and image-diff features) and selected no leak-like OCR/reference/oracle columns. Trained logistic, HistGradient and random-forest classifiers on N127 cohorts; tuned thresholds on train source-mod OOF scores with false/wrong budgets `0/1/3`; evaluated validation with full train-cohort models. `.venv` needed `scikit-learn==1.7.2` installed before this run.
- Result: The best validation policy is no-op: `0/17` exact useful changes, `0` false/wrong, Tesseract `0`, PARSeq `0`. All-change use is not a safe fallback: logistic reaches only `3/17` exact useful changes with `58` false and `14` wrong changes, Tesseract `0`; HistGradient/random forest reach `2/17` with `58` false and `15` wrong changes, Tesseract `+1`.
- Why it failed or was negative: Hard-cohort supervision improves label cleanliness but does not produce a transferable score whose source-OOF threshold can separate useful hard positives from no-headroom groups. The validation rank signal remains weak (`2-3/17` rank1 and `10/17` rank<=4), and train OOF safety pushes thresholds to no-op.
- Valid negative result or invalid run: valid diagnostic negative. Feature scrub passes, validation rows are excluded from training/threshold selection, W&B is logged, manifest hashes check, and no counted `.oscr` export was attempted because table-level policy is below the current actual-bitstream floor.
- Evidence paths: `experiments/reports/eval300_top8_hard_cohort_verifier_2026_06_26.md`, `experiments/results/eval300_top8_hard_cohort_verifier_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_hard_cohort_verifier_2026_06_26.yaml`, W&B `98v72lq6`.
- Should it be retried? Do not seed/model/threshold sweep this scrubbed scalar/tree hard-cohort verifier family. Retry only if N127 is auxiliary to a materially different policy-level objective, richer local visual/code-effect representation, direct group-level false-change/no-headroom suppression, or additional clean no-headroom/hard-positive supervision.

## N129 — Matched pairwise hard-cohort scoring still does not transfer
- Date: 2026-06-26
- Experiment ID: `eval300_top8_hard_cohort_pairwise_verifier_2026_06_26`
- Hypothesis: N128 may have failed because row classification ignores N127's matched positive/negative structure. A linear pairwise utility scorer trained to rank each covered oracle-positive candidate above its matched no-headroom negatives may provide a better transferable score.
- Configuration: Added `scripts/train_top8_hard_cohort_pairwise_verifier.py`. Reused the same N117 shortlist and N128 feature scrub (`227` features, no OCR/reference/oracle leakage). Built pairwise samples from N127 matched positive-negative rows, trained logistic pairwise rankers with `C=0.05/0.2/1.0`, tuned thresholds only on train source-mod OOF scores, and evaluated validation with full pairwise models.
- Result: Pairwise train accuracy is `1.0` for all C values, but validation policies remain non-promotable. Budget0/budget1 thresholds collapse to no-op (`0/17` exact, Tesseract `0`). Budget3 can move a few train OOF groups but transfers to validation as no-op or one false change. All-change validation use gets only `3/17` exact useful changes with `58` false and `14` wrong changes, Tesseract `-2/-1`.
- Why it failed or was negative: Matched pairwise supervision is separable on train but does not create a source-robust useful-change score. The score still ranks many no-headroom validation groups above useful hard positives, so safe train OOF thresholds abstain.
- Valid negative result or invalid run: valid diagnostic negative. Feature scrub passes, validation is not used for training/thresholding, W&B is logged, manifest hashes check, and no counted `.oscr` export was attempted because table-level policy is below the actual-bitstream floor.
- Evidence paths: `experiments/reports/eval300_top8_hard_cohort_pairwise_verifier_2026_06_26.md`, `experiments/results/eval300_top8_hard_cohort_pairwise_verifier_2026_06_26.json`, manifest `experiments/manifests/eval300_top8_hard_cohort_pairwise_verifier_2026_06_26.yaml`, W&B `y61bs9uu`.
- Should it be retried? Do not seed/C-sweep linear pairwise scoring over the same scrubbed N128 feature family. Retry only with non-tabular local evidence, group-level policy loss, additional clean supervision, or compact utility-symbol learning that changes the proposal distribution itself.

## N136 — Center-utility-biased compact assignment does not beat the current compact policy
- Date: 2026-06-26
- Experiment ID: `eval300_centerutilv2_assignment_compact_2026_06_26`
- Hypothesis: Codebook centers with high train-side OCR utility may be preferable assignment targets among near-nearest centers, improving Tesseract at the same compact code-index payload without learning a heavier candidate-conditioned assignment head.
- Configuration: First precheck runs using old K64 utility-weighted codebook files were marked invalid because those `.npz` files lacked `center_utility` and therefore reduced utility-biased assignment to nearest. Regenerated train225/top75 K64 utility-weighted codebooks with valid nonzero center utility, verified centers were identical to the earlier K64 utility-weighted centers, and generated compact `TEXT_SYMBOLS` validation streams for `assignment_max_relative_error` `1.01` and `1.05` across seed0/1/2 val75.
- Result: Relative-error `1.01` changed `7/75` assignments and `1.05` changed `23/75`; both policies matched compact actual mean bpp `3.774026` and enhancement payload bpp `0.003781`. PARSeq was unchanged, but Tesseract unicode was only `-1` character versus nearest and `+7` characters worse than the current compact policy.
- Why it failed or was negative: Center-level utility is too coarse: it can choose globally high-utility centers, but it does not know when a specific candidate/image/text instance has no headroom or when a near-nearest alternative damages the glyph. The result is weaker than the current simple pooled image-diff policy even with real compact byte accounting fixed.
- Valid negative result or invalid run: valid negative after the corrected `centerutilv2` run. The earlier old-codebook precheck roots are invalid diagnostics, not evidence, because the codebooks had no `center_utility` metadata.
- Evidence paths: `experiments/reports/eval300_centerutilv2_assignment_compact_2026_06_26.md`, `experiments/results/eval300_centerutilv2_assignment_compact_2026_06_26.json`, manifest `experiments/manifests/eval300_centerutilv2_assignment_compact_2026_06_26.yaml`, W&B `i0aqw89l`; N137 mechanism audit `experiments/reports/eval300_center_utility_assignment_rank_audit_2026_06_26.md`, result `experiments/results/eval300_center_utility_assignment_rank_audit_2026_06_26.json`, manifest `experiments/manifests/eval300_center_utility_assignment_rank_audit_2026_06_26.yaml`, W&B `4wdrz9ws`.
- Should it be retried? Do not sweep scalar relative-error thresholds or center-utility weights over this assignment family. Retry only if center utility is used as an auxiliary feature inside a candidate-conditioned or policy-level no-headroom-aware verifier, with held-out OCR and actual compact `.oscr` validation.

## N139 — Appending K16 exception centers is a no-op under nearest assignment
- Date: 2026-06-26
- Experiment ID: `eval300_k80_exception16_compact_stream_audit_2026_06_26`
- Hypothesis: N138 suggested K16 train-positive residual exception centers were the safest compact-symbol distribution change, so appending them to K64 and regenerating actual compact streams might select exception centers on useful validation cases without requiring a larger learned assignment policy.
- Configuration: Built K80=K64+16 codebooks from N138's K16 exception centers and ran the existing MLIC++ latent-codebook stream generator with `--compact-text-section`, explicit seed0/1/2 val75 selections and nearest assignment. Added `scripts/summarize_k80_exception_stream_audit.py` to verify real `.oscr` bytes, selected-candidate totals and assignment-change counts across the three run directories.
- Result: All three seeds generated valid compact `.oscr` streams and selected `25` candidates each (`75` total), with mean actual total bpp `3.774026` and enhancement payload bpp `0.003781`. However, `assignment_changed_total=0`; the appended exception centers were not selected by the existing nearest assignment path.
- Why it failed or was negative: The feasibility audit measured closeness of exception centers to oracle residual vectors, but the actual stream path still uses the selected candidate's nearest-code assignment. Without an explicit exception selector/no-headroom verifier, the new centers do not alter transmitted codes or decoded reconstructions, so OCR evaluation would be uninformative.
- Valid negative result or invalid run: valid diagnostic negative. Actual `.oscr` streams were generated, result rows match stream byte counts, manifest hashes check, and W&B is logged. It is a no-op actual-stream diagnostic, not an OCR result.
- Evidence paths: `experiments/reports/eval300_k80_exception16_compact_stream_audit_2026_06_26.md`, `experiments/results/eval300_k80_exception16_compact_stream_audit_2026_06_26.json`, manifest `experiments/manifests/eval300_k80_exception16_compact_stream_audit_2026_06_26.yaml`, W&B `bzg5slyt`.
- Should it be retried? Do not rerun nearest-assignment K80/K-plus-center streams or OCR sweeps. Retry only after adding an explicit exception-center proposal/selection policy with no-headroom suppression, and first verify that it produces nonzero useful assignment changes before running OCR.

## N140 — Distance-only exception-center selectors are false-heavy on validation
- Date: 2026-06-26
- Experiment ID: `eval300_exception_center_selector_audit_2026_06_26`
- Hypothesis: Since N138 found many useful oracle residuals close to K16 exception centers, simple exception-center eligibility by distance, margin, or center utility may separate useful oracle-change groups from no-headroom groups well enough to justify an actual OCR stream.
- Configuration: Added `scripts/analyze_exception_center_selector_audit.py`. It joins the Eval300 top8 trainval residual artifact with K80=K64+16 codebooks, computes group-level exception-center distance/margin/center-utility features, sweeps `extra_relative_error` thresholds, then trains shallow logistic/HistGradient/random-forest scores with train-tuned no-headroom false budgets. This is a table diagnostic only; no new OCR claims are made.
- Result: Distance eligibility is not source-robust. `extra_relative_error <= 1.0` covers `44/63` train oracle-change groups with only `1` no-headroom false, but on validation covers only `9/17` oracle-change groups while creating `13` no-headroom false changes. Train-tuned threshold budget 1 chooses `1.02` and leaks `19` validation false changes; higher budgets leak `45`. Shallow learned scores over the same features no-op on validation under all train-tuned budgets.
- Why it failed or was negative: The exception centers encode a train-positive residual geometry that is not enough to recognize held-out no-headroom groups. Distance/margin/center-utility features can indicate proximity to an exception symbol, but not whether using it will preserve OCR or avoid false glyph changes.
- Valid negative result or invalid run: valid diagnostic negative. The manifest hash-checks 8 referenced files and W&B is logged. It is not an OCR result and does not decode streams.
- Evidence paths: `experiments/reports/eval300_exception_center_selector_audit_2026_06_26.md`, `experiments/results/eval300_exception_center_selector_audit_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_center_selector_audit_2026_06_26.yaml`, W&B `nfck52vf`.
- Should it be retried? Do not rerun distance thresholds, distance-margin calibration or shallow distance/center-utility classifiers. Retry only with materially richer candidate-local visual/code-effect evidence and a group-level no-headroom objective, then require nonzero useful validation changes with near-zero false changes before actual `.oscr` OCR evaluation.

## N141 — Conservative explicit exception-center OCR smoke lowers rate but worsens Tesseract
- Date: 2026-06-26
- Experiment ID: `eval300_exception_center_rel100_actual_ocr_2026_06_26`
- Hypothesis: Even if N140's `extra_relative_error<=1.0` selector is false-heavy as a proxy, the actual decoded exception-center reconstructions may still preserve PARSeq and improve or at least not harm Tesseract at lower enhancement rate.
- Configuration: Exported explicit assignment rows for the N140 `rel<=1.0` policy (`22` selected rows across seed1/2 val150; `9` proxy oracle-change and `13` proxy no-headroom), generated compact `.oscr` streams with K80=K64+16 codebooks, built OCR splits, and evaluated Tesseract PSM8 and PARSeq under the same conditions as a matched seed1/2 current compact re-evaluation.
- Result: The exception policy lowers rate because it selects fewer candidates (`22` vs current `50`): mean actual bpp `3.823677 -> 3.793834`, enhancement bpp `0.003645 -> 0.001563`. It does not improve OCR. PARSeq is unchanged exactly, while Tesseract worsens by `+3` unicode and `+6` latin-alnum chars aggregate versus current. On the 22 selected rows only, Tesseract unicode nets `0` (`1` improved, `1` worsened) and latin worsens `+2`.
- Why it failed or was negative: The sparse exception-only policy drops many current useful enhancements and does not provide enough positive OCR utility on the selected exception rows to compensate. Lower rate alone is not an OSCAR-LIC win when OCR preservation worsens.
- Valid negative result or invalid run: valid diagnostic negative. Actual compact `.oscr` streams were generated, Tesseract/PARSeq were run, manifests hash-check, and W&B is logged. This is not a promotion comparison because the selected-candidate budget is lower than current; it is a branch-closing diagnostic for simple exception-only policies.
- Evidence paths: `experiments/reports/eval300_exception_center_rel100_actual_ocr_2026_06_26.md`, `experiments/results/eval300_exception_center_rel100_actual_ocr_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_center_rel100_actual_ocr_2026_06_26.yaml`, W&B `x7imee7s`; policy export manifest `experiments/manifests/eval300_exception_center_rel100_policy_export_2026_06_26.yaml`.
- Should it be retried? Do not run OCR for more simple exception-distance thresholds or sparse exception-only policies. Retry only with a policy that can either preserve current useful selections while replacing bad assignments, or explicitly learn replace/drop decisions from richer candidate-local evidence.

## N144 — Rel<=1.0 proxy replacement labels are not enough for a verifier
- Date: 2026-06-26
- Experiment ID: `eval300_exception_action_supervision_gap_2026_06_26`
- Hypothesis: The N143 current-preserving action table can train a replacement verifier from the rel<=1.0 proxy-positive exception cohort.
- Configuration: Audited `experiments/results/eval300_exception_center_action_readiness_2026_06_26.jsonl`, which joins current selected groups with K80 exception eligibility.
- Result: The replacement cohort is train-high-positive but validation-mixed. Train has `45` replacement candidates with `44` proxy true and only `1` proxy false; validation has `22` replacement candidates with `9` proxy true and `13` proxy false. Train IAM is `22/22` proxy true, while validation IAM is `4/12` proxy true and `8/12` proxy false.
- Why it failed or was negative: There are too few train replacement negatives to learn the validation no-headroom boundary, and N141 already shows proxy-oracle replacements are nearly neutral under actual current-relative OCR.
- Valid negative result or invalid run: valid supervision diagnostic negative.
- Evidence paths: `experiments/reports/eval300_exception_action_supervision_gap_2026_06_26.md`, result `experiments/results/eval300_exception_action_supervision_gap_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_action_supervision_gap_2026_06_26.yaml`, W&B `7pa1s2if`.
- Should it be retried? Not as direct proxy-positive classifier training. Derive explicit current-relative replacement/drop labels or add hard replacement-negative supervision first.

## N145 — Executed exception replace/drop actions are not current-relative safe
- Date: 2026-06-26
- Experiment ID: `eval300_exception_center_current_relative_labels_2026_06_26`
- Hypothesis: The executed N141 exception-center policy may contain enough current-relative replacement/drop signal to justify training a verifier from rel<=1.0 action labels.
- Configuration: Added `scripts/build_current_relative_exception_labels.py` and joined the matched current compact streams, exception compact streams, Tesseract rows, PARSeq rows and exported exception policy rows for seed1/2 val150. Each row is labeled as `replace_with_exception`, `drop_current` or `keep_base`, and OCR deltas are computed against current under `unicode_strict_v1` and `latin_alnum_ci_v1`.
- Result: The executed policy has `22` replacements, `28` drops and `100` keep-base rows. Strict labels across Tesseract/PARSeq and both profiles are `5` beneficial, `137` neutral, `7` harmful and `1` mixed. Replacements are mostly neutral but worsen Tesseract latin by `+2`; drops reduce rate but worsen Tesseract by `+3` unicode and `+4` latin.
- Why it failed or was negative: The labels show that the N141 policy is dominated by sparse/drop behavior, not by reliable current-preserving replacement. It does not provide enough safe current-relative positives to train a deployable replacement/drop verifier by itself.
- Valid negative result or invalid run: valid diagnostic negative. It reuses counted compact `.oscr` streams and OCR outputs, computes actual current-relative row labels, logs W&B, and the manifest hash-checks 22 referenced files.
- Evidence paths: `experiments/reports/eval300_exception_center_current_relative_labels_2026_06_26.md`, result `experiments/results/eval300_exception_center_current_relative_labels_2026_06_26.json`, table `experiments/results/eval300_exception_center_current_relative_labels_2026_06_26.jsonl`, manifest `experiments/manifests/eval300_exception_center_current_relative_labels_2026_06_26.yaml`, W&B `tvpq70xz`.
- Should it be retried? Do not train a verifier directly from executed rel<=1.0 actions. Retry only after building counterfactual current-relative labels for current-selected units or adding hard negative replacement/drop supervision.

## N146 — N143 proxy/action features do not explain current-relative safety
- Date: 2026-06-26
- Experiment ID: `eval300_exception_current_relative_feature_gap_2026_06_26`
- Hypothesis: Joining N145 labels to N143 features may show that proxy-true replacement or exception-distance features are sufficient to identify safe exception replace/drop actions.
- Configuration: Added `scripts/analyze_current_relative_exception_feature_gap.py`, joined the 50 current-selected seed1/2 validation rows from N145 to the N143 action table, and summarized actual OCR deltas by action, proxy label, source and feature strata.
- Result: Proxy-true replacement remains mostly neutral and still contains a harmful row (`1` beneficial, `7` neutral, `1` harmful). Proxy-false replacement is also mostly neutral (`12` neutral, `1` harmful). The aggregate regression is driven by dropping current rows: `28` drops save only `1204` bytes but cause Tesseract deltas `+3` unicode and `+4` latin. IAM accounts for `7/8` harmful-or-mixed rows.
- Why it failed or was negative: Exception eligibility/proxy labels explain proximity to exception centers, not whether changing or dropping the current enhancement is current-relative safe. The main missing supervision is hard negative drop/replacement labels, not another distance threshold.
- Valid negative result or invalid run: valid diagnostic negative. It reuses N143/N145 manifest-backed artifacts, logs W&B, and its manifest hash-checks 5 referenced files.
- Evidence paths: `experiments/reports/eval300_exception_current_relative_feature_gap_2026_06_26.md`, result `experiments/results/eval300_exception_current_relative_feature_gap_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_current_relative_feature_gap_2026_06_26.yaml`, W&B `qhqzghzd`.
- Should it be retried? Do not retry proxy/distance-only feature audits as a promotion path. Retry only with counterfactual current-relative labels, especially IAM hard negative drop/replacement examples.

## N148 — Current N143 train current-only rows lack close hard negatives
- Date: 2026-06-26
- Experiment ID: `eval300_exception_current_only_distribution_gap_2026_06_26`
- Hypothesis: The current N143 train current-only rows may contain enough hard negatives to train a replace/drop verifier that preserves current selections on validation.
- Configuration: Added `scripts/analyze_exception_current_only_distribution_gap.py` and audited `extra_relative_error` for `current_only` rows across the N143 train/validation action table, with N145 labels joined for seed1/2 validation rows.
- Result: Train current-only rows are mostly far from exception centers: only `14/180` are at `extra_relative_error<=1.5`, and the median is `3445.97`. Validation current-only rows are all close: `53/53` at `<=1.5`, median `1.0733`. The labeled seed1/2 validation current-only subset contains `28` drops with `5` harmful/mixed rows and the aggregate `+3/+4` Tesseract regression.
- Why it failed or was negative: A verifier trained on the current N143 train current-only negatives would mostly see easy far-away negatives, not the close current-only cases that dominate validation drop harm.
- Valid negative result or invalid run: valid distribution diagnostic negative. It uses manifest-backed N143/N145 artifacts, logs W&B, and its manifest hash-checks 5 referenced files.
- Evidence paths: `experiments/reports/eval300_exception_current_only_distribution_gap_2026_06_26.md`, result `experiments/results/eval300_exception_current_only_distribution_gap_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_current_only_distribution_gap_2026_06_26.yaml`, W&B `rsngelv2`.
- Should it be retried? Do not train on the current N143 split alone. Retry after mining/generating close current-only hard negatives, especially IAM handwriting cases, and then label them with actual compact stream OCR.

## N155 — 53-row exception probe labels are too small for learned veto training
- Date: 2026-06-26
- Experiment ID: `eval300_exception_probe_veto_diagnostic_2026_06_26`
- Hypothesis: The N152/N153 actual-labeled probe set may be sufficient to train a tiny deployable-feature harmful-row veto that approximates the N154 luma/diff/rate rule frontier.
- Configuration: Added `scripts/train_exception_probe_veto_diagnostic.py`. It deduplicates the N153 feature table to the latest 53 N152 rows, excludes source/reference/text metadata, and trains leave-one-out logistic regression plus shallow random forest harmful-row scorers over action, image-difference, luminance, PSNR and rate features.
- Result: Both learned scorers fail the N154 frontier. Under zero/one/two/three retained-harm budgets, logistic and random forest collapse to vetoing all 53 rows, yielding Tesseract `0/0`, rate delta `0`, and all 16 beneficial rows vetoed. Unconstrained thresholds mostly no-op and retain all four harmful rows.
- Why it failed or was negative: Four harmful positives among 53 rows are not enough to learn a transferable harmful veto from these feature families. The learned score does not reproduce even the simple hand-searched N154 frontier, so model/seed sweeps would be premature.
- Valid negative result or invalid run: valid diagnostic negative. The run uses actual compact-stream OCR labels, excludes source/reference text from inference features, logs W&B, and its manifest hash-checks 5 referenced files.
- Evidence paths: `experiments/reports/eval300_exception_probe_veto_diagnostic_2026_06_26.md`, result `experiments/results/eval300_exception_probe_veto_diagnostic_2026_06_26.json`, scores `experiments/results/eval300_exception_probe_veto_diagnostic_scores_2026_06_26.jsonl`, manifest `experiments/manifests/eval300_exception_probe_veto_diagnostic_2026_06_26.yaml`, W&B `vwffygur`.
- Should it be retried? Do not seed-sweep logistic, shallow forest, or similar tiny learned vetoes on this 53-row table. Retry only after adding more actual-labeled hard negatives or changing the objective/representation substantially, and benchmark against the N154 harm-budget frontier.

## N156 — Widening exception probes to rel<=4 dilutes label efficiency
- Date: 2026-06-26
- Experiment ID: `eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26`
- Hypothesis: Increasing the close-current-only/proxy-control threshold from rel<=2 to rel<=4 may add enough useful and harmful actual OCR labels to make exception replace/drop verifier training more viable.
- Configuration: Selected `71` probe rows (`63` training-use, `8` audit-only), prepared per-seed current/exception stream inputs, generated matched current K64 compact streams and K80 explicit exception/drop streams for seed0/1/2, then ran Tesseract PSM8 and PARSeq over all 63 matched current/exception reconstructions. Added v3 harmful evidence, v3 veto-rule search and harmful/beneficial contact sheets.
- Result: The exception/drop stream lowers weighted bpp `3.340284 -> 3.245654` and selected candidates `63 -> 23`, with PARSeq unchanged and Tesseract aggregate `-14` unicode / `-11` latin. Labels are `17` beneficial, `41` neutral and `5` harmful. N157 directly compares N152 to N156 and confirms all 53 N152 rows are preserved; the 10 extra training-use rows add only one beneficial and one harmful row, with added-row Tesseract unicode `+1`. Zero-harm deployable-feature veto performance weakens to Tesseract `-12/-12` while vetoing seven beneficial rows.
- Why it failed or was negative: More rows from a looser distance threshold mostly add neutral current-only/drop cases, not clearer supervision. The active problem is not row count alone; it is hard-negative/current-preserving decision quality and a stronger verifier objective.
- Valid negative result or invalid run: valid diagnostic negative. Actual compact streams and OCR labels were generated, all N156 manifests hash-check, W&B runs were logged, and no deployable verifier/promotion claim was made.
- Evidence paths: actual OCR report `experiments/reports/eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26.md`, result `experiments/results/eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26.json`, label table `experiments/results/eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26.jsonl`, manifest `experiments/manifests/eval300_exception_counterfactual_probe_actual_ocr_v3_rel4_2026_06_26.yaml`, W&B `8pfcdtsv`; harmful evidence `experiments/reports/eval300_exception_probe_harmful_evidence_v3_rel4_2026_06_26.md`, manifest `experiments/manifests/eval300_exception_probe_harmful_evidence_v3_rel4_2026_06_26.yaml`, W&B `pxz0ztxx`; veto report `experiments/reports/eval300_exception_probe_veto_rule_search_v3_rel4_2026_06_26.md`, manifest `experiments/manifests/eval300_exception_probe_veto_rule_search_v3_rel4_2026_06_26.yaml`, W&B `oui8k7ce`; expansion comparison `experiments/reports/eval300_exception_probe_expansion_v2_to_v3_rel4_comparison_2026_06_26.md`, result `experiments/results/eval300_exception_probe_expansion_v2_to_v3_rel4_comparison_2026_06_26.json`, manifest `experiments/manifests/eval300_exception_probe_expansion_v2_to_v3_rel4_comparison_2026_06_26.yaml`, W&B `uprhtlh3`.
- Should it be retried? Do not keep widening rel thresholds or sampling more close-current rows from the same feature distribution just to increase N. Retry only if the selection deliberately targets new hard-negative modes, source-balanced current-preserving labels, or richer candidate-local evidence, and benchmark against the N152/N154 frontier.

Record each negative result rather than deleting it.

```markdown
## N### — Short title
- Date:
- Experiment ID:
- Hypothesis:
- Configuration:
- Result:
- Why it failed or was negative:
- Valid negative result or invalid run:
- Evidence paths:
- Should it be retried? Under what changed condition?
```

A result is **invalid**, not negative, if rate omitted side information, test data leaked into tuning, checkpoint was wrong, metric code was faulty, or the run was interrupted before its predefined criterion.

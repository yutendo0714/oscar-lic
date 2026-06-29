# Next Experiment Plan

**Date:** 2026-06-25  
**Principle:** do not start large training until actual-bitstream baseline reproduction and oracle utility evidence are clean.

## G1: Base Codec Reproduction

Goal: produce actual compressed files and bpp for MLIC++ and CompressAI smoke baselines.

Steps:

1. Download corrected MLIC++ checkpoint at lambda 0.0250.
2. Record checkpoint URL, size, SHA256, architecture variant, and expected command.
3. Run MLIC++ test/compress/decompress on a tiny CLIC subset.
4. Compute bpp from complete produced files, not likelihood estimates.
5. Store one JSON result per image using `evaluation/REPORTING_SCHEMA.json`.
6. Log run to wandb with environment lock and external commit lock.

Exit criteria:

- decoded images are valid;
- complete-file bpp is available;
- output is deterministic for repeated decode;
- GPU is visible before running.

## G2: OCR Evaluation Foundation

Goal: lock one training teacher and one held-out evaluator.

Initial split:

- Train teacher: PARSeq.
- First held-out smoke evaluator: Tesseract PSM8.
- Stronger held-out evaluator still needed: PaddleOCR or TrOCR.

Steps:

1. PARSeq pretrained checkpoint is locked in `baselines/checkpoint_manifest.yaml`.
2. Checkpoint, tokenizer/charset configuration and preprocessing are recorded in `baselines/ocr_models.yaml`.
3. Original word-crop upper bound for `text_crops_parseq_smoke_v1` is stored in `experiments/results/parseq_text_crops_smoke_v1_2026_06_25`.
4. Raw and normalized OCR outputs are stored with model revision; see `experiments/reports/ocr_text_crop_smoke_v1_2026_06_25.md`.
5. Never use held-out OCR logits/confidence for training or oracle labels.

## G3: Adapter And Text-Crop Smoke

Goal: create the base codec adapter and a crop-level evaluation loop.

Steps:

1. Wrap MLIC++ behind `BaseCodecAdapter` without modifying upstream files broadly.
2. Define `CodecResult` with actual total bytes, section bytes, encode/decode time, metadata, and image shape.
3. Build a small text-rich validation manifest from TextOCR/ICDAR/TotalText local data.
4. Verify original/reconstructed OCR metrics on compressed outputs.

Current adapter decision:

- upstream corrected MLIC++ emits one opaque main `y` ANS stream, so per-slice bytes are not available without reserialization;
- use frozen-base enhancement sections first;
- keep likelihood slice diagnostics as candidate-design evidence only.
- a minimal `.oscr` wrapper around the upstream MLIC++ body now decodes successfully; one-section overhead is 88 bytes.
- counted dummy `TEXT_GATE` and `TEXT_MAIN` sections now decode base-only successfully through the same wrapper; three-section structural overhead is 168 bytes, and the incremental two-section cost is table bytes plus counted optional payload.
- a first raw latent residual `TEXT_MAIN` stub now works end to end, but it is too high-rate and only weakly useful; treat it as a payload-path validation and negative control, not as the proposed model.

## G4: Counterfactual Oracle Feasibility

Goal: test the central OSCAR-LIC hypothesis before large model training.

Candidate layout:

- start with latent spatial tiles and channel groups matching MLIC++ feature shape;
- keep candidate count small for exact greedy trajectories.

Controls:

- random allocation;
- uniform text-mask allocation;
- detector confidence allocation;
- base uncertainty/residual energy;
- cheap stroke/edge energy.

Measurements:

- delta OCR loss per actual incremental serialized bit;
- held-out OCR CER/WER after allocation;
- side-index overhead;
- oracle interaction variance.

Exit criteria:

- oracle beats controls at matched total bytes on held-out OCR;
- gains survive gate/index/header accounting;
- improvement is not confined to one synthetic domain.

## Training Direction After G4

If G4 passes:

1. Train a lightweight utility predictor with ranking and calibrated benefit heads.
2. Train conditional feature-residual enhancement stream with frozen base.
3. Add multi-teacher invariance and topology only if needed.
4. Jointly fine-tune a small base tail only after frozen-base gains are stable.

Current evidence:

- PARSeq-only quality oracle improves the teacher but hurts Tesseract.
- Equal-weight PARSeq+Tesseract quality oracle improves both recognizers at lower actual bpp than uniform q2 on the 24-crop smoke split.
- With a simple 40-byte/candidate plus 40-byte global overhead penalty, the multi-teacher oracle still beats uniform q2 on both recognizers.
- The deterministic 100-crop split repeats the result: overhead-aware multi-teacher oracle beats uniform q2/q3 and random/confidence/uniform-cost controls at matched bytes.
- Concrete `TEXT_GATE` payloads and dummy `TEXT_MAIN` sections are now serialized in `.oscr`, with complete-file bpp measured after CRC/table/header bytes. This validates the accounting path needed for a frozen-base learned enhancement.
- Candidate-overhead analysis from real MLIC++ latent shapes suggests `8x8` or `4x4` slice-level tiles for full images. Isolated word crops should remain diagnostics because fixed section overhead dominates their bpp.
- Raw int8 latent residuals are not the right enhancement representation. The next prototype should use a compact learned residual/codebook payload and select candidates by multi-teacher utility rather than residual magnitude.
- True single-candidate latent labels on Eval100 confirm that the residual candidate generator has real utility: the oracle improves PARSeq CER from 0.0791 to 0.0640 and Tesseract CER from 0.4030 to 0.3427.
- A tiny linear candidate classifier reaches Eval100 validation AUC 0.7085, but actual one-candidate selection still leaves most oracle gain unused. The next learned component must rank candidates within an image, not only classify candidate rows globally.
- A small within-image ranker and approximate image/base patch statistics improve non-oracle Tesseract selection, but only modestly. Validation AUC rises to about 0.67 with image patch features, while the oracle remains far stronger.
- Wide flattened per-channel latent vectors overfit and degrade actual selection. Prefer compact/shared local encoders or dimensionality-reduced latent tokens over raw tabular channel dumps.
- A tiny shared patch-CNN ranker improves candidate AUC/AP substantially, but actual one-candidate transmitted OCR gains remain modest. Candidate classification metrics are therefore diagnostics only; policy-level OCR and actual bpp remain the main validation target.
- Eval100 K16 decoder-known codebook reduces enhancement payload to about 0.0865 bpp, but residual-MSE codebook symbols do not improve OCR. The compact payload must be utility-trained, not just reconstruction-trained.
- Oracle-selected K16 codebook is the first positive compact-payload result: only 0.0223 enhancement bpp improves PARSeq CER to 0.0734 and Tesseract CER to 0.3766 over MLIC++ base. This should become the immediate target for learned gating/codebook training.
- Oracle-selected K64/26-center codebook improves PARSeq further to 0.0659 but weakens Tesseract to 0.3917, so codebook capacity should be controlled by multi-teacher utility rather than residual fidelity alone.
- Eval300 repeats the central headroom: raw single-candidate oracle improves MLIC++ base from PARSeq/Tesseract CER 0.0661/0.4326 to 0.0561/0.3665.
- Eval300 learned selectors remain the bottleneck. The patch-CNN ranker is strongest among current non-oracle selectors (validation AUC 0.6170/AP 0.1316, top-10 policy 0.0634/0.4254 CER at 3.9439 bpp), but it captures only a small fraction of the oracle.
- An image-level listwise patch gate with an explicit no-op class improves the non-oracle Tesseract policy further (top-25 Tesseract CER 0.4197 at 4.4421 bpp) and raises validation AUC/AP to 0.6315/0.1450. It slightly worsens PARSeq, so target labels alone are insufficient; add OCR-family-specific no-worse constraints or separate weighted heads.
- A generic harm-logit penalty is not enough. With penalty 0.5, validation AUC/AP drops to 0.6243/0.1430; top-20/top-30 Tesseract CER reaches 0.4199/0.4195, but PARSeq still worsens to 0.0647/0.0649 versus the split base 0.0641. The next gate should protect specific evaluator families explicitly rather than penalizing all non-target positives generically.
- A first evaluator-specific protected target is better than the generic penalty. `tesseract_parseq_safe` reaches top-25 Tesseract CER 0.4186 at 4.3970 bpp and keeps PARSeq at 0.0645, slightly better than standard listwise/harm-0.5 at similar Tesseract quality. This is still not strict no-worse behavior, so add validation-time constrained thresholding or a two-head selector before promoting it to the main model.
- Train-split constrained thresholding is a useful calibration control. It enforces zero PARSeq-distance violation on train and yields PARSeq/Tesseract CER 0.0643/0.4196 at 4.4173 bpp on Eval300 validation, which is safer than top-k but slightly weaker on Tesseract. This suggests the next model needs better-calibrated protected scores, not only a post-hoc threshold.
- The learned-selection compact bitstream path is now positive on seed0/1/2 held-out val75 splits. `tesseract_parseq_safe` top-25 gates export 25 candidates per split; K16 explicit codebook `.oscr` streams average 3.8496 actual bpp with 0.0290 enhancement bpp, preserve PARSeq CER at base level, and improve Tesseract CER from 0.4178 to 0.4091. This keeps about 84% of the raw top-25 Tesseract gain while replacing raw residual payloads with a compact counted codebook/index payload. The K16 codebook is still trained on selected validation vectors, so the next step is a proper train-codebook/eval-selection split.
- Train-codebook/eval-selection capacity controls sharpen the diagnosis. Seed0 K16 residual-MSE codebook trained on train225 selected vectors and loaded for val75 drops PSNR delta to -0.0026 dB and removes the Tesseract gain, returning to base CER 0.4243 while preserving PARSeq. Raising train-side codebooks to K64 preserves PARSeq and improves three-seed mean Tesseract from 0.4178 to 0.4126, but retains only about half of the raw top-25 Tesseract gain and slightly lowers held-out PSNR. Capacity is therefore a real bottleneck, and the final representation should use enough symbols plus utility-aware assignments/centers instead of residual-MSE-only clustering.
- A first utility-weighted K64 diagnostic is positive but unstable. Reweighting train residual K-means by Tesseract improvement with PARSeq-harm penalty improves mean Tesseract to 0.4117 and raises raw-gain retention from about 50% to 59% while preserving PARSeq, but it helps seeds 0/1 and hurts seed2. Sweeping the global alpha shows the limitation: `alpha=4` rescues seed2 but weakens seeds 0/1 and returns the mean to residual-MSE K64 level, while `alpha=16` worsens seed2. This supports utility-aware codebook training, but scalar weighted MSE with one global alpha is too weak; next try assignment-aware or prototype/diversity objectives.
- Pairwise Tesseract diffs show utility-weighting changes only 12/225 samples relative to residual-MSE K64 (6 improved, 6 worsened, net -1 edit distance). The next symbol learner should explicitly suppress rare harmful assignments, for example with no-worse validation constraints, per-source/domain stratified prototypes, or assignment losses that distinguish helpful and harmful nearest centers.
- Hard train-candidate exclusion is too blunt. Removing train candidates that harmed Tesseract or PARSeq before K64 clustering drops retained raw Tesseract gain to about 16.5%, worse than residual-MSE K64 and utility-weighted K64. Keep no-worse labels, but use them in assignment/routing losses or validation constraints rather than deleting prototype diversity.
- Center-level utility-biased assignment is also too coarse. Re-routing seed2 val candidates to train-utility-high centers within a tight residual-error margin keeps payload size unchanged, but re1.05 changes 11/25 indices and re1.01 changes 1/25; both worsen Tesseract to 0.4036. The next assignment model must be candidate-conditioned and no-worse calibrated, not just center-prior based.
- The assignment problem has large headroom. A PARSeq-safe top-8 code-index oracle keeps the selected candidates and payload size fixed and improves three-seed mean Tesseract from 0.4117 to 0.3910 while preserving PARSeq. Seed0 has no headroom, seed1 improves 0.4178 -> 0.3916, and seed2 improves 0.4010 -> 0.3650. This is stronger than the current learned-selection/codebook gains and should become the next main target.
- Train-side top-8 assignment labels are now available but reject a naive patch-difference learner. Train225 top-75 variants provide 1800 rows and 63 PARSeq-safe Tesseract-improving reassignment groups, yet a tiny `[nearest, variant, |diff|]` patch ranker overfits train labels and fails on val75: train AUC/AP 0.933/0.659 versus val AUC/AP 0.410/0.084, with train-tuned threshold worsening Tesseract by +2.11 edit characters on average. Adding codebook-center geometry and code one-hot features improves val AUC/AP to 0.551/0.131 and reduces threshold harm to +1.22 characters, but still does not beat nearest. The next assignment head should use richer latent residual/codebook geometry and calibrated no-op objectives rather than only reconstructed bitmap differences.
- Richer hand-built assignment features are not enough yet. Candidate residual-vector/codebook relation features reduce codegeom val AUC/AP to 0.492/0.118 for the CNN ranker and 0.482/0.112 for a tabular-only control, so the raw feature set mainly adds overfit. Conservative threshold and score-margin tuning improves the worst policy slightly but still averages +0.78 Tesseract edit characters versus nearest assignment's 0 and the oracle's -8. Candidate-conditioned assignment remains a major headroom item, but the next attempt should use stronger cross-fitted no-op/group-head training or train a small latent-token model directly, not add more tabular descriptors.
- The key assignment fix is the target, not more features. Training against `label_assignment_oracle_choice` makes nearest/no-op the positive class in no-headroom groups and produces the first learned assignment gain: codegeom oracle-choice reaches val AUC/AP 0.887/0.652 and mean Tesseract edit delta -0.889 at zero PARSeq delta, compared with nearest 0 and oracle -8. Latent-vector features do not improve the main policy, so keep the simple codegeom oracle-choice target as the current assignment baseline and improve it with better group/listwise modeling.
- The codegeom oracle-choice assignment gain survives real counted `.oscr` compress/decompress. With selected candidates and code-index payload size fixed, seed0/1/2 actual bpp is identical to nearest assignment, PARSeq outputs are unchanged exactly, and Tesseract improves by 3 unicode edit characters / 5 latin-alnum edit characters over 225 val samples. The same actual-output comparison shows the oracle still has much larger headroom (-24 unicode / -19 latin-alnum), so the current learned policy is a real but small baseline rather than a final result.
- Model-seed ensembling ties the simple model0 oracle-choice policy, and pooled seed0/1/2 training is slightly worse. The next improvement should therefore change the assignment objective/architecture and calibration, not just average more rowwise models or pool domains.
- A first group/listwise assignment-loss implementation is negative with the current features. Pure listwise CE improves one policy on seed1 but is unstable on seed2, and BCE plus train-tuned top-changed calibration does not beat the rowwise oracle-choice baseline. Keep the listwise code path, but the next real change should improve the candidate-conditioned representation or add stronger no-worse calibration, not only reshape the loss.
- Assignment oracle headroom is concentrated rather than uniform. On the 75 selected val groups, seed0 has no assignment headroom, while seed1/2 provide all -24 Tesseract unicode edit-character gain. IAM/ICDAR-style sources dominate the gain, and nearest Tesseract distance >=3 contributes -19 characters while distance 0 contributes none. The next assignment head should therefore condition on difficulty/source proxies and learn a strong no-op for easy groups.
- A diagnostic source/length/OCR-distance feature upper-control is negative on the unstable seed2 split. Even with table-derived OCR-distance features, CNN+metadata and tabular-only metadata policies worsen Tesseract. Difficulty tells where headroom is, but not which code center fixes the glyph. Prioritize candidate/code-effect representation over more global metadata.
- Local bitmap differencing is also insufficient by itself. A diff-crop assignment ranker that focuses the CNN on the union of code-variant pixel changes reaches reasonable AUC/AP but does not improve the seed2 policy. The next representation should model the direction from nearest center to candidate center, the selected residual vector, and local latent/glyph context together.
- A tabular-only codegeom oracle-choice diagnostic is slightly better than the CNN+codegeom table baseline on mean policy delta (-1.0 vs -0.889 Tesseract edit characters, PARSeq unchanged), but still far from the oracle and still unstable on seed2. This says the current image-difference CNN is not carrying decisive signal; the missing piece is likely a better candidate/code-effect latent representation, not simply more pixels.
- A smaller strongly regularized tabular-only head is too conservative and weaker (-0.333 mean Tesseract edit-character delta). The seed2 instability should not be treated as a generic over-capacity problem; it needs better utility/no-worse evidence for which changed center actually fixes the glyph.
- Policy-error analysis of the promoted oracle-choice model0 makes the remaining failure mode concrete: among 75 val groups it exactly finds only 4/17 oracle code changes, misses 9, makes 15 false changes in no-headroom groups, and makes 4 wrong changes. IAM words dominate both misses and false changes. A better assignment head must therefore solve two coupled tasks, not one: find deeper useful top-k code ranks in hard OCR cases and reject changes in no-headroom groups.
- A non-deployable OCR-distance gate shows that difficulty gating can help if it is predicted well: requiring nearest Tesseract distance >= 2 before accepting a non-nearest score improves the model0 table policy from -3 to -5 Tesseract edit characters while preserving PARSeq. This should be converted into a deployable learned no-op/difficulty proxy, not used directly.
- A first deployable-ish group-level gate using codegeom group features weakly confirms this direction: a fixed 0.5 gate improves mean table policy from -1.0 to -1.33 Tesseract edit characters and reduces false-change-like behavior, but train-tuned thresholds overfit. The next version should share representation with the assignment scorer or use cross-fitted calibration rather than a separately tuned post-hoc threshold.
- Training the same codegeom group gate directly on the OCR-distance proxy is negative, so code geometry alone is not enough to infer text difficulty. Use local image/latent glyph features for any deployable difficulty/no-op head.
- The fixed-0.5 group-level gate is now promoted through real counted `.oscr` evaluation and becomes the current learned assignment baseline. It keeps the selected candidate count and code-index payload length fixed, so actual bpp is unchanged versus nearest/model0 assignment. Across seed0/1/2 val75, it improves Tesseract by -5 unicode and -7 latin-alnum edit characters versus nearest with PARSeq unchanged exactly; versus the previous model0 actual baseline it adds -2/-2 Tesseract edit-character gain with no worsened samples. The caveat is calibration: threshold 0.5 is promising, but it should be cross-fitted or validated on a fresh split before becoming a paper claim.
- A coarse threshold sweep does not improve OCR beyond the promoted group gate, but it slightly cleans up the policy. Thresholds 0.1 through 0.9 all produce the same aggregate Tesseract/PARSeq deltas on the table; threshold 0.8/0.9 reduces changed groups from 15 to 14 versus 0.5. Seed0 threshold 0.8 was regenerated through actual `.oscr` and OCR, matching threshold 0.5 exactly while making zero assignment changes on seed0. Treat threshold 0.8 as the conservative equivalent baseline, not as a new performance jump.
- Adding existing `latent_vec_*` tabular features to the group gate is negative. It overfits train, lowers usable validation behavior, and reduces the best thresholded three-split gain to only -1 Tesseract edit character. The next difficulty/no-op representation should not be a wider hand-built latent-stat table; use a structured local latent-token or code-effect encoder instead.
- Score-only confidence gates are also negative. Best-vs-nearest score margin and top-score margin can reduce changed groups, but neither beats the promoted group gate. The no-op decision needs external evidence about local glyph/code effect, not only assignment scorer confidence.
- A linear-only group gate is a useful simplicity control but not a replacement. It reduces training overfit, yet its best threshold family reaches only -3 Tesseract edit characters with zero PARSeq delta, below the promoted nonlinear group gate at -5. Keep the group head small, but do not collapse it to a linear score; spend complexity on local latent/code-effect representation and cross-fitted calibration.
- Pooling seed0/1/2 train groups into one larger group gate raises validation AUC but does not improve the policy. Train-tuned and fixed 0.5 pooled gates stay at -3 Tesseract edit characters, below the split-specific promoted gate. Do not select assignment gates by AUC alone; select by row-wise OCR deltas, false-change counts and actual `.oscr` OCR comparison.
- Lightweight reconstructed-image difference features are the first deployable-ish signal to beat the promoted group gate after actual `.oscr` verification. The pooled image-diff gate keeps actual bpp unchanged and preserves PARSeq exactly, while improving Tesseract by -8 unicode/-8 latin edit characters versus nearest and by -3/-1 versus the previous group-gated baseline. This validates the local code-effect direction; replace the hand-built image-diff statistics with a compact local latent/glyph encoder and cross-fitted calibration next.
- Explicit policy-error comparison shows the image-diff gate wins by precision, not coverage: it keeps the same 4 exact oracle hits as the old group gate, but cuts false/wrong changes from 11 to 1 while changing only 5/75 groups. The next model must recover more of the 13 missed oracle-change groups without reintroducing false changes in no-headroom samples.
- A second pooled image-diff gate seed/threshold reproduces the same actual OCR aggregate (`-8` Tesseract unicode / `-8` latin, PARSeq `0`) and ties the preferred model1 policy exactly in row-wise OCR comparison. Model1's 5 non-nearest choices are a same-code subset of model2's 7 choices, while model2 adds only 2 non-useful changes. The promotion rule should therefore break OCR ties by fewer changed groups and lower false/wrong-change count.
- Score-coverage analysis shows the current bottleneck is the assignment scorer, not gate recall. For the preferred image-diff policy, all 13 missed-oracle groups occur because the scorer fails to rank the oracle code first; there are zero cases where the gate rejects an oracle-best code. Relaxing the gate would expose bad scorer choices, so the next architecture should improve candidate-conditioned code-effect ranking.
- Direct tabular image-diff scorers are a negative control. They raise validation AUC/AP but reduce oracle-change rank1 coverage to 1/17 and give only weak policy gains, so the next scorer should use a structured local latent/glyph/code-effect encoder rather than more hand-built scalar statistics.
- Score top-M upper-controls show that a short-list stage is still useful: the current scorer recovers `-15` Tesseract characters if an oracle can choose within score top4, and tabular image-diff reaches `-16` at top4 despite poor rank1 behavior. Build the next head as top-4 shortlist generation plus local code-effect reranking, not a single scalar threshold policy.
- A first top-4 shortlist diff-crop/image-feature reranker is negative despite high AUC/AP: it ranks 0/17 val oracle-change groups first. Do not spend more time on the same bitmap-diff CNN; the reranker needs explicit latent/code-center direction, local glyph context, or a stronger group-aware objective.
- Feature-family audits reinforce the same conclusion. The promoted codegeom scorer reaches only `-5` table Tesseract edits versus a `-24` oracle before gating, while the failed top4 reranker has `-15` shortlist oracle headroom but a `+1` learned best-score policy. Core rank/error features and isolated image/codebook scalar features are useful filters, not a true glyph repair representation. The next scorer should use full code-center direction, local frozen-base latent tokens and local glyph context together; see `docs/ASSIGNMENT_RERANKER_DESIGN_2026_06_25.md`.
- A first tensor-backed nearest-plus-top4 code-effect reranker is a weak positive smoke, not a promotion. Full center-direction tensors with a pure listwise MLP improve the top4 table policy from `-5.0` to `-5.67` Tesseract edits with PARSeq unchanged and best seed `-7`, while image-augmented/no-change-penalized versions are weaker. The next improvement should add true local frozen-base latent-token windows and better calibration, not rely on the current image triplet branch.
- Simple score averaging/fusion is not enough. Averaging tensor-reranker seeds gives `-6`, and fusing with the baseline score does not exceed the best single seed `-7`. Treat this as a representation/calibration problem rather than an ensemble problem.
- Full residual-coordinate tabular features are a high-AUC trap. Adding 512-D residual coordinates raises validation AUC/AP to `0.894/0.727`, but the policy worsens Tesseract by `+2.67` edits and finds zero improving groups. Do not flatten local latents into a wider table; use structured token encoders with explicit no-op/group calibration.
- The promoted image-diff gate should not simply be reused with a new scorer. On the tensor code-effect seed1 scores, the fixed gate drops `-7` table Tesseract edits to about `-1`. Train the no-op/gate jointly with the code-effect scorer or recalibrate it on the new score distribution.
- Score-margin calibration over the tensor code-effect scorer is helpful but insufficient. A nearest-vs-best-candidate margin gate raises the best seed1 policy from `-7` to `-8` Tesseract edits at zero PARSeq delta when tuned on val, but it uses `11` changes with `7` false changes and one Tesseract-worsen group, whereas the promoted actual-bitstream image-diff gate reaches the same aggregate gain with only `5` changes and one false change. Train-tuned thresholds fall back to `-7` at best, so this should be treated as a calibration diagnostic, not a promotion.
- Direct logit penalties for Tesseract non-improve/harm candidates are negative. Penalizing `delta>=0` non-nearest candidates lowers mean policy gain to around `-5`, and penalizing only `delta>0` harmful candidates is worse. This suggests the no-op problem is not solved by scalar suppression losses over the current representation; it needs group-aware calibration and better local latent/code-effect context.
- Directly supervising the best-non-nearest-vs-nearest group-change margin is also negative with the current representation. It does not reduce false changes enough and stays below the code-only pure-listwise smoke. The next calibration attempt should be architectural, with an explicit group/no-op head sharing a better local context encoder, rather than another scalar loss over the same row logits.
- A 10-seed h96 code-only sweep does not reveal a hidden strong model seed: best remains `-7` Tesseract edits and the mean drops to `-4.6`. Do not spend more time on seed sweeps of the same MLP.
- Local latent context has now been exported, but raw concatenation, a simple separated branch, a minimal latent-CNN branch and a first structured latent-window CNN branch are negative controls. The tensor artifact includes per-candidate `latent_yhat`, `latent_target` and `latent_residual` vectors, yet a simple concat MLP gets high validation AP while dropping policy gain to only mean `-2` Tesseract edits. A separated branch with stronger regularization recovers only to mean `-5.33` and keeps many false changes; a tiny latent-CNN branch reaches only `-5.0`; the structured `[300, 5, 32, 12, 12]` window-CNN smoke reaches only mean `-4.33`/`-4.0`; increasing scalar no-change pressure worsens to `-3.0` despite higher AUC/AP; post-hoc margin gating over window scores reaches only best `-6` with `15` false changes. The next latent path should use these structured windows inside an explicit group/no-op calibrated architecture, not as a standalone pooled branch, scalar no-op penalty variant or post-hoc margin policy.
- A zero-residual escape code is not sufficient by itself. On the harmful seed2 split, appending a zero center to the utility-weighted K64 codebook leaves Tesseract CER unchanged and slightly worsens PSNR. Residual-relative thresholding also fails: threshold 0.8 zeroes most selected candidates and removes the Tesseract gain, while threshold 1.0 zeroes fewer candidates and leaves CER unchanged. A useful no-op path needs utility-aware assignment or a learned confidence rule, not just another nearest-neighbor center or residual-error margin.
- Eval300 oracle-selected K64/K80 compact codebook is the best compact-payload result so far: 3.7860 actual bpp, 0.0253 enhancement bpp, PARSeq CER 0.0567 and Tesseract CER 0.4079/0.4052. It nearly preserves the raw PARSeq oracle but still misses most Tesseract raw-oracle gain.
- Multi-teacher utility should be the default for the next oracle/proxy target. Add pairwise/listwise losses, image-conditioned scores, and explicit rate cost terms before large training.

Immediate next prototype:

1. Build a small utility-aware gate/head over compact frozen MLIC++ latent tokens and local CNN patch features using candidate counterfactual labels.
2. Use within-image/listwise ranking metrics and actual transmitted bpp as the primary validation target.
3. Add a candidate-conditioned code-assignment head for selected candidates. The immediate target is to imitate the top-8 PARSeq-safe assignment oracle while preserving actual payload size; use oracle-choice/no-op group labels as the default target, and focus on code-effect/local glyph representation rather than only difficulty/source metadata, bitmap-difference patches, hand-built image-diff statistics, raw tabular latent descriptors, flat latent concatenation, tiny pooled latent CNNs, standalone window-CNN branches or post-hoc score margins. The first success criterion is to raise oracle-code rank-1 coverage above the current 4/17 oracle-change groups without increasing false/wrong changes; the next concrete route is to extend the tensor-backed nearest-plus-top4 code-effect reranker with structured local frozen-base latent-token windows, an explicit group/no-op head and cross-fitted calibration.
4. Add explicit diagnostics for missed-oracle, false-change and wrong-change groups before promotion. A candidate that increases row AUC but increases false changes in no-headroom IAM-like cases should not be promoted. The current image-diff baseline should be treated as the precision floor: false/wrong changes should stay near 1/75 while missed-oracle coverage improves.
5. Promote assignment models only after actual `.oscr` reconstruction plus row-wise OCR comparison, because table-level policy sums can overstate gains. Current promotion baseline is unchanged actual bpp, exact PARSeq preservation, and a Tesseract gain larger than the pooled image-diff group-gated baseline (-8 unicode / -8 latin-alnum chars versus nearest).
6. Tune the selector on policy-level objectives, not only row-wise AUC/AP. Binary multi-teacher improvement and listwise no-op training are stronger than residual-magnitude selection; `tesseract_parseq_safe` is the current best non-oracle direction but still needs strict validation-time constraints.
7. Replace raw int8 residual tensors with OCR-utility-trained codebook or low-dimensional residual symbols; keep raw residual and residual-MSE codebook paths as negative controls.
8. Close the gap between cross-fitted utility-weighted K64 (mean Tesseract 0.4117) and selected-val/oracle-selected compact codebooks using stronger utility-aware codebook/symbol training, with candidate-conditioned rare-harm/no-worse assignment control.
9. Evaluate on Eval300 before expanding to larger scene/document crops.

If G4 fails:

1. Recheck candidate layout and exact serialization overhead.
2. Test a simpler detector-mask residual stream as a negative/diagnostic result.
3. Do not scale a learned predictor on an uninformative oracle.

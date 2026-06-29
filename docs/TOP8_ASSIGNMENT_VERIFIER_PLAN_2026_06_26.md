# Top-8 Assignment Verifier Plan

Last updated: 2026-06-26

## Context

The strongest current non-deployable headroom is the actual `.oscr` score-seed-2 top-8 shortlist oracle:

- same actual transmitted bpp as the current learned assignment policy;
- PARSeq unchanged exactly;
- Tesseract PSM8 `-24` unicode edits versus nearest and `-16` versus the current learned policy;
- all `17/75` oracle assignment changes recovered.

The deployable current best remains the pooled image-diff group gate:

- `5/75` changed groups;
- `4` exact useful changes;
- `1` false change;
- Tesseract `-8` unicode edits versus nearest;
- PARSeq unchanged exactly.

The next performance jump is therefore not a bitrate/container issue. It is a top-8 assignment decision problem.

## Recent Evidence

N082 fixes the current target inventory:

- `57` keep-noop groups;
- `4` keep-current-good-change groups;
- `13` recover-top8-oracle-change groups;
- `1` reject-current-bad-change group.

N083 closes existing score reuse:

- existing code-only, diff-crop, latent-window, latent-context and tabular image-diff score families rank `0/13` recover targets first;
- the single bad current change is rank `1` under all score families.

N084 shows glyph-code features have auxiliary value:

- recover rank<=4 improves to `9/13` with nearest-ink adjacency and `10/13` with largest-diff-component concentration;
- the same features remain unsafe as direct selectors because no-op groups often rank non-nearest candidates above nearest.

N085-N087 close the glyphcode-only verifier branch:

- N085 direct logistic/margin verifier changes only `1.0` group per fold and reaches mean Tesseract delta `-0.67`;
- N086 explains the collapse: `is_nearest` is the largest positive coefficient, recover oracle codes are `0/13` score-rank 1;
- N087 removes nearest rows and still recovers only `1/13` recover oracle codes at non-nearest score-rank 1.

## Design Decision

Do not force no-op acceptance and hard-positive candidate ranking through one scalar score.

The next assignment verifier should be two-stage or multi-head:

1. Candidate scorer:
   - ranks non-nearest top-8 codes;
   - does not train nearest/no-op rows as the same positive class as useful non-nearest codes;
   - uses richer local code-effect evidence than glyph-code scalars.
2. No-op/change acceptor:
   - decides whether any non-nearest candidate is safe enough to transmit;
   - explicitly protects the current false-change floor;
   - uses group-level uncertainty and source/domain calibration.

Promotion remains blocked until a candidate passes table-level policy checks and then actual counted `.oscr` compress/decompress with row-wise OCR.

## Candidate Evidence To Build Next

Prefer evidence that observes the local code-effect transformation, not only scalar summaries:

- small structured patches around the changed glyph region:
  - source crop;
  - nearest reconstruction crop;
  - candidate reconstruction crop;
  - nearest-to-candidate signed difference;
  - source-to-nearest and source-to-candidate residuals.
- compact codebook geometry:
  - nearest center;
  - candidate center;
  - candidate-minus-nearest center;
  - residual alignment with local latent context.
- optional train-teacher auxiliary signals:
  - PARSeq/CRNN/ABINet only as auxiliary regularization or verifier features with cost/provenance recorded;
  - do not use confidence argmax, teacher-loss argmax, hard loss-no-worse filters, source-pseudo-label loss argmax or fixed teacher-loss vetoes as primary selectors.

## Minimal Next Experiments

E1. Build a top-8 candidate-local evidence cache for the N082 target bank:

- one row per top-8 candidate;
- fixed-size image/code-effect patches;
- codebook center/residual vectors;
- N084 glyph-code scalars as auxiliary features;
- no held-out OCR strings as features.

E2. Train a non-nearest candidate ranker only on non-nearest rows:

- positive target: assignment oracle non-nearest code;
- negatives: other non-nearest top-8 codes;
- evaluate only candidate-ranking quality first:
  - recover oracle rank<=1/2/4;
  - wrong-change count under oracle-change-only isolation;
  - source-stratified failure table.

E3. Train a separate no-op acceptor:

- input: top candidate score distribution, margin, group uncertainty, source/difficulty metadata allowed by protocol, and auxiliary features;
- target: safe change versus no change, tuned only on train folds;
- validation: false/wrong/missed groups and Tesseract/PARSeq deltas.

E4. Only if E2/E3 beat table-level current best:

- export policy rows;
- generate counted `.oscr` streams;
- run PARSeq/Tesseract row-wise OCR;
- run paired bootstrap against nearest/current/top8 oracle.

## Stop Rules

Stop a branch if any of the following holds:

- candidate ranker has recover oracle rank1 below `4/13` on the N082 bank;
- no-op acceptor introduces more than `1/75` false or wrong changes without at least matching current Tesseract `-8`;
- PARSeq worsens on the actual `.oscr` promotion check;
- gain is only visible in estimated bpp or table labels but not actual byte-counted streams.

## Immediate Recommendation

N088 built the candidate-local evidence cache, N089 tested a tiny BCE patch/code-effect ranker, and N090 tested a listwise CE plus pairwise-margin variant. Both cache rankers fail the first stop rule: recover oracle rank1 is `1/13` for BCE and `0/13` for listwise, with many wrong changes under oracle-change-only isolation.

N091 moved to broader Eval300 train-partition top8 supervision and finally produced useful candidate-ranking signal (`7/17` validation oracle-change rank1; oracle-change-only Tesseract `-12`). N092 then showed that a single train-tuned best-score threshold does not transfer as a no-op acceptor: safe train thresholds recover no useful validation changes, while aggressive thresholds create many false/wrong changes. N093 added score margins, group uncertainty summaries, source one-hot and best-candidate deployable tabular features to small acceptor models; those acceptors fit train perfectly but recover zero useful validation changes. N094 shows the mismatch is only partly explained by deep first-stage rank: all rank>4 val oracle changes fail, but rank<=1/2 buckets also contain wrong changes, and source/text overlap diagnostics do not support a simple source or reference filter. N095 extracted full Eval300 top8 CRNN/ABINet teacher OCR/loss and tested fixed PARSeq/CRNN/ABINet string-consensus rules; strict consensus recovers `0/17` val oracle changes and collapses to no-op, so direct teacher string consensus is closed as a selector. N096 then holds out each source domain and shows N091's candidate ranking is itself not source-robust: only `20/80` held-out-source oracle changes are rank1 and `54/80` are rank<=4. N097 decomposes that source failure: widening scalar/tabular evidence and source-balanced loss do not fix it, while the simple `topk_rank + assignment_relative_error` prior reaches `36/80` rank1 and `67/80` rank<=4 under source holdout. Keep this prior as a source-robust anchor, but do not treat it as a selector because oracle-change-only use still makes `44` wrong changed selections. N098 tests that anchor with train-tuned shallow no-op acceptors; it still reaches only validation Tesseract `-2` with six false changes, so top-k prior plus one-dimensional/logistic acceptors is closed before any counted `.oscr` export. N099 then adds richer local latent/code evidence to the same top-k anchor, including code-center geometry, latent target/residual vectors and local latent windows. It fits train exactly but reaches only validation Tesseract `-3` with `1/17` exact useful changes and `2` false/wrong changes, while safe-improve targets collapse to no-op. N100 replaces in-sample thresholding with source-domain OOF calibration plus hard-negative weighting; the OOF ensemble collapses to no-op, and a full-train model with the OOF threshold reaches only validation Tesseract `-2` with three false/wrong changes. N101 scales that same OOF verifier to the merged Eval600+Eval300 calibration artifact; all tested conditions collapse to no-op on `109` validation groups (`0/21` exact useful changes). This closes the current small latent/code-token verifier, including simple source-OOF/hard-negative-threshold variants and larger same-kind source-balanced calibration, as a seed/capacity/data-scaling path. N102 then tests the remaining teacher path properly: CRNN/ABINet loss is used only as an auxiliary head over deployable latent/code evidence. It yields a small best-seed rank1 improvement (`8/17` vs `7/17`) but is unstable and still makes `9` wrong choices among oracle-change groups, so it is a representation hint rather than a selector. N103 source leave-out confirms the hint is not exact-rank robustness: teacher weight `0.2` reduces exact changed groups (`24/80 -> 23/80`) but improves aggregate Tesseract (`-41 -> -46`), implying the next objective should model OCR utility directly with false/wrong penalties rather than optimizing exact oracle-code identity alone. N104 tries a no-op-inclusive train OCR utility hard target and fails validation (`2` exact useful, `4` false, `1` wrong, Tesseract `0`), so policy-level utility is necessary but train-Tesseract hard labels are not the answer. N105 then tries the teacher-separated version with CRNN/ABINet loss utility hard targets; it also fails (`0` exact useful, `3` false, `1` wrong, Tesseract `0` at the best conservative setting), so direct hard teacher-loss utility targets are closed too. N106 keeps teacher loss as an auxiliary ranker signal and trains source-OOF no-op guards over ranker-derived group features; this is safer but collapses recall (`1` exact useful, `1` false, Tesseract `-1`), so shallow OOF guards over the same score summaries are also closed. N107 extracts CRNN/ABINet logit/probability summaries as richer OCR-aware features; fixed summary classifiers still produce no deployable policy and oracle-change-only use remains wrong-heavy, so shallow fixed logit summaries are closed. N112 preserves more sequence-position evidence with binned CRNN/ABINet profiles, but it also fails as a shallow selector: train-tuned policies collapse to no-op, while oracle-change-only Tesseract `-11` requires `13` wrong choices among `17` oracle-change groups. N113 audits whether those wrong choices might be useful alternatives under OCR utility; on this top8 artifact they are not. Exact assignment-oracle labels and the Tesseract/PARSeq-safe utility oracle agree at group level in both train and validation. N114 then separates candidate-set coverage from no-op acceptance across 32 score sources. The clean validation topk-rank upper control is still strong (`12/17` exact, Tesseract `-17` at rank<=4; `17/17`, `-24` at top8), and perfect candidate choice with only 5 changes could reach `-12`. But the strongest-looking `source_mod5_r4` score is not clean-val held-out evidence: its truly source-mod-held-out clean-val slice has rank1 `0/6` and rank<=4 `5/6`, while its in-train clean-val slice has rank1 `11/11`. Use source_mod5 only as evidence that top4 candidate sets may contain the right code, not as a deployable score source. N115 tests the clean deterministic rankcap4 set directly; a shallow random forest chooser improves covered-only Tesseract to `-13` and rank1 to `7/17`, but still has `5` wrong and `5` missed groups. Therefore rankcap4 is a useful substrate, but shallow tabular candidate choice is not enough. N116 adds local MLIC++ latent-window evidence through a small CNN over the same clean rankcap4 set; it ties rank1/rank<=4 (`7/17`, `12/17`) but weakens covered-only Tesseract to `-7`, so this exact small latent-window listwise CNN is also closed as a seed/capacity path.

N117 builds the first leakage-controlled first-stage improvement after N114: source-modulo OOF scores are filtered so every clean-val candidate is scored only by a model that held out its source-modulo slice. As a candidate set, OOF score top4 reaches `14/17` exact oracle changes and Tesseract `-21`, while topk4-union-OOF-top4 reaches `15/17` and `-22`, both above deterministic rankcap4's `12/17` and `-17`. This is not a deployable selector because it uses an oracle chooser inside the shortlist, but it gives a better next substrate than pure rankcap4.

N118 then tests the obvious nearest/no-op-inclusive selector over that N117 shortlist, using OOF score/rank/margins plus deployable codebook/image-diff features. It is a clean negative: shallow logistic, HistGradient and random-forest policies cannot recover the N117 headroom safely. The best moving policy recovers only `1/17` exact useful change and adds false/wrong changes, while conservative policies collapse to no-op.

N119 audits that negative. The union shortlist covers `15/17` validation oracle changes, but all covered oracle candidates score below nearest under the rebuilt N118 models (`0/15` oracle>nearest for logistic, HistGradient and random forest). Random forest does rank oracle above wrong non-nearest in `8/15`, so the signal is partly present among non-nearest candidates but gets suppressed at the nearest/no-op boundary.

N120 implements that decoupling. A hist-gradient non-nearest proposer reaches rank1 `6/17` and rank<=4 `14/17`, and separate hist/RF risk abstention gives a safe partial policy: `2/17` exact useful changes, no false/wrong changes, Tesseract `-5`, PARSeq `0`. This is below the current actual-bitstream floor (`-8`), so it is not promotable, but it is a useful directional improvement over N118.

N121 audits N120's misses. The safe hist-gradient path has `6` exact proposals, but risk can accept only `2` at zero false/wrong; `9` oracle groups are already wrong at proposal rank1 and `2` are missing from the shortlist. Therefore the immediate bottleneck is proposal rank/recall, not a risk-threshold knob.

N122 tries the cheap hard-positive route. It sweeps 36 scalar weighting settings for the hist-gradient proposer, but the best safe risk policies still tie N120 at `2/17` exact and Tesseract `-5`; proposal rank1 can rise to `8/17`, but those settings do not survive risk gating without false/wrong pressure.

N123 tries the richer local-representation route inside the N117/N120 shape. It adds candidate-local latent windows, raw code/latent vectors and deployable table features to a listwise non-nearest proposal model, with teacher/OCR fields excluded from inference. This also fails as a path to promotion: validation proposal rank1 drops to `4/17` versus N120's `6/17`, all-proposal use creates `58` false and `13` wrong changes, and every risk gate collapses to no-op. The lesson is not that local evidence is useless; it is that high-capacity latent-window scoring on this small clean-positive set overfits unless the objective/calibration or supervision source changes materially.

N124 tries the low-variance alternative to N123. Deterministic topk/rate anchors recover proposal rank1 `6/17`, tying N120, but risk transfer is still weak: the best observed policy is only Tesseract `-2` with `2/17` exact useful changes plus `3` false and `1` wrong changes. This means the current problem is no longer just proposal ranker capacity or anchor choice; it is no-headroom verification/calibration.

N125 tries source-held-out risk threshold calibration over deterministic N117 anchors. It is safer mostly by abstaining: topk and z-fused anchors collapse to no-op, and the best moving oof-rank/random-forest case reaches only Tesseract `-1` with `1` false and `1` wrong change. This closes the current shallow risk-calibration path.

N126 quantifies why that path is exhausted. On the N117 union shortlist, validation still covers `15/17` oracle-change groups and has a non-deployable covered-oracle upper control of Tesseract `-22`, but the supervision is strongly no-headroom dominated: `58` validation groups have no useful non-nearest change, and candidate-level negative/positive balance is `24.8`. A future verifier must accept at least `3` exact useful changes with near-zero bad moves to beat the current actual-bitstream floor (`-8`). This makes the next step a supervision/objective problem, not another proposal-score or threshold problem.

N127 creates the train-only supervision substrate for that objective change. It exports `60` covered train oracle-positive candidates and `162` matched no-headroom negatives from the N117 shortlist, with validation excluded. Treat this as a hard-cohort training set, not as policy evidence. The next verifier must prove that OCR strings, references, edit deltas, evaluator outputs and validation rows are excluded from inference features before any table-level validation result is trusted.

N128 tests exactly that first scrubbed hard-cohort path. The feature scrub succeeds (`227` features, no OCR/reference/oracle leakage), but train source-mod OOF thresholding collapses to no-op on validation. All-change policies are false-heavy (`58` false changes and `14-15` wrong changes), so N127 hard-cohort supervision alone is not enough. Use N127 only as auxiliary supervision inside a materially different policy-level verifier or richer local evidence model; do not sweep N128's scalar/tree classifier family.

N129 changes N128's row-classification objective to matched pairwise ranking over N127 positives versus no-headroom negatives. It reaches perfect train pair accuracy, but the transferred validation policies still collapse to no-op or a false validation change. This closes linear pairwise scoring over the same scrubbed feature family; the next verifier must change evidence or optimize group-level false/wrong/missed tradeoffs directly.

N130 adds an actual `.oscr` section-accounting constraint to the same top8 work. Current, top4 oracle and top8 oracle streams have identical mean actual bpp (`3.849649`) and enhancement payload bpp (`0.028989`), so selector headroom is not from extra payload. However, selected streams average only about `3` logical enhancement bytes while optional table/header overhead makes the selected optional total `103` bytes. N131 implements the compact `TEXT_SYMBOLS` path and confirms exact reconstruction preservation on a 5-crop smoke with `60` bytes saved per one-candidate stream. N132/N133 scale compact packing to current/top4/top8 Eval300 streams, N134 adds the nearest-assignment baseline, and N135 joins those matched compact rates with the held-out OCR bootstrap evidence. Nearest/current/top4/top8 now share mean actual total bpp `3.774026` and enhancement payload bpp `0.003781`; Tesseract unicode improves nearest -> current/top4/top8 by `-8`/`-20`/`-24` chars while PARSeq, TrOCR, PaddleOCR recognizer and PaddleOCR E2E stay unchanged. This improves rate credibility and removes split-vs-compact oracle mismatch, but it does not relax the N126-N129 verifier stop rules.

N136 revisits compact symbol learning from the center-utility side. After regenerating K64 utility-weighted train codebooks with valid nonzero `center_utility`, utility-biased assignment under relative-error `1.01` and `1.05` preserves PARSeq and matches compact bytes, but gives only Tesseract `-1` versus nearest and is `+7` worse than the current compact policy. This closes scalar center-utility assignment as a selector path; center utility is useful metadata, not sufficient policy evidence.

N137 explains why. On the validation top8 table, center utility ranks only `3/17` oracle-change codes first and only `9/17` in the top4; relative-error guards make most useful oracle codes infeasible (`1/17` eligible at `1.01`, `5/17` at `1.05`). Dropping the guard creates a false-change flood (`54` false, `12` wrong out of `75` groups). The next compact-symbol policy must therefore change either candidate-local evidence or the symbol/proposal distribution, not just the scalar center prior.

N138 tests the symbol/proposal-distribution idea without making OCR claims. Adding train oracle-change residual exception centers to K64 makes K4/K8 very good at reaching useful validation residuals (`16/17`) but also too permissive on no-headroom groups (`31-36` close exception opportunities). K16 is the most plausible smoke-test point (`9/17` useful, `13` no-headroom close opportunities), while K32/K64 overfit train positives. This suggests a small exception-center bank plus no-headroom gating may be more promising than center utility, but it must be validated through decoded compact streams before becoming a result.

N139 performs that first actual-stream check for K16 exception centers by building K80=K64+16 codebooks and regenerating compact validation `.oscr` streams. The result is a clean no-op: `75` selected candidates are transmitted across seed0/1/2, but `assignment_changed_total=0`. Therefore appending exception centers to the codebook does not alter the current nearest assignment path. The next compact-symbol experiment must implement an explicit exception selector/no-headroom verifier before any OCR evaluation.

N140 tests the simplest possible selector for that explicit exception path. It is a negative: distance-only eligibility is high precision on train (`44/63` useful, `1` false at `rel<=1.0`) but false-heavy on validation (`9/17` useful, `13` false), train-tuned thresholds leak false changes, and shallow learned scores over distance/margin/center utility no-op. Exception centers now require candidate-local visual/code-effect evidence, not threshold polishing.

N141 runs the conservative `rel<=1.0` policy through actual compact streams and OCR. It lowers rate by selecting only `22` candidates versus current's `50`, but PARSeq is merely unchanged and Tesseract worsens by `+3` unicode / `+6` latin-alnum chars aggregate. On selected rows, extra centers do not deliver a net Tesseract unicode gain. Therefore future exception-center work must preserve useful current selections or learn replace/drop decisions; lower enhancement rate alone is not an OSCAR result.

The next coding task should not be another tiny N088-cache BCE/listwise/pairwise variant, scalar threshold, score consensus, glyphcode-only model, logistic calibration, small tree/logistic acceptor over N091 score summaries, fixed teacher string-consensus selector, source/rank filter, wider all-tabular feature family, source-balanced loss-only rerun, shallow top-k-prior no-op acceptor, N099-style latent/code-token verifier seed/capacity sweep, source-OOF threshold/hard-negative-weight variant over that same verifier, larger same-kind calibration append to that verifier, direct shallow threshold on the N102/N103 teacher-auxiliary ranker, train-Tesseract hard utility target, hard CRNN/ABINet teacher-loss utility target, N106-style shallow OOF guard over ranker score summaries, N107-style shallow fixed logit-summary classifier, N112-style binned sequence-profile classifier, use of source_mod5 clean-val in-train scores as validation evidence, capacity/seed sweeps of the N115 shallow tabular rankcap chooser, capacity/seed sweeps of the N116 small latent-window rankcap CNN, source-modulo top-score policies, seed/capacity sweeps of the N118 shallow OOF-shortlist selector, direct promotion of N120, risk-only threshold sweeps on current N120 scores, scalar sample-weight sweeps like N122, seed/capacity/epoch/dropout sweeps of the N123 high-capacity latent-window proposal, more deterministic rank/score-fusion anchors like N124, source-OOF thresholding of the same shallow N120 risk features like N125, N128-style shallow hard-cohort verifiers, N129-style linear pairwise hard-cohort scorers, scalar center-utility assignment threshold sweeps like N136, nearest-assignment K-plus-exception-center stream reruns like N139, distance-only exception-center selector thresholds or shallow distance/margin/center-utility classifiers like N140, sparse exception-only OCR policies like N141, or any N117 selector that cannot explain how it handles N126's candidate negative/positive ratio. Keep the N117 OOF shortlist and N120 two-stage shape as a scaffold only if the next verification objective is materially different or uses substantially more clean supervision. Use `--compact-text-section` for future actual streams; N134/N135 establish nearest/current/top4/top8 compact rates and OCR summary, while any new-policy compact rate still needs its own actual-byte accounting.

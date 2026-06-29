# Eval600 Source-OOF to Eval300 Top-4 Topology Selector

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream selector diagnostic; no counted `.oscr` promotion stream

## Question

After N053, direct OCR-teacher encoder features are not a useful shallow selector input. This diagnostic asks whether deployable image-processing features that are closer to text structure can help:

> Do connected-component, hole, projection, approximate stroke-width and candidate-vs-nearest dark-pixel topology features recover useful top-4 assignment choices?

No OCR model is used by the feature extractor. Features are computed only from the source crop, nearest reconstruction and candidate reconstruction.

## Setup

- Candidate set: nearest/no-op plus union top-4 from source-OOF oracle-choice and safe-improve scores.
- Split: train all non-Eval300-seed1 groups, validate Eval300 seed1.
- Feature cache:
  - existing source-OOF selector/scalar image/code features: `143` dims;
  - new topology features: `299` dims;
  - final feature matrix: `(4384, 442)`.
- Topology threshold: grayscale `< 0.5` as dark/text-ish pixels.
- New feature families:
  - connected components and component areas;
  - hole/euler proxies;
  - dark row/column projections and run counts;
  - stroke-width approximation from distance transform;
  - boundary and Sobel statistics;
  - added/removed dark-pixel fractions between candidate and nearest;
  - local diff-bbox topology around candidate-vs-nearest changes.

Two selector variants were run:

- `topology_ce`: standard group CE.
- `topology_balanced`: oracle-change weight `8`, positive margin weight `1`, no-headroom and harmful margins `0.2`.

## Results

Means are over 3 selector seeds on Eval300 seed1 val75.

| Policy / variant | Changed groups | Tesseract delta | Tesseract worsen groups | False changes | Wrong changes | Exact oracle changes | Missed oracle changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| shortlist oracle | 15.00 | -20.00 | 0.00 | 0.00 | 0.00 | 15.00 | 2.00 | 0.00 |
| topology CE argmax | 6.00 | -0.33 | 0.67 | 5.67 | 5.67 | 0.33 | 15.33 | 0.00 |
| topology CE train-tuned margin | 6.00 | -0.33 | 0.67 | 5.67 | 5.67 | 0.33 | 15.33 | 0.00 |
| topology balanced argmax | 10.67 | -1.33 | 0.00 | 10.00 | 10.00 | 0.67 | 13.67 | 0.00 |
| topology balanced train-tuned margin | 10.67 | -1.33 | 0.00 | 10.00 | 10.00 | 0.67 | 13.67 | 0.00 |

## Interpretation

Topology features provide a weak directionally useful signal: unlike the direct PARSeq encoder feature selector, the balanced topology model produces a small mean Tesseract improvement and no Tesseract-worsen groups. However, the precision is far below the promotion threshold. Recovering `-1.33` edit characters while emitting `10.0` false/wrong changes is not a safe assignment policy, and it is far below the current actual-bitstream learned baseline (`-8`) and the validated shortlist oracle (`-20`) at the same payload size.

This is therefore a valid negative result for hand-crafted topology features as a shallow top-4 MLP input. The signal may be useful as an auxiliary feature family, but it is not enough by itself and should not be seed/capacity swept without a different policy objective or calibration setup.

## Next Action

Do not run counted `.oscr` streams for this selector. If topology features are reused, pair them with a high-precision abstention/verifier objective or source/domain-disjoint calibration; do not treat this weak table-level gain as promotion evidence.

W&B offline runs: CE `zfmoo46f`, balanced `ozo0gg97`.

## Files

- Feature extractor: `scripts/extract_topology_topk_feature_cache.py`
- Selector trainer: `scripts/train_assignment_topk_balanced_selector.py`
- Feature cache: `experiments/features/eval600_source_oof_to_eval300_top4_topology_features_2026_06_26.npz`
- CE result: `experiments/results/eval600_source_oof_to_eval300_top4_topology_selector_ce_2026_06_26.json`
- Balanced result: `experiments/results/eval600_source_oof_to_eval300_top4_topology_selector_balanced_2026_06_26.json`

# Eval600 Source-OOF to Eval300 Top-4 Failure Strata

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream  
W&B: `ox7xrbee`

## Question

N049/N050 localized the failure to deployable candidate scoring: the top-4 shortlist contains most useful oracle assignments, but learned top-1 choices are false or wrong too often. This N051 diagnostic asks where the failure concentrates:

1. Which sources and difficulty strata contain the recoverable oracle headroom?
2. Which strata create false or wrong non-nearest selections?
3. Does a simple source/length/base-OCR filter look plausible, or does the next model need genuinely richer evidence?

## Setup

- Table: `eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl`
- Scores: source-OOF mixed oracle-choice and safe-improve scores, score model seed `2`
- Eval split: Eval300 seed1, `75` groups
- Shortlist: nearest plus union top-4 from oracle-choice and safe-improve scores
- New script: `scripts/analyze_top4_failure_strata.py`

OCR deltas are analysis labels only. No deployable selector and no `.oscr` promotion stream were produced.

## Overall

| Policy | Changed | Tesseract delta | Exact oracle | False | Wrong | Missed oracle |
|---|---:|---:|---:|---:|---:|---:|
| nearest | 0 | 0 | 0 | 0 | 0 | 17 |
| full oracle | 17 | -24 | 17 | 0 | 0 | 0 |
| top-4 shortlist oracle | 15 | -20 | 15 | 0 | 0 | 2 |
| oracle-score argmax | 4 | +1 | 0 | 4 | 0 | 17 |
| safe-score argmax | 23 | +3 | 2 | 19 | 2 | 15 |
| fused-score argmax | 10 | +1 | 0 | 10 | 0 | 17 |
| conservative-score argmax | 11 | +1 | 0 | 11 | 0 | 17 |

The result reinforces N049: top-4 coverage is good enough to beat the current learned actual-bitstream baseline if selection were calibrated, but current top-1 scores are not usable.

## Source Strata

| Source | Groups | Oracle changes | Shortlist delta | Safe exact | Safe false | Safe wrong | Safe delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| IAM words | 36 | 10 | -11 | 2 | 11 | 2 | +2 |
| ICDAR2013 | 8 | 3 | -3 | 0 | 0 | 0 | 0 |
| ICDAR2015 | 9 | 3 | -5 | 0 | 2 | 0 | +1 |
| ICDAR MLT 2019 | 3 | 0 | 0 | 0 | 1 | 0 | 0 |
| SynthText words | 10 | 1 | -1 | 0 | 2 | 0 | 0 |
| TextOCR | 9 | 0 | 0 | 0 | 3 | 0 | 0 |

IAM is both the largest opportunity and the largest risk. It contains `10/17` oracle-change groups and `-11` of the `-20` shortlist-oracle Tesseract edits, but the safe-score argmax creates `13` false/wrong IAM changes and worsens aggregate Tesseract by `+2` inside IAM. A source-only rule is therefore too blunt.

## Difficulty Strata

By nearest Tesseract distance:

| Distance bin | Groups | Oracle changes | Shortlist delta | Safe exact | Safe false | Safe wrong | Safe delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact | 16 | 0 | 0 | 0 | 3 | 0 | 0 |
| 1-2 | 24 | 4 | -5 | 0 | 7 | 0 | +1 |
| 3-5 | 26 | 9 | -10 | 2 | 9 | 0 | -1 |
| 6-9 | 8 | 3 | -5 | 0 | 0 | 1 | +2 |
| 10+ | 1 | 1 | 0 | 0 | 0 | 1 | +1 |

The most promising band is nearest Tesseract distance `3-5`: it holds `9` oracle-change groups and safe-score recovers the only `2` exact changes. But it still produces `9` false changes. Very hard cases (`6+`) are not automatically safe: the two wrong changes are there, including one missing-shortlist case.

By reference length:

| Length bin | Groups | Oracle changes | Shortlist delta | Safe exact | Safe false | Safe wrong | Safe delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-4 | 39 | 5 | -5 | 2 | 11 | 0 | 0 |
| 5-8 | 24 | 8 | -11 | 0 | 3 | 2 | +3 |
| 9-12 | 10 | 2 | -2 | 0 | 5 | 0 | 0 |
| 13+ | 2 | 2 | -2 | 0 | 0 | 0 | 0 |

Short words are dangerous because many false changes are visually plausible but OCR-neutral. Mid-length words contain most headroom, but also the wrong-change cases. This argues for a verifier that reasons about code-effect on text evidence, not just a length/difficulty gate.

## Interpretation

N051 rules out a comforting shortcut. The useful changes are concentrated enough to diagnose, but not enough to filter with source, word length or base OCR difficulty alone. IAM and nearest-distance `3-5` are high-yield, yet they are also high-false-pressure. Exact/no-headroom and short-word strata still create false non-nearest choices.

The next top-4 selector should not be another shallow score/rank/patch summary model. Better next moves are:

- an OCR/text-aware fixed verifier whose weights and license can be recorded, used as a teacher/verifier without contaminating held-out Tesseract labels;
- a structured MLIC++ latent/code-token code-effect representation with source-disjoint calibration;
- a two-stage policy objective that first proposes top-4 candidates and then explicitly rejects false changes in short/no-headroom cases.

TrOCR-small-printed remains a held-out evaluator in the current registry, so it should not be used as a selector feature or training target unless the OCR protocol is explicitly revised.

## Files

- Result JSON: `experiments/results/eval600_source_oof_to_eval300_top4_failure_strata_2026_06_26.json`
- Details JSONL: `experiments/results/eval600_source_oof_to_eval300_top4_failure_strata_details_2026_06_26.jsonl`
- Script: `scripts/analyze_top4_failure_strata.py`

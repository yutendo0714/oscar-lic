# Eval600 to Eval300 External Patch-Gate Diagnostic

Date: 2026-06-25

## Purpose

The Eval600 internal patch-gate split recovered only a small fraction of oracle headroom. This run removes the internal-split optimism by training the same `tesseract_parseq_safe` one-candidate patch-listwise gate on the Eval300-disjoint Eval600 table and evaluating directly on the fixed Eval300 table.

This is a table-level single-candidate diagnostic. Policy bpp uses measured serialized single-candidate `.oscr` rows, but no combined counted promotion stream was generated.

## Configuration

- Train table: `candidate_utility_table_eval600_noeval300_tile4_scale025_q005_with_image_features_2026_06_25.jsonl`
- Eval table: `candidate_utility_table_eval300_tile4_scale025_q005_with_image_features_2026_06_25.jsonl`
- Model: patch-listwise gate, hidden 32, dropout 0.2, 400 epochs
- Target: `tesseract_parseq_safe`
- Seeds: 0, 1, 2
- Threshold tuning: train-side multi-teacher threshold plus train-side Tesseract-primary / PARSeq-protected constrained threshold
- Device: CUDA
- W&B run: `lykbfyhg`

## External Eval300 Results

| policy | selected images | actual bpp | multi-teacher delta | PARSeq CER | Tesseract CER |
|---|---:|---:|---:|---:|---:|
| base | 0.00 | 3.672637 | 0.00 | 0.066088 | 0.432577 |
| oracle single candidate | 80.00 | 4.348269 | -113.00 | 0.056075 | 0.367156 |
| threshold 0.5 | 3.33 | 3.685344 | +0.33 | 0.066088 | 0.432799 |
| top10 | 10.00 | 3.712759 | -3.33 | 0.066088 | 0.430352 |
| top20 | 20.00 | 3.764649 | -6.33 | 0.066088 | 0.428349 |
| top40 | 40.00 | 3.906255 | -9.67 | 0.066088 | 0.426124 |
| top80 | 80.00 | 4.241551 | -14.00 | 0.066088 | 0.423231 |
| train-tuned | 41.33 | 3.933908 | -10.33 | 0.066088 | 0.425679 |
| constrained train-tuned | 41.33 | 3.933908 | -10.33 | 0.066088 | 0.425679 |
| residual RMS top40 | 40.00 | 4.039783 | +5.00 | 0.066088 | 0.435915 |

The train-tuned and constrained policies are identical in this run. They preserve PARSeq exactly and improve Tesseract CER by about `0.0069`, but they require selecting about 41 images and increase bpp by about `0.2613`. The raw single-candidate oracle still improves Tesseract by about `0.0654`, so this gate captures only a small part of the available external Eval300 headroom.

Row metrics also show the familiar row-policy gap:

| seed | train AUC | train AP | Eval300 AUC | Eval300 AP | Eval300 multi-teacher AUC | Eval300 multi-teacher AP |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7179 | 0.2084 | 0.6606 | 0.1318 | 0.6531 | 0.1341 |
| 1 | 0.7182 | 0.2155 | 0.6455 | 0.1254 | 0.6374 | 0.1270 |
| 2 | 0.7007 | 0.1965 | 0.6327 | 0.1275 | 0.6275 | 0.1300 |

## Interpretation

The external split confirms that additional data helps enough to beat residual-RMS controls and preserve PARSeq, but the existing one-candidate patch gate is still not strong enough for promotion. It does not beat the current promoted actual-bitstream Eval300 assignment result, and its high-rate single-candidate simulation is not the compact OSCAR stream used by the current best method.

The most useful consequence is infrastructure: `scripts/train_candidate_patch_listwise_gate_external.py` now supports fixed train/eval split diagnostics. Future gate claims should use this external mode before any counted `.oscr` promotion run.

## Next Action

Do not seed-sweep this same one-candidate patch gate. The next useful work should change the evidence or objective:

- source/difficulty-aware high-precision no-headroom rejection;
- more groups with fixed external validation;
- top8 compact-symbol assignment with explicit false-change control;
- subset-aware allocation for multi-candidate interactions.

## Artifacts

- Result JSON SHA256: `e63cf65142fdcf37c9fb1985d949a83049c198aa5067f2cada05fa8de92064bf`
- External runner SHA256: `9346335913b4d756bc11ffb630e047efce5665e16e18345c048cb62350fbfdc9`
- W&B offline run: `wandb/offline-run-20260625_223817-lykbfyhg`

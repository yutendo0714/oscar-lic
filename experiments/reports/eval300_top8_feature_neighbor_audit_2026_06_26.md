# Eval300 Top8 Feature Neighbor Audit

This is a diagnostic, not a deployable selector. Tesseract deltas are used only to audit neighborhood support and policy failure modes.

## Key Aggregates

- `rank_relerr` k=5: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.7203548085901027
- `rank_relerr` k=15: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.7203548085901027
- `rank_relerr` k=31: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.7418300653594772
- `codegeom_scalar` k=5: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=-0.3333333333333333, argmax dT=-0.3333333333333333, oracle_top4_recall=0.688608776844071
- `codegeom_scalar` k=15: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.619514472455649
- `codegeom_scalar` k=31: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.4757236227824463
- `codegeom_onehot` k=5: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=-3.3333333333333335, oracle_top4_recall=0.6942110177404295
- `codegeom_onehot` k=15: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=-0.3333333333333333, argmax dT=-2.6666666666666665, oracle_top4_recall=0.6409897292250233
- `codegeom_onehot` k=31: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=-1.0, oracle_top4_recall=0.611577964519141
- `imgdiff_scalar` k=5: safe_tuned dT=-0.3333333333333333, false=1.3333333333333333, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.7633053221288515
- `imgdiff_scalar` k=15: safe_tuned dT=-1.0, false=0.3333333333333333, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.6139122315592904
- `imgdiff_scalar` k=31: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.6704014939309058
- `latent_summary` k=5: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.7497665732959851
- `latent_summary` k=15: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.7338935574229692
- `latent_summary` k=31: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.7418300653594772
- `code_img_latent` k=5: safe_tuned dT=0.3333333333333333, false=2.3333333333333335, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.7124183006535948
- `code_img_latent` k=15: safe_tuned dT=0.6666666666666666, false=0.6666666666666666, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.6512605042016807
- `code_img_latent` k=31: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.5392156862745098
- `code_img_latent_parseqteacher` k=5: safe_tuned dT=0.3333333333333333, false=2.6666666666666665, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.6750700280112045
- `code_img_latent_parseqteacher` k=15: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.7180205415499533
- `code_img_latent_parseqteacher` k=31: safe_tuned dT=0.0, false=0.0, exact=0.0; oracle_margin dT=0.0, argmax dT=0.0, oracle_top4_recall=0.6059757236227825

## Interpretation

Best train-tuned safe-threshold neighborhood policy is imgdiff_scalar k=15 with mean Tesseract delta -1.000 over the three seed-held-out top8 diagnostic splits. This is not the same population as the actual-bitstream 225-row current-best/shortlist evaluation, where the learned baseline is -8 total Tesseract edits and the shortlist oracle is -20.
Best neighborhood policy exact/false means are 0.0/0.3333333333333333; if false changes remain high, feature-neighborhood support is insufficient for promotion without a different verifier or more calibration data.
All results that use Tesseract-derived labels are diagnostic only and must not be promoted as deployable OCR-invariant training.

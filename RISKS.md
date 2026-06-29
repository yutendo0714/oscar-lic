# Risk Register

| Risk | Probability | Severity | Early signal | Mitigation |
|---|---:|---:|---|---|
| Single-OCR overfitting | high | high | train OCR improves, held-out worsens | teacher dropout, held-out main metric, worst-OCR objective |
| Hidden side rate | high | high | estimated bpp gain vanishes in files | serialize every stream early; bit breakdown tests |
| Utility interaction non-additivity | high | medium | greedy ranking changes with S | conditional utility, sampled subsets, Shapley approximation |
| OCR teacher data leakage | medium | high | suspiciously high benchmark accuracy | disclose pretraining, alternate datasets, cross-domain test |
| Synthetic-to-real gap | high | high | gains limited to rendered text | real-domain fine-tuning, font/background holdout |
| Tiny text detector failure | high | high | oracle crop much better than end-to-end | detector-free utility path or joint detection-aware training |
| Generator hallucination | medium | high | exact sequence varies by seed | deterministic seed, sequence consistency, disable generator in critical mode |
| Encoder cost too high | medium | high | OCR teacher dominates runtime | offline distillation to lightweight predictor |
| Dataset license conflict | medium | high | redistribution prohibited | scripts/manifests only; no bundled data |
| Baseline environment rot | high | medium | installation failure | pinned commits, isolated envs, patches documented |
| Statistical cherry-picking | medium | high | selected images drive gains | preregister primary metrics, bootstrap CIs, all-image outputs |
| Privacy leakage | medium | high | attacker decodes text from latent | privacy track, encryption option, no plaintext main track |
| Row-metric false confidence | high | high | AUC/AP improves while actual `.oscr` OCR worsens | promote only actual-byte reconstruct/OCR policies; keep negative controls |
| Eval300 pilot overclaim | medium | high | result does not scale to larger approved domains | label as pilot, expand domains and OCR families before paper claims |
| Unsafe corrupted-base decode | medium | high | CRC-bypassed base payload causes entropy decoder crash/OOM | fail closed by default; never decode `BASE_MAIN` without CRC; add explicit recovery profile for optional-section damage only |
| Container GPU visibility loss | medium | medium | `nvidia-smi` fails or CUDA disappears | stop GPU experiments and wait for container restart instruction |

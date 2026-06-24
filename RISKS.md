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

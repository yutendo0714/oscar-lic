# Loss Function Specification

## 1. Total objective

\[
\begin{aligned}
L ={}& R_{all}
+\lambda_{rgb}L_{rgb}
+\lambda_{perc}L_{perc}
+\lambda_{ocr}L_{ocr-inv}
+\lambda_{seq}L_{seq}\\
&+\lambda_{feat}L_{feat}
+\lambda_{top}L_{top}
+\lambda_u L_u
+\lambda_{budget}L_{budget}
+\lambda_{rob}L_{rob}.
\end{aligned}
\]

Every run stores all component means and gradient norms. A gain from silently increasing total loss scale or OCR compute is not accepted.

## 2. Rate

\[
R_{all}=R_{base-main}+R_{base-hyper}+R_{index}+R_{text}+R_{metadata}+R_{protection}.
\]

Training likelihood losses omit fixed container bytes; evaluation uses actual file size. To reduce train/test mismatch, add an empirical overhead model fitted only on training/validation streams.

## 3. RGB distortion

Primary candidate:

\[
L_{rgb}=\alpha\|x-\hat x\|_1 + \beta(1-MS\text{-}SSIM(x,\hat x)).
\]

Report MSE/PSNR even if not trained for it. Use separate text/non-text diagnostics but avoid an unbounded hard text weight that destroys backgrounds.

## 4. OCR-model-invariant distillation

Let teacher `k` produce character posterior `P_k`, intermediate feature `F_k`, detector map `D_k` and confidence `c_k`.

\[
L_{ocr-inv}=\frac{1}{|S|}\sum_{k\in S} w_k
\left[KL(sg[P_k(x)]\|P_k(\hat x))+
\eta\|sg[F_k(x)]-F_k(\hat x)\|_1\right].
\]

Rules:

- Use ground-truth sequence loss when labels exist; teacher-on-original is fallback, not truth.
- Downweight a teacher if it is wrong/low-confidence on the original.
- Sample a teacher subset `S` each step to reduce cost and encourage invariance.
- Teacher dropout must be independent of image content unless explicitly modeled.
- Project different teacher charsets into a canonical Unicode/token space only where mapping is valid; otherwise use feature-level or ground-truth losses.

## 5. Sequence loss

Depending on recognizer:

- CTC negative log likelihood
- token cross entropy with EOS
- soft edit-distance approximation
- alignment consistency for insertion/deletion

Primary supervised loss:

\[
L_{seq}=\frac{1}{K}\sum_k CE/CTC(T_{gt},P_k(\hat x)).
\]

Do not optimize hard CER directly with a biased straight-through estimator without an ablation.

## 6. Feature loss

Feature layers should cover early stroke features and later semantic sequence features. Normalize per-layer scale:

\[
L_{feat}=\sum_{k,l}\omega_{k,l}
\frac{\|\bar F^l_k(x)-\bar F^l_k(\hat x)\|_1}{N_l}.
\]

A teacher-independent alternative is a self-supervised visual encoder, but it may ignore tiny glyph differences; test explicitly.

## 7. Topology loss

For text regions generated from ground truth or teacher consensus:

\[
L_{top}=\gamma_s\|S(x)-S(\hat x)\|_1+
\gamma_e\|E(x)-E(\hat x)\|_1+
\gamma_c L_{clDice}(K(x),K(\hat x)).
\]

- `S`: signed distance or soft foreground map
- `E`: oriented edge map
- `K`: soft skeleton

Topology extraction used for training must not depend on test labels. If a text mask is required, distinguish ground-truth training mask from deployment predictor.

## 8. Utility prediction

Oracle target `u_i` may be heavy-tailed and negative. Use clipped/log-scaled targets plus ranking:

\[
L_u=L_{Huber}(\hat u_i,\tilde u_i)+
\gamma_r\sum_{(i,j)}\log(1+\exp[-s_{ij}(\hat u_i-\hat u_j)])
+\gamma_v L_{NLL}(u_i;\mu_i,\sigma_i).
\]

Sample pairs within the same image and budget trajectory. Across-image absolute utility is less important than within-image ordering, but calibration is needed for global budget allocation.

## 9. Budget and sparsity

For relaxed gate `g_i`:

\[
L_{budget}=\left(\sum_i g_i\hat r_i-B_t\right)_+^2
+\rho\left|\sum_i g_i\hat r_i-B_t\right|.
\]

Do not use L1 sparsity alone; it ignores variable candidate bit cost.

## 10. Perceptual loss

LPIPS/DISTS-like losses may improve visual quality but can encourage text changes. Introduce only after exact OCR metrics are stable, and use lower or zero weight on critical text regions if it degrades sequence correctness.

## 11. Robustness loss

Sample a channel corruption `η` over serialized/dequantized text representations:

\[
L_{rob}=E_\eta[L_{seq}(\hat x_\eta)+\kappa L_{rgb}(\hat x_\eta)].
\]

Training corruptions:

- selected latent erasure
- burst candidate loss
- quantization perturbation
- packet drop

Raw arbitrary bit flips should be tested in serialized streams; differentiable surrogates are training approximations.

## 12. Weighting schedule

Recommended stages:

1. freeze base, train enhancement with RD + supervised OCR
2. add utility loss and budget
3. add multi-teacher invariance
4. unfreeze adapters/base tail at 0.1× LR
5. add robustness/perceptual terms

Use gradient-norm logging. PCGrad or alternating updates are options only after demonstrating gradient conflict; do not add them by default.

# Counterfactual OCR Utility: Formal Definition and Label Generation

## 1. Why this document exists

The utility target is the main scientific variable. An underspecified target would let the implementation drift into text saliency, OCR confidence or reconstruction residual, none of which is equivalent to marginal recognition value per bit.

## 2. State-dependent utility

For selected candidate set `S`, image `x`, reconstruction `x_hat(S)`, OCR loss `L_OCR` and actual serialized rate `R(S)`:

\[
u_i(S)=\frac{L_{OCR}(\hat x(S))-L_{OCR}(\hat x(S\cup\{i\}))}
{R(S\cup\{i\})-R(S)+\epsilon}.
\]

Utility is conditional on `S`; candidates interact. A single static score is an approximation to a policy along likely budget trajectories.

## 3. OCR loss used for oracle labels

Primary oracle score on labeled data:

```text
0.50 * normalized soft sequence loss
+ 0.25 * teacher-ensemble feature discrepancy
+ 0.25 * expected edit proxy
```

Hard CER is recorded but not the sole label because it is piecewise constant. On unlabeled training data, use confidence-weighted teacher consensus and exclude samples with high teacher disagreement.

Held-out OCR models are never used to construct utility labels.

## 4. Actual incremental rate

The denominator includes:

- incremental payload bytes
- change in index/gate bytes
- alignment/padding change
- header length change if any
- protection overhead assigned by policy

Since entropy coder packet overhead can be non-additive, compute `R(S∪{i})-R(S)` by serialization in the oracle prototype. Later, train a rate predictor for scalability.

## 5. Primary label algorithm: greedy trajectories

For each training crop/image and target budget:

1. Encode `B0` and all candidate quantized payloads.
2. Initialize `S=∅`.
3. For a sampled subset of remaining candidates, serialize `S∪{i}`, decode and calculate utility.
4. Select candidate with maximum positive utility.
5. Store `(state summary, candidate, Δloss, Δbits, utility, rank)`.
6. Repeat until hard budget or no positive utility.
7. Run trajectories with multiple candidate-subset seeds and budgets.

This yields labels near the deployment policy rather than only at `S=∅`.

## 6. Interaction diagnostics

### 6.1 Sampled subset marginal values

Sample states `S` from different trajectory stages and evaluate the same candidate. Compute variance of `u_i(S)`. High variance implies the predictor needs state/context features.

### 6.2 Approximate Shapley value

For a small candidate set, sample random permutations and average marginal contribution. Use only as a diagnostic because exact Shapley computation is exponential.

### 6.3 Pairwise synergy

\[
I_{ij}=\Delta L(i)+\Delta L(j)-\Delta L(i,j).
\]

Strong positive or negative interaction often occurs for adjacent stroke fragments. Measure by distance and same-word membership where ground truth exists.

## 7. Candidate pruning before expensive evaluation

Use a high-recall union of:

- text detector proposals dilated by context margin
- high edge/stroke energy
- high base uncertainty
- OCR teacher attention during offline label generation

Report oracle coverage: fraction of truly useful candidates excluded by pruning on a small exhaustive set. A cheap pruner that misses critical candidates invalidates the upper bound.

## 8. Negative utility

Additional bits may worsen OCR due to texture/artifacts or teacher instability. Preserve negative labels. Deployment normally selects only positive expected utility, but uncertainty-aware exploration during training may sample them.

## 9. Target normalization

Store raw `Δloss`, raw `Δbits`, raw ratio and a training transform:

```text
u_train = sign(u_raw) * log1p(abs(u_raw) / scale)
```

`scale` is fitted on training data only and recorded in config.

## 10. Label storage schema

JSONL/Parquet row:

```json
{
  "image_id": "...",
  "image_sha256": "...",
  "base_checkpoint_sha256": "...",
  "teacher_ensemble_id": "...",
  "normalization_profile": "...",
  "budget_id": "...",
  "trajectory_seed": 0,
  "step": 4,
  "state_selected_ids": [1, 8, 15, 21],
  "candidate_id": 33,
  "delta_ocr_loss": 0.071,
  "delta_actual_bits": 96,
  "utility_raw": 0.0007395833,
  "cer_before": 0.25,
  "cer_after": 0.125,
  "valid": true
}
```

Large state lists may be stored as bitsets referenced by hash.

## 11. Feasibility benchmark before scaling

Run 100 representative crops with 32 candidates and 3 trajectory budgets. Record:

- decode/OCR time per candidate
- storage per label
- candidate utility distribution
- oracle advantage over mask/random
- interaction variance

Extrapolate total GPU hours. If cost is excessive, reduce candidate set or use teacher feature proxies, then revalidate ranking against exact labels.

## 12. G4 decision

Proceed only if the oracle significantly outperforms matched-budget controls on held-out images using training teachers for labels but held-out OCR for evaluation. This is the strongest early test of the central hypothesis.

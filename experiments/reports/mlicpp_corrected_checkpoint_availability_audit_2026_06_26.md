# MLIC++ Corrected Checkpoint Availability Audit

This audit checked whether MLIC++ corrected-architecture checkpoints beyond lambda `0.0250` are available from the pinned official repository.

## Result

- Local MLIC repository commit: `4aa9d2a512eb9db382172058e6c278fb0d376e17`
- Remote `origin/main` HEAD: `4aa9d2a512eb9db382172058e6c278fb0d376e17`
- Official README SHA256: `b32f5a08ba540a27a86d8cb4b73bdc96146cb011d743bab7f2e4eee0ee1e5325`
- Corrected-architecture checkpoint listed: lambda `0.0250` only
- Already resolved local checkpoint: `mlicpp_mse_0025_corrected`, SHA256 `67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559`

## Decision

No new checkpoint was downloaded. The README's multi-rate MSE links are explicitly under "Old Weights" and require `LatentResidualPredictionOld` and `SynthesisTransformOld`. Mixing those weights with the corrected lambda `0.0250` architecture would break the fixed baseline boundary.

## Impact

The full corrected-architecture MLIC++ RD curve remains unresolved. Continue using corrected lambda `0.0250` as the frozen base for OSCAR-LIC experiments. A multi-rate MLIC++ curve needs either author-provided corrected checkpoints or a separately declared old-weight baseline track with explicit comparability caveats.

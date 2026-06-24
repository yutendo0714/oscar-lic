"""Transparent linear-interpolation BD-rate helper for OCR curves.

For final papers, use the interpolation method frozen in the evaluation config.
This implementation is intentionally dependency-light and reports its common
interval; it never extrapolates.
"""

from __future__ import annotations

import math
from typing import Iterable
import numpy as np


class BDError(ValueError):
    pass


def _pareto(metric: Iterable[float], rate: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    points = sorted((float(m), float(r)) for m, r in zip(metric, rate))
    if len(points) < 2:
        raise BDError("at least two points are required")
    if any(r <= 0 or not math.isfinite(r) or not math.isfinite(m) for m, r in points):
        raise BDError("metric and rate must be finite; rate must be positive")
    # For duplicate metric values, retain lowest rate.
    dedup: dict[float, float] = {}
    for m, r in points:
        dedup[m] = min(r, dedup.get(m, r))
    ms = np.array(sorted(dedup), dtype=float)
    rs = np.array([dedup[m] for m in ms], dtype=float)
    if len(ms) < 2:
        raise BDError("metric values must span an interval")
    return ms, rs


def bd_rate_linear(
    anchor_rate: Iterable[float],
    anchor_metric: Iterable[float],
    candidate_rate: Iterable[float],
    candidate_metric: Iterable[float],
    samples: int = 1000,
) -> dict:
    """Compute average rate difference using linear interpolation of log-rate.

    Metric orientation does not change the integral; lower-is-better CER curves
    are valid as long as both curves overlap and use Pareto-filtered points.
    """
    am, ar = _pareto(anchor_metric, anchor_rate)
    cm, cr = _pareto(candidate_metric, candidate_rate)
    low = max(float(am.min()), float(cm.min()))
    high = min(float(am.max()), float(cm.max()))
    if not high > low:
        raise BDError("curves have no nonzero common metric interval")
    if samples < 10:
        raise BDError("samples must be >= 10")
    grid = np.linspace(low, high, samples)
    log_a = np.interp(grid, am, np.log(ar))
    log_c = np.interp(grid, cm, np.log(cr))
    mean_log_diff = float(np.trapezoid(log_c - log_a, grid) / (high - low))
    return {
        "bd_rate_percent": (math.exp(mean_log_diff) - 1.0) * 100.0,
        "interpolation": "linear_log_rate",
        "common_interval": [low, high],
        "samples": samples,
        "anchor_points": len(am),
        "candidate_points": len(cm),
        "extrapolation": False,
    }

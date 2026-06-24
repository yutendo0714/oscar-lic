"""Reference state-dependent greedy utility oracle for small feasibility studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Iterable


CandidateID = Hashable
LossFn = Callable[[frozenset[CandidateID]], float]
CostFn = Callable[[frozenset[CandidateID], CandidateID], int]


@dataclass(frozen=True)
class OracleStep:
    step: int
    selected_before: frozenset[CandidateID]
    candidate_id: CandidateID
    loss_before: float
    loss_after: float
    delta_loss: float
    delta_bytes: int
    utility_per_bit: float
    spent_bytes_after: int


@dataclass(frozen=True)
class OracleTrajectory:
    steps: tuple[OracleStep, ...]
    selected: frozenset[CandidateID]
    spent_bytes: int
    final_loss: float


def greedy_state_dependent_oracle(
    candidates: Iterable[CandidateID],
    budget_bytes: int,
    loss_fn: LossFn,
    incremental_cost_fn: CostFn,
    *,
    stop_on_nonpositive_utility: bool = True,
) -> OracleTrajectory:
    """Enumerate every remaining candidate at each state and select max utility.

    This exact greedy reference is intentionally expensive. It is suitable for
    G4 feasibility on small crops, and serves as a correctness target for
    shortlist/Shapley approximations. ``incremental_cost_fn`` must return actual
    serialized byte growth, including index/header consequences.
    """
    if budget_bytes < 0:
        raise ValueError("budget_bytes cannot be negative")
    remaining = set(candidates)
    selected: set[CandidateID] = set()
    spent = 0
    loss_before = float(loss_fn(frozenset()))
    steps: list[OracleStep] = []

    while remaining:
        evaluated: list[tuple[float, str, CandidateID, float, int, float]] = []
        state = frozenset(selected)
        for candidate in remaining:
            delta_bytes = int(incremental_cost_fn(state, candidate))
            if delta_bytes <= 0:
                raise ValueError(f"candidate {candidate!r} has nonpositive incremental cost")
            if spent + delta_bytes > budget_bytes:
                continue
            loss_after = float(loss_fn(frozenset((*selected, candidate))))
            delta_loss = loss_before - loss_after
            utility = delta_loss / (8.0 * delta_bytes)
            evaluated.append((utility, str(candidate), candidate, loss_after, delta_bytes, delta_loss))
        if not evaluated:
            break
        # Highest utility, then stable lexical ID for deterministic ties.
        utility, _, candidate, loss_after, delta_bytes, delta_loss = max(evaluated, key=lambda x: (x[0], x[1]))
        if stop_on_nonpositive_utility and utility <= 0:
            break
        before = frozenset(selected)
        selected.add(candidate)
        remaining.remove(candidate)
        spent += delta_bytes
        steps.append(
            OracleStep(
                step=len(steps), selected_before=before, candidate_id=candidate,
                loss_before=loss_before, loss_after=loss_after,
                delta_loss=delta_loss, delta_bytes=delta_bytes,
                utility_per_bit=utility, spent_bytes_after=spent,
            )
        )
        loss_before = loss_after

    return OracleTrajectory(tuple(steps), frozenset(selected), spent, loss_before)

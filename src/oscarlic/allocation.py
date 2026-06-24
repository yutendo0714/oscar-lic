"""Budgeted candidate allocation controls used by OSCAR-LIC experiments."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    cost_bytes: int
    score: float
    text_mask: bool = False
    detector_confidence: float = 0.0
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_bytes <= 0:
            raise ValueError("candidate cost_bytes must be positive")


@dataclass(frozen=True)
class Allocation:
    selected: tuple[Candidate, ...]
    spent_bytes: int
    budget_bytes: int

    @property
    def remaining_bytes(self) -> int:
        return self.budget_bytes - self.spent_bytes


def allocate_ranked(candidates: Iterable[Candidate], budget_bytes: int, key) -> Allocation:
    if budget_bytes < 0:
        raise ValueError("budget_bytes cannot be negative")
    ordered = sorted(candidates, key=key, reverse=True)
    selected: list[Candidate] = []
    spent = 0
    for candidate in ordered:
        if spent + candidate.cost_bytes <= budget_bytes:
            selected.append(candidate)
            spent += candidate.cost_bytes
    return Allocation(tuple(selected), spent, budget_bytes)


def utility_allocation(candidates: Iterable[Candidate], budget_bytes: int) -> Allocation:
    return allocate_ranked(candidates, budget_bytes, key=lambda c: (c.score, -c.cost_bytes, c.candidate_id))


def detector_confidence_allocation(candidates: Iterable[Candidate], budget_bytes: int) -> Allocation:
    return allocate_ranked(candidates, budget_bytes, key=lambda c: (c.detector_confidence, -c.cost_bytes, c.candidate_id))


def uncertainty_allocation(candidates: Iterable[Candidate], budget_bytes: int) -> Allocation:
    return allocate_ranked(candidates, budget_bytes, key=lambda c: (c.uncertainty, -c.cost_bytes, c.candidate_id))


def uniform_text_allocation(candidates: Iterable[Candidate], budget_bytes: int) -> Allocation:
    # Stable text-first order. Candidate IDs must already encode the frozen raster/channel order.
    ordered = sorted(candidates, key=lambda c: (not c.text_mask, c.candidate_id))
    selected: list[Candidate] = []
    spent = 0
    for candidate in ordered:
        if spent + candidate.cost_bytes <= budget_bytes:
            selected.append(candidate)
            spent += candidate.cost_bytes
    return Allocation(tuple(selected), spent, budget_bytes)


def random_allocation(candidates: Iterable[Candidate], budget_bytes: int, seed: int) -> Allocation:
    items = list(candidates)
    random.Random(seed).shuffle(items)
    selected: list[Candidate] = []
    spent = 0
    for candidate in items:
        if spent + candidate.cost_bytes <= budget_bytes:
            selected.append(candidate)
            spent += candidate.cost_bytes
    return Allocation(tuple(selected), spent, budget_bytes)

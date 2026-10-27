"""Discrete golden-section search matching the pinned reference conventions."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class BandwidthSearch:
    """Selected neighbor count, objective, winner history, and evaluated scores."""

    bandwidth: int
    score: float
    history: list[tuple[int, float]]
    evaluations: dict[int, float]


def golden_section(
    objective: Callable[[int], float],
    lower: int,
    upper: int,
    tolerance: float = 1.0e-6,
    max_iterations: int = 200,
) -> BandwidthSearch:
    """Minimize an objective over a finite, discrete neighbor-count interval.

    Bounds are canonical [2, n] when the notebook's bw_min=2 overrides mgwr's
    default lower bound 40+2*p. Interior points use delta=0.38197 and NumPy's
    ties-to-even rounding. Ties favor the lower interior point. Stop when the
    two objective values differ by at most tolerance, or at max_iterations.
    This reproduces the reference search; it does not prove a global minimum.
    """
    if not 2 <= lower <= upper or tolerance < 0 or max_iterations < 1:
        raise ValueError("Invalid search bounds, tolerance, or iteration limit.")
    contraction = 0.38197
    left_bound = float(lower)
    right_bound = float(upper)
    left_candidate = left_bound + contraction * abs(right_bound - left_bound)
    right_candidate = right_bound - contraction * abs(right_bound - left_bound)
    evaluations: dict[int, float] = {}
    history: list[tuple[int, float]] = []
    for _ in range(max_iterations):
        left_candidate = int(np.round(left_candidate))
        right_candidate = int(np.round(right_candidate))
        for candidate in (left_candidate, right_candidate):
            if candidate not in evaluations:
                score = float(objective(candidate))
                if not np.isfinite(score):
                    raise ValueError(f"Non-finite objective at bandwidth {candidate}.")
                evaluations[candidate] = score
        left_score = evaluations[left_candidate]
        right_score = evaluations[right_candidate]
        if left_score <= right_score:
            selected_bandwidth, selected_score = left_candidate, left_score
            right_bound = right_candidate
            right_candidate = left_candidate
            left_candidate = left_bound + contraction * abs(right_bound - left_bound)
        else:
            selected_bandwidth, selected_score = right_candidate, right_score
            left_bound = left_candidate
            left_candidate = right_candidate
            right_candidate = right_bound - contraction * abs(right_bound - left_bound)
        history.append((selected_bandwidth, selected_score))
        if abs(left_score - right_score) <= tolerance:
            break
    return BandwidthSearch(selected_bandwidth, selected_score, history, evaluations)

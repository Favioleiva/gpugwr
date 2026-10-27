"""Bandwidth selection using the validated discrete search or bounded scalar search."""

import warnings

import numpy as np
from scipy.linalg import LinAlgWarning
from scipy.optimize import minimize_scalar

from .bandwidth import golden_section
from .distances import pairwise_distances
from .model import fit_gaussian
from .validation import ModelData


def select_bandwidth(
    data: ModelData,
    kernel: str,
    fixed: bool,
    intercept: bool,
    criterion: str,
    bounds: tuple[float, float] | None,
    *,
    fitter=None,
    fixed_bounds=None,
) -> tuple[float, dict]:
    """Minimize AICc or analytic leave-one-out CV and retain search evidence.

    Invalid candidate fits receive a finite penalty so the original discrete
    algorithm can traverse them. Every rejection is recorded. A selected
    invalid fit is an error. These are local searches, not global guarantees.
    """
    if fitter is None:
        fitter = fit_gaussian
    observation_count, parameter_count = data.design.shape
    if bounds is None:
        if fixed and fixed_bounds is not None:
            bounds = fixed_bounds()
        elif fixed:
            distances = pairwise_distances(data.coordinates)
            neighbor_index = min(parameter_count + 2, observation_count - 1)
            lower = np.partition(distances, neighbor_index, axis=1)[:, neighbor_index]
            bounds = (float(lower.max() * 1.0000001), float(distances.max() * 2))
        else:
            bounds = (2, observation_count)
    lower, upper = bounds
    if not np.isfinite([lower, upper]).all() or lower <= 0 or lower > upper:
        raise ValueError("bandwidth_bounds must be finite, positive and ordered.")
    if not fixed and (
        int(lower) != lower
        or int(upper) != upper
        or lower < 2
        or upper > observation_count
    ):
        raise ValueError("Adaptive search bounds must be integers from 2 through n.")
    rejected = {}
    evaluations = {}
    penalty = 1.0e100

    def objective(bandwidth: float) -> float:
        """Evaluate one bandwidth with the same solves used for final fitting."""
        if bandwidth in evaluations:
            return evaluations[bandwidth]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                fitted = fitter(
                    data.design,
                    data.response,
                    data.coordinates,
                    bandwidth,
                    full_diagnostics=False,
                    kernel=kernel,
                    fixed=fixed,
                    intercept=intercept,
                )
            if observation_count - fitted["tr_S"] - 2 <= 0:
                raise ValueError("Insufficient effective degrees of freedom for AICc.")
            if criterion == "AICc":
                score = fitted["aicc"]
            else:
                residuals = fitted["resid_response"]
                denominator = 1.0 - fitted["influ"].ravel()
                if np.any(denominator <= np.finfo(float).eps):
                    raise ValueError("CV requires influence strictly below one.")
                score = float(np.mean((residuals / denominator) ** 2))
            if not np.isfinite(score):
                raise ValueError("Non-finite optimization criterion.")
        except (ValueError, np.linalg.LinAlgError, LinAlgWarning) as error:
            rejected[str(bandwidth)] = str(error)
            score = penalty
        evaluations[bandwidth] = float(score)
        return float(score)

    if fixed:
        if lower == upper:
            selected = lower
        else:
            optimized = minimize_scalar(
                objective,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": 1.0e-6, "maxiter": 200},
            )
            if not optimized.success:
                raise ValueError(
                    f"Fixed bandwidth search did not converge: {optimized.message}"
                )
            selected = min([lower, float(optimized.x), upper], key=objective)
        history = [(selected, objective(selected))]
    else:
        search = golden_section(objective, int(lower), int(upper))
        selected = search.bandwidth
        history = search.history
    if objective(selected) >= penalty:
        raise ValueError("Search found no valid fit; widen bandwidth_bounds.")
    evidence = {
        "search_history": np.asarray(history),
        "search_evaluations": np.asarray(list(evaluations.items())),
    }
    if criterion == "AICc":
        # Preserve the original winner-history convention, including duplicates.
        ordered = evidence["search_history"]
        ordered = ordered[np.argsort(ordered[:, 1])]
        weights = np.exp(-0.5 * (ordered[:, 1] - ordered[:, 1].min()))
        weights /= weights.sum()
        weights /= weights.sum()
        count = int(np.sum(np.cumsum(weights) < 0.95)) + 1
        included = ordered[:count, 0]
        evidence["bandwidth_interval"] = np.array([included.min(), included.max()])
    metadata = {
        "bounds": list(bounds),
        "criterion": criterion,
        "rejected_candidates": rejected,
        "arrays": evidence,
    }
    return selected, metadata

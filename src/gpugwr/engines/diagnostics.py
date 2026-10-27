"""Batched canonical local collinearity diagnostics evaluated on CUDA."""

from itertools import combinations

import cupy as cp
import cupyx


def local_collinearity(design, weights, intercept: bool) -> dict:
    """Return correlation, VIF, condition and VDP arrays for a weight block.

    Weighted centering cancels the common np.cov normalization when forming
    correlations. SVD uses normalized weights (not their square roots), exactly
    as in the reference convention. Economy SVD omits unused left null vectors;
    it retains every singular value and right vector used by these diagnostics.
    """
    normalized = weights / cp.sum(weights, axis=1)[:, None]
    predictors = design[:, int(intercept) :]
    means = (normalized @ predictors) / cp.sum(normalized, axis=1)[:, None]
    centered = predictors[None, :, :] - means[:, None, :]
    if not intercept:
        # A locally constant predictor is valid in an uncentered model, but
        # its centered variance is exactly zero. Enforce that identity so
        # reduction roundoff cannot turn undefined VIF into a spurious one.
        supported = weights[:, :, None] > 0
        local_minimum = cp.min(
            cp.where(supported, predictors[None, :, :], cp.inf), axis=1
        )
        local_maximum = cp.max(
            cp.where(supported, predictors[None, :, :], -cp.inf), axis=1
        )
        constant = local_minimum == local_maximum
        centered = cp.where(constant[:, None, :], 0.0, centered)
    covariance = centered.transpose(0, 2, 1) @ (centered * normalized[:, :, None])
    variances = cp.diagonal(covariance, axis1=1, axis2=2)
    scales = cp.sqrt(variances)
    correlation = covariance / (scales[:, :, None] * scales[:, None, :])
    pairs = list(combinations(range(predictors.shape[1]), 2))
    correlations = cp.empty((len(weights), len(pairs)), dtype=cp.float64)
    for position, (left, right) in enumerate(pairs):
        correlations[:, position] = correlation[:, left, right]
    variance_inflation = cp.full(variances.shape, cp.nan)
    with cupyx.errstate(linalg="raise"):
        singular_covariance = cp.linalg.svd(covariance, compute_uv=False)
        threshold = (
            predictors.shape[1] * cp.finfo(cp.float64).eps * singular_covariance[:, 0]
        )
        valid = cp.all(variances > 0, axis=1) & (singular_covariance[:, -1] > threshold)
        if bool(cp.any(valid)):
            identity = cp.broadcast_to(
                cp.eye(predictors.shape[1]), correlation[valid].shape
            )
            inverse = cp.linalg.solve(correlation[valid], identity)
            variance_inflation[valid] = cp.diagonal(inverse, axis1=1, axis2=2)
        weighted = design[None, :, :] * normalized[:, :, None]
        column_norms = cp.sqrt(cp.sum(weighted**2, axis=1))
        scaled = weighted / column_norms[:, None, :]
        _, singular_values, right_vectors = cp.linalg.svd(scaled, full_matrices=False)
    components = (right_vectors.transpose(0, 2, 1) / singular_values[:, None, :]) ** 2
    proportions = components[:, :, -1] / cp.sum(components, axis=2)
    return {
        "local_correlations": correlations,
        "local_vif": variance_inflation,
        "local_condition": (singular_values[:, 0] / singular_values[:, -1])[:, None],
        "local_vdp": proportions,
    }

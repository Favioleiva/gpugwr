"""Canonical residual variance and local coefficient covariance."""

import numpy as np

from .distances import FloatArray


def residual_variance(residuals: FloatArray, trace: float) -> float:
    """Return sum(e_i^2)/(n-tr(S)) for the canonical sigma2_v1=True setting."""
    denominator = len(residuals) - trace
    if denominator <= 0:
        raise ValueError("Residual variance requires n > tr(S).")
    return float(np.sum(residuals**2) / denominator)


def unscaled_covariance(coefficient_operator: FloatArray) -> FloatArray:
    """Return A_i A_i' (p, p) for a local coefficient operator A_i (p, n).

    These are sampling covariances before multiplication by residual variance.
    The weights appear squared through A_i A_i'; (X' W_i X)^-1 alone would
    not reproduce the GWR coefficient standard errors.
    """
    return coefficient_operator @ coefficient_operator.T

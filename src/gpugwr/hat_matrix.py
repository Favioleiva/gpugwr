"""Smoothing rows and effective degrees of freedom for Gaussian GWR."""

import numpy as np

from .distances import FloatArray


def hat_row(design_row: FloatArray, coefficient_operator: FloatArray) -> FloatArray:
    """Return S_i = X_i A_i, given shapes (p,) and (p, n)."""
    return design_row @ coefficient_operator


def hat_diagnostics(smoother: FloatArray) -> dict[str, float | FloatArray]:
    """Return diagonal influence, tr(S), tr(S'S), and reference degrees of freedom.

    The input is the full n by n smoothing matrix. The canonical sigma2_v1
    convention sets ENP=tr(S); mgwr's df_model is n-tr(S), despite its name.
    """
    observation_count = len(smoother)
    influence = np.diag(smoother).reshape(-1, 1).copy()
    trace = float(np.sum(influence))
    squared_trace = float(np.sum(smoother**2))
    return {
        "influ": influence,
        "tr_S": trace,
        "tr_STS": squared_trace,
        "ENP": trace,
        "df_model": observation_count - trace,
        "df_resid": observation_count - 2.0 * trace + squared_trace,
    }

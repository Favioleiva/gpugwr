"""Reference t-statistics and multiple-comparison correction conventions."""

import numpy as np
from scipy.stats import t

from .distances import FloatArray


def coefficient_inference(
    coefficients: FloatArray,
    covariance_diagonal: FloatArray,
    effective_parameters: float,
) -> dict[str, FloatArray | float]:
    """Return standard errors, t-values, adjusted alphas, and filtered t-values.

    Inputs are n by p coefficient and scaled variance arrays, and ENP=tr(S).
    The reference uses [0.1, 0.05, 0.001]*p/ENP (including its third-alpha
    docstring discrepancy) and Student t quantiles with n-1 degrees of freedom.
    Corrected t-values preserve significant t-values and zero the others.
    """
    observation_count, parameter_count = coefficients.shape
    standard_errors = np.sqrt(covariance_diagonal)
    t_statistics = coefficients / standard_errors
    adjusted_alpha = np.array([0.1, 0.05, 0.001])
    adjusted_alpha = adjusted_alpha * parameter_count / effective_parameters
    critical_values = t.ppf(1.0 - adjusted_alpha / 2.0, observation_count - 1)
    critical_value = float(critical_values[1])
    corrected_t = t_statistics.copy()
    corrected_t[np.abs(corrected_t) < critical_value] = 0.0
    return {
        "bse": standard_errors,
        "tvalues": t_statistics,
        "adj_alpha": adjusted_alpha,
        "critical_t": critical_value,
        "critical_t_all": critical_values,
        "filtered_tvalues": corrected_t,
    }

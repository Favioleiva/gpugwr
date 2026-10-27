"""Gaussian information criteria, local fit, and collinearity diagnostics."""

from itertools import combinations

import numpy as np

from .distances import FloatArray


def information_criteria(
    response: FloatArray, residuals: FloatArray, trace: float
) -> dict[str, float]:
    """Return Gaussian profile likelihood and global fit diagnostics.

    Inputs y and e have shape (n, 1). Likelihood uses RSS/n, whereas reported
    sigma2 uses RSS/(n-tr(S)). The information-criterion penalty counts the
    variance parameter in addition to tr(S). See the mathematical report.
    """
    observation_count = len(response)
    residual_sum_squares = float(np.sum(residuals**2))
    total_sum_squares = float(np.sum((response - response.mean()) ** 2))
    half_count = observation_count / 2.0
    log_likelihood = -np.log(residual_sum_squares) * half_count
    log_likelihood -= (1.0 + np.log(np.pi / half_count)) * half_count
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (observation_count - 1.0) / (
        observation_count - trace - 1.0
    )
    return {
        "resid_ss": residual_sum_squares,
        "null_deviance": total_sum_squares,
        "global_deviance": residual_sum_squares,
        "llf": float(log_likelihood),
        "aic": float(-2.0 * log_likelihood + 2.0 * (trace + 1.0)),
        "aicc": float(
            -2.0 * log_likelihood
            + 2.0
            * observation_count
            * (trace + 1.0)
            / (observation_count - trace - 2.0)
        ),
        "bic": float(-2.0 * log_likelihood + (trace + 1.0) * np.log(observation_count)),
        "R2": r_squared,
        "adj_R2": adjusted_r_squared,
        "D2": r_squared,
        "adj_D2": adjusted_r_squared,
    }


def local_goodness_of_fit(
    response: FloatArray, residuals: FloatArray, weights: FloatArray
) -> dict[str, FloatArray]:
    """Return weighted local means, TSS, RSS, and local R2 as (n, 1) arrays.

    Local RSS uses each observation's own fitted residual, e_j=y_j-X_j beta_j,
    rather than residuals from fitting every observation with beta_i.
    """
    observation_count = len(response)
    local_mean = np.empty((observation_count, 1), dtype=np.float64)
    local_total = np.empty_like(local_mean)
    local_residual = np.empty_like(local_mean)
    for location, weight_row in enumerate(weights):
        column_weights = weight_row[:, None]
        local_mean[location] = np.sum(column_weights * response) / weight_row.sum()
        local_total[location] = np.sum(
            column_weights * (response - local_mean[location]) ** 2
        )
        local_residual[location] = np.sum(column_weights * residuals**2)
    return {
        "y_bar": local_mean,
        "TSS": local_total,
        "RSS": local_residual,
        "localR2": (local_total - local_residual) / local_total,
    }


def covariance_to_correlation(covariance: FloatArray) -> FloatArray:
    """Return a correlation matrix by scaling a square covariance matrix."""
    inverse_std = np.diag(1.0 / np.sqrt(np.diag(covariance)))
    return inverse_std @ covariance @ inverse_std


def local_collinearity(
    design: FloatArray, weights: FloatArray, intercept: bool = True
) -> dict[str, FloatArray]:
    """Return correlations, VIFs, condition numbers, and variance proportions.

    X is (n, p), including an intercept, and W is (n, n). Correlations and
    VIFs omit the intercept. The SVD design uses normalized weights themselves,
    NOT their square roots: this unusual reference convention is intentional.
    Variance proportions refer to the smallest singular-value component.
    """
    observation_count, parameter_count = design.shape
    predictor_start = int(intercept)
    predictor_count = parameter_count - predictor_start
    predictor_pairs = list(combinations(range(predictor_start, parameter_count), 2))
    correlations = np.empty((observation_count, len(predictor_pairs)))
    variance_inflation = np.empty((observation_count, predictor_count))
    condition_numbers = np.empty((observation_count, 1))
    variance_proportions = np.empty((observation_count, parameter_count))
    for location, weight_row in enumerate(weights):
        normalized_weights = weight_row / weight_row.sum()
        for pair_index, (left, right) in enumerate(predictor_pairs):
            covariance = np.cov(
                design[:, left], design[:, right], aweights=normalized_weights
            )
            if np.all(np.diag(covariance) > 0):
                correlations[location, pair_index] = covariance_to_correlation(
                    covariance
                )[0, 1]
            else:
                correlations[location, pair_index] = np.nan
        covariance = np.atleast_2d(
            np.cov(design[:, predictor_start:].T, aweights=normalized_weights)
        )
        # With no intercept, a full-rank design can contain constant columns
        # or become dependent after centering. Correlation/VIF then do not apply.
        if (
            np.all(np.diag(covariance) > 0)
            and np.linalg.matrix_rank(covariance) == predictor_count
        ):
            correlation = covariance_to_correlation(covariance)
            variance_inflation[location] = np.diag(
                np.linalg.solve(correlation, np.eye(predictor_count))
            )
        else:
            variance_inflation[location] = np.nan
        weighted_design = design * normalized_weights[:, None]
        column_norms = np.sqrt(np.sum(weighted_design**2, axis=0))
        scaled_design = weighted_design / column_norms
        # Only singular values and right vectors enter these diagnostics.
        # Economy SVD omits unused n-by-n left null vectors, preserving the
        # mathematical outputs while avoiding quadratic temporary storage.
        _, singular_values, right_vectors = np.linalg.svd(
            scaled_design, full_matrices=False
        )
        condition_numbers[location] = singular_values[0] / singular_values[-1]
        variance_components = (right_vectors.T @ np.diag(1.0 / singular_values)) ** 2
        proportions = variance_components.T / variance_components.sum(axis=1)
        variance_proportions[location] = proportions[-1]
    return {
        "local_correlations": correlations,
        "local_vif": variance_inflation,
        "local_condition": condition_numbers,
        "local_vdp": variance_proportions,
    }

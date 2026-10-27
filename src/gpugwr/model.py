"""Independent CPU experiment for canonical Gaussian adaptive-bisquare GWR."""

import numpy as np

from .diagnostics import information_criteria, local_collinearity, local_goodness_of_fit
from .distances import FloatArray, pairwise_distances
from .hat_matrix import hat_diagnostics, hat_row
from .inference import coefficient_inference
from .kernels import spatial_weights
from .variance import residual_variance, unscaled_covariance
from .weighted_regression import solve_local_regression


def validate_model_inputs(
    design: FloatArray,
    response: FloatArray,
    coordinates: FloatArray,
    intercept: bool = True,
) -> None:
    """Validate finite float64-ready arrays and an explicit leading intercept."""
    if design.ndim != 2 or response.shape != (len(design), 1):
        raise ValueError("Expected design (n, p) and response (n, 1).")
    if coordinates.shape != (len(design), 2):
        raise ValueError("Expected coordinates (n, 2) matching the design.")
    if intercept and (design.shape[1] < 2 or not np.all(design[:, 0] == 1.0)):
        raise ValueError("The design must include a leading intercept and predictors.")
    if design.shape[1] < 1 or len(design) <= design.shape[1] + 2:
        raise ValueError("Insufficient observations: require n > parameter_count + 2.")
    for values in (design, response, coordinates):
        if not np.all(np.isfinite(values)):
            raise ValueError("Model inputs must be finite.")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The design must have full column rank.")


def fit_gaussian(
    design: FloatArray,
    response: FloatArray,
    coordinates: FloatArray,
    bandwidth: float,
    full_diagnostics: bool = True,
    *,
    kernel: str = "bisquare",
    fixed: bool = False,
    intercept: bool = True,
    spatial_cache=None,
) -> dict[str, FloatArray | float]:
    """Fit canonical GWR and return named numerical arrays and scalar diagnostics.

    Parameters are standardized X (n, p) including an intercept, standardized
    y (n, 1), projected coordinates (n, 2), and the neighbor count. Setting
    full_diagnostics=False skips covariance, inference, and collinearity for
    bandwidth search, but keeps the identical local solves and AICc formula.
    """
    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    validate_model_inputs(design, response, coordinates, intercept)
    if spatial_cache is None:
        distances = pairwise_distances(coordinates)
        weights, radii = spatial_weights(distances, bandwidth, kernel, fixed)
    else:
        spatial_cache.validate(coordinates, bandwidth, kernel, fixed, "cpu")
        distances, weights, radii = spatial_cache.rows(0, len(design))
    observation_count, parameter_count = design.shape
    coefficients = np.empty_like(design)
    predictions = np.empty_like(response)
    smoother = np.empty((observation_count, observation_count))
    covariance = np.empty((observation_count, parameter_count, parameter_count))
    for location in range(observation_count):
        weighted_design = design * np.sqrt(weights[location])[:, None]
        if np.linalg.matrix_rank(weighted_design) < parameter_count:
            raise ValueError(f"Singular local regression at observation {location}.")
        coefficients[location], operator = solve_local_regression(
            design, response, weights[location]
        )
        predictions[location] = design[location] @ coefficients[location]
        smoother[location] = hat_row(design[location], operator)
        if full_diagnostics:
            covariance[location] = unscaled_covariance(operator)
    residuals = response - predictions
    result = {
        "X": design,
        "y": response,
        "coordinates": coordinates,
        "distances": distances,
        "W": weights,
        "radii": radii,
        "neighbor_mask": (weights > 0.0).astype(np.int64),
        "neighbor_indices": (
            np.argsort(distances, axis=1, kind="stable")
            if spatial_cache is None
            else spatial_cache.neighbor_indices
        ),
        "S": smoother,
        "params": coefficients,
        "predy": predictions,
        "mu": predictions.ravel(),
        "resid_response": residuals.ravel(),
        "bandwidth": float(bandwidth),
        "n": float(observation_count),
        "k": float(parameter_count),
    }
    result.update(hat_diagnostics(smoother))
    result.update(information_criteria(response, residuals, result["tr_S"]))
    if not full_diagnostics:
        return result
    sigma_squared = residual_variance(residuals, result["tr_S"])
    covariance *= sigma_squared
    variance_diagonal = np.diagonal(covariance, axis1=1, axis2=2).copy()
    standardized_residuals = residuals / np.sqrt(
        sigma_squared * (1.0 - result["influ"])
    )
    result.update(
        {
            "sigma2": sigma_squared,
            "scale": sigma_squared,
            "CCT": variance_diagonal,
            "covariance_full": covariance,
            "std_res": standardized_residuals,
            "cooksD": standardized_residuals**2
            * result["influ"]
            / (result["tr_S"] * (1.0 - result["influ"])),
            "CV": float(np.mean((residuals / (1.0 - result["influ"])) ** 2)),
            "w": np.ones((observation_count, 1)),
            "null": np.full(observation_count, response.mean()),
        }
    )
    result.update(coefficient_inference(coefficients, variance_diagonal, result["ENP"]))
    result.update(local_goodness_of_fit(response, residuals, weights))
    result.update(local_collinearity(design, weights, intercept))
    return result

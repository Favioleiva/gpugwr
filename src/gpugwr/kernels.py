"""Adaptive bisquare weights with the canonical self-neighbor convention."""

import numpy as np

from .distances import FloatArray


def adaptive_bisquare(
    distances: FloatArray, bandwidth: int
) -> tuple[FloatArray, FloatArray]:
    """Return weights (n, n) and radii (n,) for an integer neighbor count.

    The bandwidth includes the focal observation. The bandwidth-th smallest
    distance is multiplied by 1.0000001. Within this radius h_i, the weight is
    [1 - (d_ij/h_i)^2]^2; at or outside the radius it is zero. Distance ties
    may therefore give more than `bandwidth` positive weights.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("Distances must be a square matrix.")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("Distances must be finite and nonnegative.")
    if int(bandwidth) != bandwidth or not 2 <= bandwidth <= len(distances):
        raise ValueError("Bandwidth must be an integer from 2 through n.")
    neighbor_index = int(bandwidth) - 1
    radii = np.partition(distances, neighbor_index, axis=1)[:, neighbor_index]
    radii = radii * 1.0000001
    if np.any(radii <= 0):
        raise ValueError("Coincident coordinates produce a zero kernel radius.")
    relative_distances = distances / radii[:, None]
    weights = (1.0 - relative_distances**2) ** 2
    weights[distances >= radii[:, None]] = 0.0
    return weights, radii


def spatial_weights(
    distances: FloatArray, bandwidth: float, kernel: str, fixed: bool
) -> tuple[FloatArray, FloatArray]:
    """Return weights and radii for three kernels and either bandwidth mode.

    Adaptive radii include self and use the original epsilon convention.
    Fixed bandwidths are positive radii in the coordinate system's units.
    Gaussian weights are exp(-d²/(2h²)); exponential weights are exp(-d/h).
    """
    if kernel not in {"bisquare", "gaussian", "exponential"}:
        raise ValueError("kernel must be bisquare, gaussian, or exponential.")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Bandwidth must be finite and positive.")
    if not fixed and kernel == "bisquare":
        return adaptive_bisquare(distances, bandwidth)
    if fixed:
        radii = np.full(len(distances), float(bandwidth))
    else:
        if int(bandwidth) != bandwidth or not 2 <= bandwidth <= len(distances):
            raise ValueError("Adaptive bandwidth must be an integer from 2 through n.")
        neighbor_index = int(bandwidth) - 1
        radii = np.partition(distances, neighbor_index, axis=1)[:, neighbor_index]
        radii = radii * 1.0000001
        if np.any(radii <= 0):
            raise ValueError("Coincident coordinates produce a zero kernel radius.")
    relative_distances = distances / radii[:, None]
    if kernel == "gaussian":
        weights = np.exp(-0.5 * relative_distances**2)
    elif kernel == "exponential":
        weights = np.exp(-relative_distances)
    else:
        weights = (1.0 - relative_distances**2) ** 2
        weights[distances >= radii[:, None]] = 0.0
    return weights, radii

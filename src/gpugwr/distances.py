"""Euclidean distances for projected coordinates; no geographic reprojection."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def pairwise_distances(coordinates: FloatArray) -> FloatArray:
    """Return an n by n float64 matrix of Euclidean distances.

    Parameters
    ----------
    coordinates : array of shape (n, 2)
        Finite projected coordinates in the original dataset's units.

    Returns
    -------
    distances : array of shape (n, n)
        Entry (i, j) is sqrt((x_i-x_j)^2 + (y_i-y_j)^2).
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape (n, 2).")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Coordinates must be finite.")
    displacements = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(displacements**2, axis=2))

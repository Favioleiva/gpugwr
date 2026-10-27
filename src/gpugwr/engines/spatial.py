"""Chunked Euclidean distances and canonical spatial kernels on CUDA."""

import cupy as cp


def distance_rows(coordinates, start: int, stop: int):
    """Return float64 distances for focal rows [start, stop), including self.

    Direct coordinate subtraction avoids cancellation in the squared-norm
    identity, matching the CPU convention for large coordinate offsets.
    """
    difference = coordinates[start:stop, None, :] - coordinates[None, :, :]
    return cp.sqrt(cp.sum(difference * difference, axis=2))


def weight_rows(distances, bandwidth: float, kernel: str, fixed: bool):
    """Return weights and radii for a rectangular distance block on the device.

    Adaptive neighbor counts include self; ties and the 1.0000001 radius factor
    follow the CPU implementation. Fixed radii use the original coordinate units.
    """
    if fixed:
        radii = cp.full(distances.shape[0], bandwidth, dtype=cp.float64)
    else:
        position = int(bandwidth) - 1
        radii = cp.partition(distances, position, axis=1)[:, position] * 1.0000001
        if bool(cp.any(radii <= 0)):
            raise ValueError("Coincident coordinates produce a zero kernel radius.")
    relative = distances / radii[:, None]
    if kernel == "gaussian":
        weights = cp.exp(-0.5 * relative**2)
    elif kernel == "exponential":
        weights = cp.exp(-relative)
    else:
        weights = (1.0 - relative**2) ** 2
        weights[distances >= radii[:, None]] = 0.0
    return weights, radii

"""Batched float64 weighted least squares, with no ridge or pseudoinverse."""

import cupy as cp
import cupyx


def local_operators(design, weights, start: int):
    """Solve (X' W_i X) L_i = X' W_i for a batch of coefficient operators.

    A tall-design SVD uses the same rank threshold as NumPy matrix_rank:
    max(n,p) * eps * largest singular value. The normal equations then use
    batched pivoted LU through CuPy solve. This preserves the CPU mathematics
    without materializing inverses. Different reduction orders can change ulps.
    """
    weighted_design = design[None, :, :] * cp.sqrt(weights)[:, :, None]
    with cupyx.errstate(linalg="raise"):
        singular_values = cp.linalg.svd(weighted_design, compute_uv=False)
        threshold = max(design.shape) * cp.finfo(cp.float64).eps * singular_values[:, 0]
        deficient = singular_values[:, -1] <= threshold
        if bool(cp.any(deficient)):
            location = start + int(cp.flatnonzero(deficient)[0])
            raise ValueError(f"Singular local regression at observation {location}.")
        weighted_transpose = design.T[None, :, :] * weights[:, None, :]
        normal = weighted_transpose @ design
        normal_values = cp.linalg.svd(normal, compute_uv=False)
        if bool(
            cp.any(
                normal_values[:, -1] <= cp.finfo(cp.float64).eps * normal_values[:, 0]
            )
        ):
            raise ValueError("Singular or ill-conditioned local regression.")
        return cp.linalg.solve(normal, weighted_transpose)

"""Auditable local weighted least squares using linear solves."""

from scipy.linalg import solve

from .distances import FloatArray


def solve_local_regression(
    design: FloatArray, response: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return beta (p,) and the local linear operator A (p, n).

    For design X (n, p), response y (n, 1), and nonnegative weights w (n,),
    solve (X' W X) A = X' W, then beta = A y. This avoids an explicit inverse
    and retains A for hat rows and covariance calculations. Singular local
    systems raise a linear algebra error; no ridge or pseudoinverse is used.
    """
    weighted_transpose = (design * weights[:, None]).T
    normal_matrix = weighted_transpose @ design
    coefficient_operator = solve(normal_matrix, weighted_transpose)
    coefficients = coefficient_operator @ response
    return coefficients.ravel(), coefficient_operator

"""Strict numerical comparison with explicit non-finite and shape failures."""

import numpy as np


def compare_arrays(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-10,
    exact: bool = False,
) -> dict:
    """Compare arrays and return discrepancies, failing counts, shapes, and status.

    Relative discrepancies are |candidate-reference|/|reference| where the
    reference is nonzero. At reference zero they are zero for exact equality
    and unbounded otherwise (reported separately). Acceptance always uses
    |difference| <= atol + rtol*|reference|, or exact equality for discrete data.
    Non-finite values fail even when both sides contain the same NaN or infinity.
    """
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    report = {
        "name": name,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_compatible": reference.shape == candidate.shape,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "exact": exact,
        "rtol": 0.0 if exact else relative_tolerance,
        "atol": 0.0 if exact else absolute_tolerance,
        "max_absolute_difference": None,
        "max_relative_difference": None,
        "nonzero_difference_at_zero_reference": 0,
        "failing_elements": max(reference.size, candidate.size),
        "status": "FAIL",
    }
    if reference.shape != candidate.shape:
        return report
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not finite.all():
        report["nonfinite_elements"] = int(np.sum(~finite))
        return report
    differences = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    zero_reference = reference == 0
    unbounded_relative = int(np.sum(zero_reference & (differences != 0)))
    relative_differences = np.zeros_like(differences)
    np.divide(
        differences, np.abs(reference), out=relative_differences, where=~zero_reference
    )
    if exact:
        passing = reference == candidate
    else:
        passing = differences <= absolute_tolerance + relative_tolerance * np.abs(
            reference
        )
    report.update(
        {
            "max_absolute_difference": float(np.max(differences, initial=0)),
            "max_relative_difference": (
                "unbounded"
                if unbounded_relative
                else float(np.max(relative_differences, initial=0))
            ),
            "nonzero_difference_at_zero_reference": unbounded_relative,
            "failing_elements": int(np.sum(~passing)),
            "status": "PASS" if np.all(passing) else "FAIL",
        }
    )
    return report

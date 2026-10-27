"""Shared deterministic inputs and a controlled numerical thread count."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(autouse=True, scope="session")
def numerical_threads():
    """Keep BLAS execution reproducible and avoid oversubscribing tiny local solves."""
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def synthetic_model():
    """Return a full-rank n=30 design, noisy response, and projected coordinates."""
    generator = np.random.default_rng(20261027)
    design = np.column_stack((np.ones(30), generator.normal(size=(30, 3))))
    response = design @ np.array([1.0, 2.0, -0.7, 0.5])
    response = response[:, None] + generator.normal(scale=0.2, size=(30, 1))
    coordinates = generator.uniform(0.0, 100.0, size=(30, 2))
    return design, response, coordinates

"""Prove an installed CPU wheel works with no CuPy and rejects explicit GPU use."""

import importlib.util
from pathlib import Path

import numpy as np


def main() -> None:
    """Exercise installed CPU/auto/GPU semantics in a fresh no-CuPy environment."""
    import gpuGWR
    from gpugwr.engines.device import GPUUnavailableError

    assert "site-packages" in Path(gpuGWR.__file__).parts
    assert importlib.util.find_spec("cupy") is None
    generator = np.random.default_rng(4004)
    predictors = generator.normal(size=(30, 2))
    response = 3 + predictors @ np.array([1.0, -2.0]) + generator.normal(0, 0.1, 30)
    data = np.column_stack([response, predictors])
    coordinates = generator.uniform(0, 10, size=(30, 2))
    for engine in ("cpu", "auto"):
        result = gpuGWR.gpuGWR(
            data, 0, [1, 2], coordinates, bandwidth=25, engine=engine
        ).fit()
        assert result.engine == "cpu" and np.isfinite(result.aicc)
    try:
        gpuGWR.gpuGWR(data, 0, [1, 2], coordinates, bandwidth=25, engine="gpu").fit()
    except GPUUnavailableError:
        pass
    else:
        raise AssertionError("Explicit GPU request silently fell back to CPU.")
    print("INSTALLED_CPU_WITHOUT_CUPY_PASS", gpuGWR.__version__)


if __name__ == "__main__":
    main()

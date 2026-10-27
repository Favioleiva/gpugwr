"""Lazy CUDA capability detection with explicit failure and engine metadata."""

from functools import lru_cache


class GPUUnavailableError(RuntimeError):
    """An explicit GPU request cannot be served by a working float64 device."""


@lru_cache(maxsize=1)
def gpu_capability() -> dict:
    """Probe a real float64 device solve once per process, returning its metadata.

    Import, driver, runtime, and solver failures make the backend unavailable.
    Failures during subsequent fits are propagated; they never trigger fallback.
    """
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            raise GPUUnavailableError("No CUDA device is visible.")
        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        matrix = cp.eye(2, dtype=cp.float64)
        answer = cp.linalg.solve(matrix, cp.ones((2, 1), dtype=cp.float64))
        if answer.dtype != cp.float64 or not bool(cp.all(answer == 1)):
            raise GPUUnavailableError("Float64 device solver probe failed.")
        cp.cuda.get_current_stream().synchronize()
        return {
            "available": True,
            "gpu_framework": "CuPy",
            "gpu_framework_version": cp.__version__,
            "device_model": properties["name"].decode(),
            "device_id": device.id,
            "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "numerical_precision": "float64",
        }
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def resolve_engine(requested: str) -> dict:
    """Return actual engine metadata, raising if an explicit GPU is unavailable."""
    base = {
        "engine_requested": requested,
        "engine": "cpu",
        "gpu_framework": None,
        "device_model": None,
        "numerical_precision": "float64",
    }
    if requested == "cpu":
        return {**base, "engine_reason": "CPU explicitly requested."}
    capability = gpu_capability()
    if capability["available"]:
        return {
            **base,
            **capability,
            "engine": "gpu",
            "engine_reason": "Compatible float64 CUDA device detected.",
        }
    if requested == "gpu":
        raise GPUUnavailableError(
            "GPU execution unavailable; no CPU fallback: " + capability["reason"]
        )
    return {**base, "engine_reason": "Automatic CPU selection: " + capability["reason"]}

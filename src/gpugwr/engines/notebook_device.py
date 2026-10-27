"""Lazy CUDA instrumentation kept outside the GPU-independent CPU modules."""


def device_memory() -> tuple[int, int]:
    """Return available and total device bytes using the active CUDA context."""
    import cupy as cp

    return cp.cuda.runtime.memGetInfo()


def synchronize() -> None:
    """Wait for the active CUDA stream before recording completed wall time."""
    import cupy as cp

    cp.cuda.get_current_stream().synchronize()

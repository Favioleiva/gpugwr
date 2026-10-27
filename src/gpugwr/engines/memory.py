"""Conservative memory planning for bounded float64 GPU working batches."""


def plan_batch(
    observations: int, parameters: int, free_bytes: int, requested: int | None
) -> dict:
    """Estimate working memory and choose a batch within half of free VRAM.

    The estimate includes tall SVD workspaces, weighted designs and operators,
    distances, weights, and small dense systems. It is deliberately conservative;
    vendor workspace sizes can vary, so allocation errors are also translated.
    Full retained matrices are allocated on the host, never on the GPU.
    """
    persistent_bytes = (
        8 * observations * (parameters * parameters + parameters * 8 + 32)
    )
    bytes_per_location = 8 * (
        observations * (16 * parameters + 24) + 16 * parameters * parameters
    )
    budget = int(free_bytes * 0.5) - persistent_bytes
    capacity = budget // bytes_per_location
    if capacity < 1:
        raise MemoryError("Insufficient GPU memory for one GWR location and workspace.")
    batch = min(observations, capacity, 128 if requested is None else requested)
    if requested is not None and min(requested, observations) > capacity:
        raise MemoryError(
            "Requested GPU batch exceeds available memory; "
            f"use batch_size <= {capacity}."
        )
    return {
        "batch_size": batch,
        "estimated_device_bytes": persistent_bytes + batch * bytes_per_location,
        "free_device_bytes_at_start": free_bytes,
    }

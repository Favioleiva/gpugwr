"""CUDA-only construction of spatial structures shared across compatible models."""

from time import perf_counter

import cupy as cp

from .spatial import distance_rows, weight_rows


def initialize_gpu_cache(cache, batch_size: int) -> None:
    """Build full shared geometry in bounded rows after checking available VRAM."""
    n = len(cache.coordinates)
    budget = int(cp.cuda.runtime.memGetInfo()[0] * 0.5)
    estimate = 8 * (2 * n * n + n * batch_size * 12 + 4 * n)
    if estimate > budget:
        raise MemoryError("Shared GPU spatial cache exceeds half of free VRAM.")
    cache.pool = cp.cuda.MemoryPool()
    cache.pool.set_limit(size=budget)
    with cp.cuda.using_allocator(cache._allocate):
        uploaded = perf_counter()
        device_coordinates = cp.asarray(cache.coordinates)
        cp.cuda.get_current_stream().synchronize()
        cache.metadata["transfer_seconds"] = perf_counter() - uploaded
        cache.distances = cp.empty((n, n), dtype=cp.float64)
        cache.weights = cp.empty((n, n), dtype=cp.float64)
        cache.radii = cp.empty(n, dtype=cp.float64)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            distances = distance_rows(device_coordinates, start, stop)
            weights, radii = weight_rows(
                distances, cache.bandwidth, cache.kernel, cache.fixed
            )
            cache.distances[start:stop] = distances
            cache.weights[start:stop] = weights
            cache.radii[start:stop] = radii
        cp.cuda.get_current_stream().synchronize()

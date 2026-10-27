"""Explicit reusable geometry and kernel weights for compatible GWR fits."""

from time import perf_counter

import numpy as np

from .distances import pairwise_distances
from .engines.device import resolve_engine
from .kernels import spatial_weights


class GWRSpatialCache:
    """Own one coordinate order, distance convention, kernel and bandwidth.

    A cache may be shared across different response/design matrices only when
    their observation order and spatial specification agree exactly. It never
    caches coefficients, inverses, covariance, residuals or model diagnostics.
    """

    def __init__(
        self,
        coordinates,
        bandwidth: float,
        *,
        kernel: str = "bisquare",
        fixed: bool = False,
        engine: str = "cpu",
        batch_size: int = 128,
    ) -> None:
        """Build reusable distances/weights, explicitly charging preparation cost."""
        started = perf_counter()
        self.coordinates = np.array(coordinates, dtype=np.float64, copy=True)
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("Cache coordinates must have shape (n, 2).")
        if not np.all(np.isfinite(self.coordinates)):
            raise ValueError("Cache coordinates must be finite.")
        if engine not in {"cpu", "gpu"} or kernel not in {
            "bisquare",
            "gaussian",
            "exponential",
        }:
            raise ValueError("Unsupported cache engine or kernel.")
        if not isinstance(fixed, bool):
            raise ValueError("fixed must be boolean.")
        if (
            isinstance(bandwidth, bool)
            or not isinstance(bandwidth, (int, float, np.number))
            or not np.isfinite(bandwidth)
            or bandwidth <= 0
        ):
            raise ValueError("Cache bandwidth must be finite and positive.")
        n = len(self.coordinates)
        if not fixed and (int(bandwidth) != bandwidth or not 2 <= bandwidth <= n):
            raise ValueError("Adaptive cache bandwidth must be an integer in [2,n].")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("Cache batch_size must be a positive integer.")
        self.coordinates.flags.writeable = False
        self.bandwidth = float(bandwidth)
        self.kernel = kernel
        self.fixed = fixed
        self.engine = engine
        self.metadata = resolve_engine(engine)
        self.metadata["transfer_seconds"] = 0.0
        self.metadata["peak_allocated_bytes"] = 0
        self.metadata["peak_pool_reserved_bytes"] = 0
        self.metadata["hits"] = 0
        self._closed = False
        self.neighbor_indices = None
        if engine == "cpu":
            self.distances = pairwise_distances(self.coordinates)
            self.weights, self.radii = spatial_weights(
                self.distances, bandwidth, kernel, fixed
            )
            self.neighbor_indices = np.argsort(self.distances, axis=1, kind="stable")
            for values in (
                self.distances,
                self.weights,
                self.radii,
                self.neighbor_indices,
            ):
                values.flags.writeable = False
        else:
            self._initialize_gpu(batch_size)
        self.metadata["preparation_seconds"] = perf_counter() - started
        self.metadata["distance_convention"] = "Euclidean in supplied coordinate units"
        self.metadata["resident_array_bytes"] = int(
            self.distances.nbytes
            + self.weights.nbytes
            + self.radii.nbytes
            + (0 if self.neighbor_indices is None else self.neighbor_indices.nbytes)
        )

    def _allocate(self, size: int):
        """Record exact allocation high-water marks for the cache's private pool."""
        allocation = self.pool.malloc(size)
        self.metadata["peak_allocated_bytes"] = max(
            self.metadata["peak_allocated_bytes"], self.pool.used_bytes()
        )
        self.metadata["peak_pool_reserved_bytes"] = max(
            self.metadata["peak_pool_reserved_bytes"], self.pool.total_bytes()
        )
        return allocation

    def _initialize_gpu(self, batch_size: int) -> None:
        """Delegate CUDA allocation to its engine, keeping CPU code isolated."""
        from .engines.spatial_cache import initialize_gpu_cache

        initialize_gpu_cache(self, batch_size)

    def validate(self, coordinates, bandwidth, kernel, fixed, engine) -> None:
        """Reject any incompatible observation order or spatial specification."""
        if self._closed:
            raise ValueError("This spatial cache has been closed.")
        if (
            self.engine != engine
            or self.bandwidth != bandwidth
            or self.kernel != kernel
            or self.fixed != fixed
            or not np.array_equal(self.coordinates, coordinates)
        ):
            raise ValueError(
                "Spatial cache does not match this model's geometry/kernel."
            )
        self.metadata["hits"] += 1

    def __enter__(self):
        """Keep shared geometry resident for an explicitly bounded lifetime."""
        if self._closed:
            raise ValueError("This spatial cache has been closed.")
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        """Release the cache after a group of fits, including exceptional exits."""
        self.close()

    def close(self) -> None:
        """Release arrays and private CUDA reservations; repeated calls are harmless."""
        if getattr(self, "_closed", True):
            return
        self.distances = None
        self.weights = None
        self.radii = None
        self.neighbor_indices = None
        pool = getattr(self, "pool", None)
        if pool is not None:
            pool.free_all_blocks()
        self._closed = True

    def __del__(self):
        """Release forgotten caches; context management remains the preferred API."""
        self.close()

    def rows(self, start: int, stop: int):
        """Return compatible focal views without copying weights or distances."""
        if self._closed:
            raise ValueError("This spatial cache has been closed.")
        return (
            self.distances[start:stop],
            self.weights[start:stop],
            self.radii[start:stop],
        )

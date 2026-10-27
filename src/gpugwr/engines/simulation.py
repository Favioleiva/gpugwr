"""Minimal float64 Gaussian fits for batched coordinate-randomization tests.

For each focal location, N = X' W X and q = X' W y. Solving N beta = q
gives coefficients; solving N v = x_i gives influence w_ii * x_i' v.
These suffice for Gaussian AICc and coefficient SD. Covariance, local R2,
collinearity and off-diagonal hat entries are not statistics of this test.
"""

from contextlib import nullcontext
from time import perf_counter

import numpy as np

from ..distances import pairwise_distances
from .device import resolve_engine


def search_requests(lower: int, upper: int):
    """Yield uncached candidates using the validated discrete golden-section rules.

    Sending back an AICc value resumes the search. The StopIteration value is
    the selected bandwidth. This lets independent searches share GPU launches
    without forcing them to take identical branches or reuse incompatible fits.
    """
    left_bound = float(lower)
    right_bound = float(upper)
    contraction = 0.38197
    left = left_bound + contraction * abs(right_bound - left_bound)
    right = right_bound - contraction * abs(right_bound - left_bound)
    scores = {}
    for _ in range(200):
        left = int(np.round(left))
        right = int(np.round(right))
        for candidate in (left, right):
            if candidate not in scores:
                scores[candidate] = yield candidate
        left_score = scores[left]
        right_score = scores[right]
        if left_score <= right_score:
            selected = left
            right_bound = right
            right = left
            left = left_bound + contraction * abs(right_bound - left_bound)
        else:
            selected = right
            left_bound = left
            left = right
            right = right_bound - contraction * abs(right_bound - left_bound)
        if abs(left_score - right_score) <= 1.0e-6:
            break
    return selected


class SimulationEngine:
    """Reuse geometry and fit bounded batches with NumPy or the CuPy CUDA backend."""

    def __init__(self, data, engine: str, kernel: str, replica_batch_size: int):
        """Initialize one allocator and upload invariant inputs exactly once."""
        started = perf_counter()
        self.engine = engine
        self.kernel = kernel
        self.profile = resolve_engine(engine)
        self.profile["initialization_seconds"] = perf_counter() - started
        self.profile["transfer_seconds"] = 0.0
        self.profile["peak_allocated_bytes"] = 0
        self.profile["peak_pool_reserved_bytes"] = 0
        self.n, self.p = data.design.shape
        self.pool = None
        self._closed = False
        self.xp = np
        budget = 512 * 1024**2
        if engine == "gpu":
            import cupy as cp

            self.xp = cp
            budget = int(cp.cuda.runtime.memGetInfo()[0] * 0.5)
            self.pool = cp.cuda.MemoryPool()
            self.pool.set_limit(size=budget)
        # Conservative working-storage bound includes gathered distances,
        # weights, weighted designs, solves and linalg temporaries.
        persistent_bytes = 8 * (2 * self.n**2 + self.n * (self.p + 1))
        replica_bytes = 8 * (
            self.n**2 * (self.p + 8) + self.n * (8 * self.p**2 + 16 * self.p)
        )
        allowed = (budget - persistent_bytes) // replica_bytes
        if allowed < 1:
            raise MemoryError("Simulation geometry exceeds the bounded memory budget.")
        self.batch_size = min(replica_batch_size, int(allowed))
        self.profile["replica_batch_size"] = self.batch_size
        self.profile["memory_budget_bytes"] = budget
        self.profile["estimated_working_bytes"] = (
            persistent_bytes + self.batch_size * replica_bytes
        )
        self.profile["candidate_evaluations"] = 0
        self.profile["cuda_event_seconds"] = 0.0
        self.cuda_start = None
        with self.allocator():
            if engine == "gpu":
                from .spatial import distance_rows

                uploaded = perf_counter()
                self.design = self.xp.asarray(data.design)
                self.response = self.xp.asarray(data.response)
                coordinates = self.xp.asarray(data.coordinates)
                self.synchronize()
                self.profile["transfer_seconds"] += perf_counter() - uploaded
                self.cuda_start = self.xp.cuda.Event()
                self.cuda_start.record()
                self.distances = distance_rows(coordinates, 0, self.n)
            else:
                self.design = data.design
                self.response = data.response
                self.distances = pairwise_distances(data.coordinates)
            self.sorted_distances = self.xp.sort(self.distances, axis=1)

    def __enter__(self):
        """Keep invariant arrays resident for exactly one simulation call."""
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        """Release private CUDA reservations on normal and exceptional exits."""
        self.close()

    def close(self) -> None:
        """Drop resident arrays before returning cached blocks to the device."""
        if getattr(self, "_closed", True):
            return
        self.design = None
        self.response = None
        self.distances = None
        self.sorted_distances = None
        if self.pool is not None:
            self.pool.free_all_blocks()
        self._closed = True

    def __del__(self):
        """Fallback for partially constructed engines; public calls close explicitly."""
        self.close()

    def _allocate(self, size: int):
        """Track exact high-water allocated/reserved bytes in this private pool."""
        allocation = self.pool.malloc(size)
        self.profile["peak_allocated_bytes"] = max(
            self.profile["peak_allocated_bytes"], self.pool.used_bytes()
        )
        self.profile["peak_pool_reserved_bytes"] = max(
            self.profile["peak_pool_reserved_bytes"], self.pool.total_bytes()
        )
        return allocation

    def allocator(self):
        """Select the private CUDA pool; CPU computations need no allocator context."""
        if self.engine == "gpu":
            return self.xp.cuda.using_allocator(self._allocate)
        return nullcontext()

    def synchronize(self) -> None:
        """Synchronize before wall-clock measurement or host consumption."""
        if self.engine == "gpu":
            self.xp.cuda.get_current_stream().synchronize()

    def to_host(self, values):
        """Download only search scores or final dispersion vectors and measure it."""
        if self.engine == "cpu":
            return np.asarray(values)
        started = perf_counter()
        result = self.xp.asnumpy(values)
        self.profile["transfer_seconds"] += perf_counter() - started
        return result

    def fit(self, distance_batch, permutations, bandwidths):
        """Return AICc and coefficient surfaces for one candidate per permutation.

        All matrices stay float64. Positive-definite normal systems are solved
        directly, without explicit inversion. Cholesky verifies positive
        definiteness of each normal matrix; no regularization is introduced.
        """
        xp = self.xp
        bandwidths = xp.asarray(bandwidths, dtype=xp.int64)
        radii = self.sorted_distances[permutations, bandwidths[:, None] - 1] * 1.0000001
        if bool(xp.any(radii <= 0)):
            raise ValueError("Adaptive kernels require positive neighborhood radii.")
        scaled = distance_batch / radii[:, :, None]
        if self.kernel == "bisquare":
            weights = xp.where(scaled < 1, (1 - scaled**2) ** 2, 0.0)
        elif self.kernel == "gaussian":
            weights = xp.exp(-0.5 * scaled**2)
        else:
            weights = xp.exp(-scaled)
        weighted_transpose = weights[:, :, None, :] * self.design.T
        normal = weighted_transpose @ self.design
        if self.engine == "gpu":
            from cupyx import errstate

            with errstate(linalg="ignore"):
                factor = xp.linalg.cholesky(normal)
            diagonal = xp.diagonal(factor, axis1=-2, axis2=-1)
            valid_locations = xp.all(xp.isfinite(diagonal) & (diagonal > 0), axis=-1)
        else:
            valid_locations = self.cpu_positive_definite(normal)
        valid_replicas = xp.all(valid_locations, axis=1)
        normal = xp.where(valid_locations[..., None, None], normal, xp.eye(self.p))
        weighted_response = weighted_transpose @ self.response
        focal_design = xp.broadcast_to(
            self.design[None, :, :, None], weighted_response.shape
        )
        right_hand_side = xp.concatenate((weighted_response, focal_design), axis=-1)
        solution = xp.linalg.solve(normal, right_hand_side)
        coefficients = solution[..., 0]
        influence = xp.sum(self.design * solution[..., 1], axis=-1)
        predictions = xp.sum(coefficients * self.design, axis=-1)
        residuals = self.response[:, 0] - predictions
        residual_ss = xp.sum(residuals**2, axis=1)
        trace = xp.sum(influence, axis=1)
        # -2*log L + 2*n*(tr(S)+1)/(n-tr(S)-2), Gaussian ML variance RSS/n.
        negative_twice_loglike = self.n * (
            xp.log(residual_ss / self.n) + np.log(2 * np.pi) + 1
        )
        aicc = negative_twice_loglike + (
            2 * self.n * (trace + 1) / (self.n - trace - 2)
        )
        valid_replicas &= (self.n - trace - 2 > 0) & xp.isfinite(aicc)
        aicc = xp.where(valid_replicas, aicc, 1.0e100)
        if not bool(xp.all(xp.isfinite(aicc))):
            raise ValueError("Non-finite AICc in simulation.")
        self.profile["candidate_evaluations"] += len(bandwidths)
        return aicc, coefficients

    def cpu_positive_definite(self, normal):
        """Identify invalid local systems without rejecting other batch members."""
        try:
            np.linalg.cholesky(normal)
            return np.ones(normal.shape[:2], dtype=bool)
        except np.linalg.LinAlgError:
            valid = np.ones(normal.shape[:2], dtype=bool)
            for replica in range(len(normal)):
                for location in range(self.n):
                    try:
                        np.linalg.cholesky(normal[replica, location])
                    except np.linalg.LinAlgError:
                        valid[replica, location] = False
            return valid

    def run_batch(self, indices, bounds):
        """Advance independent golden-section searches together within one batch."""
        xp = self.xp
        uploaded = perf_counter()
        permutations = xp.asarray(indices)
        self.synchronize()
        if self.engine == "gpu":
            self.profile["transfer_seconds"] += perf_counter() - uploaded
        distance_batch = self.distances[
            permutations[:, :, None], permutations[:, None, :]
        ]
        searches = [search_requests(*bounds) for _ in indices]
        requests = {position: next(search) for position, search in enumerate(searches)}
        selected = np.empty(len(indices), dtype=np.int64)
        while requests:
            positions = list(requests)
            device_positions = xp.asarray(positions)
            scores, _ = self.fit(
                distance_batch[device_positions],
                permutations[device_positions],
                list(requests.values()),
            )
            next_requests = {}
            for position, score in zip(positions, self.to_host(scores), strict=True):
                try:
                    next_requests[position] = searches[position].send(float(score))
                except StopIteration as completed:
                    selected[position] = completed.value
            requests = next_requests
        scores, coefficients = self.fit(distance_batch, permutations, selected)
        if bool(xp.any(scores >= 1.0e100)):
            raise ValueError("Simulation bandwidth search found no valid fit.")
        dispersion = self.to_host(xp.std(coefficients, axis=1, ddof=0))
        return dispersion, selected

    def run(self, indices, bounds):
        """Process all permutations in bounded batches, returning compact evidence."""
        dispersions = []
        bandwidths = []
        with self.allocator():
            for start in range(0, len(indices), self.batch_size):
                dispersion, selected = self.run_batch(
                    indices[start : start + self.batch_size], bounds
                )
                dispersions.append(dispersion)
                bandwidths.append(selected)
            if self.engine == "gpu":
                finished = self.xp.cuda.Event()
                finished.record()
                finished.synchronize()
                self.profile["cuda_event_seconds"] = (
                    self.xp.cuda.get_elapsed_time(self.cuda_start, finished) / 1000
                )
        self.synchronize()
        return np.concatenate(dispersions), np.concatenate(bandwidths), self.profile

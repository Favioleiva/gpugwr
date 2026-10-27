"""Memory-bounded Gaussian GWR with genuine float64 CUDA estimation."""

from time import perf_counter

import cupy as cp
import numpy as np

from ..diagnostics import information_criteria
from ..inference import coefficient_inference
from ..model import validate_model_inputs
from .diagnostics import local_collinearity
from .memory import plan_batch
from .regression import local_operators
from .spatial import distance_rows, weight_rows


class GPUEngine:
    """Keep model inputs on one CUDA device across bandwidth evaluations.

    Local solves, hat rows, covariance, local fit and collinearity run on GPU.
    Final O(n*p) inference and scalar information criteria reuse validated CPU
    formulas. No n-by-n device arrays are retained. Requested full matrices are
    copied batchwise to host arrays; their host memory remains quadratic.
    """

    def __init__(
        self,
        data,
        batch_size: int | None,
        store_matrices: bool,
        intercept: bool,
        spatial_cache=None,
    ) -> None:
        """Validate inputs, plan working memory, and transfer the data once."""
        validate_model_inputs(data.design, data.response, data.coordinates, intercept)
        self.data = data
        self.spatial_cache = spatial_cache
        self.store_matrices = store_matrices
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        self.plan = plan_batch(*data.design.shape, free_bytes, batch_size)
        self.pool = cp.cuda.MemoryPool()
        self._closed = False
        self.pool.set_limit(size=int(free_bytes * 0.5))
        self.transfer_seconds = 0.0
        self.last_profile = {}
        self.device_seconds = 0.0
        started = perf_counter()
        with cp.cuda.using_allocator(self.pool.malloc):
            self.design = cp.asarray(data.design, dtype=cp.float64)
            self.response = cp.asarray(data.response, dtype=cp.float64)
            self.coordinates = cp.asarray(data.coordinates, dtype=cp.float64)
            cp.cuda.get_current_stream().synchronize()
        self.upload_seconds = perf_counter() - started

    def __enter__(self):
        """Keep resident model arrays alive throughout bandwidth search and fitting."""
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        """Release the private pool on success or failure without hiding errors."""
        self.close()

    def close(self) -> None:
        """Release resident arrays and cached blocks explicitly.

        CuPy pools can retain device reservations after their Python owners die.
        Explicitly returning all unused blocks prevents accumulation across fits.
        Numerical results have already been copied to host before normal closure.
        """
        if getattr(self, "_closed", True):
            return
        self.design = None
        self.response = None
        self.coordinates = None
        self.pool.free_all_blocks()
        self._closed = True

    def __del__(self):
        """Lifetime fallback; normal public fits use explicit context exit."""
        self.close()

    def host(self, values):
        """Copy a device array to NumPy, recording synchronized transfer time."""
        cp.cuda.get_current_stream().synchronize()
        started = perf_counter()
        result = cp.asnumpy(values)
        self.transfer_seconds += perf_counter() - started
        return result

    def blocks(self):
        """Yield contiguous focal slices in original observation order."""
        count = len(self.data.response)
        for start in range(0, count, self.plan["batch_size"]):
            yield start, min(start + self.plan["batch_size"], count)

    def fixed_bounds(self) -> tuple[float, float]:
        """Compute the CPU fixed-search bounds without a dense distance matrix."""
        position = min(self.design.shape[1] + 2, len(self.design) - 1)
        lower, upper = 0.0, 0.0
        with cp.cuda.using_allocator(self.pool.malloc):
            for start, stop in self.blocks():
                distances = distance_rows(self.coordinates, start, stop)
                neighbors = cp.partition(distances, position, axis=1)[:, position]
                lower = max(lower, float(cp.max(neighbors)))
                upper = max(upper, float(cp.max(distances)))
        return lower * 1.0000001, upper * 2.0

    def fit(
        self,
        design,
        response,
        coordinates,
        bandwidth: float,
        full_diagnostics: bool = True,
        *,
        kernel: str = "bisquare",
        fixed: bool = False,
        intercept: bool = True,
    ) -> dict:
        """Match the CPU fitter call signature for reusable bandwidth search.

        Input arguments identify the resident model; they must be the original
        arrays. OOM is explicit and never triggers CPU computation or float32.
        """
        if (
            design is not self.data.design
            or response is not self.data.response
            or coordinates is not self.data.coordinates
        ):
            raise ValueError("GPU engine inputs differ from the resident model.")
        started = perf_counter()
        self.transfer_seconds = 0.0
        self.device_seconds = 0.0
        if self.spatial_cache is not None:
            self.spatial_cache.validate(coordinates, bandwidth, kernel, fixed, "gpu")
        try:
            with cp.cuda.using_allocator(self.pool.malloc):
                result = self._fit(
                    bandwidth, full_diagnostics, kernel, fixed, intercept
                )
                cp.cuda.get_current_stream().synchronize()
        except cp.cuda.memory.OutOfMemoryError as error:
            raise MemoryError(
                "Insufficient GPU memory during GWR; reduce batch_size or free VRAM."
            ) from error
        self.last_profile = {
            **self.plan,
            "fit_seconds": perf_counter() - started,
            "device_to_host_seconds": self.transfer_seconds,
            "device_phase_seconds": self.device_seconds,
            "host_to_device_seconds": self.upload_seconds,
            "peak_pool_reserved_bytes": self.pool.total_bytes(),
            "memory_measurement": (
                "Private CuPy pool reserved high-water mark; "
                "excludes CUDA context and library allocations."
            ),
        }
        return result

    def _fit(
        self, bandwidth: float, full: bool, kernel: str, fixed: bool, intercept: bool
    ) -> dict:
        """Run two bounded passes: local estimation, then residual diagnostics."""
        count, parameters = self.design.shape
        retained = full and self.store_matrices
        device_outputs = {
            "params": cp.empty((count, parameters)),
            "predy": cp.empty((count, 1)),
            "radii": cp.empty(count),
            "influ": cp.empty((count, 1)),
            "row_square_sum": cp.empty(count),
        }
        covariance = cp.empty((count, parameters, parameters)) if full else None
        host_matrices = {}
        if retained:
            for name in ("distances", "W", "S", "neighbor_mask", "neighbor_indices"):
                dtype = np.int64 if name.startswith("neighbor") else np.float64
                host_matrices[name] = np.empty((count, count), dtype=dtype)
        for start, stop in self.blocks():
            event_start = cp.cuda.Event()
            event_stop = cp.cuda.Event()
            event_start.record()
            if self.spatial_cache is None:
                distances = distance_rows(self.coordinates, start, stop)
                weights, radii = weight_rows(distances, bandwidth, kernel, fixed)
            else:
                distances, weights, radii = self.spatial_cache.rows(start, stop)
            operator = local_operators(self.design, weights, start)
            coefficients = (operator @ self.response).squeeze(axis=2)
            smoother = (self.design[start:stop, None, :] @ operator).squeeze(axis=1)
            device_outputs["params"][start:stop] = coefficients
            device_outputs["predy"][start:stop, 0] = cp.sum(
                self.design[start:stop] * coefficients, axis=1
            )
            device_outputs["radii"][start:stop] = radii
            device_outputs["influ"][start:stop, 0] = smoother[
                cp.arange(stop - start), cp.arange(start, stop)
            ]
            device_outputs["row_square_sum"][start:stop] = cp.sum(smoother**2, axis=1)
            if full:
                covariance[start:stop] = operator @ operator.transpose(0, 2, 1)
                diagnostics = local_collinearity(self.design, weights, intercept)
                for name, values in diagnostics.items():
                    if name not in device_outputs:
                        device_outputs[name] = cp.empty((count, *values.shape[1:]))
                    device_outputs[name][start:stop] = values
                del diagnostics
            event_stop.record()
            event_stop.synchronize()
            self.device_seconds += (
                cp.cuda.get_elapsed_time(event_start, event_stop) / 1000
            )
            if retained:
                host_matrices["distances"][start:stop] = self.host(distances)
                host_matrices["W"][start:stop] = self.host(weights)
                host_matrices["S"][start:stop] = self.host(smoother)
                host_matrices["neighbor_mask"][start:stop] = self.host(
                    (weights > 0).astype(cp.int64)
                )
                # CuPy argsort is stable, retaining input order for tied distances.
                host_matrices["neighbor_indices"][start:stop] = self.host(
                    cp.argsort(distances, axis=1)
                )
            del operator, smoother, weights, distances, coefficients
        residuals = self.response - device_outputs["predy"]
        if full:
            for name in ("y_bar", "TSS", "RSS", "localR2"):
                device_outputs[name] = cp.empty((count, 1))
            for start, stop in self.blocks():
                event_start.record()
                if self.spatial_cache is None:
                    distances = distance_rows(self.coordinates, start, stop)
                    weights, _ = weight_rows(distances, bandwidth, kernel, fixed)
                else:
                    distances, weights, _ = self.spatial_cache.rows(start, stop)
                means = (weights @ self.response) / cp.sum(weights, axis=1)[:, None]
                totals = cp.sum(weights * (self.response.T - means) ** 2, axis=1)[
                    :, None
                ]
                local_rss = weights @ (residuals**2)
                device_outputs["y_bar"][start:stop] = means
                device_outputs["TSS"][start:stop] = totals
                device_outputs["RSS"][start:stop] = local_rss
                device_outputs["localR2"][start:stop] = (totals - local_rss) / totals
                event_stop.record()
                event_stop.synchronize()
                self.device_seconds += (
                    cp.cuda.get_elapsed_time(event_start, event_stop) / 1000
                )
                del distances, weights
        result = {name: self.host(values) for name, values in device_outputs.items()}
        result.update(host_matrices)
        result.update(
            {
                "X": self.data.design,
                "y": self.data.response,
                "coordinates": self.data.coordinates,
                "bandwidth": float(bandwidth),
                "n": float(count),
                "k": float(parameters),
            }
        )
        residual_host = self.data.response - result["predy"]
        trace = float(result["influ"].sum())
        squared_trace = float(result.pop("row_square_sum").sum())
        result.update(
            {
                "mu": result["predy"].ravel(),
                "resid_response": residual_host.ravel(),
                "tr_S": trace,
                "tr_STS": squared_trace,
                "ENP": trace,
                "df_model": count - trace,
                "df_resid": count - 2 * trace + squared_trace,
            }
        )
        result.update(information_criteria(self.data.response, residual_host, trace))
        if not full:
            return result
        sigma_squared = result["resid_ss"] / (count - trace)
        covariance *= sigma_squared
        diagonal = self.host(cp.diagonal(covariance, axis1=1, axis2=2))
        if retained:
            result["covariance_full"] = self.host(covariance)
        standardized = residual_host / np.sqrt(sigma_squared * (1 - result["influ"]))
        result.update(
            {
                "sigma2": sigma_squared,
                "scale": sigma_squared,
                "CCT": diagonal,
                "std_res": standardized,
                "cooksD": standardized**2
                * result["influ"]
                / (trace * (1 - result["influ"])),
                "CV": float(np.mean((residual_host / (1 - result["influ"])) ** 2)),
                "w": np.ones((count, 1)),
                "null": np.full(count, self.data.response.mean()),
            }
        )
        result.update(coefficient_inference(result["params"], diagonal, trace))
        return result

"""Researcher-facing model configuration and explicit CPU/CUDA engine dispatch."""

import warnings
from contextlib import nullcontext
from functools import partial
from time import perf_counter
from typing import Any

import numpy as np
from scipy.linalg import LinAlgWarning

from .engines.device import resolve_engine
from .model import fit_gaussian
from .results import GWRResults
from .selection import select_bandwidth
from .validation import prepare_data


class gpuGWR:
    """Configure Gaussian GWR with validated data and independent CPU/CUDA engines.

    DataFrame selectors are column names; NumPy selectors are column indices.
    Coordinates are two column names, an (n, 2) array, or None for Point geometry.
    See Documentation/Contract02/API.md for parameters and numerical conventions.
    """

    def __init__(
        self,
        data: Any,
        dependent: str | int,
        independent: list[str] | list[int],
        coordinates: Any = None,
        kernel: str = "bisquare",
        bandwidth: float | str = "auto",
        engine: str = "cpu",
        *,
        fixed: bool = False,
        intercept: bool = True,
        criterion: str = "AICc",
        bandwidth_bounds: tuple[float, float] | None = None,
        standardize: bool = False,
        crs: Any = None,
        store_matrices: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """Copy and validate inputs; reject unsupported options before fitting."""
        if engine not in {"cpu", "gpu", "auto"}:
            raise ValueError("engine must be cpu, gpu, or auto.")
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer or None.")
        if kernel not in {"bisquare", "gaussian", "exponential"}:
            raise ValueError("kernel must be bisquare, gaussian, or exponential.")
        if criterion not in {"AICc", "CV"}:
            raise ValueError("criterion must be AICc or CV.")
        for name, value in {
            "fixed": fixed,
            "intercept": intercept,
            "standardize": standardize,
            "store_matrices": store_matrices,
        }.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean.")
        automatic = isinstance(bandwidth, str) and bandwidth == "auto"
        if not automatic:
            if (
                isinstance(bandwidth, (str, bool))
                or not isinstance(bandwidth, (int, float, np.number))
                or not np.isfinite(bandwidth)
                or bandwidth <= 0
            ):
                raise ValueError(
                    "bandwidth must be 'auto' or a positive finite number."
                )
        if bandwidth_bounds is not None:
            if not automatic:
                raise ValueError(
                    "bandwidth_bounds applies only to automatic selection."
                )
            if (
                not isinstance(bandwidth_bounds, (list, tuple))
                or len(bandwidth_bounds) != 2
            ):
                raise ValueError("bandwidth_bounds must contain two numbers.")
        self._data = prepare_data(
            data, dependent, independent, coordinates, intercept, standardize, crs
        )
        if not automatic and not fixed:
            if int(bandwidth) != bandwidth or not 2 <= bandwidth <= len(
                self._data.response
            ):
                raise ValueError(
                    "Adaptive bandwidth must be an integer from 2 through n."
                )
        self.configuration = {
            "kernel": kernel,
            "bandwidth": bandwidth,
            "engine_requested": engine,
            "engine": "cpu",
            "fixed": fixed,
            "intercept": intercept,
            "criterion": criterion,
            "bandwidth_bounds": bandwidth_bounds,
            "standardize": standardize,
            "store_matrices": store_matrices,
            "crs": self._data.crs,
            "family": "Gaussian",
            "sigma2_v1": True,
            "batch_size": batch_size,
        }

    def fit(self, *, spatial_cache=None) -> GWRResults:
        """Select the engine and bandwidth, then return aligned statistical results.

        spatial_cache optionally supplies a GWRSpatialCache with exactly matching
        coordinates, ordering, kernel, explicit bandwidth and engine. Its separate
        preparation time must be included when timing a sequence of reused fits.
        Omitting it preserves the independent fitting path.
        """
        configuration = self.configuration.copy()
        started = perf_counter()
        configuration.update(resolve_engine(configuration["engine_requested"]))
        initialization_seconds = perf_counter() - started
        fitter = fit_gaussian
        if spatial_cache is not None:
            if configuration["bandwidth"] == "auto":
                raise ValueError(
                    "Spatial reuse requires an explicitly specified bandwidth."
                )
            spatial_cache.validate(
                self._data.coordinates,
                configuration["bandwidth"],
                configuration["kernel"],
                configuration["fixed"],
                configuration["engine"],
            )
            fitter = partial(fit_gaussian, spatial_cache=spatial_cache)
        gpu_engine = None
        if configuration["engine"] == "gpu":
            from .engines.gpu import GPUEngine

            gpu_engine = GPUEngine(
                self._data,
                configuration["batch_size"],
                configuration["store_matrices"],
                configuration["intercept"],
                spatial_cache=spatial_cache,
            )
            fitter = gpu_engine.fit
        with gpu_engine if gpu_engine is not None else nullcontext():
            bandwidth = configuration["bandwidth"]
            search = None
            if bandwidth == "auto":
                search_started = perf_counter()
                bandwidth, search = select_bandwidth(
                    self._data,
                    configuration["kernel"],
                    configuration["fixed"],
                    configuration["intercept"],
                    configuration["criterion"],
                    configuration["bandwidth_bounds"],
                    fitter=fitter,
                    fixed_bounds=None
                    if gpu_engine is None
                    else gpu_engine.fixed_bounds,
                )
                search["execution_seconds"] = perf_counter() - search_started
                search["evaluation_count"] = len(search["arrays"]["search_evaluations"])
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                try:
                    numerical = fitter(
                        self._data.design,
                        self._data.response,
                        self._data.coordinates,
                        bandwidth,
                        kernel=configuration["kernel"],
                        fixed=configuration["fixed"],
                        intercept=configuration["intercept"],
                    )
                except (np.linalg.LinAlgError, LinAlgWarning) as error:
                    raise ValueError(
                        f"Singular or ill-conditioned local regression: {error}"
                    ) from error
            if len(self._data.response) - numerical["tr_S"] - 2 <= 0:
                raise ValueError(
                    "Insufficient effective degrees of freedom for AICc; "
                    "increase bandwidth."
                )
            optional_diagnostics = {"local_correlations", "local_vif"}
            if any(
                not np.all(np.isfinite(value))
                for name, value in numerical.items()
                if name not in optional_diagnostics
            ):
                raise ValueError(
                    "Undefined numerical diagnostics; "
                    "inspect response variance and local rank."
                )
            unavailable = [
                name
                for name in optional_diagnostics
                if not np.all(np.isfinite(numerical[name]))
            ]
            configuration["unavailable_diagnostics"] = sorted(unavailable)
            if unavailable:
                warnings.warn(
                    "Some centered collinearity diagnostics are undefined; "
                    "see configuration['unavailable_diagnostics'].",
                    UserWarning,
                    stacklevel=2,
                )
            if search is not None:
                numerical.update(search.pop("arrays"))
            configuration["bandwidth"] = float(bandwidth)
            configuration["search"] = search
            configuration["initialization_seconds"] = initialization_seconds
            if gpu_engine is not None:
                configuration["gpu_execution"] = gpu_engine.last_profile
            configuration["fit_end_to_end_seconds"] = perf_counter() - started
            if spatial_cache is not None:
                configuration["spatial_reuse"] = spatial_cache.metadata.copy()
            return GWRResults(numerical, self._data, configuration)

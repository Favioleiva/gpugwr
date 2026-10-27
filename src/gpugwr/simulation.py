"""Reproducible coordinate-randomization inference for Gaussian GWR.

Simulation is separate from model estimation. PySAL is never imported here.
The reference convention is population coefficient SD and strict exceedance/R.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .model import fit_gaussian
from .selection import select_bandwidth


@dataclass
class SpatialVariabilityResults:
    """Observed SD, replicate SDs, bandwidths, permutations and timing evidence."""

    observed: np.ndarray
    reference_distribution: np.ndarray
    pvalues: np.ndarray
    bandwidths: np.ndarray
    permutations: np.ndarray
    metadata: dict

    def export(self, directory, *, overwrite: bool = False) -> list[Path]:
        """Export lossless arrays and JSON metadata without silently overwriting."""
        directory = Path(directory)
        paths = [directory / "spatial_variability.npz", directory / "metadata.json"]
        if not overwrite and any(path.exists() for path in paths):
            raise FileExistsError("Simulation output already exists.")
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            paths[0],
            observed=self.observed,
            reference_distribution=self.reference_distribution,
            pvalues=self.pvalues,
            bandwidths=self.bandwidths,
            permutations=self.permutations,
        )
        paths[1].write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return paths


def permutation_indices(n: int, n_iters: int, seed, permutations) -> np.ndarray:
    """Generate legacy MT19937 permutations without changing global RNG state.

    Permuting arange(n) with RandomState uses the same row order as the original
    np.random.seed(seed); np.random.permutation(coordinates) implementation.
    Explicit indices take precedence; every row must be a permutation of 0..n-1.
    """
    if isinstance(n_iters, bool) or not isinstance(n_iters, int) or n_iters < 1:
        raise ValueError("n_iters must be a positive integer.")
    if permutations is None:
        generator = np.random.RandomState(5536 if seed is None else seed)
        return np.asarray([generator.permutation(n) for _ in range(n_iters)])
    indices = np.asarray(permutations)
    if indices.shape != (n_iters, n) or indices.dtype.kind not in "iu":
        raise ValueError("permutations must be an integer array of shape (n_iters, n).")
    if not np.all(np.sort(indices, axis=1) == np.arange(n)):
        raise ValueError("Every permutation must contain each observation index once.")
    return indices.astype(np.int64, copy=True)


def baseline_replicates(data, configuration, indices, bounds):
    """Use the validated independent CPU fitter and reselect every bandwidth.

    This deliberately unoptimized implementation is retained as an audit route.
    Only diagnostics unused by this test are excluded from the final refit.
    """
    dispersions = []
    bandwidths = []
    for permutation in indices:
        permuted_data = replace(data, coordinates=data.coordinates[permutation])
        bandwidth, _ = select_bandwidth(
            permuted_data,
            configuration["kernel"],
            False,
            configuration["intercept"],
            "AICc",
            bounds,
        )
        fitted = fit_gaussian(
            permuted_data.design,
            permuted_data.response,
            permuted_data.coordinates,
            bandwidth,
            full_diagnostics=False,
            kernel=configuration["kernel"],
            intercept=configuration["intercept"],
        )
        dispersions.append(np.std(fitted["params"], axis=0, ddof=0))
        bandwidths.append(bandwidth)
    return np.asarray(dispersions), np.asarray(bandwidths), {}


def spatial_variability(
    results,
    n_iters: int = 1000,
    *,
    seed: int | None = 5536,
    permutations=None,
    engine: str | None = None,
    bandwidth_bounds=None,
    replica_batch_size: int = 64,
    implementation: str = "minimal",
) -> SpatialVariabilityResults:
    """Test coefficient surfaces by permuting locations and reselecting bandwidth.

    Initially supports adaptive Gaussian GWR, AICc golden-section search, and
    the existing three kernels. Search bounds inherit the fitted model search;
    for a specified-bandwidth fit, use mgwr's default [40 + 2*p, n]. Explicit
    bandwidth_bounds override this choice. Unsupported conventions are rejected.
    The null randomizes the association between locations and fixed (X, y).
    """
    started = perf_counter()
    configuration = results.configuration
    if configuration["fixed"] or configuration["criterion"] != "AICc":
        raise NotImplementedError("Simulation currently requires adaptive AICc GWR.")
    engine = results.engine if engine is None else engine
    if engine not in {"cpu", "gpu"}:
        raise ValueError("Simulation engine must be cpu or gpu.")
    if implementation not in {"baseline", "minimal"}:
        raise ValueError("implementation must be baseline or minimal.")
    if implementation == "baseline" and engine != "cpu":
        raise ValueError("The audit baseline uses only CPU.")
    if (
        isinstance(replica_batch_size, bool)
        or not isinstance(replica_batch_size, int)
        or replica_batch_size < 1
    ):
        raise ValueError("replica_batch_size must be a positive integer.")
    data = results._simulation_data
    n, parameter_count = data.design.shape
    indices = permutation_indices(n, n_iters, seed, permutations)
    search = configuration.get("search")
    bounds = bandwidth_bounds
    if bounds is None:
        bounds = search["bounds"] if search else (40 + 2 * parameter_count, n)
    if (
        len(bounds) != 2
        or any(int(value) != value for value in bounds)
        or not 2 <= bounds[0] <= bounds[1] <= n
    ):
        raise ValueError("Provide integer bandwidth_bounds within [2, n].")
    observed = np.std(results.params, axis=0, ddof=0)
    if implementation == "baseline":
        dispersions, bandwidths, profile = baseline_replicates(
            data, configuration, indices, bounds
        )
    else:
        from .engines.simulation import SimulationEngine

        with SimulationEngine(
            data, engine, configuration["kernel"], replica_batch_size
        ) as estimator:
            dispersions, bandwidths, profile = estimator.run(indices, bounds)
    # Identity permutations with the observed bandwidth describe exactly the
    # observed fit. Preserve that exact tie instead of letting a different
    # factorization's roundoff change a strict comparison into an exceedance.
    identities = np.all(indices == np.arange(n), axis=1)
    identical_fits = identities & (bandwidths == results.bandwidth)
    dispersions[identical_fits] = observed
    pvalues = np.mean(dispersions > observed, axis=0)
    metadata = {
        "engine": engine,
        "implementation": implementation,
        "n_iters": n_iters,
        "seed": 5536 if seed is None else int(seed),
        "permutation_source": "explicit" if permutations is not None else "MT19937",
        "bandwidth_bounds": list(bounds),
        "kernel": configuration["kernel"],
        "statistic": "population standard deviation (ddof=0)",
        "pseudo_p": "count(reference SD > observed SD) / n_iters; no +1",
        "variable_names": results.variable_names,
        "float_precision": "float64",
        "exact_identity_ties": int(np.sum(identical_fits)),
        "total_seconds": perf_counter() - started,
        **profile,
    }
    return SpatialVariabilityResults(
        observed, dispersions, pvalues, bandwidths, indices, metadata
    )

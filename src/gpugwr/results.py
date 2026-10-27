"""Structured statistical results, readable summaries, and aligned local tables."""

import numpy as np
import pandas as pd

from .validation import ModelData


class GWRResults:
    """Expose numerical outputs and metadata with the original observation index.

    Numerical names follow Contract 01 (params, bse, tvalues, localR2, etc.).
    Large matrices are retained only when store_matrices=True. The CPU engine
    constructs dense matrices internally; the GPU engine processes bounded rows.
    """

    def __init__(self, numerical: dict, data: ModelData, configuration: dict) -> None:
        """Own results and metadata so repeated fits do not alias earlier outputs."""
        large_names = {
            "distances",
            "W",
            "S",
            "neighbor_mask",
            "neighbor_indices",
            "covariance_full",
        }
        self.numerical = {}
        for name, value in numerical.items():
            if name in large_names and not configuration["store_matrices"]:
                continue
            self.numerical[name] = (
                value.copy() if isinstance(value, np.ndarray) else value
            )
        self.configuration = configuration
        self._simulation_data = data
        self.observation_ids = data.observation_ids.copy()
        self.variable_names = data.variable_names.copy()
        self.dependent_name = data.dependent_name
        self.scaling = data.scaling.copy()
        self.geometry = None if data.geometry is None else data.geometry.copy(deep=True)
        self.engine = configuration["engine"]
        self.n_observations = len(data.response)
        self.n_independent = len(data.variable_names) - int(configuration["intercept"])
        self.bandwidth = configuration["bandwidth"]
        self.kernel = configuration["kernel"]
        self.bandwidth_type = "fixed" if configuration["fixed"] else "adaptive"

    def __getattr__(self, name: str):
        """Provide attribute access to documented Contract 01 numerical names."""
        numerical = self.__dict__.get("numerical", {})
        if name in numerical:
            return numerical[name]
        raise AttributeError(f"No retained result named {name!r}.")

    @property
    def global_diagnostics(self) -> dict[str, float]:
        """Return scalar numerical diagnostics as JSON-ready Python numbers."""
        return {
            name: float(value)
            for name, value in self.numerical.items()
            if np.ndim(value) == 0
        }

    def local_results(self) -> pd.DataFrame:
        """Return aligned identifiers, estimates, and local diagnostics."""
        table = pd.DataFrame(index=self.observation_ids.copy())
        table["row_position"] = np.arange(self.n_observations)
        table["observation_id"] = list(self.observation_ids)
        parameter_fields = {
            "params": "coefficient",
            "bse": "standard_error",
            "tvalues": "t_statistic",
            "filtered_tvalues": "corrected_t",
        }
        for field, label in parameter_fields.items():
            for position, variable in enumerate(self.variable_names):
                table[f"{label}[{position}]:{variable}"] = self.numerical[field][
                    :, position
                ]
        for name in (
            "predy",
            "resid_response",
            "localR2",
            "std_res",
            "influ",
            "cooksD",
            "y_bar",
            "TSS",
            "RSS",
            "local_condition",
            "radii",
        ):
            table[name] = np.asarray(self.numerical[name]).ravel()
        for name in ("local_correlations", "local_vif", "local_vdp"):
            for position in range(self.numerical[name].shape[1]):
                table[f"{name}[{position}]"] = self.numerical[name][:, position]
        return table

    def summary(self) -> str:
        """Return a readable statistical summary, suitable for print or a report."""
        lines = [
            "gpuGWR — Gaussian geographically weighted regression",
            f"Response: {self.dependent_name}",
            f"Observations: {self.n_observations}; predictors: {self.n_independent}",
            f"Engine: {self.engine}; kernel: {self.kernel}",
            f"Requested engine: {self.configuration['engine_requested']}; "
            f"precision: {self.configuration.get('numerical_precision', 'float64')}",
            self.configuration.get("engine_reason", ""),
            f"Bandwidth: {self.bandwidth:g} ({self.bandwidth_type})",
            f"Intercept: {self.configuration['intercept']}; "
            f"standardized: {self.configuration['standardize']}",
            f"Selection criterion: {self.configuration['criterion']}",
            "",
            "Global diagnostics:",
        ]
        for name in (
            "resid_ss",
            "sigma2",
            "llf",
            "aic",
            "aicc",
            "bic",
            "ENP",
            "R2",
            "adj_R2",
            "df_model",
            "df_resid",
            "CV",
        ):
            lines.append(f"  {name}: {self.numerical[name]:.12g}")
        lines.extend(["", "Local coefficients (mean, std, min, median, max):"])
        for position, name in enumerate(self.variable_names):
            values = self.params[:, position]
            statistics = [
                values.mean(),
                values.std(),
                values.min(),
                np.median(values),
                values.max(),
            ]
            lines.append(
                f"  {name}: " + ", ".join(f"{value:.6g}" for value in statistics)
            )
        lines.extend(
            [
                "",
                f"Corrected 95% critical |t|: {self.critical_t:.6g}",
                f"Adjusted alpha levels: {self.adj_alpha.tolist()}",
                "Inference uses Student t(n-1); corrected t values "
                "zero nonsignificant estimates.",
            ]
        )
        return "\n".join(lines)

    def export(self, directory, *, overwrite: bool = False) -> list:
        """Write CSV, JSON, NPZ, summary and optional GeoPackage; return file paths."""
        from .serialization import export_results

        return export_results(self, directory, overwrite=overwrite)

    def spatial_variability(self, n_iters: int = 1000, **options):
        """Return reproducible Monte Carlo spatial variability evidence.

        Call result.spatial_variability(500, seed=5536, engine="cpu"). Options
        include explicit permutations, bandwidth_bounds and replica_batch_size.
        The result contains pvalues, observed/reference SDs and an export method.
        """
        from .simulation import spatial_variability

        return spatial_variability(self, n_iters, **options)

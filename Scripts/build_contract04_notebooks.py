"""Write seven readable notebooks from explicit, reviewable cell templates.

This is an authoring tool, not runtime code generation. Saved notebooks contain
ordinary Markdown and Python cells. No encoded or generated executable payload
is downloaded or evaluated by the notebooks.
"""

import argparse
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind: str, source: str, tag: str | None = None) -> dict:
    """Construct one notebook cell with stable plain-text source and no saved output."""
    source = textwrap.dedent(source).strip() + "\n"
    result = {
        "cell_type": kind,
        "metadata": {"tags": [tag]} if tag else {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def common_cells(title: str, commit: str) -> list[dict]:
    """Explain source pinning and install the reviewed package in each notebook."""
    return [
        cell(
            "markdown",
            f"""
        # {title}

        Select **Runtime → Change runtime type → NVIDIA GPU**, then run all cells
        in a fresh session. Each notebook installs its own immutable public package
        checkout and verifies every input. No Python-source upload is required.

        Package source: `Favioleiva/gpugwr@{commit}`. These links become usable only
        after owner-authorized publication. Local tests are not Colab verification.
        A failed GPU request or numerical comparison stops execution visibly.
        Save the executed notebook and download the output ZIP as evidence.
        """,
        ),
        cell(
            "code",
            f'''
            import json
            import os
            import subprocess
            import sys
            import tempfile
            from pathlib import Path

            SOURCE_COMMIT = "{commit}"
            REPOSITORY = "https://github.com/Favioleiva/gpugwr.git"
            RUN = Path(tempfile.mkdtemp(prefix="gpugwr-"))
            CHECKOUT = RUN / "source"
            OUTPUT = RUN / "outputs"
            OUTPUT.mkdir()
            subprocess.run(["git", "init", str(CHECKOUT)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(CHECKOUT),
                    "remote",
                    "add",
                    "origin",
                    REPOSITORY,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(CHECKOUT),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    SOURCE_COMMIT,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(CHECKOUT),
                    "checkout",
                    "--detach",
                    "FETCH_HEAD",
                ],
                check=True,
            )
            os.environ["CUPY_CACHE_DIR"] = str(RUN / "cupy-cache")
            with (OUTPUT / "bootstrap.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(CHECKOUT / "Colab/bootstrap.py"),
                        "--checkout",
                        str(CHECKOUT),
                        "--commit",
                        SOURCE_COMMIT,
                        "--output",
                        str(OUTPUT / "bootstrap.json"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            print((OUTPUT / "bootstrap.log").read_text("utf-8")[-12000:])
            completed.check_returncode()
            ''',
            "bootstrap",
        ),
        cell(
            "code",
            """
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from threadpoolctl import threadpool_limits
            from gpuGWR import GWRSpatialCache, gpuGWR
            from gpugwr.notebook_measurement import environment, measure
            from gpugwr.reproducibility import (
                ReproductionAssets,
                export_geography,
                model_options,
                validate_arrays,
                validate_fit,
                write_json,
                zip_outputs,
            )

            assets = ReproductionAssets(CHECKOUT, RUN / "data-cache")
            runtime = environment()
            assert runtime["gpu"]["available"], (
                "A genuine GPU is required; no CPU fallback."
            )
            runtime["source_commit"] = SOURCE_COMMIT
            write_json(OUTPUT / "environment.json", runtime)
            print(json.dumps(runtime, indent=2))
            """,
            "environment",
        ),
    ]


def download_cell() -> dict:
    """Offer a download on Colab and a normal file link in other Jupyter runtimes."""
    return cell(
        "code",
        """
        archive = zip_outputs(OUTPUT)
        print(f"Evidence archive: {archive}")
        try:
            from google.colab import files
        except ImportError:
            from IPython.display import FileLink

            display(FileLink(str(archive)))
        else:
            files.download(str(archive))
        """,
        "download",
    )


def empirical_cells(name: str, explanation: str) -> list[dict]:
    """Fit and export one complete canonical empirical comparison."""
    return [
        cell(
            "markdown",
            explanation
            + """

        All fits use a Gaussian response, intercept, adaptive bisquare kernel and
        AICc bandwidth selection. Global diagnostics and every retained local
        reference array are compared at `atol=1e-10`, `rtol=1e-8`. Bandwidth and
        neighbor decisions require exact equality. References remain unchanged.
        """,
        ),
        cell(
            "code",
            f'''
            DATASET = "{name}"
            frame, specification = assets.load_case(DATASET)
            options = model_options(specification)
            print(json.dumps(specification, indent=2))
            display(frame.head())
            with threadpool_limits(limits=1):
                result, timing = measure(
                    lambda: gpuGWR(
                        frame, **options, engine="gpu", store_matrices=True
                    ).fit(),
                    "gpu",
                )
            assert result.engine == "gpu"
            print(result.summary())
            display(result.local_results().head(10))
            result.export(OUTPUT / DATASET)
            export_geography(
                result, frame, specification, assets, OUTPUT / DATASET
            )
            validation = validate_fit(
                result,
                assets.acquire(specification["reference"]),
                OUTPUT / "canonical_validation.json",
            )
            write_json(
                OUTPUT / "performance.json",
                {{
                    **timing,
                    "dataset": DATASET,
                    "phase": "first_fit_after_bootstrap",
                    "configuration": result.configuration,
                }},
            )
            print("Canonical validation:", validation["passed"])
            print("GPU wall seconds:", timing["wall_seconds"])
            ''',
            "experiment",
        ),
        cell(
            "markdown",
            """
        The export contains aligned geographic identifiers and variable names in
        CSV/NPZ, JSON metadata, a statistical summary, the validation report and
        runtime measurements. Geographic files use only documented coordinates;
        no EPSG code is invented. Sampled device memory is a lower bound on peak
        usage and includes other processes. This single fit is not a speedup claim.
        """,
        ),
        download_cell(),
    ]


def brazil_cells() -> list[dict]:
    """Reproduce every historical Brazil specification with and without reuse."""
    return [
        cell(
            "markdown",
            """
        ## Brazil: literal historical reproduction, 5,562 municipalities

        Response `DfHRElc` and the four original predictor sets are displayed below.
        Existing logged columns are not transformed again. All models use an
        intercept, adaptive bisquare weights and **1,345 neighbors**, without
        standardization or bandwidth selection. `COD_MUN_I` identifies municipalities.
        Variable names are preserved; the source does not establish a complete
        authoritative units dictionary, so units are not invented.

        **Historical coordinate convention:** polygon centroids computed directly
        in EPSG:4326, then Euclidean distances in degrees. This intentionally
        reproduces the original experiment; it is not a geographically corrected
        distance model. No corrected alternative is mixed into these results.

        The 34.8 MB GeoJSON is requested from the original pinned Gist and must
        match the approved SHA-256. Its upstream availability/identity was not
        established in the local historical record. Acquisition failure is a
        failed reproducibility step, never permission to substitute other data.
        """,
        ),
        cell(
            "code",
            '''
            frame, specification = assets.load_case("brazil")
            print(json.dumps(specification, indent=2))
            assert len(frame) == 5562
            assert specification["bandwidth"] == 1345
            display(frame.head())


            def fit_sequence(reuse):
                """Time cache preparation, all four fits and cache release."""
                cache = None
                fitted_models = {}
                if reuse:
                    cache = GWRSpatialCache(
                        frame[specification["coordinates"]].to_numpy(),
                        bandwidth=1345,
                        kernel="bisquare",
                        fixed=False,
                        engine="gpu",
                    )
                try:
                    for model in specification["models"]:
                        options = model_options(specification, model=model)
                        fitted_models[model["name"]] = gpuGWR(
                            frame,
                            **options,
                            engine="gpu",
                        ).fit(spatial_cache=cache)
                finally:
                    if cache is not None:
                        cache.close()
                return fitted_models


            timings = []
            independent = None
            with threadpool_limits(limits=1):
                for reuse in (False, True):
                    fitted_models, timing = measure(
                        lambda: fit_sequence(reuse), "gpu"
                    )
                    label = "reused" if reuse else "independent"
                    timing.update(
                        {"reuse": reuse, "dataset": "brazil", "models": {}}
                    )
                    for model in specification["models"]:
                        name = model["name"]
                        result = fitted_models[name]
                        folder = OUTPUT / label / name
                        result.export(folder)
                        export_geography(
                            result, frame, specification, assets, folder
                        )
                        validate_fit(
                            result,
                            assets.acquire(model["reference"]),
                            folder / "canonical_validation.json",
                        )
                        assert (
                            result.bandwidth == 1345
                            and result.n_observations == 5562
                        )
                        if reuse:
                            validate_arrays(
                                result.numerical,
                                independent[name].numerical,
                                folder / "reuse_validation.json",
                            )
                        timing["models"][name] = result.configuration
                        print(label, name, result.summary())
                        display(result.local_results().head())
                    if not reuse:
                        independent = fitted_models
                    timings.append(timing)
                    write_json(OUTPUT / "performance.json", timings)
            display(
                pd.DataFrame(
                    [
                        {
                            "reuse": item["reuse"],
                            "four_model_seconds": item["wall_seconds"],
                            "sampled_peak_mib": item[
                                "sampled_peak_device_memory_mib"
                            ],
                        }
                        for item in timings
                    ]
                )
            )
            ''',
            "experiment",
        ),
        download_cell(),
    ]


def monte_carlo_cells() -> list[dict]:
    """Explain and reproduce the approved strict-exceedance simulation convention."""
    return [
        cell(
            "markdown",
            r"""
        ## Georgia: 500 coordinate permutations

        This experiment uses **unstandardized** `PctBach ~ PctFB + PctBlack + PctRural`
        and an intercept, matching Contract 03 Addendum 001. It differs from the
        standardized canonical fit in notebook 01. AICc bandwidth selection uses
        bounds 48–159 and is repeated for every permutation of coordinates.

        The statistic is population coefficient SD (`ddof=0`). For coefficient j,
        `pseudo_p[j] = count(reference_SD[:, j] > observed_SD[j]) / 500`.
        The comparison is strictly greater; there is **no +1 correction**.
        Seed 5536 uses the legacy MT19937 ordering. The complete 500 approved
        permutations, selected bandwidths, SDs and pseudo-p values are checked.
        """,
        ),
        cell(
            "code",
            """
            frame, specification = assets.load_case("georgia")
            options = model_options(specification)
            options.update(standardize=False, bandwidth_bounds=(48, 159))
            with threadpool_limits(limits=1):
                observed_fit, fit_timing = measure(
                    lambda: gpuGWR(frame, **options, engine="gpu").fit(), "gpu"
                )
                simulation, simulation_timing = measure(
                    lambda: observed_fit.spatial_variability(
                        500, seed=5536, engine="gpu"
                    ),
                    "gpu",
                )
            print(observed_fit.summary())
            display(observed_fit.local_results().head())
            observed_fit.export(OUTPUT / "observed_fit")
            export_geography(
                observed_fit,
                frame,
                specification,
                assets,
                OUTPUT / "observed_fit",
            )
            simulation.export(OUTPUT / "monte_carlo")
            with np.load(
                assets.acquire(assets.manifest["monte_carlo"]["reference"]),
                allow_pickle=False,
            ) as archive:
                reference = {name: archive[name] for name in archive.files}
            actual = {name: getattr(simulation, name) for name in reference}
            validation = validate_arrays(
                actual, reference, OUTPUT / "canonical_validation.json"
            )
            table = pd.DataFrame(
                {
                    "variable": observed_fit.variable_names,
                    "observed_sd": simulation.observed,
                    "pseudo_p": simulation.pvalues,
                }
            )
            table.to_csv(OUTPUT / "pseudo_pvalues.csv", index=False)
            display(table)
            write_json(
                OUTPUT / "performance.json",
                {
                    "fit": fit_timing,
                    "simulation": simulation_timing,
                    "configuration": observed_fit.configuration,
                    "simulation_metadata": simulation.metadata,
                },
            )
            print("All 500-iteration canonical checks:", validation["passed"])
            """,
            "experiment",
        ),
        download_cell(),
    ]


def benchmark_cells() -> list[dict]:
    """Measure cold and warmed fits with canonical checks for every trial."""
    return [
        cell(
            "markdown",
            """
        ## CPU/GPU benchmark protocol

        Test all five empirical datasets; Brazil uses historical model1 here
        (notebook 05 reproduces all four). Each engine gets one fresh-process fit
        and three warmed repetitions. Fresh GPU workers receive empty on-disk
        CuPy caches. Record process wall time separately from fit wall time.
        BLAS is limited to one thread; CPU/GPU ordering alternates each repeat.

        Model construction, fitting, transfers, synchronization and pool cleanup
        are timed; data acquisition, validation and exporting are excluded from
        fit time. Cold subprocess total time additionally includes imports,
        loading, validation and exports. The environment probe warms the parent
        CUDA context, but never the fresh worker. Every timed result is validated.
        Report medians of measured trials, without extrapolating to other GPUs.
        Whole-device sampled memory is a lower bound, not an exact peak; private
        CuPy pool high-water measurements are retained in GPU configuration.
        """,
        ),
        cell(
            "code",
            """
            from time import perf_counter

            DATASETS = ["georgia", "columbus", "baltimore", "boston", "brazil"]
            REPEATS = 3
            records = []
            for dataset in DATASETS:
                frame, specification = assets.load_case(dataset)
                model = (
                    specification["models"][0] if dataset == "brazil" else None
                )
                options = model_options(specification, model=model)
                reference = assets.acquire(
                    (model or specification)["reference"]
                )
                for engine in ("cpu", "gpu"):
                    cold_output = OUTPUT / dataset / (engine + "_cold")
                    worker_environment = os.environ.copy()
                    worker_environment["CUPY_CACHE_DIR"] = str(
                        RUN / (dataset + "_" + engine + "_cold_cache")
                    )
                    started = perf_counter()
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "gpugwr.notebook_benchmark",
                            "--checkout",
                            str(CHECKOUT),
                            "--cache",
                            str(assets.cache),
                            "--dataset",
                            dataset,
                            "--engine",
                            engine,
                            "--output",
                            str(cold_output),
                        ],
                        env=worker_environment,
                        text=True,
                        capture_output=True,
                    )
                    cold_output.parent.mkdir(parents=True, exist_ok=True)
                    (cold_output.parent / (engine + "_cold.log")).write_text(
                        completed.stdout + completed.stderr, encoding="utf-8"
                    )
                    completed.check_returncode()
                    timing = json.loads(
                        (cold_output / "timing.json").read_text("utf-8")
                    )
                    timing["subprocess_total_seconds"] = (
                        perf_counter() - started
                    )
                    records.append(timing)

                with threadpool_limits(limits=1):
                    for engine in ("cpu", "gpu"):
                        warmup = gpuGWR(
                            frame,
                            **options,
                            engine=engine,
                            store_matrices=model is None,
                        ).fit()
                        validate_fit(
                            warmup,
                            reference,
                            OUTPUT
                            / dataset
                            / (engine + "_warmup_validation.json"),
                        )
                    for repetition in range(REPEATS):
                        order = (
                            ("cpu", "gpu")
                            if repetition % 2 == 0
                            else ("gpu", "cpu")
                        )
                        for engine in order:
                            result, timing = measure(
                                lambda: gpuGWR(
                                    frame,
                                    **options,
                                    engine=engine,
                                    store_matrices=model is None,
                                ).fit(),
                                engine,
                            )
                            label = f"{engine}_warm_{repetition + 1}"
                            validate_fit(
                                result,
                                reference,
                                OUTPUT / dataset / (label + "_validation.json"),
                            )
                            timing.update(
                                {
                                    "dataset": dataset,
                                    "phase": "warm",
                                    "repeat": repetition + 1,
                                    "configuration": result.configuration,
                                    "model_options": options,
                                }
                            )
                            records.append(timing)
                            if repetition == 0:
                                result.export(OUTPUT / dataset / label)
                                export_geography(
                                    result,
                                    frame,
                                    specification,
                                    assets,
                                    OUTPUT / dataset / label,
                                )
                                print(result.summary())
                                display(result.local_results().head())
                write_json(OUTPUT / "benchmark_trials.json", records)
            table = pd.DataFrame(
                [
                    {
                        "dataset": item["dataset"],
                        "engine": item["engine"],
                        "phase": item["phase"],
                        "wall_seconds": item["wall_seconds"],
                        "effective_batch_size": item["configuration"]
                        .get("gpu_execution", {})
                        .get("batch_size"),
                        "sampled_peak_mib": item[
                            "sampled_peak_device_memory_mib"
                        ],
                    }
                    for item in records
                ]
            )
            table.to_csv(OUTPUT / "benchmark_trials.csv", index=False)
            summary = (
                table[table.phase == "warm"]
                .groupby(["dataset", "engine"])
                .wall_seconds.agg(["count", "median", "min", "max"])
            )
            summary.to_csv(OUTPUT / "benchmark_summary.csv")
            display(summary)
            """,
            "experiment",
        ),
        download_cell(),
    ]


def main() -> None:
    """Write independently executable notebooks pinned to the supplied real commit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", arguments.commit) or set(arguments.commit) == {
        "0"
    }:
        raise ValueError("Use a real full source commit SHA.")
    descriptions = {
        "Georgia": (
            "159 counties: PctBach (bachelor's-degree percentage) explained by "
            "PctFB (foreign-born), PctBlack and PctRural percentages. Response "
            "and predictors are standardized with population SD, matching "
            "Contract 01. Original projected X/Y distances; AreaKey retains "
            "county identity."
        ),
        "Columbus": (
            "49 neighborhoods: CRIME (residential burglaries and vehicle thefts "
            "per 1,000 households) ~ INC (household income) + HOVAL (housing "
            "value), in original units. INC/HOVAL are thousands of dollars. "
            "Original arbitrary Cartesian X/Y units; no EPSG is assigned. "
            "Bounds 10–49, no standardization."
        ),
        "Baltimore": (
            "211 house sales: PRICE ~ NROOM (rooms) + AGE + SQFT. Original scales "
            "are retained; the source does not fully specify monetary/area "
            "units. Original Maryland grid X/Y has no established full CRS. "
            "Bounds 10–211, no standardization; IDs remain aligned."
        ),
        "Boston": (
            "506 census tracts: corrected response CMEDV ~ RM (rooms) + AGE + "
            "DIS (distance measure), in original scales. Uses author-supplied "
            "boston.utm X/Y, zone 19 with no verified datum; no EPSG is invented. "
            "Bounds 10–506, no standardization. The original studies are not "
            "being reinterpreted causally."
        ),
    }
    notebooks = []
    for number, (name, explanation) in enumerate(descriptions.items(), start=1):
        notebooks.append(
            (f"{number:02d}_{name}_GWR", empirical_cells(name.lower(), explanation))
        )
    notebooks.extend(
        [
            ("05_Brazil_GWR", brazil_cells()),
            ("06_Georgia_Monte_Carlo", monte_carlo_cells()),
            ("07_CPU_GPU_Benchmarks", benchmark_cells()),
        ]
    )
    destination = ROOT / "Notebooks"
    destination.mkdir(exist_ok=True)
    for name, experiment in notebooks:
        cells = common_cells(name.replace("_", " "), arguments.commit) + experiment
        for index, item in enumerate(cells):
            item["id"] = f"cell-{index:02d}"
            if item["cell_type"] == "code":
                compile("".join(item["source"]), f"{name}:cell-{index}", "exec")
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.12"},
                "colab": {"name": name + ".ipynb"},
                "accelerator": "GPU",
                "gpugwr_source_commit": arguments.commit,
            },
        }
        (destination / (name + ".ipynb")).write_text(
            json.dumps(notebook, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Wrote {len(notebooks)} notebooks pinned to {arguments.commit}.")


if __name__ == "__main__":
    main()

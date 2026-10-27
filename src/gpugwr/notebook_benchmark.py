"""Fresh-process benchmark worker used by the independently executable notebook.

Launching this module in a new process with a new CUPY_CACHE_DIR measures a
process-cold model fit. The parent separately records total subprocess latency.
"""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from .api import gpuGWR
from .notebook_measurement import environment, measure
from .reproducibility import ReproductionAssets, model_options, validate_fit, write_json


def main() -> None:
    """Fit and validate one dataset, exporting evidence even for a cold measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--engine", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    assets = ReproductionAssets(arguments.checkout, arguments.cache)
    frame, specification = assets.load_case(arguments.dataset)
    model = specification["models"][0] if arguments.dataset == "brazil" else None
    options = model_options(specification, model=model)
    arguments.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        fitted, timing = measure(
            lambda: gpuGWR(
                frame, **options, engine=arguments.engine, store_matrices=model is None
            ).fit(),
            arguments.engine,
        )
    reference = (model or specification)["reference"]
    validate_fit(
        fitted, assets.acquire(reference), arguments.output / "validation.json"
    )
    fitted.export(arguments.output / "results")
    write_json(
        arguments.output / "timing.json",
        {
            **timing,
            "phase": "fresh_process_empty_cupy_disk_cache",
            "dataset": arguments.dataset,
            "model_options": options,
            "configuration": fitted.configuration,
            "environment": environment(),
        },
    )


if __name__ == "__main__":
    main()

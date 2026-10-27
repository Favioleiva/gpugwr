"""Contract 04 portability checks, input tampering and unchanged empirical results."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gpuGWR import gpuGWR
from gpugwr.reproducibility import (
    ReproductionAssets,
    model_options,
    validate_arrays,
    validate_fit,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset", ["georgia", "columbus", "baltimore", "boston"])
def test_portable_empirical_cpu(dataset, tmp_path):
    """Independent assets preserve every canonical output in a new CPU fit."""
    assets = ReproductionAssets(ROOT, tmp_path)
    frame, specification = assets.load_case(dataset)
    result = gpuGWR(
        frame, **model_options(specification), engine="cpu", store_matrices=True
    ).fit()
    report = validate_fit(
        result, assets.acquire(specification["reference"]), tmp_path / "validation.json"
    )
    assert report["passed"]
    assert len(report["comparisons"]) >= 59
    assert list(result.observation_ids) == list(frame.index)


def test_reject_corrupt_data_before_parsing(tmp_path):
    """Reject a same-size one-byte edit through SHA-256 validation."""
    checkout = tmp_path / "checkout"
    shutil.copytree(ROOT / "Colab/assets", checkout / "Colab/assets")
    path = checkout / "Colab/assets/georgia.csv"
    payload = bytearray(path.read_bytes())
    payload[-3] ^= 1
    path.write_bytes(payload)
    assets = ReproductionAssets(checkout, tmp_path / "cache")
    with pytest.raises(ValueError, match="integrity"):
        assets.load_case("georgia")


@pytest.mark.parametrize(
    "value", [np.array([1.1]), np.array([np.nan]), np.array([1.0, 1.0])]
)
def test_discrepancies_saved_before_failure(value, tmp_path):
    """Numerical, nonfinite and shape failures are retained in downloadable evidence."""
    output = tmp_path / "failure.json"
    with pytest.raises(AssertionError):
        validate_arrays({"params": value}, {"params": np.array([1.0])}, output)
    assert not json.loads(output.read_text("utf-8"))["passed"]


def test_discrete_pvalues_not_toleranced(tmp_path):
    """A tiny pseudo-p difference must fail even if float tolerances would accept it."""
    with pytest.raises(AssertionError):
        validate_arrays(
            {"pvalues": np.array([0.1 + 1e-12])},
            {"pvalues": np.array([0.1])},
            tmp_path / "pvalues.json",
        )


def test_metadata_alignment_required(tmp_path):
    """A reordered geographic identifier cannot pass on numerical arrays alone."""
    assets = ReproductionAssets(ROOT, tmp_path)
    frame, specification = assets.load_case("columbus")
    result = gpuGWR(
        frame, **model_options(specification), engine="cpu", store_matrices=True
    ).fit()
    result.observation_ids = result.observation_ids[::-1]
    with pytest.raises(AssertionError, match="metadata"):
        validate_fit(
            result, assets.acquire(specification["reference"]), tmp_path / "ids.json"
        )


def test_cuda_family_selection():
    """Do not select CUDA 13 for pre-Turing hardware or a CUDA 12-only driver."""
    specification = importlib.util.spec_from_file_location(
        "bootstrap", ROOT / "Colab/bootstrap.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert module.select_cuda_major(12, 8.0) == 12
    assert module.select_cuda_major(13, 7.0) == 12
    assert module.select_cuda_major(13, 7.5) == 13
    with pytest.raises(RuntimeError):
        module.select_cuda_major(11, 8.6)


def test_notebooks_independent_and_pinned():
    """Each notebook has independent installation and downloadable evidence."""
    notebooks = list((ROOT / "Notebooks").glob("*.ipynb"))
    assert len(notebooks) == 7
    for path in notebooks:
        notebook = json.loads(path.read_text("utf-8"))
        tags = []
        for index, cell in enumerate(notebook["cells"]):
            tags.extend(cell["metadata"].get("tags", []))
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:{index}", "exec")
                assert not cell["outputs"]
        assert {"bootstrap", "environment", "experiment", "download"}.issubset(tags)
        commit = notebook["metadata"]["gpugwr_source_commit"]
        assert len(commit) == 40 and set(commit) != {"0"}

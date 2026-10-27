"""Notebook data acquisition, canonical comparisons and export helpers.

This module orchestrates the validated public API. It contains no alternative
GWR estimator and never changes reference arrays or widens tolerances.
"""

import hashlib
import json
import shutil
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .comparison import compare_arrays

DISCRETE_OUTPUTS = {
    "bandwidth",
    "n",
    "k",
    "neighbor_mask",
    "neighbor_indices",
    "bandwidth_interval",
    "bandwidths",
    "pvalues",
    "permutations",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file without loading large geometries into memory."""
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write finite, readable JSON into a run-specific output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


class ReproductionAssets:
    """Access hash-checked assets from an immutable checkout and a download cache."""

    def __init__(self, checkout: Path, cache: Path) -> None:
        """Read the reviewed manifest; never infer dataset identity from filenames."""
        self.checkout = Path(checkout).resolve()
        self.cache = Path(cache).resolve()
        self.manifest = json.loads(
            (self.checkout / "Colab/assets/manifest.json").read_text("utf-8")
        )
        self.cache.mkdir(parents=True, exist_ok=True)

    def acquire(self, name: str) -> Path:
        """Return verified bytes, downloading only a declared HTTPS source if absent.

        Existing corrupt files fail immediately. Interrupted downloads are staged
        separately and are never treated as valid cached inputs.
        """
        entry = self.manifest["assets"][name]
        if "path" in entry:
            path = (self.checkout / entry["path"]).resolve()
            if not path.is_relative_to(self.checkout):
                raise ValueError("Asset path escapes the checkout.")
        else:
            path = self.cache / name
            if not path.exists():
                if not entry["url"].startswith("https://"):
                    raise ValueError("Dataset acquisition requires HTTPS.")
                temporary = path.with_suffix(path.suffix + ".part")
                request = urllib.request.Request(
                    entry["url"], headers={"User-Agent": "gpuGWR-reproducibility/0.4"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    with temporary.open("wb") as target:
                        shutil.copyfileobj(response, target)
                self._verify(temporary, entry)
                temporary.replace(path)
        self._verify(path, entry)
        return path

    @staticmethod
    def _verify(path: Path, entry: dict) -> None:
        """Reject size or digest changes before parsing any scientific input."""
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Dataset/reference integrity failure: {path.name}")

    def load_case(self, name: str) -> tuple[pd.DataFrame, dict]:
        """Load original row order and identifiers with exactly the approved fields."""
        specification = self.manifest["cases"][name]
        path = self.acquire(specification["data"])
        if name == "brazil":
            import geopandas as gpd

            geographic = gpd.read_file(path)
            if geographic.crs is None:
                geographic = geographic.set_crs(4326)
            if geographic.crs.to_epsg() != 4326:
                raise ValueError("Brazil's historical geometry must be EPSG:4326.")
            # Literal historical operation: planar centroids of degree geometry.
            # This is intentionally not a geographically corrected alternative.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Geometry is in a geographic CRS"
                )
                centers = geographic.geometry.centroid
            frame = pd.DataFrame(geographic.drop(columns="geometry"))
            frame["x_coor"] = centers.x.to_numpy()
            frame["y_coor"] = centers.y.to_numpy()
            predictors = specification["models"][0]["independent"]
        else:
            frame = pd.read_csv(
                path,
                dtype={specification["id_column"]: str},
                float_precision="round_trip",
            )
            predictors = specification["independent"]
        frame[specification["id_column"]] = frame[specification["id_column"]].astype(
            str
        )
        frame = frame.set_index(specification["id_column"], drop=False)
        if len(frame) != specification["observations"] or not frame.index.is_unique:
            raise ValueError(
                "Observation count or unique geographic identifiers changed."
            )
        columns = [
            specification["dependent"],
            *predictors,
            *specification["coordinates"],
        ]
        if not np.isfinite(frame[columns].to_numpy(dtype=np.float64)).all():
            raise ValueError("Scientific inputs contain nonfinite values.")
        return frame, specification


def model_options(specification: dict, *, model: dict | None = None) -> dict:
    """Translate an approved specification without changing its mathematical options."""
    options = {
        "dependent": specification["dependent"],
        "independent": (model or specification)["independent"],
        "coordinates": specification["coordinates"],
        "kernel": "bisquare",
        "fixed": False,
        "standardize": specification.get("standardize", False),
    }
    for name in ("bandwidth", "bandwidth_bounds"):
        if name in specification:
            options[name] = specification[name]
    return options


def validate_arrays(actual: dict, expected: dict, output: Path) -> dict:
    """Save all discrepancies before raising; discrete outputs require exact equality.

    Every numeric reference field must be present. Metadata arrays are checked
    separately by the caller; they are never converted to numeric quantities.
    """
    comparisons = []
    for name, reference in expected.items():
        if np.asarray(reference).dtype.kind not in "biufc":
            continue
        if name not in actual:
            comparisons.append(
                {"name": name, "status": "FAIL", "reason": "missing output"}
            )
            continue
        comparisons.append(
            compare_arrays(
                name, reference, actual[name], exact=name in DISCRETE_OUTPUTS
            )
        )
        if name in {"search_history", "search_evaluations"}:
            comparisons.append(
                compare_arrays(
                    name + "_bandwidths",
                    reference[:, 0],
                    actual[name][:, 0],
                    exact=True,
                )
            )
    report = {
        "passed": bool(comparisons)
        and all(item["status"] == "PASS" for item in comparisons),
        "rtol": 1e-8,
        "atol": 1e-10,
        "comparisons": comparisons,
    }
    write_json(output, report)
    if not report["passed"]:
        failed = [item["name"] for item in comparisons if item["status"] != "PASS"]
        raise AssertionError(f"Canonical numerical discrepancy: {failed}; see {output}")
    return report


def validate_fit(result, reference_path: Path, output: Path) -> dict:
    """Compare all retained reference outputs, including available names and IDs."""
    with np.load(reference_path, allow_pickle=False) as archive:
        expected = {name: archive[name] for name in archive.files}
    for name, actual in (
        ("observation_ids", np.asarray(result.observation_ids).astype(str)),
        ("variable_names", np.asarray(result.variable_names)),
    ):
        if name in expected:
            if not np.array_equal(expected[name].astype(str), actual.astype(str)):
                write_json(output, {"passed": False, "metadata_mismatch": name})
                raise AssertionError(f"Canonical metadata mismatch: {name}")
            del expected[name]
    return validate_arrays(result.numerical, expected, output)


def export_geography(
    result,
    frame: pd.DataFrame,
    specification: dict,
    assets: ReproductionAssets,
    directory: Path,
) -> None:
    """Export known geographic coordinates or original planar coordinates honestly.

    Georgia has explicit longitude/latitude columns. Brazil retains its original
    polygons. For the other datasets a coordinate CSV is appropriate because the
    sources do not establish a complete CRS for the fitted Cartesian coordinates.
    """
    coordinates = frame[[specification["id_column"], *specification["coordinates"]]]
    coordinates.to_csv(directory / "model_coordinates.csv", index=False)
    write_json(
        directory / "coordinate_convention.json",
        {
            "description": specification.get(
                "coordinate_convention", specification.get("coordinate_description")
            ),
            "geographic_correction": False,
            "id_column": specification["id_column"],
        },
    )
    if specification["data"] == "brazil.geojson":
        import geopandas as gpd

        geographic = gpd.read_file(assets.acquire("brazil.geojson"))
        identifiers = geographic[specification["id_column"]].astype(str).to_numpy()
        if not np.array_equal(identifiers, frame.index.to_numpy()):
            raise AssertionError("Brazil geographic feature alignment changed.")
        geographic = gpd.GeoDataFrame(
            result.local_results().reset_index(drop=True),
            geometry=geographic.geometry,
            crs=geographic.crs,
        )
        geographic.to_file(
            directory / "local_results.geojson", driver="GeoJSON", index=False
        )
    elif {"Longitud", "Latitude"}.issubset(frame.columns):
        import geopandas as gpd

        geographic = gpd.GeoDataFrame(
            result.local_results().reset_index(drop=True),
            geometry=gpd.points_from_xy(frame["Longitud"], frame["Latitude"]),
            crs=4326,
        )
        geographic.to_file(
            directory / "local_results.geojson", driver="GeoJSON", index=False
        )


def zip_outputs(directory: Path) -> Path:
    """Create one downloadable archive containing tables, evidence and metadata."""
    directory = Path(directory)
    target = directory.with_suffix(".zip")
    if target.exists():
        raise FileExistsError(f"Archive already exists: {target}")
    return Path(shutil.make_archive(str(directory), "zip", root_dir=directory))

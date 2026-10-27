"""Explicit, collision-checked serialization of aligned GWR outputs."""

import json
import platform
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np


def json_value(value: Any) -> Any:
    """Convert NumPy scalars, tuple identifiers and dates to portable JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}.")


def portable_values(value: Any) -> Any:
    """Recursively represent unavailable floating-point values as JSON null."""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: portable_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_values(item) for item in value]
    return value


def export_results(results, directory, *, overwrite: bool = False) -> list[Path]:
    """Stage exports, refuse existing targets by default, then install each file.

    Identifiers are present as JSON values and as JSON-encoded strings in NPZ.
    row_position disambiguates duplicate IDs. Geometry uses GeoPackage without
    coordinate transformation. Explicit overwrite replaces only named outputs.
    """
    from . import __version__

    destination = Path(directory)
    names = [
        "local_results.csv",
        "global_diagnostics.csv",
        "results.json",
        "results.npz",
        "summary.txt",
    ]
    if results.geometry is not None:
        names.append("local_results.gpkg")
    targets = [destination / name for name in names]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Export targets exist; use overwrite=True: {existing}")
    destination.mkdir(parents=True, exist_ok=True)
    table = results.local_results()
    encoded_ids = [
        json.dumps(
            portable_values(value),
            default=json_value,
            ensure_ascii=False,
            allow_nan=False,
        )
        for value in results.observation_ids
    ]
    metadata = {
        "configuration": results.configuration,
        "dependent": results.dependent_name,
        "variable_names": results.variable_names,
        "observation_ids": list(results.observation_ids),
        "index_names": list(results.observation_ids.names),
        "index_dtype": str(results.observation_ids.dtype),
        "scaling": results.scaling,
        "global_diagnostics": results.global_diagnostics,
        "numerical": {
            name: np.asarray(value).tolist()
            for name, value in results.numerical.items()
        },
        "numerical_metadata": {
            name: {
                "shape": list(np.shape(value)),
                "dtype": str(np.asarray(value).dtype),
            }
            for name, value in results.numerical.items()
        },
        "environment": {
            "python": platform.python_version(),
            "gpuGWR": __version__,
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "pandas": version("pandas"),
            "pyproj": version("pyproj"),
        },
        "identifier_encoding": "JSON values; CSV/NPZ/GeoPackage use JSON strings",
    }
    with tempfile.TemporaryDirectory(prefix=".gpugwr-", dir=destination) as temporary:
        staging = Path(temporary)
        csv_table = table.copy()
        csv_table["observation_id"] = encoded_ids
        csv_table.to_csv(staging / "local_results.csv", index=False)
        import pandas as pd

        pd.DataFrame([results.global_diagnostics]).to_csv(
            staging / "global_diagnostics.csv", index=False
        )
        (staging / "results.json").write_text(
            json.dumps(
                portable_values(metadata),
                indent=2,
                ensure_ascii=False,
                default=json_value,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            staging / "results.npz",
            **results.numerical,
            observation_ids_json=np.asarray(encoded_ids),
            variable_names=np.asarray(results.variable_names),
        )
        (staging / "summary.txt").write_text(results.summary() + "\n", encoding="utf-8")
        if results.geometry is not None:
            import geopandas as gpd

            geographic = gpd.GeoDataFrame(
                csv_table.reset_index(drop=True),
                geometry=results.geometry.reset_index(drop=True),
                crs=results.geometry.crs,
            )
            geographic.to_file(
                staging / "local_results.gpkg", driver="GPKG", index=False
            )
        for name, target in zip(names, targets, strict=True):
            if overwrite:
                (staging / name).replace(target)
            else:
                # Exclusive creation also prevents a race from silently overwriting.
                with target.open("xb") as output, (staging / name).open("rb") as source:
                    shutil.copyfileobj(source, output)
    return targets

"""Validate user data without dropping rows, mutating inputs, or guessing a CRS."""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .model import validate_model_inputs


@dataclass
class ModelData:
    """Owned model arrays and observation metadata, in original row order."""

    design: np.ndarray
    response: np.ndarray
    coordinates: np.ndarray
    observation_ids: pd.Index
    variable_names: list[str]
    dependent_name: str
    geometry: Any
    crs: str | None
    scaling: dict


def numeric_array(values: Any, name: str) -> np.ndarray:
    """Copy real numerical values to float64; reject strings, booleans and NaNs."""
    if isinstance(values, (pd.Series, pd.DataFrame)):
        dtypes = [values.dtype] if isinstance(values, pd.Series) else values.dtypes
        if any(
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
            for dtype in dtypes
        ):
            raise ValueError(f"{name} must contain real numeric values.")
        array = values.to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values.")
    result = array.astype(np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains missing or infinite values.")
    return result


def prepare_data(
    data: Any,
    dependent: str | int,
    independent: list[str] | list[int],
    coordinates: Any,
    intercept: bool,
    standardize: bool,
    crs: Any,
) -> ModelData:
    """Resolve columns or array indices and validate model arrays and CRS.

    NumPy data must be a two-dimensional table; selectors are integer columns.
    Coordinate arrays must have shape (n, 2). GeoDataFrames may use Point
    geometry when coordinates is None. All coordinates are Euclidean/projected.
    Standardization, when requested, uses population standard deviations.
    """
    if not isinstance(independent, (list, tuple)) or not independent:
        raise ValueError("independent must be a nonempty list of columns or indices.")
    if len(set(independent)) != len(independent) or dependent in independent:
        raise ValueError("Dependent and independent selectors must be distinct.")
    geometry = None
    metadata_crs = None
    if isinstance(data, pd.DataFrame):
        if not data.columns.is_unique:
            raise ValueError("Input column names must be unique.")
        selectors = [dependent, *independent]
        if not all(isinstance(name, str) for name in selectors):
            raise ValueError("DataFrame regression selectors must be column names.")
        missing = [name for name in selectors if name not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        response = numeric_array(data[dependent], "Dependent variable")[:, None]
        predictors = numeric_array(data[list(independent)], "Independent variables")
        observation_ids = data.index.copy()
        # Check class descriptors: an ordinary DataFrame may have columns named
        # geometry or crs, which must not be mistaken for geospatial metadata.
        if hasattr(type(data), "geometry") and hasattr(type(data), "crs"):
            geometry = data.geometry.copy(deep=True)
            metadata_crs = data.crs
        if coordinates is None:
            if geometry is None:
                raise ValueError("Supply two coordinate columns or a coordinate array.")
            if (
                geometry.isna().any()
                or geometry.is_empty.any()
                or not (geometry.geom_type == "Point").all()
            ):
                raise ValueError(
                    "Automatic coordinates require nonempty Point geometry."
                )
            coordinate_values = np.column_stack((geometry.x, geometry.y))
        elif (
            isinstance(coordinates, (list, tuple))
            and len(coordinates) == 2
            and all(isinstance(name, str) for name in coordinates)
        ):
            if any(name not in data.columns for name in coordinates):
                raise ValueError("Missing coordinate columns.")
            coordinate_values = data[list(coordinates)]
        else:
            coordinate_values = coordinates
        names = list(independent)
        dependent_name = dependent
    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("NumPy data must be a two-dimensional numerical table.")
        selectors = [dependent, *independent]
        if not all(
            isinstance(index, (int, np.integer))
            and not isinstance(index, (bool, np.bool_))
            and 0 <= index < data.shape[1]
            for index in selectors
        ):
            raise ValueError(
                "NumPy selectors must be valid nonnegative column indices."
            )
        response = numeric_array(data[:, dependent], "Dependent variable")[:, None]
        predictors = numeric_array(data[:, independent], "Independent variables")
        coordinate_values = coordinates
        observation_ids = pd.RangeIndex(len(data))
        names = [f"x{index}" for index in independent]
        dependent_name = f"column_{dependent}"
    else:
        raise TypeError("data must be a pandas/GeoPandas DataFrame or NumPy array.")
    coordinate_array = numeric_array(coordinate_values, "Coordinates")
    resolved_crs = validate_crs(crs, metadata_crs)
    if np.ptp(response) == 0:
        raise ValueError("A constant dependent variable has undefined R-squared.")
    scaling = {"standardize": standardize}
    if standardize:
        predictor_scale = predictors.std(axis=0, ddof=0)
        if np.any(predictor_scale == 0):
            raise ValueError("Cannot standardize constant independent variables.")
        scaling.update(
            {
                "predictor_mean": predictors.mean(axis=0).tolist(),
                "predictor_scale": predictor_scale.tolist(),
                "response_mean": float(response.mean()),
                "response_scale": float(response.std(ddof=0)),
            }
        )
        predictors = (predictors - predictors.mean(axis=0)) / predictor_scale
        response = (response - response.mean()) / response.std(ddof=0)
    design = predictors
    if intercept:
        design = np.column_stack((np.ones(len(predictors)), predictors))
        intercept_name = "Intercept"
        while intercept_name in names:
            intercept_name = "_" + intercept_name
        names = [intercept_name, *names]
    validate_model_inputs(design, response, coordinate_array, intercept)
    return ModelData(
        design,
        response,
        coordinate_array,
        observation_ids,
        names,
        dependent_name,
        geometry,
        resolved_crs,
        scaling,
    )


def validate_crs(explicit_crs: Any, metadata_crs: Any) -> str | None:
    """Return a projected CRS string or warn if the caller supplies no CRS.

    CRS assignment never reprojects coordinates. Geographic or conflicting
    CRS metadata raise errors; unknown coordinates require caller diligence.
    """
    if explicit_crs is None and metadata_crs is None:
        warnings.warn(
            "CRS is unavailable: Euclidean distances assume projected linear "
            "coordinates. Do not supply longitude/latitude; provide a projected CRS.",
            UserWarning,
            stacklevel=3,
        )
        return None
    from pyproj import CRS

    resolved = CRS.from_user_input(
        explicit_crs if explicit_crs is not None else metadata_crs
    )
    if metadata_crs is not None and resolved != CRS.from_user_input(metadata_crs):
        raise ValueError("Explicit CRS conflicts with GeoDataFrame CRS metadata.")
    if not resolved.is_projected:
        raise ValueError(
            "Only projected CRS coordinates are supported; reproject first."
        )
    return resolved.to_string()

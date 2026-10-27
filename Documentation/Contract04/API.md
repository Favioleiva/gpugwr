# gpuGWR 0.4.0 public API

```python
from gpuGWR import gpuGWR

model = gpuGWR(
    data, dependent, independent, coordinates=None,
    kernel="bisquare", bandwidth="auto", engine="cpu",
    fixed=False, intercept=True, criterion="AICc", bandwidth_bounds=None,
    standardize=False, crs=None, store_matrices=False, batch_size=None,
)
result = model.fit()
```

Construction copies and validates data; fit() returns GWRResults.
There is no backend substitute for engine. Core defaults and mathematics are unchanged.

| Parameter | Meaning |
|---|---|
| data | pandas/GeoPandas table or 2D NumPy array |
| dependent / independent | Response and list of predictors; integer column indices for NumPy |
| coordinates | Two column names, ordered (n,2) array, or None for point geometry |
| kernel | bisquare, gaussian, exponential |
| bandwidth | auto, adaptive integer neighbors, or fixed positive distance |
| engine | cpu, gpu, auto; explicit GPU never silently falls back |
| fixed | False: adaptive neighbors; True: distance bandwidth |
| intercept | Add a constant when True |
| criterion | AICc or CV for automatic selection |
| bandwidth_bounds | Optional lower/upper search bounds, not for specified bandwidth |
| standardize | Center and scale X and y using population SD |
| crs | Verified projected CRS; the general API rejects geographic CRS |
| store_matrices | Retain distances, weights, hat matrix, neighbors and full covariance |
| batch_size | Positive GPU working batch; None uses available-memory planning |

No missing values, silent row removal or automatic reprojection is allowed.
Designs must have full rank and defined diagnostics. Unknown CRS warns.
Coordinate arrays are positional. Identifiers come from the input table index;
notebooks assign the approved geographic ID column before fitting.

```python
print(result.summary())
local = result.local_results()
global_values = result.global_diagnostics
result.export("outputs/my_run")
```

Exports: CSV, JSON, NPZ, summary text, and GeoPackage when geometry is retained.
Existing files are refused unless overwrite=True. Local outputs include
coefficients, standard errors, t/corrected-t, predictions, residuals, local R²,
influence and collinearity. Global diagnostics include ENP, AIC/AICc/BIC,
R², residual variance and CV. Configuration records actual/requested engine,
precision and GPU execution profile.

## Spatial variability

```python
simulation = result.spatial_variability(
    500, seed=5536, engine="gpu", replica_batch_size=64,
)
simulation.export("outputs/simulation")
```

Adaptive AICc GWR is required. Every coordinate permutation reselects bandwidth.
Population SD uses ddof=0; pseudo-p is strict exceedance count / iterations,
without +1. Explicit permutations override the seed. implementation="baseline"
with engine="cpu" selects the full audit path. Results retain permutations,
bandwidths, observed/reference SD, pseudo-p and timing metadata.

## Spatial reuse

```python
from gpuGWR import GWRSpatialCache

with GWRSpatialCache(coords, bandwidth=1345, engine="gpu") as cache:
    first = model1.fit(spatial_cache=cache)
    second = model2.fit(spatial_cache=cache)
```

Reuse requires identical ordered coordinates, engine, kernel and explicit
bandwidth. Automatic selection and incompatible models are rejected.
X and y may change; inference is recalculated. Include cache preparation and
release when comparing the cost of a model sequence.

# Scientific inputs and reference provenance

Colab/assets/manifest.json records size, SHA-256, original source, license context
and model specifications. The four smaller tables and all reference archives
are byte-for-byte copies, retrieved with the pinned checkout and checked before
parsing. Original datasets and historical results remain unchanged locally.

| Dataset | Rows | Response | Predictors | Convention |
|---|---:|---|---|---|
| Georgia | 159 | PctBach | PctFB, PctBlack, PctRural | Standardized X/y; original UTM X/Y |
| Columbus | 49 | CRIME | INC, HOVAL | Original Cartesian units; bounds 10–49 |
| Baltimore | 211 | PRICE | NROOM, AGE, SQFT | Maryland grid; bounds 10–211 |
| Boston | 506 | CMEDV | RM, AGE, DIS | Supplied UTM; bounds 10–506 |
| Brazil | 5,562 | DfHRElc | Four sets below | Historical degrees; 1,345 neighbors |

All empirical models use Gaussian GWR, intercept and adaptive bisquare.
Only notebook 01 standardizes. Georgia Monte Carlo uses raw percentages,
bounds 48–159, seed 5536 and 500 permutations.

Georgia comes from libpysal commit 0d3deb5cf4a79d67ae75225247f0517e6de7b71f,
libpysal/examples/georgia/GData_utm.csv; preserve its BSD license and AreaKey.
Canonical arrays come from Contract 01's approved mgwr run.

Columbus, Baltimore and Boston derive from spData 2.3.5,
commit 6d9c2a8d087822463f6571016826fe45efa749c0, declared CC0 in its DESCRIPTION.
Contract 02 extracted the original RDA files with R 4.3.0 into these exact CSVs.
observation_id, row order and double precision values are retained.
Columbus's second AREA was renamed AREA_duplicate_1. Boston uses corrected
CMEDV and boston.utm. No imputation or row filtering occurred.
The original DESCRIPTION, extraction procedure and source/processed manifests
are included under Colab/provenance.

Original studies: Anselin (1988), Spatial Econometrics, Columbus; Dubin (1992),
Regional Science and Urban Economics 22(3), Baltimore; Harrison and Rubinfeld
(1978), Gilley and Pace (1996), Pace and Gilley (1997), Boston.
These notebooks establish numerical reproduction, not causal conclusions.

## Brazil

Common predictors: margin, stalign, Abstntn, lppdnst, lpctppy, GINI, IDHM,
HHsinpr, Ocp18ml, CoverBF. Additional party predictors:

| Model | Party predictors |
|---|---|
| model1 | PMDB, PSDB, PT |
| model2 | PMDB |
| model3 | PSDB |
| model4 | PT |

Exact predictor order is in the manifest. COD_MUN_I identifies municipalities.
Existing logged columns are not logged again. No complete authoritative units
dictionary was supplied, so units are not invented.
The original GeoJSON is 34,807,331 bytes; SHA-256:
0d36191a0849acf5f7312e2361ddeeb40b8563feda341f8c2fb43f0aaa325022.
Its pinned Gist URL is in the manifest. The approved record used an owner-supplied
copy; remote identity was not independently verified. Acquisition must match
that hash or fail. Original and derived Brazil inputs are excluded from
publication because upstream licensing remains unspecified.

Planar centroids are computed directly from EPSG:4326 geometry, then Euclidean
distances use degrees, exactly as in the historical experiment. No geographic
correction is mixed into these outputs. Brazil exports retain original polygons;
Georgia exports explicit longitude/latitude points separately from fitted X/Y.
Other datasets get coordinate CSVs rather than an invented geographic CRS.

## Canonical checks

Georgia and the three external archives retain all approved numeric arrays,
including matrices and search paths. Brazil's approved independent mgwr archives
retain 11 outputs per model: params, bse, tvalues, predy, influ, CCT, localR2,
R2, aicc, ENP, sigma2. All are checked; cache-reuse checks also compare every
retained gpuGWR numerical output. Brazil bandwidth/count are exact assertions.

Monte Carlo references are the approved Contract 03 CPU outputs, whose conventions
were checked against mgwr. Every permutation, bandwidth, SD and pseudo-p is
compared. Discrete quantities require exact equality; continuous comparisons
use abs(actual-reference) <= 1e-10 + 1e-8*abs(reference).
Failures are saved before raising. NaN/Inf and shape differences fail.

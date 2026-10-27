# gpuGWR

Scientifically validated Gaussian geographically weighted regression with
independent CPU and optional float64 CUDA engines. Principal investigator:
Favio Leiva. Original code is [MIT licensed](LICENSE).

**Version 0.4.0 is a local publication candidate.** Contract 04 prepares Colab
integration; publication and fresh Colab GPU acceptance are still pending.
The numerical estimators from completed Contracts 01–03 remain unchanged.

```bash
python -m pip install .
```

```python
from gpuGWR import gpuGWR

result = gpuGWR(
    data=df, dependent="y", independent=["x1", "x2"],
    coordinates=["east", "north"], bandwidth="auto", engine="cpu",
).fit()
print(result.summary())
result.export("outputs/my_run")
```

Provide verified projected coordinates for ordinary analysis. Choose
`engine="gpu"` for explicit CUDA execution or `engine="auto"` for a reported
capability decision. Explicit GPU requests never silently fall back to CPU.
GPU acceleration can be slower for small workloads; measure on your actual GPU.

- [Installation and bootstrap](Documentation/Contract04/INSTALLATION.md)
- [Public API, simulation and spatial reuse](Documentation/Contract04/API.md)
- [Datasets and model specifications](Documentation/Contract04/DATASETS.md)
- [Reproduction and benchmarks](Documentation/Contract04/REPRODUCIBILITY.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Reproducible Colab notebooks

Launch buttons target the intended public repository and become usable after
owner-authorized publication. Each notebook independently installs a full commit,
checks inputs, requires a real GPU, validates outputs and downloads an evidence ZIP.

| Notebook | Data | Launch |
|---|---:|---|
| [Georgia GWR](Notebooks/01_Georgia_GWR.ipynb) | 159 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpugwr/blob/main/Notebooks/01_Georgia_GWR.ipynb) |
| [Columbus GWR](Notebooks/02_Columbus_GWR.ipynb) | 49 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpugwr/blob/main/Notebooks/02_Columbus_GWR.ipynb) |
| [Baltimore GWR](Notebooks/03_Baltimore_GWR.ipynb) | 211 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpugwr/blob/main/Notebooks/03_Baltimore_GWR.ipynb) |
| [Boston GWR](Notebooks/04_Boston_GWR.ipynb) | 506 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpugwr/blob/main/Notebooks/04_Boston_GWR.ipynb) |
| [Brazil GWR](Notebooks/05_Brazil_GWR.ipynb) | 5,562; four models | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpugwr/blob/main/Notebooks/05_Brazil_GWR.ipynb) |
| [Georgia Monte Carlo](Notebooks/06_Georgia_Monte_Carlo.ipynb) | 500 iterations | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpugwr/blob/main/Notebooks/06_Georgia_Monte_Carlo.ipynb) |
| [CPU/GPU benchmarks](Notebooks/07_CPU_GPU_Benchmarks.ipynb) | Five datasets | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpugwr/blob/main/Notebooks/07_CPU_GPU_Benchmarks.ipynb) |

Brazil explicitly reproduces the historical degree-coordinate calculation;
it is not a geographically corrected alternative. Its upstream acquisition
and licensing limitations are documented. Comparisons retain the approved
`rtol=1e-8`, `atol=1e-10`; discrete decisions require exact equality.

Contract 04 remains active. Development approval does not authorize remote
publication. No performance claim is extrapolated to untested Colab GPUs.

# Installation and Colab bootstrap

Version 0.4.0 prepares the existing estimator for distribution; its numerical
implementation is unchanged from 0.3.1. Original project code is MIT licensed.
The publication candidate is local until the owner separately authorizes upload.

## CPU installation

From a reviewed checkout:

```bash
python -m pip install .
python -c "from gpuGWR import gpuGWR; print(gpuGWR.__name__)"
```

Base dependencies: NumPy >=1.26, SciPy >=1.11, pandas >=2.1, pyproj >=3.6.
CuPy and PySAL are not CPU requirements. The base package declares Python >=3.10.
Pinned notebooks require Python >=3.11 because their NumPy/pandas pins are newer.
Local validation uses Python 3.12; proposed Linux CI is not evidence until run.

## GPU installation

Install exactly one compatible CuPy family:

```bash
python -m pip install '.[cuda12]'
# OR, for compatible CUDA 13 hardware and driver:
python -m pip install '.[cuda13]'
```

The existing `[gpu]` extra remains a CUDA 13 alias. Extras cannot inspect hardware.
The notebook bootstrap reads the NVIDIA driver, supported CUDA version, GPU
compute capability, model and total/free VRAM. It selects one wheel family,
installs pinned dependencies, checks dependency consistency, then executes and
synchronizes a nontrivial float64 solve on a real CuPy device array.
The actual runtime version is recorded separately from the driver limit.

CuPy publishes separate CUDA 12 and 13 wheels; `[ctk]` installs NVIDIA runtime
components. Do not combine wheel families.
[CuPy installation documentation](https://docs.cupy.dev/en/stable/install.html).

Each notebook fetches its full 40-character package-source commit from
`https://github.com/Favioleiva/gpugwr.git` and installs that checkout. No other
notebook or manual Python-source upload is required. The package-source commit
precedes the final notebook commit to avoid a self-referential hash; both must
be published. The bootstrap runs in a subprocess, so it never replaces CuPy in
a process that has already imported it. Use a fresh interactive runtime.

Outputs include bootstrap.log, bootstrap.json, environment.json, source commit,
resolved versions, model configurations, numerical comparisons and performance
records. The final cell downloads a ZIP; also save the executed notebook.
New run directories preserve previous results.

Colab hardware, memory and runtime availability vary.
[Colab FAQ](https://research.google.com/colaboratory/faq.html).
Local CUDA tests cannot substitute for fresh Colab verification. Missing
resources, memory errors, dependency conflicts and numerical mismatches stop
execution visibly; they do not trigger CPU fallback or tolerance changes.


### Colab dependency correction (2026-11-03)

A reported Python 3.13 Colab A100 installation stopped at `pip check`: the
original pandas 3.0.2 pin conflicted with google-colab's pandas 2.2.3 requirement
and RAPIDS' pandas <2.4 constraint. The scientific stack now pins pandas 2.2.3
and includes Jedi 0.19.2 for IPython. The strict dependency check remains enabled.
No estimator, reference data, or numerical tolerance was changed.

Open the updated notebook from GitHub in a fresh GPU runtime. A saved copy of an
older notebook still pins the older installer. If the notebook detects packages
already imported at different versions, restart the Python session and rerun all
cells. Preserve the failed notebook as run01 and the retry as run02.

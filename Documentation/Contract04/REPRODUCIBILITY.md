# Reproduction and evidence boundaries

1. Open any notebook in a fresh GPU runtime.
2. Run its installation and retain the log and JSON probe.
3. Run all remaining cells, including canonical validation.
4. Download the ZIP and save the executed notebook.
5. Identify the actual hardware/software; success applies to that combination.

Notebooks pin source commits and hash data/reference assets. Scientific pins
are in Colab/requirements-cpu.txt; CuPy 14.1.1 is pinned per CUDA family.
NVIDIA component wheels and ancillary packages are resolved by pip, then exact
versions are saved. This is not a universal cross-platform lockfile. Preserve
the complete runtime environment record when reproducing a measured run.

Notebook 05 covers all four Brazil models and spatial reuse. Notebook 06 checks
500 Monte Carlo iterations. Notebook 07 measures five datasets, one cold process
and three warmed trials per engine, synchronizes CUDA and alternates engine order.
BLAS uses one thread. Model construction, transfers and cleanup are timed;
validation, data loading and exports are outside fit time. Cold subprocess total
time additionally includes imports, loading, validation and exports.

Memory evidence includes sampled whole-device usage (a lower bound on peak,
including other processes) and private CuPy pool reserved high-water measurements.
Neither is an exact total-process peak. Do not extrapolate timings to other GPUs.

Distinguish actual fresh Colab runs, local notebook-body execution, historical
reference outputs, and untested proposed combinations. Only actual Colab evidence
satisfies its acceptance criterion. Numerical mismatches must be reported without
changing references or widening tolerances.

Fresh CPU installation test:

```bash
python -m build
python -m venv .venv-cpu
.venv-cpu/bin/python -m pip install dist/*.whl
.venv-cpu/bin/python -I tests/installed_contract04_probe.py
```

Use Scripts/python.exe on Windows. The CI definition tests installation and
four empirical datasets on CPU; it does not claim a GPU or Colab runtime.

"""Install a pinned checkout and verify genuine float64 CUDA in a fresh runtime.

This script runs in a subprocess so CuPy is never replaced in a process that
already imported it. Package source must be an exact, clean Git commit.
"""

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path


def command(arguments: list[str], *, cwd: Path | None = None) -> str:
    """Run a command without a shell, returning output or propagating its failure."""
    completed = subprocess.run(
        arguments, cwd=cwd, text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def select_cuda_major(driver_cuda_major: int, compute_capability: float) -> int:
    """Select a wheel family supported by the driver and device architecture.

    CUDA 13 removed offline compilation for architectures below compute capability
    7.5. Those devices keep the CUDA 12 family even with a newer driver. The runtime
    probe, rather than this selection rule, establishes actual execution capability.
    """
    if driver_cuda_major < 12:
        raise RuntimeError("This bootstrap requires a CUDA 12-capable NVIDIA driver.")
    if compute_capability < 5.0:
        raise RuntimeError(
            "This reproduction has no supported wheel plan for this GPU."
        )
    if driver_cuda_major >= 13 and compute_capability >= 7.5:
        return 13
    return 12


def inspect_driver() -> dict:
    """Read actual NVIDIA hardware, driver limit and free memory before installation."""
    summary = command(["nvidia-smi"])
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", summary)
    if match is None:
        raise RuntimeError("Cannot establish NVIDIA driver's supported CUDA version.")
    query = command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    for line in query.splitlines():
        index, name, driver, total, free, capability = line.split(", ")
        devices.append(
            {
                "index": int(index),
                "name": name,
                "driver_version": driver,
                "vram_mib": int(total),
                "free_vram_mib": int(free),
                "compute_capability": float(capability),
            }
        )
    if not devices:
        raise RuntimeError("No NVIDIA GPU; select a Colab GPU runtime explicitly.")
    # Use the least capable visible GPU for an installation compatible with all.
    cuda_major = select_cuda_major(
        int(match[1]), min(device["compute_capability"] for device in devices)
    )
    return {
        "devices": devices,
        "driver_supported_cuda": match[0].split(":")[1].strip(),
        "selected_cuda_major": cuda_major,
        "nvidia_smi": summary,
    }


def verify_gpu() -> dict:
    """Execute and synchronize a nontrivial float64 solve on a CUDA array."""
    import cupy as cp
    import numpy as np

    matrix = cp.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=cp.float64)
    response = cp.asarray([9.0, 8.0], dtype=cp.float64)
    solution = cp.linalg.solve(matrix, response)
    cp.cuda.get_current_stream().synchronize()
    if not isinstance(solution, cp.ndarray) or solution.dtype != cp.float64:
        raise AssertionError("CUDA float64 array verification failed.")
    np.testing.assert_allclose(cp.asnumpy(solution), [2.0, 3.0], rtol=0, atol=1e-14)
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "cuda_calculation_verified": True,
        "dtype": str(solution.dtype),
        "device_id": solution.device.id,
        "cupy_version": cp.__version__,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
        "total_vram_bytes": total,
        "available_vram_bytes": free,
        "probe_solution": cp.asnumpy(solution).tolist(),
    }


def install(checkout: Path, cpu_only: bool, driver: dict) -> None:
    """Install the project and one compatible CuPy wheel family with pinned inputs."""
    pip = [sys.executable, "-m", "pip"]
    subprocess.run(
        [*pip, "install", "-r", str(checkout / "Colab/requirements-cpu.txt")],
        check=True,
    )
    subprocess.run([*pip, "install", "--no-deps", str(checkout)], check=True)
    if not cpu_only:
        installed_cupy = [
            package.metadata["Name"]
            for package in importlib.metadata.distributions()
            if package.metadata["Name"].lower()
            in {"cupy", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x"}
        ]
        if installed_cupy:
            subprocess.run([*pip, "uninstall", "-y", *installed_cupy], check=True)
        requirements = (
            checkout / f"Colab/requirements-cuda{driver['selected_cuda_major']}.txt"
        )
        subprocess.run([*pip, "install", "-r", str(requirements)], check=True)
    subprocess.run([*pip, "check"], check=True)


def main() -> None:
    """Install/probe one immutable checkout and retain success or failure evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    arguments = parser.parse_args()
    record = {"status": "FAIL", "python": sys.version, "platform": platform.platform()}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", arguments.commit):
            raise ValueError("Provide an exact 40-character Git commit, not a branch.")
        actual = command(["git", "rev-parse", "HEAD"], cwd=arguments.checkout)
        if actual != arguments.commit:
            raise ValueError("Checkout does not match the requested source commit.")
        dirty = command(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=arguments.checkout,
        )
        if dirty:
            raise ValueError("Tracked source changes would invalidate provenance.")
        record["source_commit"] = actual
        record["repository"] = "https://github.com/Favioleiva/gpugwr"
        driver = {} if arguments.cpu else inspect_driver()
        record["driver"] = driver
        if not arguments.probe_only:
            install(arguments.checkout, arguments.cpu, driver)
        from gpuGWR import gpuGWR
        from gpugwr import __version__

        record["gpuGWR_version"] = __version__
        record["public_class"] = gpuGWR.__name__
        record["gpu"] = {"status": "NOT REQUESTED"} if arguments.cpu else verify_gpu()
        record["versions"] = {
            package.metadata["Name"]: package.version
            for package in importlib.metadata.distributions()
        }
        record["status"] = "PASS"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        arguments.output.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

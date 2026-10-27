"""Transparent wall-clock and sampled device-memory measurement for notebooks."""

import importlib.metadata
import platform
import subprocess
import threading
from time import perf_counter
from typing import Callable, TypeVar

from .engines.device import gpu_capability
from .engines.notebook_device import device_memory, synchronize

ResultType = TypeVar("ResultType")


def environment() -> dict:
    """Record package versions and actual CUDA device properties without host names."""
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {},
        "gpu": gpu_capability(),
    }
    for name in (
        "gpuGWR",
        "numpy",
        "scipy",
        "pandas",
        "pyproj",
        "geopandas",
        "threadpoolctl",
    ):
        try:
            record["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record["versions"][name] = None
    if record["gpu"]["available"]:
        free, total = device_memory()
        record["gpu"].update({"available_vram_bytes": free, "total_vram_bytes": total})
    return record


class DeviceMemorySampler:
    """Sample device memory and report the observed lower bound on peak usage."""

    def __init__(self, interval: float = 0.05) -> None:
        """Configure the polling interval; no subprocess persists after finish()."""
        self.interval = interval
        self.samples: list[int] = []
        self.errors: list[str] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        """Query nvidia-smi for all visible device memory until signaled to stop."""
        while not self.stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                # Sum across devices avoids mislabeling a CUDA_VISIBLE_DEVICES index.
                self.samples.append(
                    sum(int(line) for line in completed.stdout.splitlines())
                )
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.errors.append(type(error).__name__)
                return
            self.stop.wait(self.interval)

    def finish(self) -> dict:
        """Stop sampling and return the scope, count, errors and observed maximum."""
        self.stop.set()
        self.thread.join(timeout=6)
        return {
            "sampled_peak_device_memory_mib": max(self.samples)
            if self.samples
            else None,
            "memory_samples": len(self.samples),
            "memory_sampling_interval_seconds": self.interval,
            "memory_scope": (
                "Sum over visible NVIDIA devices; includes other processes; "
                "sampled lower bound on peak"
            ),
            "memory_errors": self.errors,
        }


def measure(
    operation: Callable[[], ResultType], engine: str
) -> tuple[ResultType, dict]:
    """Time construction, computation, transfers and cleanup; exclude exports.

    CUDA synchronization is inside the timed interval. The sampler is separate
    from the timing interval, but its overhead may affect measured performance.
    """
    sampler = DeviceMemorySampler() if engine == "gpu" else None
    if sampler is not None:
        sampler.thread.start()
    started = perf_counter()
    try:
        result = operation()
        if engine == "gpu":
            synchronize()
        seconds = perf_counter() - started
    finally:
        memory = (
            sampler.finish()
            if sampler is not None
            else {
                "sampled_peak_device_memory_mib": None,
                "memory_scope": "CPU: not applicable",
            }
        )
    return result, {"engine": engine, "wall_seconds": seconds, **memory}

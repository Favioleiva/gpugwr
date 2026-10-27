"""Public gpuGWR interface; the historical gpugwr package holds the CPU code."""

from gpugwr import __version__ as __version__
from gpugwr.api import gpuGWR
from gpugwr.results import GWRResults
from gpugwr.simulation import SpatialVariabilityResults
from gpugwr.spatial_cache import GWRSpatialCache

__all__ = ["gpuGWR", "GWRResults", "GWRSpatialCache", "SpatialVariabilityResults"]

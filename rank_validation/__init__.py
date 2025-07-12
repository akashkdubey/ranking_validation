"""Top-level package for rank_validation."""

from importlib.metadata import version, PackageNotFoundError

try:                                    # Installed case
    __version__: str = version("rank_validation")
except PackageNotFoundError:            # Running from a git checkout
    __version__ = "0.0.0.dev0"

# Re-export public API
from .metrics import (
    ndcg,
    recall,
    kendall_tau,
    tau_ap,
    rbo_sim,
    METRIC_REGISTRY,
)

__all__ = [
    "__version__",
    "ndcg",
    "recall",
    "kendall_tau",
    "tau_ap",
    "rbo_sim",
    "METRIC_REGISTRY",
]

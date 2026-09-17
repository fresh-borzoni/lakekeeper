"""Python client for Lakekeeper datasets.

A dataset is a versioned collection of files. Commits move a branch, tags pin a
snapshot forever, and reading resolves a ref to the exact file list it named.
"""

from .client import LakekeeperClient
from .dataset import Dataset, DatasetFile, ImportResult, Snapshot
from .errors import LakekeeperError, CommitConflict, NotFound

__all__ = [
    "LakekeeperClient",
    "Dataset",
    "DatasetFile",
    "ImportResult",
    "Snapshot",
    "LakekeeperError",
    "CommitConflict",
    "NotFound",
]

"""
Cache state for precomputed math artifacts.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CacheState:
    spans: Tuple = ()
    decompositions: Tuple = ()
    projections: Tuple = ()

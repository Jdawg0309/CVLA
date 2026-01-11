"""
Plane data model.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PlaneData:
    """Immutable plane representation (ax + by + cz + d = 0)."""
    id: str
    equation: Tuple[float, float, float, float]
    color: Tuple[float, float, float, float]
    label: str
    visible: bool = True

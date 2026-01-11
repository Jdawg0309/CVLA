"""
Vector data model.
"""

from dataclasses import dataclass
from typing import Tuple
from uuid import uuid4
import numpy as np


@dataclass(frozen=True)
class VectorData:
    """
    Immutable vector representation.
    """
    id: str
    coords: Tuple[float, float, float]
    color: Tuple[float, float, float]
    label: str
    visible: bool = True

    @staticmethod
    def create(coords: Tuple[float, float, float],
               color: Tuple[float, float, float],
               label: str) -> 'VectorData':
        """Factory method that generates a UUID."""
        return VectorData(
            id=str(uuid4()),
            coords=coords,
            color=color,
            label=label,
            visible=True
        )

    def to_numpy(self) -> np.ndarray:
        """Convert coords to numpy array for math operations."""
        return np.array(self.coords, dtype=np.float32)

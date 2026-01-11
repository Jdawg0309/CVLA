"""
Image data model.
"""

from dataclasses import dataclass
from typing import Tuple
from uuid import uuid4
import numpy as np


@dataclass(frozen=True)
class ImageData:
    """
    Immutable image representation.
    """
    id: str
    pixels: np.ndarray
    name: str
    shape: Tuple[int, ...]
    history: Tuple[str, ...] = ()

    def __post_init__(self):
        """Ensure pixels array is a fresh copy."""
        object.__setattr__(self, "pixels", self.pixels.copy())
        object.__setattr__(self, "shape", tuple(self.pixels.shape))

    @staticmethod
    def create(pixels: np.ndarray, name: str,
               history: Tuple[str, ...] = ()) -> 'ImageData':
        """Factory method that generates UUID and copies pixels."""
        return ImageData(
            id=str(uuid4()),
            pixels=pixels.copy(),
            name=name,
            shape=tuple(pixels.shape),
            history=history
        )

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def channels(self) -> int:
        return self.shape[2] if len(self.shape) > 2 else 1

    @property
    def is_grayscale(self) -> bool:
        return self.channels == 1

    def as_matrix(self) -> np.ndarray:
        """Return grayscale matrix (copy)."""
        if self.is_grayscale:
            return self.pixels.copy() if len(self.pixels.shape) == 2 else self.pixels[:, :, 0].copy()
        return (0.299 * self.pixels[:, :, 0] +
                0.587 * self.pixels[:, :, 1] +
                0.114 * self.pixels[:, :, 2]).copy()

    def with_history(self, operation: str) -> 'ImageData':
        """Return new ImageData with operation appended to history."""
        return ImageData(
            id=self.id,
            pixels=self.pixels,
            name=self.name,
            shape=self.shape,
            history=self.history + (operation,)
        )

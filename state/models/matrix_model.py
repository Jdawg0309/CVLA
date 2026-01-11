"""
Matrix data model.
"""

from dataclasses import dataclass
from typing import Tuple, List
from uuid import uuid4
import numpy as np


@dataclass(frozen=True)
class MatrixData:
    """
    Immutable matrix representation.
    """
    id: str
    values: Tuple[Tuple[float, ...], ...]
    label: str
    visible: bool = True

    @staticmethod
    def create(values: List[List[float]], label: str) -> 'MatrixData':
        """Factory method that generates UUID and converts to tuples."""
        tuple_values = tuple(tuple(row) for row in values)
        return MatrixData(
            id=str(uuid4()),
            values=tuple_values,
            label=label,
            visible=True
        )

    @staticmethod
    def identity(size: int, label: str = "I") -> 'MatrixData':
        """Create an identity matrix."""
        values = tuple(
            tuple(1.0 if i == j else 0.0 for j in range(size))
            for i in range(size)
        )
        return MatrixData(id=str(uuid4()), values=values, label=label, visible=True)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for math operations."""
        return np.array(self.values, dtype=np.float32)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (rows, cols)."""
        if not self.values:
            return (0, 0)
        return (len(self.values), len(self.values[0]))

    def with_cell(self, row: int, col: int, value: float) -> 'MatrixData':
        """Return a new MatrixData with one cell changed."""
        new_values = list(list(r) for r in self.values)
        new_values[row][col] = value
        return MatrixData(
            id=self.id,
            values=tuple(tuple(r) for r in new_values),
            label=self.label,
            visible=self.visible
        )

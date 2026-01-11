"""
Matrix-related action definitions.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AddMatrix:
    """Add a new matrix to the scene."""
    values: Tuple[Tuple[float, ...], ...]
    label: str


@dataclass(frozen=True)
class DeleteMatrix:
    """Delete a matrix by ID."""
    id: str


@dataclass(frozen=True)
class UpdateMatrixCell:
    """Update a single cell in a matrix."""
    id: str
    row: int
    col: int
    value: float


@dataclass(frozen=True)
class SelectMatrix:
    """Select a matrix by ID."""
    id: str


@dataclass(frozen=True)
class ApplyMatrixToSelected:
    """Apply a matrix transformation to the selected vector."""
    matrix_id: str


@dataclass(frozen=True)
class ApplyMatrixToAll:
    """Apply a matrix transformation to all vectors."""
    matrix_id: str

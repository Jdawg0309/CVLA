"""
Vector-related action definitions.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class AddVector:
    """Add a new vector to the scene."""
    coords: Tuple[float, float, float]
    color: Tuple[float, float, float]
    label: str


@dataclass(frozen=True)
class DeleteVector:
    """Delete a vector by ID."""
    id: str


@dataclass(frozen=True)
class UpdateVector:
    """Update vector properties. Only non-None fields are changed."""
    id: str
    coords: Optional[Tuple[float, float, float]] = None
    color: Optional[Tuple[float, float, float]] = None
    label: Optional[str] = None
    visible: Optional[bool] = None


@dataclass(frozen=True)
class SelectVector:
    """Select a vector by ID."""
    id: str

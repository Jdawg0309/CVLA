"""
Immutable Data Models for CVLA

These dataclasses are the ONLY representation of domain data.
All are frozen (immutable) to prevent accidental mutation.

INVARIANT: UI and reducers must NEVER mutate these objects.
           Always create new instances via dataclasses.replace().
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np
from uuid import uuid4


# =============================================================================
# DOMAIN DATA MODELS (Immutable)
# =============================================================================

@dataclass(frozen=True)
class VectorData:
    """
    Immutable vector representation.

    Note: coords is a tuple, NOT a numpy array.
    This ensures true immutability.
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


@dataclass(frozen=True)
class MatrixData:
    """
    Immutable matrix representation.

    Values stored as nested tuples for true immutability.
    """
    id: str
    values: Tuple[Tuple[float, ...], ...]  # NxM matrix as nested tuples
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


@dataclass(frozen=True)
class PlaneData:
    """Immutable plane representation (ax + by + cz + d = 0)."""
    id: str
    equation: Tuple[float, float, float, float]  # (a, b, c, d)
    color: Tuple[float, float, float, float]     # RGBA
    label: str
    visible: bool = True


@dataclass(frozen=True)
class ImageData:
    """
    Immutable image representation.

    IMPORTANT: NumPy arrays cannot be truly frozen.
    We enforce immutability by:
    1. Copying the array in __post_init__
    2. Convention: NEVER mutate pixels after creation
    """
    id: str
    pixels: np.ndarray  # Treat as immutable by convention
    name: str
    shape: Tuple[int, ...]
    history: Tuple[str, ...] = ()  # Operations applied

    def __post_init__(self):
        """Ensure pixels array is a fresh copy."""
        # frozen=True prevents normal assignment, use object.__setattr__
        object.__setattr__(self, "pixels", self.pixels.copy())
        object.__setattr__(self, "shape", tuple(self.pixels.shape))

    @staticmethod
    def create(pixels: np.ndarray, name: str,
               history: Tuple[str, ...] = ()) -> 'ImageData':
        """Factory method that generates UUID and copies pixels."""
        return ImageData(
            id=str(uuid4()),
            pixels=pixels.copy(),  # Copy here
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
        # RGB to grayscale
        return (0.299 * self.pixels[:, :, 0] +
                0.587 * self.pixels[:, :, 1] +
                0.114 * self.pixels[:, :, 2]).copy()

    def with_history(self, operation: str) -> 'ImageData':
        """Return new ImageData with operation appended to history."""
        return ImageData(
            id=self.id,
            pixels=self.pixels,  # Will be copied in __post_init__
            name=self.name,
            shape=self.shape,
            history=self.history + (operation,)
        )


# =============================================================================
# EDUCATIONAL STEP MODEL
# =============================================================================

@dataclass(frozen=True)
class EducationalStep:
    """
    Represents one step in an educational pipeline.

    Combines:
    - What operation happened
    - The math involved
    - Human-readable explanation
    - Visual data to display
    """
    id: str
    title: str                          # e.g., "Apply Sobel X Kernel"
    explanation: str                    # One sentence explanation
    operation: str                      # e.g., "convolution", "transform"

    # Math details
    input_data: Optional['ImageData'] = None
    output_data: Optional['ImageData'] = None
    kernel_name: Optional[str] = None
    kernel_values: Optional[Tuple[Tuple[float, ...], ...]] = None
    transform_matrix: Optional[Tuple[Tuple[float, ...], ...]] = None

    # Position for visualization
    kernel_position: Optional[Tuple[int, int]] = None

    @staticmethod
    def create(title: str, explanation: str, operation: str,
               input_data: Optional['ImageData'] = None,
               output_data: Optional['ImageData'] = None,
               kernel_name: Optional[str] = None,
               kernel_values: Optional[np.ndarray] = None,
               transform_matrix: Optional[np.ndarray] = None) -> 'EducationalStep':
        """Factory method."""
        kv = None
        if kernel_values is not None:
            kv = tuple(tuple(row) for row in kernel_values)
        tm = None
        if transform_matrix is not None:
            tm = tuple(tuple(row) for row in transform_matrix)

        return EducationalStep(
            id=str(uuid4()),
            title=title,
            explanation=explanation,
            operation=operation,
            input_data=input_data,
            output_data=output_data,
            kernel_name=kernel_name,
            kernel_values=kv,
            transform_matrix=tm
        )

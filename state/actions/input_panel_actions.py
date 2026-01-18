"""
Input panel actions for the new UI layout.

These actions control the state of the input panel widgets.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


# Input method switching

@dataclass(frozen=True)
class SetInputMethod:
    """Switch between input methods: text, file, grid."""
    method: str  # "text", "file", "grid"


# Text input actions

@dataclass(frozen=True)
class SetTextInput:
    """Update the text input content."""
    content: str


@dataclass(frozen=True)
class ClearTextInput:
    """Clear the text input field."""
    pass


@dataclass(frozen=True)
class ParseTextInput:
    """
    Parse the current text input and update parsed_type.

    The parser will detect:
    - Vector: "1, 2, 3" or "[1, 2, 3]" or "1 2 3"
    - Matrix: "1, 2; 3, 4" or "[[1, 2], [3, 4]]" or multi-line
    """
    pass


# File input actions

@dataclass(frozen=True)
class SetFilePath:
    """Update the file path."""
    path: str


@dataclass(frozen=True)
class LoadFile:
    """Load the file at the current path."""
    pass


@dataclass(frozen=True)
class ClearFilePath:
    """Clear the file path."""
    pass


# Grid input actions

@dataclass(frozen=True)
class SetGridSize:
    """Set the grid dimensions."""
    rows: int
    cols: int


@dataclass(frozen=True)
class SetGridCell:
    """Set a single grid cell value."""
    row: int
    col: int
    value: float


@dataclass(frozen=True)
class SetGridRow:
    """Set an entire grid row."""
    row: int
    values: Tuple[float, ...]


@dataclass(frozen=True)
class SetGridColumn:
    """Set an entire grid column."""
    col: int
    values: Tuple[float, ...]


@dataclass(frozen=True)
class ClearGrid:
    """Clear all grid cells to zero."""
    pass


@dataclass(frozen=True)
class ApplyGridTemplate:
    """
    Apply a template to the grid.

    Templates:
    - "identity": Identity matrix
    - "zeros": All zeros
    - "ones": All ones
    - "random": Random values
    - "diagonal": 1s on diagonal, 0s elsewhere
    """
    template: str


@dataclass(frozen=True)
class TransposeGrid:
    """Transpose the grid (swap rows and columns)."""
    pass


# Actions that create tensors from current input

@dataclass(frozen=True)
class CreateTensorFromTextInput:
    """Create a tensor from the current text input."""
    label: str
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)


@dataclass(frozen=True)
class CreateTensorFromFileInput:
    """Create a tensor from the current file input."""
    label: str = ""  # Empty means use filename


@dataclass(frozen=True)
class CreateTensorFromGridInput:
    """
    Create a tensor from the current grid input.

    Args:
        tensor_type: "vector" (uses first row), "matrix" (uses full grid)
        label: Label for the tensor
        color: Color for the tensor
    """
    tensor_type: str  # "vector" or "matrix"
    label: str
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)

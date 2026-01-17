"""
UI input action definitions for controlled form fields.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class SetInputVector:
    """Update the vector input form fields."""
    coords: Optional[Tuple[float, float, float]] = None
    label: Optional[str] = None
    color: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class SetInputMatrixCell:
    """Update a cell in the matrix input form."""
    row: int
    col: int
    value: float


@dataclass(frozen=True)
class SetInputMatrixShape:
    """Resize the matrix input form."""
    rows: int
    cols: int


@dataclass(frozen=True)
class SetInputMatrixLabel:
    """Update the matrix input label."""
    label: str


@dataclass(frozen=True)
class SetEquationCell:
    """Update a cell in the linear equation input."""
    row: int
    col: int
    value: float


@dataclass(frozen=True)
class SetEquationCount:
    """Update the number of equations."""
    count: int


@dataclass(frozen=True)
class SetImagePath:
    """Update the image path input field."""
    path: str


@dataclass(frozen=True)
class SetSamplePattern:
    """Update the sample pattern selection."""
    pattern: str


@dataclass(frozen=True)
class SetSampleSize:
    """Update the sample size input."""
    size: int


@dataclass(frozen=True)
class SetTransformRotation:
    """Update the rotation input."""
    rotation: float


@dataclass(frozen=True)
class SetTransformScale:
    """Update the scale input."""
    scale: float


@dataclass(frozen=True)
class SetSelectedKernel:
    """Update the selected kernel."""
    kernel_name: str


@dataclass(frozen=True)
class SetImageRenderScale:
    """Update the image render scale (pixel spacing)."""
    scale: float


@dataclass(frozen=True)
class SetImageRenderMode:
    """Set image render mode ('plane' or 'height-field')."""
    mode: str


@dataclass(frozen=True)
class SetImageColorMode:
    """Set image color mode ('grayscale' or 'heatmap')."""
    mode: str


@dataclass(frozen=True)
class SetImageNormalizeMean:
    """Set the normalization mean used before preprocessing."""
    mean: float


@dataclass(frozen=True)
class SetImageNormalizeStd:
    """Set the normalization standard deviation used before preprocessing."""
    std: float


@dataclass(frozen=True)
class ToggleImageGridOverlay:
    """Toggle pixel grid overlay for the image plane."""
    pass


@dataclass(frozen=True)
class ToggleImageDownsample:
    """Toggle downsampling for large image loads."""
    pass


@dataclass(frozen=True)
class SetImagePreviewResolution:
    """Set preview/downsample resolution for image loading."""
    size: int


@dataclass(frozen=True)
class SetSelectedPixel:
    """Set the selected pixel coordinate (row, col)."""
    row: int
    col: int


@dataclass(frozen=True)
class SetInputExpression:
    """Set raw input expression for vectors/matrices."""
    expression: str

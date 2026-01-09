"""
Action Definitions for CVLA

Actions are immutable descriptions of state changes.
They do NOT perform the change — the reducer does.

RULE: Every user interaction that changes state must go through an action.
      No direct mutation of AppState is allowed.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Union


# =============================================================================
# VECTOR ACTIONS
# =============================================================================

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


# =============================================================================
# MATRIX ACTIONS
# =============================================================================

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


# =============================================================================
# IMAGE ACTIONS (Images Tab)
# =============================================================================

@dataclass(frozen=True)
class LoadImage:
    """Load an image from file path."""
    path: str
    max_size: Tuple[int, int] = (128, 128)


@dataclass(frozen=True)
class CreateSampleImage:
    """Create a sample/test image."""
    pattern: str  # 'gradient', 'checkerboard', 'circle', 'edges', 'noise'
    size: int


@dataclass(frozen=True)
class ApplyKernel:
    """Apply a convolution kernel to the current image."""
    kernel_name: str


@dataclass(frozen=True)
class ApplyTransform:
    """Apply an affine transform to the current image."""
    rotation: float  # degrees
    scale: float


@dataclass(frozen=True)
class FlipImageHorizontal:
    """Flip the current image horizontally."""
    pass


@dataclass(frozen=True)
class UseResultAsInput:
    """Use the processed image as the new input."""
    pass


@dataclass(frozen=True)
class ClearImage:
    """Clear both current and processed images."""
    pass


# =============================================================================
# PIPELINE / EDUCATIONAL STEP ACTIONS
# =============================================================================

@dataclass(frozen=True)
class StepForward:
    """Move to the next step in the educational pipeline."""
    pass


@dataclass(frozen=True)
class StepBackward:
    """Move to the previous step in the educational pipeline."""
    pass


@dataclass(frozen=True)
class JumpToStep:
    """Jump to a specific step index."""
    index: int


@dataclass(frozen=True)
class ResetPipeline:
    """Clear all pipeline steps and reset to initial state."""
    pass


# =============================================================================
# UI INPUT ACTIONS (Controlled Inputs)
# =============================================================================

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
class SetInputMatrixSize:
    """Resize the matrix input form."""
    size: int


@dataclass(frozen=True)
class SetInputMatrixLabel:
    """Update the matrix input label."""
    label: str


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


# =============================================================================
# NAVIGATION / UI STATE ACTIONS
# =============================================================================

@dataclass(frozen=True)
class SetActiveTab:
    """Switch to a different tab."""
    tab: str  # 'vectors', 'matrices', 'systems', 'images', 'visualize'


@dataclass(frozen=True)
class ToggleMatrixEditor:
    """Toggle the matrix editor visibility."""
    pass


@dataclass(frozen=True)
class ToggleMatrixValues:
    """Toggle showing matrix values in the Images tab."""
    pass


@dataclass(frozen=True)
class TogglePreview:
    """Toggle matrix preview mode."""
    pass


@dataclass(frozen=True)
class ClearSelection:
    """Clear the current selection."""
    pass


# =============================================================================
# HISTORY ACTIONS
# =============================================================================

@dataclass(frozen=True)
class Undo:
    """Undo the last action."""
    pass


@dataclass(frozen=True)
class Redo:
    """Redo the last undone action."""
    pass


# =============================================================================
# ACTION UNION TYPE
# =============================================================================

Action = Union[
    # Vector actions
    AddVector, DeleteVector, UpdateVector, SelectVector,
    # Matrix actions
    AddMatrix, DeleteMatrix, UpdateMatrixCell, SelectMatrix,
    ApplyMatrixToSelected, ApplyMatrixToAll,
    # Image actions
    LoadImage, CreateSampleImage, ApplyKernel, ApplyTransform,
    FlipImageHorizontal, UseResultAsInput, ClearImage,
    # Pipeline actions
    StepForward, StepBackward, JumpToStep, ResetPipeline,
    # UI input actions
    SetInputVector, SetInputMatrixCell, SetInputMatrixSize, SetInputMatrixLabel,
    SetImagePath, SetSamplePattern, SetSampleSize,
    SetTransformRotation, SetTransformScale, SetSelectedKernel,
    # Navigation actions
    SetActiveTab, ToggleMatrixEditor, ToggleMatrixValues, TogglePreview,
    ClearSelection,
    # History actions
    Undo, Redo,
]

"""
Action Definitions for CVLA

Actions are immutable descriptions of state changes.
They do NOT perform the change — the reducer does.
"""

from typing import Union

from state.actions.vector_actions import (
    AddVector, DeleteVector, UpdateVector, SelectVector,
    ClearAllVectors, DuplicateVector, DeselectVector,
)
from state.actions.matrix_actions import (
    AddMatrix, DeleteMatrix, UpdateMatrixCell, UpdateMatrix, SelectMatrix,
    ApplyMatrixToSelected, ApplyMatrixToAll,
    ToggleMatrixPlot,
)
from state.actions.image_actions import (
    LoadImage, CreateSampleImage, ApplyKernel, ApplyTransform,
    FlipImageHorizontal, UseResultAsInput, ClearImage,
    NormalizeImage,
    NormalizeImage,
)
from state.actions.pipeline_actions import StepForward, StepBackward, JumpToStep, ResetPipeline
from state.actions.input_actions import (
    SetInputVector, SetInputMatrixCell, SetInputMatrixSize, SetInputMatrixLabel,
    SetImagePath, SetSamplePattern, SetSampleSize,
    SetTransformRotation, SetTransformScale, SetSelectedKernel,
    SetImageRenderScale, SetImageRenderMode, SetImageColorMode, ToggleImageGridOverlay,
    ToggleImageDownsample, SetImagePreviewResolution,
    SetImageNormalizeMean, SetImageNormalizeStd,
    SetImageNormalizeMean, SetImageNormalizeStd,
    SetImageNormalizeMean, SetImageNormalizeStd,
)
from state.actions.navigation_actions import (
    SetActiveTab, ToggleMatrixEditor, ToggleMatrixValues, ToggleImageOnGrid,
    TogglePreview, ClearSelection, SetTheme, SetActiveTool,
    SetActiveImageTab,
)
from state.actions.history_actions import Undo, Redo


Action = Union[
    # Vector actions
    AddVector, DeleteVector, UpdateVector, SelectVector,
    ClearAllVectors, DuplicateVector, DeselectVector,
    # Matrix actions
    AddMatrix, DeleteMatrix, UpdateMatrixCell, UpdateMatrix, SelectMatrix,
    ApplyMatrixToSelected, ApplyMatrixToAll,
    ToggleMatrixPlot,
    # Image actions
    LoadImage, CreateSampleImage, ApplyKernel, ApplyTransform,
    FlipImageHorizontal, UseResultAsInput, ClearImage,
    # Pipeline actions
    StepForward, StepBackward, JumpToStep, ResetPipeline,
    # UI input actions
    SetInputVector, SetInputMatrixCell, SetInputMatrixSize, SetInputMatrixLabel,
    SetImagePath, SetSamplePattern, SetSampleSize,
    SetTransformRotation, SetTransformScale, SetSelectedKernel,
    SetImageRenderScale, SetImageRenderMode, SetImageColorMode, ToggleImageGridOverlay,
    ToggleImageDownsample, SetImagePreviewResolution,
    SetImageNormalizeMean, SetImageNormalizeStd,
    # Navigation actions
    SetActiveTab, ToggleMatrixEditor, ToggleMatrixValues, ToggleImageOnGrid,
    TogglePreview, ClearSelection, SetTheme, SetActiveTool,
    # History actions
    Undo, Redo,
]

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
from state.actions.pipeline_actions import StepForward, StepBackward, JumpToStep, ResetPipeline, SetPipeline
from state.actions.input_actions import (
    SetInputVector, SetInputMatrixCell, SetInputMatrixShape, SetInputMatrixLabel,
    SetEquationCell, SetEquationCount,
    SetImagePath, SetSamplePattern, SetSampleSize,
    SetTransformRotation, SetTransformScale, SetSelectedKernel,
    SetImageRenderScale, SetImageRenderMode, SetImageColorMode, ToggleImageGridOverlay,
    ToggleImageDownsample, SetImagePreviewResolution, SetSelectedPixel,
    SetImageNormalizeMean, SetImageNormalizeStd, SetInputExpression,
)
from state.actions.navigation_actions import (
    SetActiveTab, SetActiveMode, ToggleMatrixEditor, ToggleMatrixValues, ToggleImageOnGrid,
    TogglePreview, ClearSelection, SetTheme, SetActiveTool,
    SetActiveImageTab,
    SetViewPreset, SetViewUpAxis, ToggleViewGrid, ToggleViewAxes, ToggleViewLabels,
    SetViewGridSize, SetViewMajorTick, SetViewMinorTick, ToggleViewAutoRotate,
    SetViewRotationSpeed, ToggleViewCubeFaces, ToggleViewCubeCorners,
    SetViewCubicGridDensity, SetViewCubeFaceOpacity, ToggleView2D,
    ShowError, DismissError,
)
from state.actions.history_actions import Undo, Redo

# New unified tensor actions
from state.actions.tensor_actions import (
    AddTensor, DeleteTensor, UpdateTensor, SelectTensor, DeselectTensor,
    ApplyOperation, PreviewOperation, CancelPreview, ConfirmPreview,
    ClearAllTensors, DuplicateTensor,
    AddVectorTensor, AddMatrixTensor, AddImageTensor,
)

# New input panel actions
from state.actions.input_panel_actions import (
    SetInputMethod, SetTextInput, ClearTextInput, ParseTextInput,
    SetFilePath, LoadFile, ClearFilePath,
    SetGridSize, SetGridCell, SetGridRow, SetGridColumn,
    ClearGrid, ApplyGridTemplate, TransposeGrid,
    CreateTensorFromTextInput, CreateTensorFromFileInput, CreateTensorFromGridInput,
)


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
    StepForward, StepBackward, JumpToStep, ResetPipeline, SetPipeline,
    # UI input actions
    SetInputVector, SetInputMatrixCell, SetInputMatrixShape, SetInputMatrixLabel,
    SetEquationCell, SetEquationCount,
    SetImagePath, SetSamplePattern, SetSampleSize,
    SetTransformRotation, SetTransformScale, SetSelectedKernel,
    SetImageRenderScale, SetImageRenderMode, SetImageColorMode, ToggleImageGridOverlay,
    ToggleImageDownsample, SetImagePreviewResolution, SetSelectedPixel,
    SetImageNormalizeMean, SetImageNormalizeStd, SetInputExpression,
    # Navigation actions
    SetActiveTab, SetActiveMode, ToggleMatrixEditor, ToggleMatrixValues, ToggleImageOnGrid,
    TogglePreview, ClearSelection, SetTheme, SetActiveTool,
    SetActiveImageTab,
    SetViewPreset, SetViewUpAxis, ToggleViewGrid, ToggleViewAxes, ToggleViewLabels,
    SetViewGridSize, SetViewMajorTick, SetViewMinorTick, ToggleViewAutoRotate,
    SetViewRotationSpeed, ToggleViewCubeFaces, ToggleViewCubeCorners,
    SetViewCubicGridDensity, SetViewCubeFaceOpacity, ToggleView2D,
    ShowError, DismissError,
    # History actions
    Undo, Redo,
    # Tensor actions (new unified model)
    AddTensor, DeleteTensor, UpdateTensor, SelectTensor, DeselectTensor,
    ApplyOperation, PreviewOperation, CancelPreview, ConfirmPreview,
    ClearAllTensors, DuplicateTensor,
    AddVectorTensor, AddMatrixTensor, AddImageTensor,
    # Input panel actions (new UI)
    SetInputMethod, SetTextInput, ClearTextInput, ParseTextInput,
    SetFilePath, LoadFile, ClearFilePath,
    SetGridSize, SetGridCell, SetGridRow, SetGridColumn,
    ClearGrid, ApplyGridTemplate, TransposeGrid,
    CreateTensorFromTextInput, CreateTensorFromFileInput, CreateTensorFromGridInput,
]

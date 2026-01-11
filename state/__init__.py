"""
CVLA State Management

This module implements a Redux-style state management system:
- Single source of truth (AppState)
- Immutable state updates via reducer
- Actions describe what happened
- UI components read state and dispatch actions

ARCHITECTURE:
    User Interaction → Action → Reducer → New State → UI Re-render
                                   ↑
                              Pure function
                              No side effects
"""

from state.models import (
    VectorData,
    MatrixData,
    PlaneData,
    ImageData,
    EducationalStep,
    PipelineOp,
    MicroOp,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState, MAX_HISTORY, create_initial_state
    from state.selectors import (
        get_vector_by_id,
        get_matrix_by_id,
        get_selected_vector,
        get_selected_matrix,
        get_current_step,
        get_next_color,
        COLOR_PALETTE,
    )

from state.actions import (
    Action,
    # Vector actions
    AddVector, DeleteVector, UpdateVector, SelectVector,
    # Matrix actions
    AddMatrix, DeleteMatrix, UpdateMatrixCell, UpdateMatrix, SelectMatrix,
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
    SetImageRenderScale, SetImageRenderMode, SetImageColorMode, ToggleImageGridOverlay,
    ToggleImageDownsample, SetImagePreviewResolution, ToggleImageOnGrid,
    NormalizeImage, SetImageNormalizeMean, SetImageNormalizeStd,
    # Navigation actions
    SetActiveTab, ToggleMatrixEditor, ToggleMatrixValues, TogglePreview,
    ClearSelection, SetTheme, SetActiveTool, SetActiveImageTab,
    # History actions
    Undo, Redo,
)

from state.reducers import reduce

__all__ = [
    # Models
    'VectorData', 'MatrixData', 'PlaneData', 'ImageData', 'EducationalStep', 'PipelineOp', 'MicroOp',
    # State
    'AppState', 'MAX_HISTORY', 'create_initial_state',
    'get_vector_by_id', 'get_matrix_by_id',
    'get_selected_vector', 'get_selected_matrix',
    'get_current_step', 'get_next_color', 'COLOR_PALETTE',
    # Reducer
    'reduce', 'Store', 'Action',
    # All action types
    'AddVector', 'DeleteVector', 'UpdateVector', 'SelectVector',
    'AddMatrix', 'DeleteMatrix', 'UpdateMatrixCell', 'UpdateMatrix', 'SelectMatrix',
    'ApplyMatrixToSelected', 'ApplyMatrixToAll',
    'LoadImage', 'CreateSampleImage', 'ApplyKernel', 'ApplyTransform',
    'FlipImageHorizontal', 'UseResultAsInput', 'ClearImage',
    'StepForward', 'StepBackward', 'JumpToStep', 'ResetPipeline',
    'SetInputVector', 'SetInputMatrixCell', 'SetInputMatrixSize', 'SetInputMatrixLabel',
    'SetImagePath', 'SetSamplePattern', 'SetSampleSize',
    'SetTransformRotation', 'SetTransformScale', 'SetSelectedKernel',
    'SetImageRenderScale', 'SetImageRenderMode', 'SetImageColorMode', 'ToggleImageGridOverlay',
    'ToggleImageDownsample', 'SetImagePreviewResolution', 'ToggleImageOnGrid',
    'NormalizeImage', 'SetImageNormalizeMean', 'SetImageNormalizeStd',
    'SetActiveTab', 'ToggleMatrixEditor', 'ToggleMatrixValues', 'TogglePreview',
    'ClearSelection', 'SetTheme', 'SetActiveTool', 'SetActiveImageTab',
    'Undo', 'Redo',
    # Scene adapter
    'SceneAdapter', 'RendererVector', 'RendererMatrix', 'create_scene_from_state',
]


def __getattr__(name):
    if name in ("AppState", "MAX_HISTORY", "create_initial_state"):
        from state import app_state as _app_state
        return getattr(_app_state, name)
    if name in (
        "get_vector_by_id", "get_matrix_by_id",
        "get_selected_vector", "get_selected_matrix",
        "get_current_step", "get_next_color", "COLOR_PALETTE",
    ):
        from state import selectors as _state_queries
        return getattr(_state_queries, name)
    if name == "Store":
        from state.store import Store as _Store
        return _Store
    if name in ("SceneAdapter", "RendererVector", "RendererMatrix", "create_scene_from_state"):
        from engine import scene_adapter as _scene_adapter
        return getattr(_scene_adapter, name)
    raise AttributeError(f"module 'state' has no attribute {name!r}")

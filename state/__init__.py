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

from .models import (
    VectorData,
    MatrixData,
    PlaneData,
    ImageData,
    EducationalStep,
    PipelineOp,
    MicroOp,
)

from .app_state import (
    AppState,
    MAX_HISTORY,
    create_initial_state,
    get_vector_by_id,
    get_matrix_by_id,
    get_selected_vector,
    get_selected_matrix,
    get_current_step,
    get_next_color,
    COLOR_PALETTE,
)

from .actions import (
    Action,
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
    SetImageRenderScale, SetImageRenderMode, SetImageColorMode, ToggleImageGridOverlay,
    ToggleImageDownsample, SetImagePreviewResolution, ToggleImageOnGrid,
    # Navigation actions
    SetActiveTab, ToggleMatrixEditor, ToggleMatrixValues, TogglePreview,
    ClearSelection,
    # History actions
    Undo, Redo,
)

from .reducer import reduce, Store
from .scene_adapter import SceneAdapter, RendererVector, RendererMatrix, create_scene_from_state

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
    'AddMatrix', 'DeleteMatrix', 'UpdateMatrixCell', 'SelectMatrix',
    'ApplyMatrixToSelected', 'ApplyMatrixToAll',
    'LoadImage', 'CreateSampleImage', 'ApplyKernel', 'ApplyTransform',
    'FlipImageHorizontal', 'UseResultAsInput', 'ClearImage',
    'StepForward', 'StepBackward', 'JumpToStep', 'ResetPipeline',
    'SetInputVector', 'SetInputMatrixCell', 'SetInputMatrixSize', 'SetInputMatrixLabel',
    'SetImagePath', 'SetSamplePattern', 'SetSampleSize',
    'SetTransformRotation', 'SetTransformScale', 'SetSelectedKernel',
    'SetImageRenderScale', 'SetImageRenderMode', 'SetImageColorMode', 'ToggleImageGridOverlay',
    'ToggleImageDownsample', 'SetImagePreviewResolution', 'ToggleImageOnGrid',
    'SetActiveTab', 'ToggleMatrixEditor', 'ToggleMatrixValues', 'TogglePreview',
    'ClearSelection',
    'Undo', 'Redo',
    # Scene adapter
    'SceneAdapter', 'RendererVector', 'RendererMatrix', 'create_scene_from_state',
]

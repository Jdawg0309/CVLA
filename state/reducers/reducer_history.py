"""
History handling for reducer.
"""

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState
from state.actions import (
    SetInputVector, SetInputMatrixCell, SetInputMatrixShape, SetInputMatrixLabel,
    SetImagePath, SetSamplePattern, SetSampleSize,
    SetTransformRotation, SetTransformScale, SetSelectedKernel,
    SetActiveTab, SetActiveMode, ToggleMatrixEditor, ToggleMatrixValues, TogglePreview,
    ClearSelection, SetTheme, SetActiveTool,
    StepForward, StepBackward, JumpToStep,
    Undo, Redo,
    SetImageNormalizeMean, SetImageNormalizeStd, SetImageColorMode,
    SetImageRenderScale, SetImageRenderMode, SetImagePreviewResolution,
    ToggleImageGridOverlay, ToggleImageDownsample, ToggleImageOnGrid,
    SetActiveImageTab, SetSelectedPixel, SetInputExpression,
    SetViewPreset, SetViewUpAxis, ToggleViewGrid, ToggleViewAxes, ToggleViewLabels,
    SetViewGridSize, SetViewMajorTick, SetViewMinorTick, ToggleViewAutoRotate,
    SetViewRotationSpeed, ToggleViewCubeFaces, ToggleViewCubeCorners,
    SetViewCubicGridDensity, SetViewCubeFaceOpacity, ToggleView2D,
)


def reduce_history(state: "AppState", action):
    """Handle undo/redo actions."""
    if isinstance(action, Undo):
        if not state.history:
            return state
        previous = state.history[-1]
        new_history = state.history[:-1]
        new_future = (state,) + state.future
        return replace(previous, history=new_history, future=new_future)

    if isinstance(action, Redo):
        if not state.future:
            return state
        next_state = state.future[0]
        new_future = state.future[1:]
        new_history = state.history + (state,)
        return replace(next_state, history=new_history, future=new_future)

    return None


def should_record_history(action) -> bool:
    """Return True if the action should be recorded in history."""
    return not isinstance(action, (
        SetInputVector, SetInputMatrixCell, SetInputMatrixShape,
        SetInputMatrixLabel, SetImagePath, SetSamplePattern,
        SetSampleSize, SetTransformRotation, SetTransformScale,
        SetSelectedKernel, SetActiveTab, SetActiveMode, ToggleMatrixEditor,
        ToggleMatrixValues, TogglePreview, ClearSelection,
        SetTheme, SetActiveTool,
        StepForward, StepBackward, JumpToStep,
        SetImageNormalizeMean, SetImageNormalizeStd, SetImageColorMode,
        SetImageRenderScale, SetImageRenderMode, SetImagePreviewResolution,
        ToggleImageGridOverlay, ToggleImageDownsample, ToggleImageOnGrid,
        SetActiveImageTab, SetSelectedPixel, SetInputExpression,
        SetViewPreset, SetViewUpAxis, ToggleViewGrid, ToggleViewAxes, ToggleViewLabels,
        SetViewGridSize, SetViewMajorTick, SetViewMinorTick, ToggleViewAutoRotate,
        SetViewRotationSpeed, ToggleViewCubeFaces, ToggleViewCubeCorners,
        SetViewCubicGridDensity, SetViewCubeFaceOpacity, ToggleView2D,
    ))

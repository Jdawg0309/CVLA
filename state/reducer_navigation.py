"""
Navigation and view action reducers.
"""

from dataclasses import replace

from state.actions import (
    SetActiveTab, ToggleMatrixEditor, ToggleMatrixValues, TogglePreview,
    SetImageRenderScale, SetImageRenderMode, SetImageColorMode,
    ToggleImageGridOverlay, ToggleImageDownsample, SetImagePreviewResolution,
    ToggleImageOnGrid, ClearSelection,
)


def reduce_navigation(state, action):
    if isinstance(action, SetActiveTab):
        return replace(state, active_tab=action.tab)

    if isinstance(action, ToggleMatrixEditor):
        return replace(state, show_matrix_editor=not state.show_matrix_editor)

    if isinstance(action, ToggleMatrixValues):
        return replace(state, show_matrix_values=not state.show_matrix_values)

    if isinstance(action, TogglePreview):
        return replace(state, preview_enabled=not state.preview_enabled)

    if isinstance(action, SetImageRenderScale):
        return replace(state, image_render_scale=max(0.05, float(action.scale)))

    if isinstance(action, SetImageRenderMode):
        mode = action.mode if action.mode in ("plane", "height-field") else "plane"
        return replace(state, image_render_mode=mode)

    if isinstance(action, SetImageColorMode):
        mode = action.mode if action.mode in ("grayscale", "heatmap") else "grayscale"
        return replace(state, image_color_mode=mode)

    if isinstance(action, ToggleImageGridOverlay):
        return replace(state, show_image_grid_overlay=not state.show_image_grid_overlay)

    if isinstance(action, ToggleImageDownsample):
        return replace(state, image_downsample_enabled=not state.image_downsample_enabled)

    if isinstance(action, SetImagePreviewResolution):
        size = max(16, min(int(action.size), 1024))
        return replace(state, image_preview_resolution=size)

    if isinstance(action, ToggleImageOnGrid):
        return replace(state, show_image_on_grid=not state.show_image_on_grid)

    if isinstance(action, ClearSelection):
        return replace(state, selected_id=None, selected_type=None)

    return None

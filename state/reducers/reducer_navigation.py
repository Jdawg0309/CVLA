"""
Navigation and view action reducers.
"""

from dataclasses import replace

from state.actions import (
    SetActiveTab, SetActiveMode, ToggleMatrixEditor, ToggleMatrixValues, TogglePreview,
    SetImageRenderScale, SetImageRenderMode, SetImageColorMode,
    ToggleImageGridOverlay, ToggleImageDownsample, SetImagePreviewResolution,
    ToggleImageOnGrid, ClearSelection, SetTheme, SetActiveTool,
    SetSelectedPixel,
    SetActiveImageTab,
    SetViewPreset, SetViewUpAxis, ToggleViewGrid, ToggleViewAxes, ToggleViewLabels,
    SetViewGridSize, SetViewMajorTick, SetViewMinorTick, ToggleViewAutoRotate,
    SetViewRotationSpeed, ToggleViewCubeFaces, ToggleViewCubeCorners,
    ToggleViewTensorFaces, SetViewCubicGridDensity, SetViewCubeFaceOpacity, ToggleView2D,
    ShowError, DismissError,
)


def reduce_navigation(state, action):
    if isinstance(action, SetActiveTab):
        tab = action.tab if action.tab in ("vectors", "matrices", "systems", "images", "visualize") else state.active_mode
        return replace(state, active_tab=tab, active_mode=tab)

    if isinstance(action, SetActiveMode):
        mode = action.mode if action.mode in ("vectors", "matrices", "systems", "images", "visualize") else state.active_mode
        return replace(state, active_mode=mode, active_tab=mode)

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
        mode = action.mode if action.mode in ("grayscale", "heatmap", "rgb") else "grayscale"
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

    if isinstance(action, SetSelectedPixel):
        return replace(state, selected_pixel=(action.row, action.col))

    if isinstance(action, SetActiveImageTab):
        tab = action.tab if action.tab in ("raw", "preprocess") else "raw"
        return replace(state, active_image_tab=tab)

    if isinstance(action, ClearSelection):
        return replace(state, selected_id=None, selected_type=None)

    if isinstance(action, SetTheme):
        theme = action.theme if action.theme in ("dark", "light", "high-contrast") else "dark"
        return replace(state, ui_theme=theme)

    if isinstance(action, SetActiveTool):
        return replace(state, active_tool=action.tool)

    if isinstance(action, SetViewPreset):
        preset = action.preset if action.preset in ("cube", "xy", "xz", "yz") else "cube"
        if preset == "cube":
            major = state.view_major_tick
            minor = state.view_minor_tick
            if state.view_grid_mode != "cube":
                major = max(1, int(state.view_base_major_tick * state.view_cubic_grid_density))
                minor = max(1, int(state.view_base_minor_tick * state.view_cubic_grid_density))
            return replace(state,
                view_preset="cube",
                view_grid_mode="cube",
                view_grid_plane=state.view_grid_plane,
                view_major_tick=major,
                view_minor_tick=minor,
            )
        return replace(state,
            view_preset=preset,
            view_grid_mode="plane",
            view_grid_plane=preset,
        )

    if isinstance(action, SetViewUpAxis):
        axis = action.axis if action.axis in ("x", "y", "z") else state.view_up_axis
        return replace(state, view_up_axis=axis)

    if isinstance(action, ToggleViewGrid):
        return replace(state, view_show_grid=not state.view_show_grid)

    if isinstance(action, ToggleViewAxes):
        return replace(state, view_show_axes=not state.view_show_axes)

    if isinstance(action, ToggleViewLabels):
        return replace(state, view_show_labels=not state.view_show_labels)

    if isinstance(action, SetViewGridSize):
        size = max(5, min(int(action.size), 50))
        return replace(state, view_grid_size=size)

    if isinstance(action, SetViewMajorTick):
        value = max(1, min(int(action.value), 10))
        return replace(state,
            view_base_major_tick=value,
            view_major_tick=value,
        )

    if isinstance(action, SetViewMinorTick):
        value = max(1, min(int(action.value), 5))
        return replace(state,
            view_base_minor_tick=value,
            view_minor_tick=value,
        )

    if isinstance(action, ToggleViewAutoRotate):
        return replace(state, view_auto_rotate=not state.view_auto_rotate)

    if isinstance(action, SetViewRotationSpeed):
        speed = max(0.1, min(float(action.speed), 2.0))
        return replace(state, view_rotation_speed=speed)

    if isinstance(action, ToggleViewCubeFaces):
        return replace(state, view_show_cube_faces=not state.view_show_cube_faces)

    if isinstance(action, ToggleViewCubeCorners):
        return replace(state, view_show_cube_corners=not state.view_show_cube_corners)

    if isinstance(action, ToggleViewTensorFaces):
        return replace(state, view_show_tensor_faces=not state.view_show_tensor_faces)

    if isinstance(action, SetViewCubicGridDensity):
        density = max(0.5, min(float(action.density), 3.0))
        major = state.view_major_tick
        minor = state.view_minor_tick
        if state.view_grid_mode == "cube":
            major = max(1, int(state.view_base_major_tick * density))
            minor = max(1, int(state.view_base_minor_tick * density))
        return replace(state,
            view_cubic_grid_density=density,
            view_major_tick=major,
            view_minor_tick=minor,
        )

    if isinstance(action, SetViewCubeFaceOpacity):
        opacity = max(0.01, min(float(action.opacity), 0.3))
        return replace(state, view_cube_face_opacity=opacity)

    if isinstance(action, ToggleView2D):
        mode_2d = not state.view_mode_2d
        grid_mode = state.view_grid_mode
        if mode_2d:
            grid_mode = "plane"
        return replace(state,
            view_mode_2d=mode_2d,
            view_grid_mode=grid_mode,
        )

    if isinstance(action, ShowError):
        return replace(state,
            error_message=action.message,
            show_error_modal=True,
        )

    if isinstance(action, DismissError):
        return replace(state,
            error_message=None,
            show_error_modal=False,
        )

    return None

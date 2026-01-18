"""
View settings widget for visualization controls.

Provides UI for camera, grid, axes, and other visualization settings.
"""

import imgui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState

from state.actions import (
    SetViewPreset,
    SetViewUpAxis,
    ToggleViewGrid,
    ToggleViewAxes,
    ToggleViewLabels,
    ToggleView2D,
    SetViewGridSize,
    SetViewMajorTick,
    SetViewMinorTick,
    ToggleViewAutoRotate,
    SetViewRotationSpeed,
    ToggleViewCubeFaces,
    ToggleViewCubeCorners,
    ToggleViewTensorFaces,
    SetViewCubicGridDensity,
    SetViewCubeFaceOpacity,
)


class ViewSettingsWidget:
    """Widget for visualization settings."""

    PRESETS = [
        ("cube", "Cube", "3D cubic grid view"),
        ("xy", "XY Plane", "Top-down 2D view"),
        ("xz", "XZ Plane", "Front 2D view"),
        ("yz", "YZ Plane", "Side 2D view"),
    ]

    UP_AXES = [
        ("z", "Z (Standard)", "Z-up coordinate system"),
        ("y", "Y (Blender/Unity)", "Y-up coordinate system"),
        ("x", "X", "X-up coordinate system"),
    ]

    def __init__(self):
        pass

    def render(self, state: "AppState", dispatch, width: float):
        """Render view settings UI."""
        if state is None:
            imgui.text_disabled("No state available")
            return

        imgui.text("VIEW SETTINGS")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # View preset
        imgui.text("View Preset:")
        imgui.spacing()

        button_width = (width - 30) / 2
        for i, (preset_id, preset_name, tooltip) in enumerate(self.PRESETS):
            if i > 0 and i % 2 == 0:
                pass  # New row
            elif i > 0:
                imgui.same_line()

            is_active = (
                (state.view_grid_mode == "cube" and preset_id == "cube") or
                (state.view_grid_mode == "plane" and state.view_grid_plane == preset_id)
            )

            if is_active:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)

            if imgui.button(preset_name, button_width, 25):
                dispatch(SetViewPreset(preset=preset_id))

            if is_active:
                imgui.pop_style_color(1)

            if imgui.is_item_hovered():
                imgui.set_tooltip(tooltip)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Up axis
        imgui.text("Up Axis:")
        imgui.same_line(80)
        imgui.push_item_width(width - 100)

        current_axis = state.view_up_axis
        axis_names = [a[1] for a in self.UP_AXES]
        current_idx = next(
            (i for i, a in enumerate(self.UP_AXES) if a[0] == current_axis),
            0
        )

        changed, new_idx = imgui.combo("##up_axis", current_idx, axis_names)
        if changed:
            dispatch(SetViewUpAxis(axis=self.UP_AXES[new_idx][0]))
        imgui.pop_item_width()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Display toggles
        imgui.text("Display:")
        imgui.spacing()

        changed, _ = imgui.checkbox("Show Grid", state.view_show_grid)
        if changed:
            dispatch(ToggleViewGrid())

        changed, _ = imgui.checkbox("Show Axes", state.view_show_axes)
        if changed:
            dispatch(ToggleViewAxes())

        changed, _ = imgui.checkbox("Show Labels", state.view_show_labels)
        if changed:
            dispatch(ToggleViewLabels())

        changed, _ = imgui.checkbox("Show Tensor Faces", state.view_show_tensor_faces)
        if changed:
            dispatch(ToggleViewTensorFaces())

        changed, _ = imgui.checkbox("2D Mode", state.view_mode_2d)
        if changed:
            dispatch(ToggleView2D())

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Grid settings
        if state.view_show_grid:
            imgui.text("Grid Settings:")
            imgui.spacing()

            imgui.push_item_width(width - 100)

            changed, new_size = imgui.slider_int(
                "Grid Size", state.view_grid_size, 5, 50
            )
            if changed:
                dispatch(SetViewGridSize(size=new_size))

            changed, new_major = imgui.slider_int(
                "Major Ticks", state.view_major_tick, 1, 10
            )
            if changed:
                dispatch(SetViewMajorTick(value=new_major))

            changed, new_minor = imgui.slider_int(
                "Minor Ticks", state.view_minor_tick, 1, 5
            )
            if changed:
                dispatch(SetViewMinorTick(value=new_minor))

            imgui.pop_item_width()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

        # Cubic view settings
        if state.view_grid_mode == "cube":
            imgui.text("Cubic View:")
            imgui.spacing()

            changed, _ = imgui.checkbox("Auto-rotate", state.view_auto_rotate)
            if changed:
                dispatch(ToggleViewAutoRotate())

            imgui.push_item_width(width - 100)

            if state.view_auto_rotate:
                changed, new_speed = imgui.slider_float(
                    "Rotation Speed", state.view_rotation_speed, 0.1, 2.0
                )
                if changed:
                    dispatch(SetViewRotationSpeed(speed=new_speed))

            changed, _ = imgui.checkbox("Show Cube Faces", state.view_show_cube_faces)
            if changed:
                dispatch(ToggleViewCubeFaces())

            changed, _ = imgui.checkbox("Corner Indicators", state.view_show_cube_corners)
            if changed:
                dispatch(ToggleViewCubeCorners())

            changed, new_density = imgui.slider_float(
                "Grid Density", state.view_cubic_grid_density, 0.5, 3.0
            )
            if changed:
                dispatch(SetViewCubicGridDensity(density=new_density))

            if state.view_show_cube_faces:
                changed, new_opacity = imgui.slider_float(
                    "Face Opacity", state.view_cube_face_opacity, 0.01, 0.3
                )
                if changed:
                    dispatch(SetViewCubeFaceOpacity(opacity=new_opacity))

            imgui.pop_item_width()

"""
Sidebar visualization section.
"""

import imgui

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
    SetViewCubicGridDensity,
    SetViewCubeFaceOpacity,
)


def _render_visualization_options(self, state, dispatch):
    """Render visualization options section."""
    if self._section("Visualization", "👁️"):
        if state is None or dispatch is None:
            imgui.text_disabled("Visualization options unavailable (no state).")
            self._end_section()
            return
        imgui.text("View Preset:")
        imgui.same_line()

        presets = ["Cube", "XY Plane", "XZ Plane", "YZ Plane"]
        preset_values = ["cube", "xy", "xz", "yz"]

        if state.view_grid_mode == "plane":
            current_idx = preset_values.index(state.view_grid_plane)
        else:
            current_idx = preset_values.index("cube")

        imgui.push_item_width(120)
        if imgui.begin_combo("##view_preset", presets[current_idx]):
            for i, preset in enumerate(presets):
                if imgui.selectable(preset, i == current_idx)[0]:
                    preset_value = preset_values[i]
                    if dispatch:
                        dispatch(SetViewPreset(preset=preset_value))
            imgui.end_combo()
        imgui.pop_item_width()

        imgui.spacing()

        imgui.text("Up Axis:")
        imgui.same_line()

        axes = ["Z (Standard)", "Y (Blender)", "X"]
        axis_values = ["z", "y", "x"]

        current_axis_idx = axis_values.index(state.view_up_axis)

        imgui.push_item_width(120)
        if imgui.begin_combo("##up_axis", axes[current_axis_idx]):
            for i, axis in enumerate(axes):
                if imgui.selectable(axis, i == current_axis_idx)[0]:
                    if dispatch:
                        dispatch(SetViewUpAxis(axis=axis_values[i]))
            imgui.end_combo()
        imgui.pop_item_width()

        imgui.spacing()

        changed, _ = imgui.checkbox("Show Grid", state.view_show_grid)
        if changed and dispatch:
            dispatch(ToggleViewGrid())
        imgui.same_line()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle Grid")

        changed, _ = imgui.checkbox("Show Axes", state.view_show_axes)
        if changed and dispatch:
            dispatch(ToggleViewAxes())
        imgui.same_line()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle Axes")

        changed, _ = imgui.checkbox("Show Labels", state.view_show_labels)
        if changed and dispatch:
            dispatch(ToggleViewLabels())
        imgui.same_line()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle Labels")

        changed, _ = imgui.checkbox("2D Mode", state.view_mode_2d)
        if changed and dispatch:
            dispatch(ToggleView2D())
        imgui.same_line()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle 2D Mode")

        imgui.new_line()
        imgui.spacing()

        if state.view_show_grid:
            imgui.indent(10)
            imgui.text("Grid Settings:")

            imgui.push_item_width(150)
            size_changed, new_grid_size = imgui.slider_int(
                "Size", state.view_grid_size, 5, 50
            )
            if size_changed and dispatch:
                dispatch(SetViewGridSize(size=new_grid_size))

            major_changed, new_major = imgui.slider_int(
                "Major Ticks", state.view_major_tick, 1, 10
            )
            if major_changed and dispatch:
                dispatch(SetViewMajorTick(value=new_major))

            minor_changed, new_minor = imgui.slider_int(
                "Minor Ticks", state.view_minor_tick, 1, 5
            )
            if minor_changed and dispatch:
                dispatch(SetViewMinorTick(value=new_minor))

            imgui.pop_item_width()
            imgui.unindent(10)

        if state.view_grid_mode == "cube":
            imgui.separator()
            imgui.text("Cubic View Settings:")
            imgui.push_item_width(150)

            changed, _ = imgui.checkbox("Auto-rotate", state.view_auto_rotate)
            if changed and dispatch:
                dispatch(ToggleViewAutoRotate())

            rs_changed, new_speed = imgui.slider_float(
                "Rotation Speed", state.view_rotation_speed, 0.1, 2.0
            )
            if rs_changed and dispatch:
                dispatch(SetViewRotationSpeed(speed=new_speed))

            changed, _ = imgui.checkbox("Show Cube Faces", state.view_show_cube_faces)
            if changed and dispatch:
                dispatch(ToggleViewCubeFaces())

            changed, _ = imgui.checkbox("Show Corner Indicators", state.view_show_cube_corners)
            if changed and dispatch:
                dispatch(ToggleViewCubeCorners())

            gd_changed, new_density = imgui.slider_float(
                "Grid Density", state.view_cubic_grid_density, 0.5, 3.0
            )
            if gd_changed and dispatch:
                dispatch(SetViewCubicGridDensity(density=new_density))

            fo_changed, new_opacity = imgui.slider_float(
                "Face Opacity", state.view_cube_face_opacity, 0.01, 0.3
            )
            if fo_changed and dispatch:
                dispatch(SetViewCubeFaceOpacity(opacity=new_opacity))

            imgui.pop_item_width()

        self._end_section()

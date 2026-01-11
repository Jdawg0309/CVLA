"""
Sidebar visualization section.
"""

import imgui


def _render_visualization_options(self, camera, view_config):
    """Render visualization options section."""
    if self._section("Visualization", "👁️"):
        imgui.text("View Preset:")
        imgui.same_line()

        presets = ["Cube", "XY Plane", "XZ Plane", "YZ Plane"]
        preset_values = ["cube", "xy", "xz", "yz"]

        current_idx = preset_values.index(view_config.grid_plane if view_config.grid_mode == "plane" else "cube")

        imgui.push_item_width(120)
        if imgui.begin_combo("##view_preset", presets[current_idx]):
            for i, preset in enumerate(presets):
                if imgui.selectable(preset, i == current_idx)[0]:
                    preset_value = preset_values[i]
                    if preset_value == "cube":
                        view_config.update(grid_mode="cube")
                    else:
                        view_config.update(grid_mode="plane", grid_plane=preset_value)
                    camera.set_view_preset(preset_value)
            imgui.end_combo()
        imgui.pop_item_width()

        imgui.spacing()

        imgui.text("Up Axis:")
        imgui.same_line()

        axes = ["Z (Standard)", "Y (Blender)", "X"]
        axis_values = ["z", "y", "x"]

        current_axis_idx = axis_values.index(view_config.up_axis)

        imgui.push_item_width(120)
        if imgui.begin_combo("##up_axis", axes[current_axis_idx]):
            for i, axis in enumerate(axes):
                if imgui.selectable(axis, i == current_axis_idx)[0]:
                    view_config.update(up_axis=axis_values[i])
            imgui.end_combo()
        imgui.pop_item_width()

        imgui.spacing()

        toggles = [
            ("Show Grid", 'show_grid'),
            ("Show Axes", 'show_axes'),
            ("Show Labels", 'show_labels'),
            ("2D Mode", None)
        ]

        for label, attr in toggles:
            if attr is None:
                changed, mode_2d = imgui.checkbox("2D Mode", camera.mode_2d)
                if changed:
                    camera.mode_2d = mode_2d
                    if mode_2d:
                        view_config.update(grid_mode="plane")
            else:
                current_value = getattr(view_config, attr)
                changed, new_value = imgui.checkbox(label, current_value)
                if changed:
                    setattr(view_config, attr, new_value)

            imgui.same_line()
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Toggle {label.replace('Show ', '')}")

        imgui.new_line()
        imgui.spacing()

        if view_config.show_grid:
            imgui.indent(10)
            imgui.text("Grid Settings:")

            imgui.push_item_width(150)
            try:
                safe_grid_size = int(view_config.grid_size)
            except Exception:
                safe_grid_size = 20
            safe_grid_size = max(5, min(50, safe_grid_size))
            size_changed, new_grid_size = imgui.slider_int(
                "Size", safe_grid_size, 5, 50
            )
            if size_changed:
                try:
                    view_config.grid_size = int(new_grid_size)
                except Exception:
                    view_config.grid_size = safe_grid_size

            try:
                safe_major = int(view_config.major_tick)
            except Exception:
                safe_major = 5
            safe_major = max(1, min(10, safe_major))
            major_changed, new_major = imgui.slider_int(
                "Major Ticks", safe_major, 1, 10
            )
            if major_changed:
                try:
                    view_config.major_tick = int(new_major)
                except Exception:
                    view_config.major_tick = safe_major

            try:
                safe_minor = int(view_config.minor_tick)
            except Exception:
                safe_minor = 1
            safe_minor = max(1, min(5, safe_minor))
            minor_changed, new_minor = imgui.slider_int(
                "Minor Ticks", safe_minor, 1, 5
            )
            if minor_changed:
                try:
                    view_config.minor_tick = int(new_minor)
                except Exception:
                    view_config.minor_tick = safe_minor

            imgui.pop_item_width()
            imgui.unindent(10)

        if view_config.grid_mode == "cube":
            imgui.separator()
            imgui.text("Cubic View Settings:")
            imgui.push_item_width(150)

            auto_changed, view_config.auto_rotate = imgui.checkbox(
                "Auto-rotate", getattr(view_config, 'auto_rotate', False)
            )
            if auto_changed:
                pass

            try:
                rs = float(getattr(view_config, 'rotation_speed', 0.5))
            except Exception:
                rs = 0.5
            rs_changed, view_config.rotation_speed = imgui.slider_float(
                "Rotation Speed", rs, 0.1, 2.0
            )

            changed, view_config.show_cube_faces = imgui.checkbox(
                "Show Cube Faces", view_config.show_cube_faces
            )

            changed, view_config.show_cube_corners = imgui.checkbox(
                "Show Corner Indicators", view_config.show_cube_corners
            )

            try:
                gd = float(view_config.cubic_grid_density)
            except Exception:
                gd = 1.0
            gd_changed, view_config.cubic_grid_density = imgui.slider_float(
                "Grid Density", gd, 0.5, 3.0
            )
            if gd_changed:
                try:
                    view_config._setup_cubic_view()
                except Exception:
                    pass

            try:
                fo = float(view_config.cube_face_opacity)
            except Exception:
                fo = 0.05
            fo_changed, view_config.cube_face_opacity = imgui.slider_float(
                "Face Opacity", fo, 0.01, 0.3
            )
            if fo_changed:
                try:
                    for i in range(len(view_config.cube_face_colors)):
                        color = list(view_config.cube_face_colors[i])
                        color[3] = view_config.cube_face_opacity
                        view_config.cube_face_colors[i] = tuple(color)
                except Exception:
                    pass

            imgui.pop_item_width()

        self._end_section()

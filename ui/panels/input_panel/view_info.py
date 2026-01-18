"""
View info widget for the input panel when in View mode.

Shows information about the current view and camera state.
"""

import imgui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState


class ViewInfoWidget:
    """Widget showing view/camera information in View mode."""

    def __init__(self):
        pass

    def render(self, state: "AppState", dispatch, width: float):
        """Render view information."""
        if state is None:
            imgui.text_disabled("No state available")
            return

        imgui.text("Current View")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # View mode info
        if state.view_grid_mode == "cube":
            imgui.text_colored("Mode: 3D Cubic", 0.4, 0.8, 0.4, 1.0)
        else:
            plane_names = {"xy": "XY Plane", "xz": "XZ Plane", "yz": "YZ Plane"}
            plane = plane_names.get(state.view_grid_plane, state.view_grid_plane)
            imgui.text_colored(f"Mode: {plane}", 0.4, 0.8, 0.4, 1.0)

        imgui.spacing()

        # Up axis
        axis_names = {"z": "Z-up", "y": "Y-up", "x": "X-up"}
        axis = axis_names.get(state.view_up_axis, state.view_up_axis)
        imgui.text(f"Up Axis: {axis}")

        # 2D mode
        if state.view_mode_2d:
            imgui.text_colored("2D Mode: On", 0.8, 0.8, 0.4, 1.0)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Display settings summary
        imgui.text("Display:")
        imgui.spacing()

        settings = []
        if state.view_show_grid:
            settings.append("Grid")
        if state.view_show_axes:
            settings.append("Axes")
        if state.view_show_labels:
            settings.append("Labels")

        if settings:
            imgui.text_colored(", ".join(settings), 0.6, 0.6, 0.6, 1.0)
        else:
            imgui.text_colored("(none)", 0.4, 0.4, 0.4, 1.0)

        imgui.spacing()

        # Grid info
        if state.view_show_grid:
            imgui.text(f"Grid: {state.view_grid_size}x{state.view_grid_size}")

        # Cubic view info
        if state.view_grid_mode == "cube":
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text("Cubic Settings:")
            imgui.spacing()

            if state.view_auto_rotate:
                imgui.text_colored(
                    f"Auto-rotate: {state.view_rotation_speed:.1f}x",
                    0.6, 0.8, 0.6, 1.0
                )

            if state.view_show_cube_faces:
                imgui.text(f"Face opacity: {state.view_cube_face_opacity:.0%}")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Help text
        imgui.text_colored("Tip:", 0.6, 0.6, 0.6, 1.0)
        imgui.push_text_wrap_pos(width - 10)
        imgui.text_colored(
            "Use the Operations panel on the right to adjust view settings.",
            0.5, 0.5, 0.5, 1.0
        )
        imgui.pop_text_wrap_pos()

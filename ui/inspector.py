"""
Inspector panel for detailed object inspection and editing.
"""

import imgui
from core.vector import Vector3D

from ui.inspector_header import _render_header
from ui.inspector_coordinates import _render_coordinate_editor
from ui.inspector_properties import _render_properties
from ui.inspector_transform_history import _render_transform_history
from ui.inspector_computed import _render_computed_properties


class Inspector:
    def __init__(self):
        self.window_width = 320
        self.show_transform_history = True
        self.show_computed_properties = True

    def render(self, scene, selected_vector, screen_width, screen_height):
        """Render inspector panel."""
        if not selected_vector or not isinstance(selected_vector, Vector3D):
            return

        imgui.set_next_window_position(screen_width - self.window_width - 10, 30)
        imgui.set_next_window_size(self.window_width, screen_height - 40)

        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 8.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (12, 12))

        if imgui.begin("Inspector",
                      flags=imgui.WINDOW_NO_RESIZE |
                            imgui.WINDOW_NO_MOVE |
                            imgui.WINDOW_NO_TITLE_BAR):

            self._render_header(selected_vector)
            imgui.separator()

            self._render_coordinate_editor(selected_vector)
            imgui.separator()

            self._render_properties(selected_vector, scene)
            imgui.separator()

            if self.show_transform_history:
                self._render_transform_history(selected_vector)
                imgui.separator()

            if self.show_computed_properties:
                self._render_computed_properties(selected_vector, scene)

        imgui.end()
        imgui.pop_style_var(2)

    _render_header = _render_header
    _render_coordinate_editor = _render_coordinate_editor
    _render_properties = _render_properties
    _render_transform_history = _render_transform_history
    _render_computed_properties = _render_computed_properties

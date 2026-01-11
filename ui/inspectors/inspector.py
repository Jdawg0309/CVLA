"""
Inspector panel for detailed object inspection and editing.
"""

import imgui
from state.selectors import get_selected_vector

from ui.utils import set_next_window_position, set_next_window_size

_SET_WINDOW_POS_FIRST = getattr(imgui, "SET_WINDOW_POS_FIRST_USE_EVER", 0)
_SET_WINDOW_SIZE_FIRST = getattr(imgui, "SET_WINDOW_SIZE_FIRST_USE_EVER", 0)
_WINDOW_RESIZABLE = getattr(imgui, "WINDOW_RESIZABLE", 1)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 1)

from ui.inspectors.inspector_header import _render_header
from ui.inspectors.inspector_coordinates import _render_coordinate_editor
from ui.inspectors.inspector_properties import _render_properties
from ui.inspectors.inspector_transform_history import _render_transform_history
from ui.inspectors.inspector_computed import _render_computed_properties


class Inspector:
    def __init__(self):
        self.show_transform_history = True
        self.show_computed_properties = True

    def render(self, state, dispatch, rect):
        """Render inspector panel."""
        if state is None or dispatch is None:
            return

        selected_vector = get_selected_vector(state)

        x, y, width, height = rect
        set_next_window_position(x, y, cond=_SET_WINDOW_POS_FIRST)
        set_next_window_size((width, height), cond=_SET_WINDOW_SIZE_FIRST)
        imgui.set_next_window_size_constraints(
            (260, 240),
            (width + 80, height + 140),
        )

        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 2.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (10, 8))

        flags = _WINDOW_RESIZABLE | _WINDOW_NO_COLLAPSE
        if imgui.begin("Inspector", flags=flags):

            imgui.text("Inspector")
            imgui.same_line()
            if selected_vector:
                imgui.text_disabled(f"({selected_vector.label})")
            else:
                imgui.text_disabled("(No selection)")
            imgui.separator()

            if not selected_vector:
                imgui.text_wrapped("Select a vector to see detailed properties.")
            else:
                if imgui.collapsing_header("Coordinates", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    self._render_coordinate_editor(selected_vector, dispatch)
                if imgui.collapsing_header("Properties", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    self._render_properties(selected_vector, dispatch)
                if self.show_transform_history and imgui.collapsing_header("Transform History", 0)[0]:
                    self._render_transform_history(selected_vector, dispatch)
                if self.show_computed_properties and imgui.collapsing_header("Computed", 0)[0]:
                    self._render_computed_properties(selected_vector, state, dispatch)

        imgui.end()
        imgui.pop_style_var(2)

    _render_header = _render_header
    _render_coordinate_editor = _render_coordinate_editor
    _render_properties = _render_properties
    _render_transform_history = _render_transform_history
    _render_computed_properties = _render_computed_properties

"""
Sidebar main render method.

This module renders the main sidebar UI. It reads from AppState and
dispatches actions to modify state.
"""

import imgui
from ui.panels.images.images_tab import render_images_tab
from ui.utils import set_next_window_position, set_next_window_size

_SET_WINDOW_POS_FIRST = getattr(imgui, "SET_WINDOW_POS_FIRST_USE_EVER", 0)
_SET_WINDOW_SIZE_FIRST = getattr(imgui, "SET_WINDOW_SIZE_FIRST_USE_EVER", 0)
_WINDOW_RESIZABLE = getattr(imgui, "WINDOW_RESIZABLE", 1)
_WINDOW_NO_SCROLLBAR = getattr(imgui, "WINDOW_NO_SCROLLBAR", 0)


def render(self, rect, camera, view_config, state=None, dispatch=None):
    """
    Main render method.

    Args:
        rect: (x, y, width, height) for the panel
        camera: Camera object
        view_config: View configuration
        state: AppState (single source of truth)
        dispatch: Function to dispatch actions
    """
    self._state = state
    self._dispatch = dispatch

    x, y, width, height = rect
    set_next_window_position(x, y, cond=_SET_WINDOW_POS_FIRST)
    set_next_window_size((width, height), cond=_SET_WINDOW_SIZE_FIRST)
    imgui.set_next_window_size_constraints(
        (max(240, width - 40), max(220, height - 60)),
        (width + 120, height + 40),
    )

    imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 6.0)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (12, 10))

    flags = _WINDOW_RESIZABLE | _WINDOW_NO_SCROLLBAR
    if imgui.begin(
        "Operations Panel",
        flags=flags,
    ):
        active_mode = state.active_mode if state is not None else self.active_tab

        tab_label = active_mode.title() if active_mode else "Operations"
        imgui.text(tab_label)
        imgui.same_line()
        imgui.text_disabled("Panel")
        imgui.same_line(width - 90)
        if imgui.small_button("More"):
            imgui.open_popup("##ops_panel_more")

        if imgui.begin_popup("##ops_panel_more"):
            if imgui.menu_item("Export...")[0]:
                self.show_export_dialog = True
            if imgui.menu_item("Reset Filters")[0]:
                self.vector_list_filter = ""
            imgui.end_popup()

        imgui.separator()

        if active_mode == "vectors":
            self._render_vector_creation()
            if state and state.selected_type == 'vector':
                self._render_vector_operations()
            self._render_vector_list()
            self._render_matrix_operations()
            self._render_linear_systems()

        elif active_mode == "images":
            if state is not None and dispatch is not None:
                render_images_tab(state, dispatch)
            else:
                imgui.text_disabled("Images panel unavailable (no state).")

        elif active_mode == "visualize":
            self._render_visualization_options(state, dispatch)

        self._render_export_dialog()

    imgui.end()
    imgui.pop_style_var(2)

    return None

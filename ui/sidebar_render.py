"""
Sidebar main render method.

This module renders the main sidebar UI. It reads from AppState and
dispatches actions to modify state.
"""

import imgui
from ui.images_tab import render_images_tab
from state.actions import Undo, Redo, SetActiveTab


def render(self, height, camera, view_config, state=None, dispatch=None):
    """
    Main render method.

    Args:
        height: Window height
        camera: Camera object
        view_config: View configuration
        state: AppState (single source of truth)
        dispatch: Function to dispatch actions
    """
    self.scene = None
    self._state = state
    self._dispatch = dispatch

    imgui.set_next_window_position(10, 30)
    imgui.set_next_window_size(self.window_width, height - 40)

    imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 8.0)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (12, 12))

    if imgui.begin("CVLA Controls",
                  flags=imgui.WINDOW_NO_RESIZE |
                        imgui.WINDOW_NO_MOVE |
                        imgui.WINDOW_NO_TITLE_BAR):

        imgui.text_colored("CVLA - Linear Algebra Visualizer", 0.9, 0.9, 1.0, 1.0)
        imgui.text_disabled("Interactive 3D Mathematics")
        imgui.same_line(300)

        # Undo/Redo via dispatch
        can_undo = state is not None and len(state.history) > 0
        can_redo = state is not None and len(state.future) > 0

        if not can_undo:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        if imgui.button("Undo") and dispatch and can_undo:
            dispatch(Undo())
        if not can_undo:
            imgui.pop_style_var()

        imgui.same_line()

        if not can_redo:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        if imgui.button("Redo") and dispatch and can_redo:
            dispatch(Redo())
        if not can_redo:
            imgui.pop_style_var()
        imgui.separator()
        imgui.spacing()

        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 4.0)
        tab_names = ["Vectors", "Matrices", "Systems", "Images", "View"]
        tab_values = ["vectors", "matrices", "systems", "images", "visualize"]
        active_tab = state.active_tab if state is not None else self.active_tab

        for i, (name, value) in enumerate(zip(tab_names, tab_values)):
            if i > 0:
                imgui.same_line()

            if active_tab == value:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.26, 0.59, 0.98, 0.6)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.26, 0.59, 0.98, 0.8)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.26, 0.59, 0.98, 1.0)
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.18, 0.18, 0.24, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.26, 0.59, 0.98, 0.4)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.26, 0.59, 0.98, 0.6)

            if imgui.button(name, width=(self.window_width - 50) / 5):
                if state is not None and dispatch is not None:
                    dispatch(SetActiveTab(tab=value))
                else:
                    self.active_tab = value

            imgui.pop_style_color(3)

        imgui.pop_style_var(1)
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if active_tab == "vectors":
            self._render_vector_creation()
            if state and state.selected_type == 'vector':
                self._render_vector_operations()
            self._render_vector_list()

        elif active_tab == "matrices":
            self._render_matrix_operations()

        elif active_tab == "systems":
            self._render_linear_systems()

        elif active_tab == "images":
            if state is not None and dispatch is not None:
                render_images_tab(state, dispatch)
            else:
                self._render_image_operations()

        elif active_tab == "visualize":
            self._render_visualization_options(camera, view_config)

        self._render_export_dialog()

    imgui.end()
    imgui.pop_style_var(2)

    return None

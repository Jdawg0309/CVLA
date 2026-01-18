"""
Main input panel orchestrator.

Combines text, file, and grid input widgets with the tensor list.
"""

import imgui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState

from state.actions.input_panel_actions import SetInputMethod
from ui.panels.input_panel.text_input import TextInputWidget
from ui.panels.input_panel.file_input import FileInputWidget
from ui.panels.input_panel.grid_input import GridInputWidget
from ui.panels.input_panel.tensor_list import TensorListWidget

# ImGui flag constants
_WINDOW_NO_TITLE_BAR = getattr(imgui, "WINDOW_NO_TITLE_BAR", 0)
_WINDOW_NO_RESIZE = getattr(imgui, "WINDOW_NO_RESIZE", 0)
_WINDOW_NO_MOVE = getattr(imgui, "WINDOW_NO_MOVE", 0)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 0)


class InputPanel:
    """
    Left input panel for creating tensors.

    Contains:
    - Tab bar for switching between input methods (text, file, grid)
    - Active input widget
    - Tensor list showing all tensors in scene
    """

    INPUT_METHODS = [
        ("text", "Text", "Enter vectors/matrices as text"),
        ("file", "File", "Load images from files"),
        ("grid", "Grid", "Spreadsheet-style entry"),
    ]

    def __init__(self):
        self.text_widget = TextInputWidget()
        self.file_widget = FileInputWidget()
        self.grid_widget = GridInputWidget()
        self.tensor_list = TensorListWidget()

    def render(self, rect, state: "AppState", dispatch):
        """
        Render the input panel.

        Args:
            rect: (x, y, width, height) tuple
            state: Current app state
            dispatch: Action dispatch function
        """
        x, y, width, height = rect

        flags = (
            _WINDOW_NO_TITLE_BAR |
            _WINDOW_NO_RESIZE |
            _WINDOW_NO_MOVE |
            _WINDOW_NO_COLLAPSE
        )

        imgui.set_next_window_position(x, y)
        imgui.set_next_window_size(width, height)

        if imgui.begin("Input Panel", flags=flags):
            self._render_header(state, dispatch, width)
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Calculate heights
            input_height = height * 0.55  # 55% for input widget
            list_height = height * 0.35   # 35% for tensor list
            # Remaining 10% for headers/separators

            # Input method tabs and widget
            self._render_input_tabs(state, dispatch, width)
            imgui.spacing()

            # Active input widget
            imgui.begin_child("input_widget", width - 10, input_height, border=False)
            self._render_active_input(state, dispatch, width - 20)
            imgui.end_child()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Tensor list
            self.tensor_list.render(state, dispatch, width - 10, list_height)

        imgui.end()

    def _render_header(self, state: "AppState", dispatch, width: float):
        """Render panel header."""
        imgui.text("INPUT")
        imgui.same_line(width - 60)
        # Could add settings button here

    def _render_input_tabs(self, state: "AppState", dispatch, width: float):
        """Render input method tab bar."""
        active_method = state.active_input_method
        tab_width = (width - 20) / len(self.INPUT_METHODS)

        for i, (method_id, method_name, tooltip) in enumerate(self.INPUT_METHODS):
            if i > 0:
                imgui.same_line()

            is_active = active_method == method_id

            # Style active tab
            if is_active:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.35, 0.55, 0.75, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.4, 0.6, 0.8, 1.0)

            if imgui.button(method_name, tab_width, 25):
                dispatch(SetInputMethod(method=method_id))

            if is_active:
                imgui.pop_style_color(3)

            # Tooltip
            if imgui.is_item_hovered():
                imgui.set_tooltip(tooltip)

    def _render_active_input(self, state: "AppState", dispatch, width: float):
        """Render the currently active input widget."""
        active_method = state.active_input_method

        if active_method == "text":
            self.text_widget.render(state, dispatch, width)
        elif active_method == "file":
            self.file_widget.render(state, dispatch, width)
        elif active_method == "grid":
            self.grid_widget.render(state, dispatch, width)

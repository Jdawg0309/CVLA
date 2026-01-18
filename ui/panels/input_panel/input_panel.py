"""
Main input panel orchestrator.

Combines manual and file input widgets with the tensor list.
"""

import imgui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState

from state.actions.input_panel_actions import SetInputMethod
from ui.panels.input_panel.text_input import TextInputWidget
from ui.panels.input_panel.file_input import FileInputWidget
from ui.panels.input_panel.tensor_list import TensorListWidget

# ImGui flag constants
_WINDOW_NO_TITLE_BAR = getattr(imgui, "WINDOW_NO_TITLE_BAR", 0)
_WINDOW_NO_RESIZE = getattr(imgui, "WINDOW_NO_RESIZE", 0)
_WINDOW_NO_MOVE = getattr(imgui, "WINDOW_NO_MOVE", 0)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 0)
_WINDOW_ALWAYS_VERTICAL_SCROLLBAR = getattr(
    imgui, "WINDOW_ALWAYS_VERTICAL_SCROLLBAR", 0
)


class InputPanel:
    """
    Left input panel for creating tensors.

    Contains:
    - Input selector for manual or file-based input
    - Active input widget
    - Tensor list showing all tensors in scene
    """

    INPUT_METHODS = [
        ("matrix", "Manual", "Enter tensor data manually"),
        ("json", "JSON", "Load a numeric tensor from JSON"),
        ("csv", "CSV", "Load a numeric tensor from CSV"),
        ("excel", "Excel", "Load a numeric tensor from Excel"),
        ("image", "Image", "Load an image tensor from file"),
    ]

    def __init__(self):
        self.text_widget = TextInputWidget()
        self.file_widget = FileInputWidget()
        self.tensor_list = TensorListWidget()
        self._last_mode = None

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
            imgui.separator()
            imgui.spacing()

            # Get active mode
            active_mode = state.active_mode if state else "vectors"
            self._last_mode = active_mode

            if imgui.begin_child(
                "##input_scroll",
                0,
                0,
                border=False,
                flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
            ):
                # Render different content based on mode
                if active_mode == "visualize":
                    # View mode - show view info
                    self._render_view_mode_content(state, dispatch, width, height)
                else:
                    # Algebra or Vision mode - show normal input
                    self._render_standard_content(state, dispatch, width, height)

            imgui.end_child()

        imgui.end()

    def _render_header(self, state: "AppState", dispatch, width: float):
        """Render panel header."""
        # Show mode-specific header
        active_mode = state.active_mode if state else "vectors"
        mode_labels = {
            "vectors": "TENSORS",
            "visualize": "VIEW",
        }
        header = mode_labels.get(active_mode, "TENSORS")
        imgui.text(header)
        imgui.same_line(width - 60)
        # Could add settings button here

    def _render_input_selector(self, state: "AppState", dispatch, width: float):
        """Render input method selector."""
        active_method = state.active_input_method
        method_ids = [m[0] for m in self.INPUT_METHODS]
        method_labels = [m[1] for m in self.INPUT_METHODS]
        method_tooltips = {m[0]: m[2] for m in self.INPUT_METHODS}

        try:
            current_index = method_ids.index(active_method)
        except ValueError:
            current_index = 0

        imgui.text("Input Source:")
        imgui.same_line()
        imgui.push_item_width(width - 110)
        changed, new_index = imgui.combo(
            "##input_type",
            current_index,
            method_labels
        )
        imgui.pop_item_width()

        if changed:
            dispatch(SetInputMethod(method=method_ids[new_index]))

        if imgui.is_item_hovered() and active_method in method_tooltips:
            imgui.set_tooltip(method_tooltips[active_method])

    def _render_active_input(self, state: "AppState", dispatch, width: float):
        """Render the currently active input widget."""
        active_method = state.active_input_method

        if active_method == "matrix":
            self.text_widget.render(state, dispatch, width, matrix_only=False)
        elif active_method in ("json", "csv", "excel", "image"):
            self.file_widget.render(state, dispatch, width, file_type=active_method)
        else:
            self.text_widget.render(state, dispatch, width, matrix_only=False)

    def _render_standard_content(self, state: "AppState", dispatch, width: float, height: float):
        """Render standard input content (tensor mode)."""
        # Input selector and widget
        self._render_input_selector(state, dispatch, width)
        imgui.spacing()

        expanded, _ = imgui.collapsing_header(
            "Create",
            imgui.TREE_NODE_DEFAULT_OPEN
        )
        if expanded:
            self._render_active_input(state, dispatch, width - 20)

        imgui.spacing()

        expanded, _ = imgui.collapsing_header(
            "Tensors",
            imgui.TREE_NODE_DEFAULT_OPEN
        )
        if expanded:
            list_height = max(220, height * 0.35)
            self.tensor_list.render(state, dispatch, width - 10, list_height)

        imgui.spacing()

    def _render_view_mode_content(self, state: "AppState", dispatch, width: float, height: float):
        """Render View mode content - simplified, just tensor list."""
        # Info text
        imgui.text_colored("View Settings", 0.6, 0.8, 0.6, 1.0)
        imgui.spacing()
        imgui.text_wrapped("Use the Operations panel on the right to adjust view settings.")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Tensor list for reference
        expanded, _ = imgui.collapsing_header(
            "Tensors",
            imgui.TREE_NODE_DEFAULT_OPEN
        )
        if expanded:
            list_height = max(300, height * 0.5)
            self.tensor_list.render(state, dispatch, width - 10, list_height)

        imgui.spacing()

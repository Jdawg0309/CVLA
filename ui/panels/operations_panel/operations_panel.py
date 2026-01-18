"""
Main operations panel for the right side of the CVLA interface.

Orchestrates tensor info, type-specific operations, preview, and history.
"""

import imgui
from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from state.app_state import AppState

from ui.utils import set_next_window_position, set_next_window_size
from state.selectors.tensor_selectors import get_selected_tensor

from ui.panels.operations_panel.tensor_info import TensorInfoWidget
from ui.panels.operations_panel.vector_ops import VectorOpsWidget
from ui.panels.operations_panel.matrix_ops import MatrixOpsWidget
from ui.panels.operations_panel.image_ops import ImageOpsWidget
from ui.panels.operations_panel.linear_systems import LinearSystemsWidget
from ui.panels.operations_panel.operation_preview import OperationPreviewWidget
from ui.panels.operations_panel.operation_history import OperationHistoryWidget


_SET_WINDOW_POS_FIRST = getattr(imgui, "SET_WINDOW_POS_FIRST_USE_EVER", 0)
_SET_WINDOW_SIZE_FIRST = getattr(imgui, "SET_WINDOW_SIZE_FIRST_USE_EVER", 0)
_WINDOW_RESIZABLE = getattr(imgui, "WINDOW_RESIZABLE", 1)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 0)
_WINDOW_ALWAYS_VERTICAL_SCROLLBAR = getattr(
    imgui, "WINDOW_ALWAYS_VERTICAL_SCROLLBAR", 0
)


class OperationsPanel:
    """
    Right-side panel for tensor operations and visualization.

    Shows:
    - Selected tensor info
    - Type-specific operations (vector/matrix/image)
    - Operation preview (before/after)
    - Operation history timeline
    """

    def __init__(self):
        self.tensor_info = TensorInfoWidget()
        self.vector_ops = VectorOpsWidget()
        self.matrix_ops = MatrixOpsWidget()
        self.image_ops = ImageOpsWidget()
        self.linear_systems = LinearSystemsWidget()
        self.preview = OperationPreviewWidget()
        self.history = OperationHistoryWidget()

        # Panel state
        self._active_section = "operations"  # "operations", "preview", "history"
        self._show_preview = True
        self._show_history = True

    def render(
        self,
        rect: tuple,
        state: "AppState",
        dispatch: Callable,
    ):
        """
        Render the operations panel.

        Args:
            rect: (x, y, width, height) for panel position
            state: Current AppState
            dispatch: Function to dispatch actions
        """
        x, y, width, height = rect

        set_next_window_position(x, y, cond=_SET_WINDOW_POS_FIRST)
        set_next_window_size((width, height), cond=_SET_WINDOW_SIZE_FIRST)
        imgui.set_next_window_size_constraints(
            (280, 300),
            (width + 100, height + 200),
        )

        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 4.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (12, 10))

        flags = _WINDOW_RESIZABLE | _WINDOW_NO_COLLAPSE

        if imgui.begin("Operations", flags=flags):
            # Get selected tensor
            selected = get_selected_tensor(state) if state else None

            # Panel header
            self._render_header(selected, width)

            imgui.separator()
            imgui.spacing()

            if imgui.begin_tab_bar("##ops_tabs"):
                if imgui.begin_tab_item("Vector Ops")[0]:
                    if imgui.begin_child(
                        "##vector_ops_content",
                        0, 0,
                        border=False,
                        flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
                    ):
                        if selected is None or selected.tensor_type != "vector":
                            self._render_no_selection(width)
                        else:
                            self.tensor_info.render(selected, state, dispatch, width - 30)
                            imgui.spacing()
                            imgui.separator()
                            imgui.spacing()
                            self.vector_ops.render(selected, state, dispatch, width - 30)
                    imgui.end_child()
                    imgui.end_tab_item()

                if imgui.begin_tab_item("Matrix Ops")[0]:
                    if imgui.begin_child(
                        "##matrix_ops_content",
                        0, 0,
                        border=False,
                        flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
                    ):
                        if selected is None or selected.tensor_type != "matrix":
                            self._render_no_selection(width)
                        else:
                            self.tensor_info.render(selected, state, dispatch, width - 30)
                            imgui.spacing()
                            imgui.separator()
                            imgui.spacing()
                            self.matrix_ops.render(selected, state, dispatch, width - 30)
                    imgui.end_child()
                    imgui.end_tab_item()

                if imgui.begin_tab_item("Image Ops")[0]:
                    if imgui.begin_child(
                        "##image_ops_content",
                        0, 0,
                        border=False,
                        flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
                    ):
                        if selected is None or selected.tensor_type != "image":
                            self._render_no_selection(width)
                        else:
                            self.tensor_info.render(selected, state, dispatch, width - 30)
                            imgui.spacing()
                            imgui.separator()
                            imgui.spacing()
                            self.image_ops.render(selected, state, dispatch, width - 30)
                    imgui.end_child()
                    imgui.end_tab_item()

                if imgui.begin_tab_item("Linear Systems")[0]:
                    if imgui.begin_child(
                        "##linear_systems_content",
                        0, 0,
                        border=False,
                        flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
                    ):
                        self.linear_systems.render(state, dispatch, width - 30, selected)
                    imgui.end_child()
                    imgui.end_tab_item()

                if selected is not None and self._show_preview and imgui.begin_tab_item("Preview")[0]:
                    if imgui.begin_child(
                        "##preview_content",
                        0, 0,
                        border=False,
                        flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
                    ):
                        self.preview.render(selected, state, dispatch, width - 30)
                    imgui.end_child()
                    imgui.end_tab_item()

                if selected is not None and self._show_history and imgui.begin_tab_item("History")[0]:
                    if imgui.begin_child(
                        "##history_content",
                        0, 0,
                        border=False,
                        flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
                    ):
                        self.history.render(selected, state, dispatch, width - 30)
                    imgui.end_child()
                    imgui.end_tab_item()

                imgui.end_tab_bar()

        imgui.end()
        imgui.pop_style_var(2)

    def _render_header(self, selected, width: float):
        """Render the panel header."""
        imgui.text("Operations")
        imgui.same_line()

        if selected:
            tensor_type = selected.tensor_type.upper()
            type_colors = {
                "vector": (0.2, 0.6, 0.8, 1.0),
                "matrix": (0.8, 0.6, 0.2, 1.0),
                "image": (0.6, 0.8, 0.2, 1.0),
            }
            color = type_colors.get(selected.tensor_type, (0.7, 0.7, 0.7, 1.0))
            imgui.text_colored(f"[{tensor_type}]", *color)
            imgui.same_line()
            imgui.text_disabled(f"({selected.label})")
        else:
            imgui.text_disabled("(No selection)")

        # Options menu
        imgui.same_line(width - 60)
        if imgui.small_button("Options"):
            imgui.open_popup("##ops_options")

        if imgui.begin_popup("##ops_options"):
            _, self._show_preview = imgui.checkbox(
                "Show Preview", self._show_preview
            )
            _, self._show_history = imgui.checkbox(
                "Show History", self._show_history
            )
            imgui.separator()
            if imgui.menu_item("Reset Layout")[0]:
                self._show_preview = True
                self._show_history = True
            imgui.end_popup()

    def _render_no_selection(self, width: float):
        """Render content when nothing is selected."""
        imgui.spacing()
        imgui.spacing()

        # Center the message
        text = "Select a tensor to see operations"
        text_width = imgui.calc_text_size(text)[0]
        imgui.set_cursor_pos_x((width - text_width) / 2)
        imgui.text_colored(text, 0.5, 0.5, 0.5, 1.0)

        imgui.spacing()
        imgui.spacing()

        # Hint text
        hint_width = width - 40
        imgui.set_cursor_pos_x(20)
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + hint_width)
        imgui.text_colored(
            "Create tensors using the Input panel on the left, "
            "then select one to apply operations.",
            0.4, 0.4, 0.4, 1.0
        )
        imgui.pop_text_wrap_pos()

        imgui.spacing()
        imgui.spacing()

        # Quick create buttons
        imgui.separator()
        imgui.spacing()
        imgui.text_disabled("Quick Create:")
        imgui.spacing()

        btn_width = (width - 50) / 3
        if imgui.button("Vector", btn_width, 25):
            # Would switch to input panel and set to vector mode
            pass
        imgui.same_line()
        if imgui.button("Matrix", btn_width, 25):
            pass
        imgui.same_line()
        if imgui.button("Image", btn_width, 25):
            pass

    def _render_type_ops(self, tensor, state, dispatch, width: float):
        """Render type-specific operations based on tensor type."""
        tensor_type = tensor.tensor_type if tensor else None

        if tensor_type == "vector":
            self.vector_ops.render(tensor, state, dispatch, width)
        elif tensor_type == "matrix":
            self.matrix_ops.render(tensor, state, dispatch, width)
        elif tensor_type == "image":
            self.image_ops.render(tensor, state, dispatch, width)
        else:
            imgui.text_colored(
                f"Unknown tensor type: {tensor_type}",
                0.8, 0.4, 0.4, 1.0
            )

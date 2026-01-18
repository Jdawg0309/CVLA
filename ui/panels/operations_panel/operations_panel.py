"""
Main operations panel for the right side of the CVLA interface.

Orchestrates tensor info, type-specific operations, and preview.
"""

import imgui
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from state.app_state import AppState

from ui.utils import set_next_window_position, set_next_window_size
from state.selectors.tensor_selectors import get_selected_tensor
from state.models.tensor_model import TensorDType

from ui.panels.operations_panel.tensor_info import TensorInfoWidget
from ui.panels.operations_panel.vector_ops import VectorOpsWidget
from ui.panels.operations_panel.matrix_ops import MatrixOpsWidget
from ui.panels.operations_panel.image_ops import ImageOpsWidget
from ui.panels.operations_panel.linear_systems import LinearSystemsWidget
from ui.panels.operations_panel.operation_preview import OperationPreviewWidget
from ui.panels.operations_panel.view_settings import ViewSettingsWidget


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
    """

    def __init__(self):
        self.tensor_info = TensorInfoWidget()
        self.vector_ops = VectorOpsWidget()
        self.matrix_ops = MatrixOpsWidget()
        self.image_ops = ImageOpsWidget()
        self.linear_systems = LinearSystemsWidget()
        self.preview = OperationPreviewWidget()
        self.view_settings = ViewSettingsWidget()

        # Panel state
        self._show_preview = True

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
            # Get selected tensor and active mode
            selected = get_selected_tensor(state) if state else None
            active_mode = state.active_mode if state else "vectors"

            # Panel header
            self._render_header(selected, width)

            imgui.separator()
            imgui.spacing()

            # Render different content based on active mode
            if active_mode == "visualize":
                # View mode - show visualization settings
                self._render_view_mode(state, dispatch, width)
            else:
                # Tensor mode (rank/shape-driven)
                self._render_tensor_mode(selected, state, dispatch, width)

        imgui.end()
        imgui.pop_style_var(2)

    def _render_header(self, selected, width: float):
        """Render the panel header."""
        imgui.text("Operations")
        imgui.same_line()

        if selected:
            type_colors = {
                "r1": (0.2, 0.6, 0.8, 1.0),
                "r2": (0.8, 0.6, 0.2, 1.0),
                "r3": (0.7, 0.7, 0.7, 1.0),
            }
            if selected.rank == 1:
                rank_label = "RANK-1"
                color = type_colors["r1"]
            elif selected.rank == 2:
                rank_label = "RANK-2"
                color = type_colors["r2"]
            else:
                rank_label = f"RANK-{selected.rank}"
                color = type_colors["r3"]
            imgui.text_colored(f"[{rank_label}]", *color)
            imgui.same_line()
            imgui.text_disabled(f"({selected.label})")
            if selected.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
                imgui.same_line()
                imgui.text_disabled(f"[{selected.dtype.value}]")
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
            imgui.separator()
            if imgui.menu_item("Reset Layout")[0]:
                self._show_preview = True
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

    def _render_type_ops(self, tensor, state, dispatch, width: float):
        """Render type-specific operations based on tensor type."""
        if tensor and tensor.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
            self.image_ops.render(tensor, state, dispatch, width)
        elif tensor and tensor.rank == 1:
            self.vector_ops.render(tensor, state, dispatch, width)
        elif tensor and tensor.rank == 2:
            self.matrix_ops.render(tensor, state, dispatch, width)
        else:
            imgui.text_colored(
                "Unknown tensor type",
                0.8, 0.4, 0.4, 1.0
            )

    def _render_view_mode(self, state, dispatch, width: float):
        """Render View mode content - visualization settings."""
        if imgui.begin_child(
            "##view_settings_content",
            0, 0,
            border=False,
            flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
        ):
            self.view_settings.render(state, dispatch, width - 30)
        imgui.end_child()

    def _render_tensor_mode(self, selected, state, dispatch, width: float):
        """Render tensor operations using rank/shape logic."""
        if imgui.begin_child(
            "##tensor_ops_content",
            0, 0,
            border=False,
            flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
        ):
            if selected is None:
                self._render_no_selection(width)
                imgui.end_child()
                return

            self.tensor_info.render(selected, state, dispatch, width - 30)
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if selected.rank == 1:
                self.vector_ops.render(selected, state, dispatch, width - 30)
            elif selected.rank == 2:
                self.matrix_ops.render(selected, state, dispatch, width - 30)
                imgui.spacing()
                imgui.separator()
                imgui.spacing()
                self.linear_systems.render(state, dispatch, width - 30, selected)
            else:
                imgui.text_colored(
                    f"No rank-{selected.rank} ops available yet.",
                    0.6, 0.6, 0.6, 1.0
                )

            if selected.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
                imgui.spacing()
                imgui.separator()
                imgui.spacing()
                self.image_ops.render(selected, state, dispatch, width - 30)

            if self._show_preview:
                imgui.spacing()
                imgui.separator()
                imgui.spacing()
                self.preview.render(selected, state, dispatch, width - 30)

        imgui.end_child()

    def _render_image_hint(self, width: float):
        """Render hint for Vision mode when no image is selected."""
        imgui.spacing()
        imgui.spacing()

        text = "Select a tensor with image dtype to see image ops"
        text_width = imgui.calc_text_size(text)[0]
        imgui.set_cursor_pos_x((width - text_width) / 2)
        imgui.text_colored(text, 0.5, 0.5, 0.5, 1.0)

        imgui.spacing()
        imgui.spacing()

        hint_width = width - 40
        imgui.set_cursor_pos_x(20)
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + hint_width)
        imgui.text_colored(
            "Use the Input panel to load an image file.",
            0.4, 0.4, 0.4, 1.0
        )
        imgui.pop_text_wrap_pos()

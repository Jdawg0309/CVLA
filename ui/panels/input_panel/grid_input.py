"""
Grid input widget for the input panel.

Provides a spreadsheet-style editor for matrices and vectors.
"""

import imgui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState

from state.actions.input_panel_actions import (
    SetGridSize, SetGridCell, ClearGrid, ApplyGridTemplate,
    TransposeGrid, CreateTensorFromGridInput
)
from state.selectors import get_next_color


class GridInputWidget:
    """Widget for grid/spreadsheet-based tensor input."""

    TEMPLATES = [
        ("zeros", "Zeros"),
        ("ones", "Ones"),
        ("identity", "Identity"),
        ("diagonal", "Diagonal (1,2,3...)"),
        ("random", "Random"),
    ]

    MAX_ROWS = 10
    MAX_COLS = 10
    MIN_SIZE = 1

    def __init__(self):
        self._label_buffer = ""
        self._cell_buffers = {}  # (row, col) -> str buffer

    def render(self, state: "AppState", dispatch, width: float, matrix_only: bool = False):
        """Render the grid input widget."""
        rows = state.input_grid_rows
        cols = state.input_grid_cols
        cells = state.input_grid_cells

        # Size controls
        imgui.text("Grid size:")
        imgui.same_line(80)

        imgui.push_item_width(60)
        changed_rows, new_rows = imgui.input_int("##rows", rows, 1, 1)
        imgui.pop_item_width()

        imgui.same_line()
        imgui.text("x")
        imgui.same_line()

        imgui.push_item_width(60)
        changed_cols, new_cols = imgui.input_int("##cols", cols, 1, 1)
        imgui.pop_item_width()

        if changed_rows or changed_cols:
            new_rows = max(self.MIN_SIZE, min(self.MAX_ROWS, new_rows))
            new_cols = max(self.MIN_SIZE, min(self.MAX_COLS, new_cols))
            dispatch(SetGridSize(rows=new_rows, cols=new_cols))
            self._cell_buffers.clear()

        imgui.spacing()

        # Template buttons
        imgui.text("Templates:")
        imgui.same_line(80)
        for i, (template_id, template_name) in enumerate(self.TEMPLATES):
            if i > 0:
                imgui.same_line()
            if imgui.small_button(template_name):
                dispatch(ApplyGridTemplate(template=template_id))
                self._cell_buffers.clear()

        imgui.spacing()

        # Grid editor
        cell_width = min(60, (width - 40) / max(cols, 1))
        cell_height = 25

        imgui.begin_child(
            "grid_editor",
            width - 10,
            min(rows * cell_height + 20, 200),
            border=True
        )

        for r in range(rows):
            for c in range(cols):
                if c > 0:
                    imgui.same_line()

                # Get current value
                current_value = 0.0
                if r < len(cells) and c < len(cells[r]):
                    current_value = cells[r][c]

                # Get or create buffer
                key = (r, c)
                if key not in self._cell_buffers:
                    self._cell_buffers[key] = f"{current_value:.3g}"

                # Render cell
                imgui.push_item_width(cell_width - 4)
                changed, new_val = imgui.input_text(
                    f"##cell_{r}_{c}",
                    self._cell_buffers[key],
                    32
                )
                imgui.pop_item_width()

                if changed:
                    self._cell_buffers[key] = new_val
                    try:
                        float_val = float(new_val) if new_val.strip() else 0.0
                        dispatch(SetGridCell(row=r, col=c, value=float_val))
                    except ValueError:
                        pass

        imgui.end_child()

        imgui.spacing()

        # Grid operations
        button_width = (width - 30) / 3

        if imgui.button("Clear", button_width, 0):
            dispatch(ClearGrid())
            self._cell_buffers.clear()

        imgui.same_line()

        if imgui.button("Transpose", button_width, 0):
            dispatch(TransposeGrid())
            self._cell_buffers.clear()

        imgui.same_line()

        if imgui.button("Refresh", button_width, 0):
            self._cell_buffers.clear()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Label input
        imgui.text("Label:")
        imgui.same_line()
        imgui.push_item_width(width - 80)
        _, self._label_buffer = imgui.input_text(
            "##grid_label",
            self._label_buffer,
            256
        )
        imgui.pop_item_width()

        imgui.spacing()

        # Create tensor buttons
        if matrix_only:
            if imgui.button("Create Matrix", width - 20, 30):
                label = self._label_buffer.strip() or self._generate_matrix_label(state)
                color, _ = get_next_color(state)
                dispatch(CreateTensorFromGridInput(
                    tensor_type="matrix",
                    label=label,
                    color=color
                ))
                self._label_buffer = ""
        else:
            half_width = (width - 30) / 2
            if imgui.button("Create Vector", half_width, 30):
                label = self._label_buffer.strip() or self._generate_vector_label(state)
                color, _ = get_next_color(state)
                dispatch(CreateTensorFromGridInput(
                    tensor_type="vector",
                    label=label,
                    color=color
                ))
                self._label_buffer = ""

            imgui.same_line()

            if imgui.button("Create Matrix", half_width, 30):
                label = self._label_buffer.strip() or self._generate_matrix_label(state)
                color, _ = get_next_color(state)
                dispatch(CreateTensorFromGridInput(
                    tensor_type="matrix",
                    label=label,
                    color=color
                ))
                self._label_buffer = ""

        # Help text
        imgui.spacing()
        if not matrix_only:
            imgui.text_colored("Note: 'Create Vector' uses first row only", 0.5, 0.5, 0.5, 1.0)

    def _generate_vector_label(self, state: "AppState") -> str:
        """Generate automatic vector label."""
        from state.selectors import get_tensor_count_by_type
        count = get_tensor_count_by_type(state, "vector")
        return f"v{count + 1}"

    def _generate_matrix_label(self, state: "AppState") -> str:
        """Generate automatic matrix label."""
        from state.selectors import get_tensor_count_by_type
        count = get_tensor_count_by_type(state, "matrix")
        return f"M{count + 1}"

"""
Inspector panel for detailed object inspection and editing.
"""

import imgui
from state.selectors import get_selected_vector, get_selected_matrix

from ui.utils import set_next_window_position, set_next_window_size

_SET_WINDOW_POS_FIRST = getattr(imgui, "SET_WINDOW_POS_FIRST_USE_EVER", 0)
_SET_WINDOW_SIZE_FIRST = getattr(imgui, "SET_WINDOW_SIZE_FIRST_USE_EVER", 0)
_WINDOW_RESIZABLE = getattr(imgui, "WINDOW_RESIZABLE", 1)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 1)

from ui.inspectors.inspector_header import _render_header
from ui.inspectors.inspector_coordinates import _render_coordinate_editor
from ui.inspectors.inspector_properties import _render_properties
from ui.inspectors.inspector_computed import _render_computed_properties
from ui.panels.images.images_tab import (
    _render_color_mode_selector,
    _render_image_info_section,
    _render_image_render_options,
    _render_image_tab_selector,
)
from state.actions import AddVector, UpdateMatrix, UpdateMatrixCell


class Inspector:
    def __init__(self):
        self.show_computed_properties = True

    def render(self, state, dispatch, rect):
        """Render inspector panel."""
        if state is None or dispatch is None:
            return

        selected_vector = get_selected_vector(state)
        selected_matrix = get_selected_matrix(state)

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
            elif selected_matrix:
                label = selected_matrix.label or "Matrix"
                imgui.text_disabled(f"({label})")
            else:
                imgui.text_disabled("(No selection)")
            imgui.separator()

            if not selected_vector and not selected_matrix:
                if state.active_mode == "images" and state.current_image is not None:
                    self._render_image_inspector(state, dispatch)
                else:
                    imgui.text_wrapped("Select a vector or matrix to see detailed properties.")
            elif selected_vector:
                if imgui.collapsing_header("Coordinates", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    self._render_coordinate_editor(selected_vector, dispatch)
                if imgui.collapsing_header("Properties", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    self._render_properties(selected_vector, dispatch)
                if self.show_computed_properties and imgui.collapsing_header("Computed", 0)[0]:
                    self._render_computed_properties(selected_vector, state, dispatch)
            else:
                self._render_matrix_details(selected_matrix, state, dispatch)


        imgui.end()
        imgui.pop_style_var(2)

    _render_header = _render_header
    _render_coordinate_editor = _render_coordinate_editor
    _render_properties = _render_properties
    _render_computed_properties = _render_computed_properties

    def _render_matrix_details(self, selected_matrix, state, dispatch):
        rows, cols = selected_matrix.shape
        imgui.text(f"Size: {rows}x{cols}")
        imgui.spacing()
        imgui.text("Label:")
        imgui.same_line()
        imgui.push_item_width(160)
        label_changed, new_label = imgui.input_text(
            "##matrix_label_edit",
            selected_matrix.label or "",
            16
        )
        imgui.pop_item_width()
        if label_changed and dispatch:
            dispatch(UpdateMatrix(id=selected_matrix.id, label=new_label))

        imgui.spacing()
        imgui.text("Values:")
        imgui.spacing()
        imgui.push_item_width(70)
        for r, row in enumerate(selected_matrix.values):
            imgui.push_id(f"row_{r}")
            for c, value in enumerate(row):
                imgui.push_id(f"cell_{c}")
                changed, new_value = imgui.input_float(
                    "##val",
                    float(value),
                    format="%.3f"
                )
                if changed and dispatch:
                    dispatch(UpdateMatrixCell(
                        id=selected_matrix.id,
                        row=r,
                        col=c,
                        value=float(new_value),
                    ))
                imgui.pop_id()
                if c < cols - 1:
                    imgui.same_line()
            imgui.pop_id()
        imgui.pop_item_width()
        imgui.spacing()
        self._render_matrix_vectors(selected_matrix, state, dispatch)

    def _render_matrix_vectors(self, selected_matrix, state, dispatch):
        rows, cols = selected_matrix.shape
        if rows == 0 or cols == 0:
            return
        imgui.text("Column Vectors:")
        for c in range(cols):
            col = [selected_matrix.values[r][c] for r in range(rows)]
            display = ", ".join(f"{v:.2f}" for v in col[:4])
            suffix = "..." if len(col) > 4 else ""
            imgui.text(f"c{c+1} = [{display}{suffix}]")
        imgui.spacing()
        if imgui.button("Add Column Vectors", width=-1) and dispatch:
            base_label = selected_matrix.label or "M"
            for c in range(cols):
                col = tuple(selected_matrix.values[r][c] for r in range(rows))
                dispatch(AddVector(
                    coords=col,
                    color=(0.8, 0.2, 0.2),
                    label=f"{base_label}_c{c+1}",
                ))
        if state and state.input_vector_coords:
            v = state.input_vector_coords
            if len(v) == cols:
                result = []
                for r in range(rows):
                    value = sum(selected_matrix.values[r][c] * v[c] for c in range(cols))
                    result.append(value)
                preview = ", ".join(f"{val:.2f}" for val in result[:4])
                suffix = "..." if len(result) > 4 else ""
                imgui.text(f"M·v = [{preview}{suffix}]")

    def _render_image_inspector(self, state, dispatch):
        imgui.text("Image Inspector")
        imgui.separator()
        _render_image_tab_selector(state, dispatch)

        if state.active_image_tab == "raw":
            _render_color_mode_selector(state, dispatch)
            _render_image_render_options(state, dispatch)
            _render_image_info_section(state, dispatch)
        else:
            imgui.text_disabled("Preprocess view active")

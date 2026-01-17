"""
Sidebar input section for vectors and matrices.
"""

import imgui

from state.actions import AddMatrix, AddVector, SetInputExpression


def _render_input_section(self):
    """Render raw input box for vectors/matrices."""
    if self._section("Input Box", "⌨️", default_open=True):
        if self._state is None or self._dispatch is None:
            imgui.text_disabled("Input unavailable (no state).")
            self._end_section()
            return

        imgui.text_disabled("Vector: 1, 3, 5   Matrix: 1,2; 3,4")
        imgui.push_item_width(-1)
        text = self._state.input_expression
        changed = False
        if hasattr(imgui, "input_text_multiline"):
            changed, text = imgui.input_text_multiline(
                "##input_expression",
                text,
                512,
                0,
                80
            )
        else:
            changed, text = imgui.input_text_with_hint(
                "##input_expression",
                "Enter vector or matrix...",
                text,
                512,
            )
        imgui.pop_item_width()
        if changed:
            self._dispatch(SetInputExpression(expression=text))

        if self._state.input_expression_type == "error":
            imgui.text_colored(self._state.input_expression_error, 0.9, 0.4, 0.4, 1.0)
        elif self._state.input_expression_type == "vector":
            dims = len(self._state.input_vector_coords)
            imgui.text(f"Parsed vector ({dims}D)")
            if imgui.button("Add Vector", width=-1):
                self._dispatch(AddVector(coords=tuple(self._state.input_vector_coords),
                                         color=self._state.input_vector_color,
                                         label=self._state.input_vector_label))
        elif self._state.input_expression_type == "matrix":
            rows = self._state.input_matrix_rows
            cols = self._state.input_matrix_cols
            imgui.text(f"Parsed matrix ({rows}x{cols})")
            if imgui.button("Add Matrix", width=-1):
                values = [list(row) for row in self._state.input_matrix]
                self._dispatch(AddMatrix(values=tuple(tuple(r) for r in values),
                                         label=self._state.input_matrix_label))

        self._end_section()
